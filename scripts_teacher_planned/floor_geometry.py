"""Vectorized combined-room floor lookup; same floor caches and interpolation."""
import json
from pathlib import Path
import numpy as np

def prepare(directory,spacing=.03):
    directory=Path(directory);path=directory/'v4_floor_height_3cm.npz'
    if path.exists():return path
    import trimesh
    from scipy.ndimage import distance_transform_edt
    meta=json.loads((directory/'scene_metadata.json').read_text());arr=np.load(directory/'scene_arrays.npz')
    ids=[o['object_id'] for o in meta['objects'] if o['label']=='floor']
    floor=trimesh.Trimesh(arr['vertices'],arr['faces'][np.isin(arr['face_object_ids'],ids)],process=False)
    used=floor.vertices[np.unique(floor.faces)]
    low=used[:,:2].min(0)-spacing*2;high=used[:,:2].max(0)+spacing*2
    size=np.ceil((high-low)/spacing).astype(int)+1
    xx,yy=np.meshgrid(low[0]+np.arange(size[0])*spacing,low[1]+np.arange(size[1])*spacing,indexing='ij')
    xy=np.stack([xx,yy],-1).reshape(-1,2);height=np.full(len(xy),-np.inf)
    for start in range(0,len(xy),2048):
        q=xy[start:start+2048];origin=np.c_[q,np.full(len(q),used[:,2].max()+1)]
        direction=np.zeros_like(origin);direction[:,2]=-1
        points,ray,_=floor.ray.intersects_location(origin,direction,multiple_hits=True)
        if len(points):np.maximum.at(height,ray+start,points[:,2])
    height=height.reshape(tuple(size));valid=np.isfinite(height)
    if not valid.any():raise RuntimeError('The annotated floor has no ray hits')
    nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
    filled=height[tuple(nearest)]
    np.savez(path,height=filled.astype(np.float32),valid=valid,origin=low.astype(np.float32),spacing=spacing)
    print('V4_FLOOR',str(path),size.tolist(),float(valid.mean()),flush=True)
    return path

class NearestOriginIndex:
    """Exact nearest grid origin, with a bounded fallback for a sparse last row.

    Normal queries inspect four enclosing grid corners. A query whose nearest
    corner is missing (or an irregular layout) uses the full original search.
    """
    def __init__(self,origins):
        import torch
        self.origins=origins
        xy=origins[:,:2]
        self.x=torch.unique(xy[:,0],sorted=True).contiguous()
        self.y=torch.unique(xy[:,1],sorted=True).contiguous()
        self.grid=None
        self.sparse=True
        # Avoid allocating an N x N index for arbitrary non-grid origins.
        if len(self.x)*len(self.y)<=4*len(origins):
            ix=torch.searchsorted(self.x,xy[:,0].contiguous())
            iy=torch.searchsorted(self.y,xy[:,1].contiguous())
            cells=ix*len(self.y)+iy
            if len(torch.unique(cells))==len(origins):
                self.grid=torch.full((len(self.x),len(self.y)),-1,device=origins.device,dtype=torch.long)
                self.grid[ix,iy]=torch.arange(len(origins),device=origins.device)
                self.sparse=len(self.x)*len(self.y)!=len(origins)

    def __call__(self,q):
        import torch
        if self.grid is None:
            return torch.cdist(q,self.origins[:,:2]).argmin(-1)
        upper_x=torch.searchsorted(self.x,q[:,0].contiguous())
        upper_y=torch.searchsorted(self.y,q[:,1].contiguous())
        ix=torch.stack([(upper_x-1).clamp(0,len(self.x)-1),upper_x.clamp(0,len(self.x)-1)],-1)
        iy=torch.stack([(upper_y-1).clamp(0,len(self.y)-1),upper_y.clamp(0,len(self.y)-1)],-1)
        cx=ix[:,:,None].expand(-1,2,2).reshape(-1,4)
        cy=iy[:,None,:].expand(-1,2,2).reshape(-1,4)
        ids=self.grid[cx,cy]
        delta_x=q[:,0,None]-self.x[cx];delta_y=q[:,1,None]-self.y[cy]
        distance=delta_x.square()+delta_y.square()
        minimum=distance.amin(-1,keepdim=True)
        valid_nearest=(distance==minimum)&(ids>=0)
        sentinel=len(self.origins)
        nearest=torch.where(valid_nearest,ids,sentinel).amin(-1)
        missing=nearest==sentinel
        if self.sparse and missing.any():
            nearest[missing]=torch.cdist(q[missing],self.origins[:,:2]).argmin(-1)
        return nearest


class MultiFloorLookup:
    def __init__(self,paths,env_origins,room_offsets,device):
        import torch
        self.count=len(paths)
        self.centers=(env_origins[:,None]+room_offsets[None]).reshape(-1,3)
        self.index=NearestOriginIndex(self.centers)
        arrays=[np.load(path) for path in paths]
        shapes=np.array([a['height'].shape for a in arrays])
        maximum=shapes.max(0)
        # Padding is never queried: per-room shapes clamp each interpolation.
        heights=np.zeros((self.count,*maximum),dtype=np.result_type(*[a['height'].dtype for a in arrays]))
        for i,a in enumerate(arrays):
            h=a['height'];heights[i,:h.shape[0],:h.shape[1]]=h
        self.height=torch.as_tensor(heights,device=device)
        self.origins=torch.as_tensor(np.stack([a['origin'] for a in arrays]),device=device)
        spacings=[float(a['spacing']) for a in arrays]
        self.common_spacing=spacings[0] if all(s==spacings[0] for s in spacings) else None
        self.spacing=torch.tensor(spacings,device=device,dtype=self.height.dtype)
        self.shapes=torch.as_tensor(shapes,device=device)
        for a in arrays:a.close()
    def __call__(self,locations):
        import torch
        shape=locations.shape[:-1];q=locations.reshape(-1,locations.shape[-1])
        index=self.index(q[:,:2]);room=index%self.count
        centers=self.centers[index]
        delta=q[:,:2]-centers[:,:2]-self.origins[room]
        # Preserve the original Python-scalar divide for standard 3 cm caches.
        # CUDA may use a different arithmetic path for tensor denominators.
        xy=delta/self.common_spacing if self.common_spacing is not None else delta/self.spacing[room,None]
        sizes=self.shapes[room]
        x=torch.minimum(xy[:,0].clamp_min(0),sizes[:,0]-1)
        y=torch.minimum(xy[:,1].clamp_min(0),sizes[:,1]-1)
        i=torch.minimum(x.long(),sizes[:,0]-2)
        j=torch.minimum(y.long(),sizes[:,1]-2)
        dx=x-i;dy=y-j
        z=(1-dx)*(1-dy)*self.height[room,i,j]+dx*(1-dy)*self.height[room,i+1,j]+(1-dx)*dy*self.height[room,i,j+1]+dx*dy*self.height[room,i+1,j+1]
        z+=centers[:,2]
        return z.reshape(*shape,1) if locations.ndim==2 else z.reshape(shape)

class FloorLookup:
    def __init__(self,path,env_origins,device):
        import torch
        a=np.load(path);self.height=torch.as_tensor(a['height'],device=device)
        self.origin=torch.as_tensor(a['origin'],device=device)
        self.spacing=float(a['spacing']);self.env_origins=env_origins
        self.index=NearestOriginIndex(env_origins)
    def __call__(self,locations):
        import torch
        shape=locations.shape[:-1];q=locations.reshape(-1,locations.shape[-1])
        idx=self.index(q[:,:2])
        z=self.at_origins(q,idx)
        return z.reshape(*shape,1) if locations.ndim==2 else z.reshape(shape)
    def at_origins(self,q,idx):
        xy=(q[:,:2]-self.env_origins[idx,:2]-self.origin)/self.spacing
        x=xy[:,0].clamp(0,self.height.shape[0]-1);y=xy[:,1].clamp(0,self.height.shape[1]-1)
        i=x.long().clamp_max(self.height.shape[0]-2);j=y.long().clamp_max(self.height.shape[1]-2)
        dx=x-i;dy=y-j
        z=(1-dx)*(1-dy)*self.height[i,j]+dx*(1-dy)*self.height[i+1,j]+(1-dx)*dy*self.height[i,j+1]+dx*dy*self.height[i+1,j+1]
        z+=self.env_origins[idx,2]
        return z

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('scene_dir');a=p.parse_args();prepare(a.scene_dir)
