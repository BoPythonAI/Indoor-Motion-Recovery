"""Route scene height-map observations through the installed physical-floor lookup.

Install after IsaacLab import and before constructing the environment. Sampling
points, yaw rotation, point ordering and root-relative height normalization match
ProtoMotions Terrain.get_height_maps / get_height_maps_jit exactly. Only the
height source changes for terrain instances with a custom get_ground_heights.
"""


def get_scene_height_maps(terrain,root_states,env_ids=None,return_all_dims=False):
    import torch
    from protomotions.utils import rotations
    height_points=terrain.height_points if env_ids is None else terrain.height_points[env_ids]
    base_rot=root_states.root_rot
    base_pos=root_states.root_pos
    if height_points.shape[0]!=base_pos.shape[0]:
        raise ValueError('root_states must already contain the selected env_ids, as in TerrainObs')
    points=rotations.quat_apply_yaw(
        base_rot.repeat(1,terrain.num_height_points),height_points,True
    )+base_pos[:,:3].unsqueeze(1)
    heights=terrain.get_ground_heights(points).reshape(len(base_pos),terrain.num_height_points)
    if return_all_dims:
        return torch.cat([points[...,:2],heights[...,None]],dim=-1).clone()
    return (base_pos[:,2:3]-heights).clone()


def install():
    from protomotions.components.terrains.terrain import Terrain
    if getattr(Terrain,'_scene_heightmap_installed',False):
        return
    original=Terrain.get_height_maps

    def get_height_maps(terrain,root_states,env_ids=None,return_all_dims=False):
        # scene_adapter installs MultiFloorLookup on this instance. Before that,
        # and for ordinary procedural terrain, keep the original runtime path.
        if 'get_ground_heights' not in terrain.__dict__:
            return original(terrain,root_states,env_ids=env_ids,return_all_dims=return_all_dims)
        return get_scene_height_maps(terrain,root_states,env_ids,return_all_dims)

    Terrain.get_height_maps=get_height_maps
    Terrain._scene_heightmap_installed=True
