"""Original ScanNet collision meshes and typed per-motion task placement."""
import json
import os
from pathlib import Path

def scene_specs(cfg):
    if cfg.get('scene_specs'):return cfg['scene_specs']
    if cfg.get('scene_dir'):return [dict(scene_dir=cfg['scene_dir'],forbidden_id=cfg['forbidden_id'])]
    return []

def install():
    from scene_heightmap import install as install_heightmap
    install_heightmap()
    cfg=json.loads(Path(os.environ['V4_CONFIG']).read_text());specs=scene_specs(cfg)
    if not cfg.get('obstacle') and not specs:return
    if specs:
        from reference_terrain_fix import install as install_reference_terrain
        install_reference_terrain()
    import isaaclab.sim as su
    from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
    from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
    metas=[json.loads((Path(s['scene_dir'])/'scene_metadata.json').read_text()) for s in specs]
    if specs:
        import torch
        from protomotions.envs.base_env.env import BaseEnv
        rows=json.loads(Path(cfg['motion_metadata']).read_text())
        row_scene=[next(i for i,s in enumerate(specs) if s['scene_dir']==r['scene_dir']) for r in rows]
        align=[json.loads((Path(s['scene_dir'])/'spawn_alignment.json').read_text())['initial_z_offset_m'] for s in specs]
        def fixed_spawn(env,env_ids,ref_state=None,sample_flat=False):
            if not getattr(env,'v4_scene_ready',False):
                env.v4_room_offsets=torch.tensor([[30.*i,0.,0.] for i in range(len(specs))],device=env.device)
                env.v4_row_scene=torch.tensor(row_scene,device=env.device)
                env.v4_translations=torch.tensor([m['motion_to_world_translation'] for m in metas],device=env.device)
                env.v4_align=torch.tensor(align,device=env.device)
                if cfg.get('scene_floor_observation',True):
                    from floor_geometry import prepare,MultiFloorLookup
                    env.terrain.get_ground_heights=MultiFloorLookup([prepare(s['scene_dir']) for s in specs],env.simulator._scene.env_origins,env.v4_room_offsets,env.device)
                env.v4_scene_ready=True
            index=env.v4_row_scene[env.motion_manager.motion_ids[env_ids]]
            offset=env.simulator._scene.env_origins[env_ids]+env.v4_room_offsets[index]+env.v4_translations[index]
            offset[:,2]+=env.v4_align[index]
            env.respawn_root_offset[env_ids]=offset
        BaseEnv.update_respawn_root_offset_by_env_ids=fixed_spawn
    original=IsaacLabSimulator._get_scene_cfg
    def build(sim):
        scene=original(sim)
        if specs:
            scene.terrain=None;scene.env_spacing=20. if len(specs)==1 else 40.*len(specs)
            ground=[];forbidden=[]
            for i,(s,m) in enumerate(zip(specs,metas)):
                base=Path(s['scene_dir']);prefix=f'/World/envs/env_.*/ScanNet{i}'
                asset=AssetBaseCfg(prim_path=prefix,spawn=su.UsdFileCfg(usd_path=str(base/(m['scene_id']+'.usdc'))),
                                   init_state=AssetBaseCfg.InitialStateCfg(pos=(30.*i,0.,0.)))
                setattr(scene,f'scannet{i}',asset)
                floor=[prefix+'/'+o['prim_name'] for o in m['objects'] if o['label']=='floor']
                forbid=[prefix+'/'+o['prim_name'] for o in m['objects'] if o['object_id']==s['forbidden_id']]
                assert len(floor)==1 and len(forbid)==1
                ground.extend(floor);forbidden.extend(forbid)
            filters=ground+forbidden
        else:
            scene.forbidden=RigidObjectCfg(prim_path='/World/envs/env_.*/Forbidden',
                spawn=su.CuboidCfg(size=(.5,.5,1.0),rigid_props=su.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                    collision_props=su.CollisionPropertiesCfg(),mass_props=su.MassPropertiesCfg(mass=10.),
                    visual_material=su.PreviewSurfaceCfg(diffuse_color=(.6,.15,.15))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-10.)))
            filters=['/World/ground/terrain/mesh','/World/envs/env_.*/Forbidden']
        for name in sim.robot_config.contact_bodies:getattr(scene,'contact_sensor_'+name).filter_prim_paths_expr=filters
        return scene
    IsaacLabSimulator._get_scene_cfg=build
