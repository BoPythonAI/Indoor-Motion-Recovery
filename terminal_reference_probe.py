"""Read-only terminal-pose geometry using the installed V5 reference placement.

Call probe(task, output_path) after construction of RecoveryTask and scene setup.
No motion buffers, reference hooks, controls, criteria or simulation state change.
"""
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

FOOT_NAMES = ('L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe')


def as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, 'torch', None)
    if tensor is not None:
        return tensor
    import warp as wp
    return wp.to_torch(value)


def foot_geometry(env, mjcf_path):
    """Exact original foot box corners in link coordinates, including geom pose."""
    from protomotions.utils import rotations as rot
    xml = ET.parse(mjcf_path).getroot()
    signs = torch.tensor(list(itertools.product([-1., 1.], repeat=3)), device=env.device)
    corners = []
    for name in FOOT_NAMES:
        body = next(b for b in xml.findall('.//body') if b.attrib['name'] == name)
        geoms = body.findall('geom')
        if len(geoms) != 1 or geoms[0].attrib['type'] != 'box':
            raise ValueError(f'{name}: expected original single-box collision geometry')
        geom = geoms[0].attrib
        tensor = lambda key, default: torch.tensor([float(v) for v in geom.get(key, default).split()], device=env.device)
        size, pos, quat_wxyz = tensor('size', ''), tensor('pos', '0 0 0'), tensor('quat', '1 0 0 0')
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        corners.append(rot.quat_rotate(quat_xyzw[None].expand(8, -1), signs*size, True)+pos)
    names = env.robot_config.kinematic_info.body_names
    return [names.index(name) for name in FOOT_NAMES], torch.stack(corners)


def world_corners(pos, quat, local):
    from protomotions.utils import rotations as rot
    frames = pos.shape[0]
    q = quat[:, :, None].expand(-1, -1, 8, -1).reshape(-1, 4)
    p = local[None].expand(frames, -1, -1, -1).reshape(-1, 3)
    return rot.quat_rotate(q, p, True).reshape(frames, 4, 8, 3)+pos[:, :, None]


def box_aabb_margin(corners, bounds):
    """Signed OBB-to-AABB margin: exact convex distance, SAT depth if intersecting.

    The foot is the true oriented collision box. The obstacle is its metadata
    AABB, not its triangle mesh; negative is overlap with that conservative box.
    """
    from scipy.optimize import lsq_linear
    c = np.asarray(corners, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.float64)
    center = c.mean(0)
    axes = np.stack([c[4]-c[0], c[2]-c[0], c[1]-c[0]], axis=1)*.5
    half = (b[1]-b[0])*.5
    other = b.mean(0)
    directions = [np.eye(3)[i] for i in range(3)]
    unit = axes/np.linalg.norm(axes, axis=0)[None]
    directions.extend(unit[:, i] for i in range(3))
    directions.extend(np.cross(unit[:, i], np.eye(3)[j]) for i in range(3) for j in range(3))
    gaps = []
    for direction in directions:
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction/norm
        gaps.append(abs(np.dot(other-center, direction))-
                    np.abs(direction@axes).sum()-(half*np.abs(direction)).sum())
    separating_gap = max(gaps)
    if separating_gap <= 0:
        return float(separating_gap)
    matrix = np.concatenate([axes, -np.diag(half)], axis=1)
    fit = lsq_linear(matrix, other-center, bounds=(-1., 1.), tol=1e-12, max_iter=500)
    if not fit.success:
        raise RuntimeError('Foot/obstacle convex-distance solve did not converge')
    return float(np.linalg.norm(matrix@fit.x-(other-center)))


@torch.no_grad()
def probe(task, output_path, source_ids=('011723', '015571'), preceding_frames=15,
          mjcf_path='/root/autodl-tmp/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml'):
    from protomotions.utils import rotations as rot
    from terminal_support_geometry import analyze_terminal_support
    env, lib = task.env, task.env.motion_lib
    if not getattr(env, 'v4_scene_ready', False):
        raise RuntimeError('Probe needs initialized scene-specific V5 placement')
    robot = env.simulator._robot
    common_names = list(env.robot_config.kinematic_info.body_names)
    sim_names = list(robot.body_names)
    sim_to_common = [sim_names.index(name) for name in common_names]
    masses_sim = as_torch(robot.root_physx_view.get_masses()).to(env.device)
    coms_sim = as_torch(robot.root_physx_view.get_coms()).to(env.device)
    if masses_sim.shape != (env.num_envs, len(sim_names)) or coms_sim.shape != (env.num_envs, len(sim_names), 7):
        raise ValueError(f'Unexpected physical mass/local COM layout: {masses_sim.shape}, {coms_sim.shape}')
    masses = masses_sim[:, sim_to_common]
    local_com = coms_sim[:, sim_to_common, :3]
    feet, local_corners = foot_geometry(env, mjcf_path)
    current_motion_ids = env.motion_manager.motion_ids
    selected = []
    for mid, row in enumerate(task.rows):
        if str(row['motion_id']) not in source_ids:
            continue
        matches = (current_motion_ids == mid).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise RuntimeError(f'{row["motion_id"]} has no current environment for its scene placement; run a single-source config')
        selected.append((mid, row, int(matches[0])))
    if not selected:
        raise ValueError(f'No requested source in current motion bank: {source_ids}')
    report = dict(probe='source_terminal_geometry_v1', placement='installed V5 dynamic reference helper',
                  mutation='none', foot_names=FOOT_NAMES, body_names=common_names,
                  mjcf_path=str(mjcf_path), sources=[])
    for mid, row, eid in selected:
        endpoint = int(row['source_frames'])-1
        # Include first synthetic hold frame to expose the exact transition.
        frame_indices = torch.arange(max(0, endpoint-preceding_frames), endpoint+2, device=env.device)
        mids = torch.full_like(frame_indices, mid)
        frames = lib.get_motion_state_exact_frame(mids, frame_indices)
        raw = frames.rigid_body_pos.clone()
        quat = frames.rigid_body_rot.clone()
        env_ids = torch.full_like(frame_indices, eid)
        offset = env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(raw, env_ids=env_ids)
        pos = raw+offset
        com_offset_world = rot.quat_rotate(quat.reshape(-1, 4),
            local_com[eid][None].expand(len(frame_indices), -1, -1).reshape(-1, 3), True).reshape_as(pos)
        body_com = pos+com_offset_world
        com = (body_com*masses[eid][None, :, None]).sum(1)/masses[eid].sum()
        com_vel_body = frames.rigid_body_vel+torch.cross(frames.rigid_body_ang_vel, com_offset_world, dim=-1)
        com_velocity = (com_vel_body*masses[eid][None, :, None]).sum(1)/masses[eid].sum()
        corners = world_corners(pos[:, feet], quat[:, feet], local_corners)
        floor = env.terrain.get_ground_heights(corners.reshape(len(frame_indices), 32, 3)).reshape(-1, 4, 8)
        scene_index = int(env.v4_row_scene[mid])
        spec = task.scenes[scene_index]
        meta_path = Path(spec['scene_dir'])/'scene_metadata.json'
        meta = json.loads(meta_path.read_text())
        obstacle = next(obj for obj in meta['objects'] if obj['object_id'] == spec['forbidden_id'])
        origin = env.simulator._scene.env_origins[eid]+env.v4_room_offsets[scene_index]
        bounds = np.asarray(obstacle['bounds'])+origin.detach().cpu().numpy()
        corners_np, floor_np, com_np = [v.detach().cpu().numpy() for v in (corners, floor, com)]
        support = analyze_terminal_support(corners_np, floor_np, com_np)
        margins = [[box_aabb_margin(foot, bounds) for foot in frame] for frame in corners_np]
        # Check COM frame conversion against the simulator's current physical pose.
        state = env.simulator.get_robot_state()
        actual_offset = rot.quat_rotate(state.rigid_body_rot[eid], local_com[eid], True)
        actual_com = ((state.rigid_body_pos[eid]+actual_offset)*masses[eid, :, None]).sum(0)/masses[eid].sum()
        physical_com = as_torch(robot.data.body_com_pose_w)[eid, :, :3]
        direct_com = (physical_com*masses_sim[eid, :, None]).sum(0)/masses_sim[eid].sum()
        frame_records = []
        for i, number in enumerate(frame_indices.tolist()):
            frame_records.append(dict(frame=number, relative_to_source_end=number-endpoint,
                root_position_world=pos[i, 0].cpu().tolist(), root_velocity=frames.rigid_body_vel[i, 0].cpu().tolist(),
                root_angular_velocity=frames.rigid_body_ang_vel[i, 0].cpu().tolist(),
                mass_weighted_com_world=com[i].cpu().tolist(), mass_weighted_com_velocity=com_velocity[i].cpu().tolist(),
                body_position_raw=raw[i].cpu().tolist(), reference_translation_xyz=offset[i, 0].cpu().tolist(),
                body_position_world=pos[i].cpu().tolist(), body_quat_xyzw=quat[i].cpu().tolist(),
                body_velocity_world=frames.rigid_body_vel[i].cpu().tolist(),
                body_angular_velocity_world=frames.rigid_body_ang_vel[i].cpu().tolist(),
                foot_position_world=pos[i, feet].cpu().tolist(), foot_quat_xyzw=quat[i, feet].cpu().tolist(),
                foot_corners_world=corners_np[i].tolist(), foot_corner_floor_height=floor_np[i].tolist(),
                foot_corner_gap_m=(corners_np[i, :, :, 2]-floor_np[i]).tolist(),
                forbidden_foot_box_to_aabb_margin_m=margins[i], min_forbidden_foot_margin_m=min(margins[i]),
                support=support[i]))
        report['sources'].append(dict(motion_id=row['motion_id'], motion_index=mid, env_id=eid,
            source_frames=int(row['source_frames']), packed_frames=int(lib.motion_num_frames[mid]),
            motion_dt=float(lib.motion_dt[mid]), scene_dir=spec['scene_dir'], scene_metadata=str(meta_path),
            floor_cache_candidates=[str(p) for p in Path(spec['scene_dir']).glob('*floor*')],
            forbidden_object_id=spec['forbidden_id'], forbidden_aabb_world=bounds.tolist(),
            masses_kg_common_order=masses[eid].cpu().tolist(), local_com_m_common_order=local_com[eid].cpu().tolist(),
            physical_com_transform_error_m=float((actual_com-direct_com).norm()),
            frame_records=frame_records))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('TERMINAL_REFERENCE_PROBE', str(path),
          [(source['motion_id'], source['physical_com_transform_error_m']) for source in report['sources']], flush=True)
    return report
