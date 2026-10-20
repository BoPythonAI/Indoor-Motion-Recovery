"""Signed terrain placement for current and every future scene reference pose.

Controlled A/B alternative to reference_alignment_fix; install exactly one.
Uses the original ProtoMotions support-body rule, restricted to foot bodies,
without its nonnegative-height assertion. Original collision meshes stay intact.
"""


def install():
    import torch
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.control.mimic_control import MimicControl
    if getattr(BaseEnv, '_v4_reference_terrain_fixed', False):
        return
    if getattr(BaseEnv, '_v4_reference_alignment_fixed', False):
        raise RuntimeError('Install only one reference placement A/B module')
    original_offset = BaseEnv.get_spawn_to_ref_pose_offset_with_terrain_height_correction
    original_context = MimicControl.populate_context

    def support_height(env, world_pos):
        names = env.robot_config.kinematic_info.body_names
        feet = [names.index(n) for n in ('L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe')]
        q = world_pos[..., feet, :]
        # The height helper accepts a flattened batch of poses, including futures.
        shape = q.shape[:-2]
        flat = q.reshape(-1, len(feet), 3)
        height = env.terrain.get_ground_heights(flat)
        index = (height - flat[..., 2]).argmax(-1)
        return height.gather(1, index[:, None]).squeeze(-1).reshape(shape)

    def scene_offset(env, target_pos, env_ids=None):
        if not getattr(env, 'v4_scene_ready', False):
            return original_offset(env, target_pos, env_ids=env_ids)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        offset = torch.zeros_like(target_pos)
        offset[..., :2] = env.respawn_root_offset[env_ids, None, :2]
        world = target_pos + offset
        # h is a world ground height, not an additional correction on spawn Z.
        # p_world = p_raw + [spawn_x, spawn_y, h(p_raw.xy + spawn.xy)].
        offset[..., 2] = support_height(env, world)[:, None]
        return offset

    def populate_context(control, ctx):
        original_context(control, ctx)
        env = control.env
        if not getattr(env, 'v4_scene_ready', False):
            return
        future = ctx.mimic.future_pos
        # Upstream applies first-future support h0 to ALL future poses. The
        # support-body argmax is invariant to that common Z translation, so we
        # can recover each future support height without another motion query.
        heights = support_height(env, future)
        future[..., 2] += (heights - heights[:, :1])[:, :, None]

    BaseEnv.get_spawn_to_ref_pose_offset_with_terrain_height_correction = scene_offset
    MimicControl.populate_context = populate_context
    BaseEnv._v4_reference_terrain_fixed = True
