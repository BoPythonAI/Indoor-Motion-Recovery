"""Preserve the original task goal when the input reference endpoint is planned.

Called only by this independent runtime, after original reset computes planned
goal/deadline and BEFORE previous-distance/history/curriculum construction.
"""
import json
from pathlib import Path

import torch
from motion_reference import file_sha256


PLAN_VERSION = 'goal_preserving_xy_reference_shift_v1'
EXPERIMENTAL_STATUS = 'PROPOSED_FOR_PHYSICS_WITH_FLOOR_PROXY_WARNING'
ACCEPTED = {'PLANNED_PENDING_PHYSICS', 'IDENTITY_ORIGINAL_TERMINAL_GEOMETRY_CLEAR', EXPERIMENTAL_STATUS}


def validate_plan_status(manifest):
    """Explicit experimental export is runnable, but never geometry-qualified."""
    if manifest.get('plan_version') != PLAN_VERSION or manifest.get('status') not in ACCEPTED:
        raise ValueError('Rejected or unrecognized reference plans are not deployable')
    entries = [manifest] + manifest.get('sources', [])
    for entry in entries:
        if entry.get('status') not in ACCEPTED:
            raise ValueError('Rejected or unrecognized source plan is not deployable')
        if entry.get('status') == EXPERIMENTAL_STATUS:
            if entry.get('geometry_qualified') is not False or entry.get('experimental_export') is not True:
                raise ValueError('Experimental plans require explicit geometry_qualified=false and experimental_export=true')
    if any(s.get('status') == EXPERIMENTAL_STATUS for s in manifest.get('sources', [])):
        if manifest.get('status') != EXPERIMENTAL_STATUS:
            raise ValueError('Experimental source must be marked experimental at manifest root')
    return manifest.get('status') == EXPERIMENTAL_STATUS


def checked_file(path, expected, label):
    actual = file_sha256(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(label + ' SHA256 does not match the planning manifest')
    return actual


class OriginalGoalContract:
    def __init__(self, task):
        cfg, env = task.cfg, task.env
        if cfg.get('reference_planning_version') != PLAN_VERSION:
            raise ValueError('Planned runtime requires explicit reference_planning_version')
        path = Path(cfg['reference_plan_manifest']).resolve()
        manifest = json.loads(path.read_text())
        manifest_sha = file_sha256(path)
        expected = cfg.get('reference_plan_manifest_sha256')
        if expected and expected != manifest_sha:
            raise ValueError('Planning manifest changed')
        self.experimental = validate_plan_status(manifest)
        sources = manifest.get('sources', [])
        if len(sources) != len(task.rows):
            raise ValueError('Planning sources must match every loaded motion metadata row')
        for source, row in zip(sources, task.rows):
            if str(source['source_id']) != str(row['motion_id']) or source.get('status') not in ACCEPTED:
                raise ValueError('Planning source order/identity/status does not match metadata')
            if int(source['source_frames']) != int(row['source_frames']):
                raise ValueError('Original source_frames may not change')
        original_path = Path(cfg.get('reference_original_bank', manifest['original_bank'])).resolve()
        checked_file(original_path, manifest['original_bank_sha256'], 'Original bank')
        checked_file(cfg['motion_bank'], manifest['planned_bank_sha256'], 'Actual planned bank')
        checked_file(cfg['motion_metadata'], manifest['original_metadata_sha256'], 'Original metadata')
        if manifest.get('mjcf_sha256') and cfg.get('surface_mjcf_sha256') != manifest['mjcf_sha256']:
            raise ValueError('Actual collision MJCF differs from the planning geometry')
        if len(task.rows) == 1:
            scene = Path(task.rows[0]['scene_dir'])
            for key, name in [('scene_metadata_sha256','scene_metadata.json'),
                              ('floor_cache_sha256','v4_floor_height_3cm.npz')]:
                if manifest.get(key):
                    checked_file(scene/name, manifest[key], name)
        device = task.device
        self.source_frames = torch.tensor([s['source_frames'] for s in sources],device=device,dtype=torch.long)
        self.total_frames = torch.tensor([s['total_frames'] for s in sources],device=device,dtype=torch.long)
        self.motion_dt = torch.tensor([s['motion_dt'] for s in sources],device=device,dtype=env.motion_lib.motion_dt.dtype)
        self.motion_length = torch.tensor([s['motion_length'] for s in sources],device=device,dtype=env.motion_lib.motion_lengths.dtype)
        self.original = torch.tensor([s['original_goal_world_in_scene'] for s in sources],device=device,dtype=task.goal.dtype)
        self.planned = torch.tensor([s['planned_reference_goal_world_in_scene'] for s in sources],device=device,dtype=task.goal.dtype)
        if self.original.shape != (len(sources),3) or self.planned.shape != self.original.shape:
            raise ValueError('Each original/planned scene goal must have three coordinates')
        if not bool(torch.isfinite(self.original).all() & torch.isfinite(self.planned).all()):
            raise ValueError('Nonfinite goal coordinate')
        self.validate_timing(task)
        # Load original raw body positions once. Scene/world CSV coordinates are
        # only diagnostics: regrouping float32 additions changes controller input.
        original_bank = torch.load(original_path, map_location='cpu', weights_only=False)
        for key, actual in [('motion_num_frames',env.motion_lib.motion_num_frames),
                            ('motion_dt',env.motion_lib.motion_dt),
                            ('motion_lengths',env.motion_lib.motion_lengths)]:
            if not torch.equal(original_bank[key].to(actual.device),actual):
                raise ValueError('Original bank timing differs from loaded library: '+key)
        ends = original_bank['length_starts'].long()+original_bank['motion_num_frames'].long()-1
        raw = original_bank['gts'][ends].clone().to(device=device)
        if raw.shape != (len(sources),24,3) or raw.dtype != task.goal.dtype or not bool(torch.isfinite(raw).all()):
            raise ValueError('Original bank must contain finite, native-dtype terminal 24-body positions')
        self.original_terminal_raw = raw
        self.identity_sources = torch.tensor([s['status']=='IDENTITY_ORIGINAL_TERMINAL_GEOMETRY_CLEAR'
                                             for s in sources],device=device,dtype=torch.bool)
        loaded_ends = env.motion_lib.length_starts+env.motion_lib.motion_num_frames-1
        if not torch.equal(env.motion_lib.gts[loaded_ends][self.identity_sources],raw[self.identity_sources]):
            raise ValueError('Identity source terminal body positions differ from original bank')
        del original_bank
        self.identity = dict(reference_planning_version=PLAN_VERSION,reference_plan_manifest=str(path),
            reference_plan_status=manifest['status'],experimental_export=self.experimental,
            geometry_qualified=manifest.get('geometry_qualified'),manifest_verbatim=manifest,
            reference_plan_manifest_sha256=manifest_sha,original_bank=str(original_path),
            original_bank_sha256=manifest['original_bank_sha256'],
            planned_bank=str(Path(cfg['motion_bank']).resolve()),planned_bank_sha256=manifest['planned_bank_sha256'],
            original_metadata_sha256=manifest['original_metadata_sha256'],
            stop_reference_version=cfg['stop_reference_version'],sources=sources,
            goal_semantics='identity preserves existing goal bitwise; changed reference uses original raw terminal + original runtime terrain offset',
            goal_numeric_contract='original_bank_terminal_runtime_order_v2',
            mutation_point='v5_task.reset after original goal/deadline, before start_step/prev_distance/history')
        cfg.update(reference_plan_manifest_sha256=manifest_sha,
            reference_plan_status=manifest['status'],reference_plan_experimental_export=self.experimental,
            reference_plan_geometry_qualified=manifest.get('geometry_qualified'),
            reference_goal_numeric_contract='original_bank_terminal_runtime_order_v2',
            reference_original_bank=str(original_path),reference_original_bank_sha256=manifest['original_bank_sha256'],
            reference_planned_bank_sha256=manifest['planned_bank_sha256'],
            original_goal_world_in_scene=self.original.cpu().tolist(),
            planned_reference_goal_world_in_scene=self.planned.cpu().tolist())
        task.original_task_goal = torch.zeros_like(task.goal)
        task.planned_reference_goal = torch.zeros_like(task.goal)
        output = Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
        identity_path = output/'reference_goal_contract.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != self.identity:
                raise ValueError('Output directory belongs to a different reference planning contract')
        else:
            identity_path.write_text(json.dumps(self.identity,indent=2)+'\n')
        self.log_path = output/'reference_goal_resets.jsonl'
        self.reset_number = 0

    def validate_timing(self, task):
        lib = task.env.motion_lib
        if not torch.equal(task.source_frames,self.source_frames):
            raise ValueError('Task source_frames changed')
        if not torch.equal(lib.motion_num_frames,self.total_frames):
            raise ValueError('Planned bank changed total frame counts')
        if not torch.allclose(lib.motion_dt,self.motion_dt,atol=1e-7,rtol=0):
            raise ValueError('Planned bank changed motion dt')
        if not torch.allclose(lib.motion_lengths,self.motion_length,atol=1e-6,rtol=0):
            raise ValueError('Planned bank changed original motion lengths')

    @torch.no_grad()
    def apply(self, task, ids, mids):
        self.validate_timing(task)
        env = task.env
        original_deadline = torch.round(self.motion_length[mids]/env.dt).long()
        if not torch.equal(task.deadline[ids],original_deadline):
            raise ValueError('Original task deadline changed')
        scene_ids = env.v4_row_scene[mids]
        world_translation = env.simulator._scene.env_origins[ids]+env.v4_room_offsets[scene_ids]
        planned_actual = task.goal[ids].clone()
        expected_planned = self.planned[mids]+world_translation
        if not torch.allclose(planned_actual,expected_planned,atol=1e-4,rtol=0):
            error = float((planned_actual-expected_planned).abs().max())
            raise ValueError(f'Actual V5 planned endpoint differs from manifest (max error {error:.6g} m)')
        identity = self.identity_sources[mids]
        has_changed = not bool(identity.all())
        original_world = planned_actual.clone()
        if has_changed:
            raw = self.original_terminal_raw[mids]
            # Match get_motion_state(mids, lengths), including its same-frame
            # interpolation arithmetic instead of assuming multiplication is exact.
            f0,f1,blend = env.motion_lib._calc_frame_blend_from_id_and_time(
                mids,env.motion_lib.motion_lengths[mids])
            end = env.motion_lib.motion_num_frames[mids]-1
            if not torch.equal(f0,end) or not torch.equal(f1,end):
                raise ValueError('Original goal query must select the terminal frame')
            weight = blend[:,None,None]
            ref_pos = (1.0-weight)*raw+weight*raw
            if task.scene_dir:
                offset = env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(ref_pos,env_ids=ids)
                computed = ref_pos[:,0]+offset[:,0]
            else:
                computed = ref_pos[:,0]+env.respawn_root_offset[ids]
            original_world[~identity] = computed[~identity]
        expected_original = self.original[mids]+world_translation
        if not torch.allclose(original_world,expected_original,atol=1e-4,rtol=0):
            error = float((original_world-expected_original).abs().max())
            raise ValueError(f'Original runtime endpoint differs from manifest (max error {error:.6g} m)')
        # This is the only mutation of task semantics. It restores the original
        # goal; all source_frames/lengths/deadlines/force/collision rules stay put.
        task.planned_reference_goal[ids] = planned_actual
        task.original_task_goal[ids] = original_world
        # Identity environments receive no task.goal write at all.
        if has_changed:
            task.goal[ids[~identity]] = original_world[~identity]
        if not torch.equal(task.deadline[ids],original_deadline):
            raise RuntimeError('Goal injection unexpectedly changed deadline')
        self.reset_number += 1
        record=dict(reset_number=self.reset_number,env_ids=ids.cpu().tolist(),motion_ids=mids.cpu().tolist(),
            reference_plan_status=self.identity['reference_plan_status'],experimental_export=self.experimental,
            geometry_qualified=self.identity['geometry_qualified'],
            identity_goal_preserved=identity.cpu().tolist(),
            goal_numeric_contract=self.identity['goal_numeric_contract'],
            original_goal_world=original_world.cpu().tolist(),planned_reference_goal_world=planned_actual.cpu().tolist(),
            source_frames=self.source_frames[mids].cpu().tolist(),deadline=original_deadline.cpu().tolist())
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(record)+'\n')


@torch.no_grad()
def apply_original_goal(task, ids, mids):
    contract = getattr(task,'_original_goal_contract',None)
    if contract is None:
        contract = task._original_goal_contract = OriginalGoalContract(task)
    contract.apply(task,ids,mids)
