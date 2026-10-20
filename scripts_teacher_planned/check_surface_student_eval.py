"""CPU-only configuration fixture. Never launches student inference or simulation."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import torch
from motion_reference import file_sha256
from prepare_surface_student_eval import prepare


def run():
    runtime=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='warm_student_eval_') as directory:
        root=Path(directory)
        flat=root/'flat.pt';flat.write_bytes(b'flat')
        terrain=root/'terrain.pt';terrain.write_bytes(b'terrain')
        mjcf=root/'robot.xml';mjcf.write_bytes(b'synthetic collision XML identity')
        bank=root/'heldout.pt';bank.write_bytes(b'heldout bank')
        metadata=root/'heldout.json'
        metadata.write_text(json.dumps([dict(motion_id='heldout',scene_dir='room',split='val')]))
        prior=dict(flat_checkpoint=str(flat),terrain_checkpoint=str(terrain),
                   flat_sha256=file_sha256(flat),terrain_sha256=file_sha256(terrain))
        feature='v3_terrain_reference__terrain_correction_v1__foot_clearance_v1'
        target='executed_total_minus_composed_terrain_prior_v1'
        state=dict(policy_kind='terrain_prior_surface_correction_student_v1',feature_version=feature,
            geometry_version='mjcf_foot_obb_forbidden_aabb_sat15_v1',
            surface_mjcf_path=str(mjcf),surface_mjcf_sha256=file_sha256(mjcf),
            action_target_version=target,teacher_provenance={'kind':'terrain_prior_surface_correction_v1'},
            teacher_identity='selected_teacher',teacher_checkpoint_sha256='a'*64,frozen_prior=prior,
            stop_reference_version='original_freeze_v5_dynamic',motion_bank_sha256='b'*64,
            motion_metadata_sha256='c'*64,kind='bc',horizon=6,steps=5000,ema_model={'synthetic':torch.zeros(1)})
        scene=dict(checkpoint='old_template_teacher.pt',teacher_provenance={'stale':True},
            motion_bank=str(bank),motion_metadata=str(metadata),feature_version=feature,
            stop_reference_version='original_freeze_v5_dynamic',scene_specs=[dict(scene_dir='room',forbidden_id=28)],
            tracker_checkpoint=str(flat),terrain_checkpoint=str(terrain),adaptive_phase=True,
            seed=17,success_bonus=250.,stationary_weight=1.,power_weight=.01,
            cases=[dict(motion=0,bw=bw,direction=d,phase=p) for bw in [0.,.45,.75] for d in range(4) for p in [.3,.6]])
        source=root/'scene.json';source.write_text(json.dumps(scene))
        checkpoint=root/'student.pt'
        args=argparse.Namespace(scene_config=str(source),student_checkpoint=str(checkpoint),output=None,
            student_runtime=str(runtime.parent/'scripts_student_surface'),teacher_runtime=str(runtime),
            execute_steps=1,num_envs=24,measure_latency=True)
        for kind,execute in [('bc',1),('diffusion',2)]:
            state['kind']=kind;torch.save(state,checkpoint)
            args.output=str(root/kind);args.execute_steps=execute
            manifest=prepare(args)
            cfg=json.loads(Path(manifest['config']).read_text())
            assert cfg['method']==kind and cfg['execute_steps']==execute and cfg['student_weights']=='ema'
            assert cfg['cases']==scene['cases'] and cfg['scene_specs']==scene['scene_specs']
            assert cfg['success_bonus']==250 and cfg['adaptive_phase'] and cfg['measure_latency']
            assert 'checkpoint' not in cfg and 'teacher_provenance' not in cfg
            assert cfg['action_target_version']==target and manifest['cases_per_strength']==8
            assert manifest['num_envs']==24
            assert cfg['surface_mjcf_sha256']==file_sha256(mjcf)
            assert manifest['motion_bank_sha256']!=state['motion_bank_sha256']
            assert str(bank) in manifest['command'] and str(flat) in manifest['command']
        changed=copy.deepcopy(state);changed['frozen_prior']['terrain_sha256']='d'*64
        torch.save(changed,checkpoint);args.output=str(root/'bad_prior')
        try:prepare(args)
        except ValueError:pass
        else:raise AssertionError('Changed prior accepted')
        assert not Path(args.output).exists()
        scene['cases']=scene['cases'][:-1];source.write_text(json.dumps(scene))
        torch.save(state,checkpoint);args.output=str(root/'bad_cases')
        try:prepare(args)
        except ValueError:pass
        else:raise AssertionError('Changed case set accepted')
    print('WARM_STUDENT_EVAL_CPU_PASSED: BC_diffusion; explicit_EMA; execute1_2; latency; exact24cases; '
          'scene_criteria_preserved; heldout_bank_allowed; frozen_prior_mismatch_rejected; launch_contract')


if __name__=='__main__':run()
