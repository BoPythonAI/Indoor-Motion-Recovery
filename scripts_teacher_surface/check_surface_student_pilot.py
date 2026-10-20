"""Synthetic CPU dry run for the preparation tool; no jobs are launched."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import torch
from collections import Counter

from motion_reference import file_sha256
from collection_sampling import BalancedCases
from prepare_surface_student_pilot import prepare, qualify, FEATURE_VERSION
from teacher_factory import POLICY_KIND
from warm_policy import WarmResidualPolicy
from surface_contract import GEOMETRY_VERSION


def run():
    runtime=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='warm_pilot_') as directory:
        root=Path(directory)
        rows=[dict(motion_id=f'train_{i:02d}',family=f'family_{min(i,9)}',split='train',
                   scene_dir=str(root/f'scene_{i:02d}'),source_frames=70+i) for i in range(11)]
        metadata=root/'train.json';metadata.write_text(json.dumps(rows))
        bank=root/'train.pt';bank.write_bytes(b'synthetic bank, never simulated')
        flat=root/'flat.pt';flat.write_bytes(b'synthetic frozen flat')
        terrain=root/'terrain.pt';terrain.write_bytes(b'synthetic frozen terrain')
        mjcf=root/'robot.xml';mjcf.write_bytes(b'synthetic original collision XML identity')
        prior=dict(kind='official_terrain_prior',flat_checkpoint=str(flat),flat_sha256=file_sha256(flat),
                   terrain_checkpoint=str(terrain),terrain_sha256=file_sha256(terrain))
        teacher=dict(feature_version=FEATURE_VERSION,stop_reference_version='original_freeze_v5_dynamic',
            geometry_version=GEOMETRY_VERSION,surface_mjcf_path=str(mjcf),surface_mjcf_sha256=file_sha256(mjcf),
            motion_bank=str(bank),motion_bank_sha256=file_sha256(bank),motion_metadata=str(metadata),
            motion_metadata_sha256=file_sha256(metadata),scene_specs=[dict(scene_dir=r['scene_dir'],forbidden_id=1) for r in rows],
            tracker_checkpoint=str(flat),terrain_checkpoint=str(terrain),adaptive_phase=True,
            reference_placement='signed_foot_support_per_future_v1',
            reward_regularizer='0.5 * mean((actual_total - composed_frozen_terrain_total)^2)',
            success_bonus=250.,stationary_weight=1.,power_weight=.01)
        model=WarmResidualPolicy(8,width=16)
        checkpoint=root/'selected.pt'
        torch.save(dict(policy_kind=POLICY_KIND,policy=model.state_dict(),model_config=model.model_config,
                        geometry_version=GEOMETRY_VERSION,surface_mjcf_sha256=file_sha256(mjcf),
                        config=teacher,frozen_prior=prior,feature_version=FEATURE_VERSION,update=150),checkpoint)
        ids=[f'dev_{i}' for i in range(5)]
        counts=[[8,8,8,7,7],[8,6,6,6,6],[8,4,4,4,4]]
        per={source:{str(bw):dict(success=counts[j][i],total=8) for j,bw in enumerate([0.,.45,.75])}
             for i,source in enumerate(ids)}
        report=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),source_ids=ids,
            summary={str(bw):dict(success=sum(counts[j]),total=40) for j,bw in enumerate([0.,.45,.75])},
            per_source=per)
        summary=root/'summary.json';summary.write_text(json.dumps(report))
        args=argparse.Namespace(teacher_checkpoint=str(checkpoint),evaluation_summary=str(summary),
            output=str(root/'pilot'),root=str(root),teacher_runtime=str(runtime),
            student_runtime=str(runtime.parent/'scripts_student_surface'),python='python',collect_timeout=1800)
        manifest=prepare(args)
        assert manifest['total_physical_attempts']==1200 and manifest['num_envs']==80
        assert [job['label'] for job in manifest['collection_jobs']]==['full','takeover','late_takeover']
        assert manifest['families']==10 and len(manifest['sources'])==11
        assert manifest['motion_bank_sha256']==teacher['motion_bank_sha256']
        assert manifest['teacher_provenance']['teacher_checkpoint_sha256']==file_sha256(checkpoint)
        assert manifest['geometry_version']==GEOMETRY_VERSION and manifest['surface_mjcf_sha256']==file_sha256(mjcf)
        expected=POLICY_KIND+':'+hashlib.sha256(json.dumps(manifest['teacher_provenance'],sort_keys=True).encode()).hexdigest()
        assert manifest['teacher_identity']==expected and manifest['teacher_update']==150
        for job in manifest['collection_jobs']:
            cfg=json.loads(Path(job['config']).read_text())
            assert cfg['fixed_attempts']==400 and cfg['checkpoint']==str(checkpoint)
            assert cfg['max_attempts']==400 and job['attempts']==400 and job['num_envs']==80
            assert cfg['scene_specs']==teacher['scene_specs'] and cfg['success_bonus']==250.
            balance=job['expected_attempt_balance']['counts']
            assert set(balance['family'].values())=={40}
            assert balance['strength']=={'0.0':80,'0.45':160,'0.75':160}
            assert set(balance['direction'].values())=={100} and set(balance['phase'].values())=={200}
            phases=[.9,1.0] if job['label']=='late_takeover' else [.3,.6]
            seed={'full':101,'takeover':103,'late_takeover':107}[job['label']]
            assert cfg['collection_phases']==phases and cfg['collection_seed']==seed and cfg['seed']==seed
            assert job['expected_attempt_balance']['phases']==phases
            assert balance['phase']=={str(phase):200 for phase in phases}
            assert cfg['collection_directions']==[0,1,2,3]
            assert cfg['collection_strength_schedule']==[0.,.45,.75,.45,.75]
            assert job['expected_attempt_balance']['cycle_size']==400
            actual_sampler=BalancedCases(rows,seed,strengths=cfg['collection_strength_schedule'],
                directions=cfg['collection_directions'],phases=cfg['collection_phases'])
            actual_cases=[]
            for _ in range(5):actual_cases.extend(actual_sampler.next_cases(80))
            assert actual_sampler.report()==job['expected_attempt_balance']
            joint=Counter((rows[case['motion']]['family'],case['bw'],case['direction'],case['phase']) for case in actual_cases)
            assert len(joint)==10*3*4*2
            assert all(count==(1 if key[1]==0 else 2) for key,count in joint.items())
            if job['label'] in ('takeover','late_takeover'):
                assert cfg['rollin_method']=='noisyteacher' and cfg['rollin_noise']==.012
                assert [item['after_push_end_steps'] for item in cfg['takeover_schedule']]==[-6,0,6,12]
            else:
                assert cfg['collector']=='full' and cfg['rollin_method']=='teacher'
        commands=(Path(args.output)/'commands.sh').read_text().splitlines()
        train=[shlex.split(line) for line in commands if 'train_student.py' in line]
        assert len(train)==2
        for tokens in train:
            assert tokens[tokens.index('--steps')+1]=='5000'
            assert tokens[tokens.index('--batch-size')+1]=='128'
            assert tokens[tokens.index('--sampling')+1]=='balanced'
            assert '--ema-decay' in tokens and '--ema-power' in tokens and '--config' not in tokens
            streams=tokens[tokens.index('--data')+1:tokens.index('--output')]
            assert [Path(path).name for path in streams]==['full','takeover','late_takeover']
        assert train[0][train[0].index('--data')+1:train[0].index('--output')]==train[1][train[1].index('--data')+1:train[1].index('--output')]
        failures=[]
        failed=copy.deepcopy(report);failed['summary']['0.0']['success']=34
        failed['per_source'][ids[0]]['0.0']['success']=4;failures.append(failed)
        failed=copy.deepcopy(report);failed['checkpoint_sha256']='0'*64;failures.append(failed)
        failed=copy.deepcopy(report);failed['summary']['0.75']['total']=48;failures.append(failed)
        failed=copy.deepcopy(report);failed['source_ids'][0]='train_00';failures.append(failed)
        for failed in failures:
            try:qualify(failed,file_sha256(checkpoint),manifest['sources'])
            except ValueError:pass
            else:raise AssertionError('An unqualified/mismatched evaluation was accepted')
    print('WARM_PILOT_CPU_PASSED: qualified_exact_weights; 11_sources_10_families; 400_plus_400_plus_400_attempts; '
          '80_envs; exact_balance; actual_factory_identity; unchanged_bank_and_scenes; same_data_BC_diffusion_EMA; '
          'late_phases_0.9_1.0_seed107; phase_reports_match_each_stream; unqualified_100_rejected; no_jobs_started')


if __name__=='__main__':run()
