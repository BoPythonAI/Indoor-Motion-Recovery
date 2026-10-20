"""Prepare the fixed warm correction pilot. Never starts collection or training."""
import argparse
import copy
import json
from pathlib import Path
import shlex

from collection_sampling import BalancedCases
from motion_reference import file_sha256, resolve_motion_reference
from teacher_factory import read_checkpoint, build_teacher_provenance
from surface_contract import GEOMETRY_VERSION


FEATURE_VERSION = 'v3_terrain_reference__terrain_correction_v1__foot_clearance_v1'
ACTION_TARGET_VERSION = 'executed_total_minus_composed_terrain_prior_v1'
DEFAULT_ROOT = '/root/autodl-tmp/method_v1_20261004/project_v4_recovery'
DEFAULT_PYTHON = '/root/autodl-tmp/IsaacLab/.venv/bin/python'
REQUIRED = {0.:38, .45:32, .75:24}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def strength_rows(rows):
    parsed = {float(key):value for key,value in rows.items()}
    if len(parsed)!=len(rows) or set(parsed)!=set(REQUIRED):
        raise ValueError('Evaluation must contain exactly nominal/0.45/0.75 BW')
    for value in parsed.values():
        if any(type(value.get(key)) is not int for key in ('success','total')):
            raise ValueError('Evaluation success and total must be integer counts')
        if not 0<=value['success']<=value['total']:
            raise ValueError('Invalid evaluation success count')
    return parsed


def qualify(report, checkpoint_sha256, train_ids):
    if report.get('checkpoint_sha256')!=checkpoint_sha256:
        raise ValueError('Five-source evaluation is not bound to this exact teacher checkpoint SHA256')
    ids=report.get('source_ids',[])
    if len(ids)!=5 or len(set(ids))!=5 or set(ids)&set(train_ids):
        raise ValueError('Qualification requires five distinct held-out sources, disjoint from teacher training')
    if set(report.get('per_source',{}))!=set(ids):
        raise ValueError('Qualification needs matching per_source results for all five sources')
    totals=strength_rows(report['summary'])
    per_source=[strength_rows(report['per_source'][source]) for source in ids]
    for bw, minimum in REQUIRED.items():
        if totals[bw]['total']!=40 or any(result[bw]['total']!=8 for result in per_source):
            raise ValueError('Each strength requires exactly five sources x eight cases = 40')
        if sum(result[bw]['success'] for result in per_source)!=totals[bw]['success']:
            raise ValueError('Per-source successes differ from the reported total')
        if totals[bw]['success']<minimum:
            raise ValueError(f'Teacher not qualified: {bw:g} BW {totals[bw]["success"]}/40 < {minimum}/40')
    return dict(passed=True,source_ids=ids,summary=report['summary'],
                thresholds={str(bw):dict(minimum_success=n,total=40) for bw,n in REQUIRED.items()},
                checkpoint_sha256=checkpoint_sha256)


def prepare(args):
    checkpoint=Path(args.teacher_checkpoint).resolve()
    state=read_checkpoint(checkpoint)
    teacher=copy.deepcopy(state['config'])
    teacher.update(checkpoint=str(checkpoint),teacher_kind='terrain_surface',method='terrain_surface')
    if state.get('feature_version')!=FEATURE_VERSION or teacher.get('feature_version')!=FEATURE_VERSION:
        raise ValueError('Pilot requires the warm terrain correction feature version')
    if state.get('geometry_version')!=GEOMETRY_VERSION or teacher.get('geometry_version')!=GEOMETRY_VERSION:
        raise ValueError('Pilot requires the selected surface foot-clearance geometry contract')
    mjcf=Path(teacher['surface_mjcf_path']).resolve()
    if file_sha256(mjcf)!=teacher.get('surface_mjcf_sha256') or state.get('surface_mjcf_sha256')!=teacher['surface_mjcf_sha256']:
        raise ValueError('Actual robot MJCF differs from the qualified surface teacher')
    teacher['surface_mjcf_path']=str(mjcf)
    if teacher.get('stop_reference_version')!='original_freeze_v5_dynamic':
        raise ValueError('This pilot is fixed to the selected original V5 dynamic reference bank')
    reference=resolve_motion_reference(teacher)
    rows=read_json(reference['motion_metadata'])
    if len(rows)!=11 or len({r['family'] for r in rows})!=10 or any(r['split']!='train' for r in rows):
        raise ValueError('Pilot must retain the original eleven train motions / ten families')
    if len({str(r['motion_id']) for r in rows})!=11:
        raise ValueError('Teacher training metadata contains duplicate motion IDs')
    if not teacher.get('scene_specs'):
        raise ValueError('Warm pilot requires the selected teacher real-scene specifications')
    if {r['scene_dir'] for r in rows}!={s['scene_dir'] for s in teacher['scene_specs']}:
        raise ValueError('Teacher training scenes and motion metadata differ')
    prior=copy.deepcopy(state['frozen_prior'])
    for config_key,prior_key in [('tracker_checkpoint','flat'),('terrain_checkpoint','terrain')]:
        path=Path(teacher[config_key]).resolve()
        if file_sha256(path)!=prior[prior_key+'_sha256']:
            raise ValueError('Selected frozen prior bytes differ from warm teacher training')
        teacher[config_key]=str(path)
        prior[prior_key+'_checkpoint']=str(path)
    reference,provenance,identity=build_teacher_provenance(
        teacher,state,prior,state['model_config'],FEATURE_VERSION)
    checkpoint_sha256=provenance['teacher_checkpoint_sha256']
    evaluation=Path(args.evaluation_summary).resolve()
    qualification=qualify(read_json(evaluation),checkpoint_sha256,[str(r['motion_id']) for r in rows])

    root=Path(args.root)
    runtime=Path(args.teacher_runtime or root/'scripts_teacher_surface').resolve()
    student=Path(args.student_runtime or root/'scripts_student_surface').resolve()
    if not (runtime/'launch.sh').is_file() or not (student/'train_student.py').is_file():
        raise ValueError('The selected warm teacher/student runtimes must exist before preparing commands')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new pilot output directory')
    output.mkdir(parents=True,exist_ok=True)
    data=output/'data'
    common=copy.deepcopy(teacher)
    for key in ('resume','updates','validation_motion','validation_metadata','cases','warm_trace',
                'record','hybrid_gate','task_owned_gate','momentum_observer','inactive_zero_student'):
        common.pop(key,None)
    common.update(reference,mode='collect',checkpoint=str(checkpoint),teacher_provenance=provenance,
        teacher_identity=identity,feature_version=FEATURE_VERSION,action_target_version=ACTION_TARGET_VERSION,
        frozen_prior=prior,teacher_checkpoint_sha256=checkpoint_sha256,
        balanced_collection=True,collection_strength_schedule=[0.,.45,.75,.45,.75],
        collection_directions=[0,1,2,3],collection_phases=[.3,.6],fixed_attempts=400,
        max_attempts=400,storage_dtype='float16',max_bw=.75,min_bw=0.)
    jobs=[]
    streams=[('full','full',101,[.3,.6]),('takeover','recovery',103,[.3,.6]),
             ('late_takeover','recovery',107,[.9,1.0])]
    for label,collector,seed,phases in streams:
        sampler=BalancedCases(rows,seed,strengths=common['collection_strength_schedule'],
                              directions=common['collection_directions'],phases=phases)
        sampler.next_cases(400)
        if sampler.cycle_size!=400:
            raise ValueError('Unexpected family balance period; the fixed pilot is 400 attempts per stream')
        cfg=dict(common,output=str(data/label),collector=collector,seed=seed,collection_seed=seed,
                 collection_phases=phases,
                 rollin_schedule=[0],rollin_method='teacher' if collector=='full' else 'noisyteacher')
        if collector=='recovery':
            cfg.update(rollin_noise=.012,takeover_schedule=[dict(name=name,after_push_end_steps=offset)
                for name,offset in [('push_start',-6),('push_end',0),('push_end_plus6',6),('push_end_plus12',12)]])
        path=output/f'collect_{label}.json'
        path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
        jobs.append(dict(label=label,config=str(path),output=cfg['output'],num_envs=80,
                         attempts=400,expected_attempt_balance=sampler.report()))
    train_configs=[]
    commands=['#!/usr/bin/env bash','set -euo pipefail',
              '# Prepared only. This file runs jobs only when explicitly executed.',
              'export SURFACE_TEACHER_SCRIPTS='+shlex.quote(str(runtime))]
    for job in jobs:
        command=['timeout','--signal=TERM','--kill-after=30s',str(args.collect_timeout)+'s','bash',
                 str(runtime/'launch.sh'),job['config'],'80',reference['motion_bank'],teacher['tracker_checkpoint']]
        commands.append(shlex.join(command))
    for kind in ['bc','diffusion']:
        train=dict(data=[job['output'] for job in jobs],output=str(output/'checkpoints'/kind),
            kind=kind,steps=5000,batch_size=128,horizon=6,seed=109,lr=2e-4,sampling='balanced',
            ema_decay=.9999,ema_power=.75,device='cuda',feature_version=FEATURE_VERSION,
            action_target_version=ACTION_TARGET_VERSION,teacher_checkpoint_sha256=checkpoint_sha256)
        path=output/f'train_{kind}.json'
        path.write_text(json.dumps(train,indent=2)+'\n',encoding='utf-8')
        train_configs.append(str(path))
        commands.append(shlex.join([args.python,str(student/'train_student.py'),'--data',*train['data'],
            '--output',train['output'],'--kind',kind,'--steps','5000','--batch-size','128','--horizon','6',
            '--seed','109','--lr','0.0002','--sampling','balanced','--ema-decay','0.9999',
            '--ema-power','0.75','--device','cuda']))
    manifest=dict(status='PREPARED_NOT_EXECUTED',teacher_checkpoint=str(checkpoint),
        teacher_checkpoint_sha256=checkpoint_sha256,teacher_update=int(state['update']),
        teacher_identity=identity,teacher_provenance=provenance,frozen_prior=prior,
        evaluation_summary=str(evaluation),evaluation_summary_sha256=file_sha256(evaluation),
        qualification=qualification,feature_version=FEATURE_VERSION,action_target_version=ACTION_TARGET_VERSION,
        geometry_version=GEOMETRY_VERSION,surface_mjcf_path=str(mjcf),surface_mjcf_sha256=teacher['surface_mjcf_sha256'],
        sources=[r['motion_id'] for r in rows],families=10,num_envs=80,
        attempts_per_stream=400,total_physical_attempts=1200,
        late_takeover_reason='Cover the diagnosed endpoint/hold states already included in surface teacher training; no successful demonstrations are presumed',
        successful_episodes='Only observed successes are saved; attempt quotas are not success quotas',
        collection_jobs=jobs,train_configs=train_configs,student_updates_per_kind=5000,
        student_batch_size=128,student_default_weights='ema',runtime_directory=str(runtime),
        student_runtime_directory=str(student),
        causal_inputs='Collection timing metadata never enters controller inputs; current state/history only',
        label='float32 executed_total minus same-state composed frozen terrain prior',**reference)
    (output/'pilot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (output/'commands.sh').write_text('\n'.join(commands)+'\n',encoding='utf-8')
    print('SURFACE_STUDENT_PILOT_PREPARED',str(output),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--teacher-checkpoint',required=True)
    parser.add_argument('--evaluation-summary',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--teacher-runtime')
    parser.add_argument('--student-runtime')
    parser.add_argument('--python',default=DEFAULT_PYTHON)
    parser.add_argument('--collect-timeout',type=int,default=1800)
    prepare(parser.parse_args())
