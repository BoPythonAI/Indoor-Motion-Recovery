"""Prepare one unchanged 24-case scene evaluation for a warm correction student."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import torch

from motion_reference import file_sha256, resolve_motion_reference


def prepare(args):
    student_runtime=Path(args.student_runtime or Path(__file__).resolve().parent.parent/'scripts_student_surface').resolve()
    teacher_runtime=Path(args.teacher_runtime or Path(__file__).resolve().parent).resolve()
    if not (student_runtime/'launch.sh').is_file() or not (teacher_runtime/'warm_task.py').is_file():
        raise ValueError('Both selected warm student and teacher runtimes must exist')
    spec=importlib.util.spec_from_file_location('_warm_eval_contracts',student_runtime/'contracts.py')
    contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)
    checkpoint=Path(args.student_checkpoint).resolve()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    contracts.validate_checkpoint(state)
    if state.get('kind') not in ('bc','diffusion') or not state.get('ema_model'):
        raise ValueError('Evaluation requires an explicit BC/diffusion EMA checkpoint')
    if args.execute_steps not in (1,2) or args.execute_steps>state['horizon']:
        raise ValueError('execute_steps must be 1 or 2, within the trained chunk horizon')
    if args.num_envs<1 or 24%args.num_envs:
        raise ValueError('num_envs must divide the original 24 cases')
    source=Path(args.scene_config).resolve()
    cfg=copy.deepcopy(json.loads(source.read_text(encoding='utf-8')))
    reference=resolve_motion_reference(cfg)
    if reference['stop_reference_version']!=state['stop_reference_version']:
        raise ValueError('Student and evaluation use different reference versions')
    if cfg.get('geometry_version',contracts.GEOMETRY_VERSION)!=contracts.GEOMETRY_VERSION:
        raise ValueError('Scene config uses a different surface geometry version')
    mjcf=Path(cfg.get('surface_mjcf_path') or state.get('surface_mjcf_path') or '').resolve()
    if not mjcf.is_file() or file_sha256(mjcf)!=state['surface_mjcf_sha256']:
        raise ValueError('Actual evaluation MJCF differs from surface student training')
    if cfg.get('surface_mjcf_sha256',state['surface_mjcf_sha256'])!=state['surface_mjcf_sha256']:
        raise ValueError('Scene config and student MJCF hashes differ')
    cfg.update(geometry_version=contracts.GEOMETRY_VERSION,surface_mjcf_path=str(mjcf),
               surface_mjcf_sha256=state['surface_mjcf_sha256'])
    rows=json.loads(Path(reference['motion_metadata']).read_text(encoding='utf-8'))
    scenes=cfg.get('scene_specs') or [dict(scene_dir=cfg.get('scene_dir'))]
    if len(rows)!=1 or len(scenes)!=1 or scenes[0]['scene_dir']!=rows[0]['scene_dir']:
        raise ValueError('Use the verified independent single-source scene configuration')
    if rows[0]['split']=='train':
        raise ValueError('This evaluation tool requires a held-out source')
    if cfg.get('feature_version',contracts.FEATURE_VERSION)!=contracts.FEATURE_VERSION:
        raise ValueError('Scene config is not a warm correction feature configuration')
    prior=state['frozen_prior']
    for config_key,key in [('tracker_checkpoint','flat'),('terrain_checkpoint','terrain')]:
        path=Path(cfg[config_key]).resolve()
        if file_sha256(path)!=prior[key+'_sha256']:
            raise ValueError('Evaluation frozen prior differs from student training: '+key)
        cfg[config_key]=str(path)
    canonical=[dict(motion=0,bw=bw,direction=direction,phase=phase)
               for bw in [0.,.45,.75] for direction in range(4) for phase in [.3,.6]]
    cases=cfg.get('cases') or canonical
    def keys(items):
        return [(int(c['motion']),float(c['bw']),int(c['direction']),float(c['phase'])) for c in items]
    if len(cases)!=24 or sorted(keys(cases))!=sorted(keys(canonical)):
        raise ValueError('Source config must preserve exactly the original 24 paired cases')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new student evaluation output directory')
    output.mkdir(parents=True,exist_ok=True)
    # Remove only obsolete policy/recording identity; inherit all physical scene/task settings.
    for key in ('checkpoint','resume','teacher_provenance','teacher_identity','warm_trace',
                'task_owned_gate','momentum_observer','hybrid_gate','inactive_zero_student'):
        cfg.pop(key,None)
    cfg.update(reference,mode='evaluate',teacher_kind='terrain_surface',method=state['kind'],
        feature_version=contracts.FEATURE_VERSION,action_target_version=contracts.ACTION_TARGET_VERSION,
        student_checkpoint=str(checkpoint),student_weights='ema',execute_steps=args.execute_steps,
        noise_mode='independent',inference_steps=8,measure_latency=bool(args.measure_latency),
        output=str(output/'evaluation'),cases=cases,strengths=[0.,.45,.75],record=False)
    cfg.setdefault('seed',17)
    config_path=output/'config.json'
    config_path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    command=shlex.join(['env','SURFACE_TEACHER_SCRIPTS='+str(teacher_runtime),'bash',
        str(student_runtime/'launch.sh'),str(config_path),str(args.num_envs),
        reference['motion_bank'],cfg['tracker_checkpoint']])
    (output/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+command+'\n',encoding='utf-8')
    manifest=dict(status='PREPARED_NOT_EXECUTED',source_config=str(source),source_config_sha256=file_sha256(source),
        source_id=rows[0]['motion_id'],student_checkpoint=str(checkpoint),student_checkpoint_sha256=file_sha256(checkpoint),
        student_kind=state['kind'],student_weights='ema',student_steps=state.get('steps'),
        teacher_checkpoint_sha256=state['teacher_checkpoint_sha256'],frozen_prior=prior,
        feature_version=contracts.FEATURE_VERSION,action_target_version=contracts.ACTION_TARGET_VERSION,
        geometry_version=contracts.GEOMETRY_VERSION,surface_mjcf_path=str(mjcf),
        surface_mjcf_sha256=state['surface_mjcf_sha256'],
        student_training_motion_bank_sha256=state['motion_bank_sha256'],**reference,
        execute_steps=args.execute_steps,inference_steps=8,noise_mode='independent',
        cases=24,cases_per_strength=8,num_envs=args.num_envs,measure_latency=bool(args.measure_latency),
        latency_semantics='residual_model_latency_ms includes student plus both frozen prior actors; '
                          'tracker_plus_residual_model_latency_ms additionally includes task flat action computation; '
                          'neither includes physics. Chunk refresh and cached steps are pooled.',
        config=str(config_path),command=command)
    (output/'prepared_evaluation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('SURFACE_STUDENT_EVALUATION_PREPARED',str(config_path),flush=True)
    print(command,flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-config',required=True)
    parser.add_argument('--student-checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--student-runtime')
    parser.add_argument('--teacher-runtime')
    parser.add_argument('--execute-steps',type=int,choices=[1,2],default=1)
    parser.add_argument('--num-envs',type=int,default=24)
    parser.add_argument('--measure-latency',action='store_true')
    prepare(parser.parse_args())
