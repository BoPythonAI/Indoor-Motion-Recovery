"""Fresh Qwen relation parsing -> original task validator -> surface physics.

This adapter does not replace the compiler or controller. --prepare-only is CPU-only.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

MOTION_ID='015317'
INSTRUCTION='Walk to the office chair farthest from the trash can; no body part may touch the table nearest the starting position.'
ROOT='/root/autodl-tmp/method_v1_20261004/project_v4_recovery'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def bind_program(program,meta,validator):
    if program.get('schema_version')!='task_program_v1' or program.get('status')!='COMPILED':
        raise ValueError('Qwen task program is not compiled')
    if program.get('instruction')!=INSTRUCTION:
        raise ValueError('Task program does not belong to this exact natural-language request')
    forbidden=validator.require_supported(program,meta)
    goal=program['goal']['instance_id_from_proposal_map']
    if goal!=10 or forbidden!=17:
        raise ValueError('Original 015317 task requires resolved goal 10 and forbidden object 17')
    if program.get('source',{}).get('model')!='Qwen3-VL-4B-Instruct':
        raise ValueError('Expected the existing Qwen3-VL relation compiler')
    relations=program.get('parsed_relations',{})
    if relations.get('goal',{}).get('relation')!='farthest' or relations.get('forbidden',{}).get('relation')!='nearest':
        raise ValueError('Compiled relations do not match the requested farthest/nearest operators')
    return dict(goal_instance_id=goal,forbidden_instance_id=forbidden)


def preflight(args):
    root=Path(args.root).resolve()
    teacher=Path(args.teacher_runtime or root/'scripts_teacher_surface').resolve()
    student=Path(args.student_runtime or root/'scripts_student_surface').resolve()
    compiler=Path(args.compiler_directory or root/'scripts_demo').resolve()
    sys.path.insert(0,str(teacher))
    from motion_reference import file_sha256,resolve_motion_reference
    from surface_contract import FEATURE_VERSION,GEOMETRY_VERSION,POLICY_KIND
    import torch
    for path in (compiler/'vlm_relations.py',compiler/'task_program.py',teacher/'launch.sh'):
        if not path.is_file():raise ValueError('Missing original compiler/runtime file: '+str(path))
    source=Path(args.scene_config).resolve();cfg=copy.deepcopy(read(source))
    reference=resolve_motion_reference(cfg)
    rows=read(reference['motion_metadata'])
    if len(rows)!=1 or str(rows[0]['motion_id'])!=MOTION_ID:
        raise ValueError('Demo must use the original single-source 015317 motion bank/metadata')
    scene=Path(rows[0]['scene_dir']).resolve();meta=read(scene/'scene_metadata.json')
    specs=cfg.get('scene_specs') or [dict(scene_dir=cfg.get('scene_dir'),forbidden_id=cfg.get('forbidden_id'))]
    if len(specs)!=1 or Path(specs[0]['scene_dir']).resolve()!=scene or meta['target_object_id']!=10:
        raise ValueError('Physical scene/reference task is not bound to the original goal-10 task')
    manifest_path=Path(args.manifest).resolve();manifest=read(manifest_path)[0]
    if manifest['scene_id']!=meta['scene_id']:
        raise ValueError('Qwen proposal manifest belongs to another scene')
    if not manifest.get('annotated_images') or any(not Path(p).is_file() for p in manifest['annotated_images'][:2]):
        raise ValueError('Existing annotated Qwen input images must be available')
    checkpoint=Path(args.checkpoint).resolve()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if args.method=='teacher':
        if state.get('policy_kind')!=POLICY_KIND:
            raise ValueError('Teacher must be the explicit surface checkpoint kind')
        launch=teacher/'launch.sh'
        if args.execute_steps!=1:raise ValueError('Teacher executes its current correction every step')
        horizon=None
    else:
        contracts=module(student/'contracts.py','surface_demo_student_contracts')
        contracts.validate_checkpoint(state)
        if state.get('kind')!=args.method or not state.get('ema_model'):
            raise ValueError('Selected student kind/EMA weights do not match the requested method')
        if args.execute_steps>state['horizon']:raise ValueError('Execution chunk exceeds trained horizon')
        horizon=state['horizon']
        launch=student/'launch.sh'
    if not launch.is_file():raise ValueError('Missing selected surface launch entry: '+str(launch))
    if state.get('feature_version')!=FEATURE_VERSION or cfg.get('feature_version',FEATURE_VERSION)!=FEATURE_VERSION:
        raise ValueError('Only current surface foot-clearance features are supported')
    if state.get('geometry_version')!=GEOMETRY_VERSION:
        raise ValueError('Surface geometry contract missing from controller checkpoint')
    trained_reference=state.get('stop_reference_version',state.get('config',{}).get('stop_reference_version'))
    if trained_reference!=reference['stop_reference_version']:
        raise ValueError('Physical task reference version differs from controller training')
    for key,component in [('tracker_checkpoint','flat'),('terrain_checkpoint','terrain')]:
        path=Path(cfg[key]).resolve()
        if file_sha256(path)!=state['frozen_prior'][component+'_sha256']:
            raise ValueError('Frozen prior does not match the selected controller: '+component)
        cfg[key]=str(path)
    mjcf=Path(cfg.get('surface_mjcf_path') or state.get('surface_mjcf_path') or state.get('config',{}).get('surface_mjcf_path','')).resolve()
    if not mjcf.is_file() or file_sha256(mjcf)!=state['surface_mjcf_sha256']:
        raise ValueError('Actual robot MJCF differs from the controller geometry')
    # A planned bank must use the task that restores the original goal. Merely
    # passing the new bank to the default surface task would move that goal.
    planned_contract=None
    planned=bool(cfg.get('reference_planning_version') or cfg.get('reference_plan_manifest'))
    contract_source=teacher/'reference_goal_contract.py'
    if planned != contract_source.is_file():
        raise ValueError('Planned scene config and selected teacher runtime must both include the original-goal contract')
    if planned:
        contract=module(contract_source,'surface_demo_selected_goal_contract')
        if cfg.get('reference_planning_version')!=contract.PLAN_VERSION:
            raise ValueError('Selected runtime and planned reference versions differ')
        plan_path=Path(cfg['reference_plan_manifest']).resolve();plan_manifest=read(plan_path)
        experimental=contract.validate_plan_status(plan_manifest)
        plan_sha=file_sha256(plan_path)
        if cfg.get('reference_plan_manifest_sha256',plan_sha)!=plan_sha:
            raise ValueError('Original-goal planning manifest changed')
        if (reference['motion_bank_sha256']!=plan_manifest['planned_bank_sha256'] or
                reference['motion_metadata_sha256']!=plan_manifest['original_metadata_sha256']):
            raise ValueError('Actual planned bank/metadata differs from the original-goal manifest')
        if len(plan_manifest['sources'])!=1 or str(plan_manifest['sources'][0]['source_id'])!=MOTION_ID:
            raise ValueError('Original-goal plan does not belong to source 015317')
        original=Path(cfg.get('reference_original_bank',plan_manifest['original_bank'])).resolve()
        if file_sha256(original)!=plan_manifest['original_bank_sha256']:
            raise ValueError('Original motion bank changed since reference planning')
        if plan_manifest.get('mjcf_sha256')!=state['surface_mjcf_sha256']:
            raise ValueError('Reference planning geometry differs from the controller')
        planned_contract=dict(reference_planning_version=contract.PLAN_VERSION,
            reference_plan_manifest=str(plan_path),reference_plan_manifest_sha256=plan_sha,
            original_bank=str(original),original_bank_sha256=plan_manifest['original_bank_sha256'],
            planned_bank_sha256=plan_manifest['planned_bank_sha256'],
            status=plan_manifest['status'],experimental_export=experimental,
            geometry_qualified=plan_manifest.get('geometry_qualified'))
    canonical=[dict(motion=0,bw=bw,direction=d,phase=p) for bw in [0.,.45,.75] for d in range(4) for p in [.3,.6]]
    cases=cfg.get('cases') or canonical
    key=lambda c:(int(c['motion']),float(c['bw']),int(c['direction']),float(c['phase']))
    if len(cases)!=24 or sorted(map(key,cases))!=sorted(map(key,canonical)):
        raise ValueError('Keep all original 24 paired cases in their original slots')
    cfg.update(reference,feature_version=FEATURE_VERSION,geometry_version=GEOMETRY_VERSION,
        surface_mjcf_path=str(mjcf),surface_mjcf_sha256=state['surface_mjcf_sha256'],cases=cases,
        strengths=[0.,.45,.75],teacher_kind='terrain_surface',mode='evaluate')
    runtime_files=['launch.sh','entry.py','teacher.py','v5_task.py','warm_task.py','warm_task_base.py',
                   'warm_prior.py','teacher_factory.py','recovery_runtime.py','scene_adapter.py',
                   'scene_heightmap.py','floor_geometry.py','reference_terrain_fix.py','foot_clearance.py',
                   'surface_contract.py','motion_reference.py']
    if planned:runtime_files.append('reference_goal_contract.py')
    runtime_evidence=dict(teacher_runtime=str(teacher),student_runtime=str(student) if args.method!='teacher' else None,
        selected_launch=str(launch),selected_launch_sha256=file_sha256(launch),
        task_sources={name:file_sha256(teacher/name) for name in runtime_files},
        student_sources={name:file_sha256(student/name) for name in
            ('entry.py','warm_dependency.py','student_runtime.py','controller.py','contracts.py','model.py')}
            if args.method!='teacher' else {},
        environment={'SURFACE_TEACHER_SCRIPTS':str(teacher)})
    inference_seed=getattr(args,'inference_seed',None)
    if inference_seed is None:inference_seed=cfg.get('inference_seed')
    if inference_seed is None:inference_seed=cfg.get('seed',17)
    controller=dict(kind=args.method,trained_horizon=horizon,execute_steps=args.execute_steps,
        weights='ema' if args.method!='teacher' else 'teacher_checkpoint',
        physics_seed=cfg.get('seed',17),inference_seed=inference_seed if args.method!='teacher' else None,
        inference_seed_role='private diffusion sampling only; BC deterministic' if args.method!='teacher' else None)
    return dict(cfg=cfg,scene=scene,meta=meta,source=source,manifest_path=manifest_path,
        teacher=teacher,student=student,compiler=compiler,launch=launch,checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),file_sha256=file_sha256,
        runtime_evidence=runtime_evidence,planned_contract=planned_contract,controller=controller)


def run(args):
    context=preflight(args);sha=context['file_sha256']
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new E2E output directory')
    out.mkdir(parents=True,exist_ok=True)
    parse_command=[sys.executable,str(context['compiler']/'vlm_relations.py'),'--instruction',INSTRUCTION,
        '--scene',str(context['scene']),'--manifest',str(context['manifest_path']),'--out',str(out/'qwen')]
    run_config=out/'run_config.json'
    physics_command=['env','SURFACE_TEACHER_SCRIPTS='+str(context['teacher']),'bash',str(context['launch']),
        str(run_config),'24',context['cfg']['motion_bank'],context['cfg']['tracker_checkpoint']]
    plan=dict(status='PREPARED_NOT_EXECUTED',motion_id=MOTION_ID,instruction=INSTRUCTION,
        method=args.method,checkpoint=str(context['checkpoint']),checkpoint_sha256=context['checkpoint_sha256'],
        scene_config=str(context['source']),scene_config_sha256=sha(context['source']),
        manifest=str(context['manifest_path']),manifest_sha256=sha(context['manifest_path']),
        compiler_sources={name:sha(context['compiler']/name) for name in ('vlm_relations.py','task_program.py')},
        runtime=context['runtime_evidence'],planned_reference=context['planned_contract'],controller=context['controller'],
        parser_command=shlex.join(parse_command),physics_command=shlex.join(physics_command),
        num_envs=24,cases=context['cfg']['cases'],record_env=args.record_env,
        compiled_program=None,evaluation=None)
    write(out/'plan.json',plan)
    if args.prepare_only:
        print(json.dumps(plan,indent=2));return plan
    # Sequential processes: Qwen exits and releases GPU before physics is launched.
    with (out/'qwen.log').open('w') as stream:
        subprocess.run(parse_command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    program_path=out/'qwen'/'task_program.json'
    if not program_path.is_file():raise ValueError('Qwen abstained; inspect fresh qwen/results.json')
    program=read(program_path);evidence=read(out/'qwen'/'results.json')
    if len(evidence)!=1 or evidence[0].get('id')!='task' or not evidence[0].get('raw') or evidence[0].get('program')!=program:
        raise ValueError('Compiled task is not bound to the fresh Qwen raw-response record')
    validator=module(context['compiler']/'task_program.py','original_surface_demo_task_program')
    binding=bind_program(program,context['meta'],validator)
    cfg=context['cfg']
    for name in ('checkpoint','student_checkpoint','teacher_provenance','teacher_identity','warm_trace','resume'):
        cfg.pop(name,None)
    # The program controls the physical forbidden-instance selection; it is not display-only metadata.
    scene_spec=copy.deepcopy((cfg.get('scene_specs') or [{}])[0])
    scene_spec.update(scene_dir=str(context['scene']),forbidden_id=binding['forbidden_instance_id'])
    cfg.update(scene_dir=str(context['scene']),forbidden_id=binding['forbidden_instance_id'],
        scene_specs=[scene_spec],
        method='terrain_surface' if args.method=='teacher' else args.method,output=str(out/'physics'),
        record=args.record_env is not None,record_env=args.record_env or 0,measure_latency=True,
        execute_steps=args.execute_steps,noise_mode='independent',inference_steps=8,
        task_program=str(program_path),task_program_sha256=sha(program_path),task_binding=binding)
    cfg['demo_runtime']=context['runtime_evidence']
    cfg.setdefault('seed',17)
    if args.method=='teacher':cfg['checkpoint']=str(context['checkpoint'])
    else:cfg.update(student_checkpoint=str(context['checkpoint']),student_weights='ema',
                    action_target_version='executed_total_minus_composed_terrain_prior_v1',
                    inference_seed=context['controller']['inference_seed'],physics_seed=cfg['seed'])
    if sha(context['checkpoint'])!=context['checkpoint_sha256']:
        raise ValueError('Selected controller checkpoint changed during language compilation')
    write(run_config,cfg)
    with (out/'physics.log').open('w') as stream:
        subprocess.run(physics_command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report_path=out/'physics'/'evaluation.json';report=read(report_path)
    episode_key=lambda row:(int(row['motion']),float(row['bw']),int(row['direction']),float(row['phase']))
    if len(report.get('episodes',[]))!=24 or sorted(map(episode_key,report['episodes']))!=sorted(map(episode_key,cfg['cases'])):
        raise ValueError('Physics did not evaluate the original 24-case task matrix')
    selected=None
    if args.record_env is not None:
        case=cfg['cases'][args.record_env]
        selected=next(row for row in report['episodes'] if episode_key(row)==episode_key(case))
        if not (out/'physics'/'trace.json').is_file():raise ValueError('Selected physical trace was not produced')
    result=dict(status='E2E_EVALUATED',instruction=INSTRUCTION,motion_id=MOTION_ID,**binding,
        method=args.method,checkpoint_sha256=context['checkpoint_sha256'],
        runtime=context['runtime_evidence'],planned_reference=context['planned_contract'],controller=context['controller'],
        program=str(program_path),task_program_sha256=sha(program_path),qwen_raw_results=str(out/'qwen'/'results.json'),
        run_config=str(run_config),run_config_sha256=sha(run_config),evaluation=str(report_path),summary=report['summary'],
        selected_record_env=args.record_env,selected_episode=selected,
        physical_trace=str(out/'physics'/'trace.json') if selected is not None else None)
    write(out/'result.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-config',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--method',choices=['teacher','bc','diffusion'],required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--root',default=ROOT)
    parser.add_argument('--compiler-directory')
    parser.add_argument('--teacher-runtime',help='Task runtime shared by teacher/BC/diffusion; defaults to scripts_teacher_surface. Use scripts_teacher_planned for an original-goal planned config.')
    parser.add_argument('--student-runtime')
    parser.add_argument('--execute-steps',choices=[1,2],type=int,default=1)
    parser.add_argument('--inference-seed',type=int,default=None,help='Private diffusion seed; defaults to the selected scene config, then its physics seed. BC is deterministic.')
    parser.add_argument('--record-env',type=int,choices=range(24))
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())
