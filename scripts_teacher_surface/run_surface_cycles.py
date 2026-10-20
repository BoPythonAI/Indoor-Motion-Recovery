"""Sequential train/checkpoint/independent-scene validation. Default prepares only."""
import argparse
import json
from pathlib import Path
import subprocess

from motion_reference import resolve_motion_reference, file_sha256


VALIDATION_IDS = {'014810','015330'}
REMOVE_KEYS = ('hybrid_gate','gate_input','momentum_observer','task_owned_gate','inactive_zero_student',
               'validation_motion','validation_metadata','observer_log_path')


def read_config(path):
    return json.loads(Path(path).read_text())


def clean_config(source):
    cfg=dict(source)
    for key in REMOVE_KEYS:cfg.pop(key,None)
    cfg.update(teacher_kind='terrain_surface',method='terrain_surface',
               feature_version='v3_terrain_reference__terrain_correction_v1__foot_clearance_v1')
    cfg.update(resolve_motion_reference(cfg))
    return cfg


def evaluation_config(source, checkpoint, output):
    cfg=clean_config(source)
    rows=read_config(cfg['motion_metadata'])
    if len(rows)!=1:
        raise ValueError('Validation/dev evaluation must use an independent single-source scene config')
    source_id=str(rows[0]['motion_id'])
    scenes=cfg.get('scene_specs') or [dict(scene_dir=cfg.get('scene_dir'))]
    if len(scenes)!=1 or scenes[0]['scene_dir']!=rows[0]['scene_dir']:
        raise ValueError('Single-source metadata and configured physical scene differ')
    cfg.update(mode='evaluate',resume=False,checkpoint=str(checkpoint) if checkpoint else None,
        output=str(output),seed=17,strengths=[0.,.45,.75],record=False,
        cases=[dict(motion=0,bw=bw,direction=d,phase=p) for bw in [0.,.45,.75] for d in range(4) for p in [.3,.6]])
    return cfg,source_id


def write_config(path,cfg):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(cfg,indent=2)+'\n'
    if path.exists() and path.read_text()!=encoded:
        raise ValueError(f'Refusing to change an existing frozen stage config: {path}')
    path.write_text(encoded)
    return str(path)


def run_job(runtime, config_path, cfg, envs, log_path, timeout_s):
    log_path=Path(log_path)
    if log_path.exists():raise ValueError(f'Use a fresh cycle output; log already exists: {log_path}')
    command=['timeout','--signal=TERM','--kill-after=30s',str(timeout_s)+'s','bash',
             str(Path(runtime)/'launch.sh'),str(config_path),str(envs),cfg['motion_bank'],cfg['tracker_checkpoint']]
    print('SURFACE_JOB_START',json.dumps(dict(config=str(config_path),output=cfg['output'],envs=envs)),flush=True)
    with log_path.open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('SURFACE_JOB_DONE',str(log_path),flush=True)


def aggregate(reports):
    return {str(bw):dict(success=sum(r['summary'][str(bw)]['success'] for r in reports),
                         total=sum(r['summary'][str(bw)]['total'] for r in reports)) for bw in [0.,.45,.75]}


def main(args):
    # launch.sh changes cwd to ProtoMotions; freeze host-absolute paths first.
    for key in ('output','runtime','checkpoint','train_config'):
        value=getattr(args,key,None)
        if value:setattr(args,key,str(Path(value).resolve()))
    for key in ('validation_config','evaluation_config'):
        value=getattr(args,key,None)
        if value:setattr(args,key,[str(Path(path).resolve()) for path in value])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.action=='evaluate':
        if not args.evaluation_config or not args.checkpoint:
            raise ValueError('evaluate requires --evaluation-config and --checkpoint')
        reports=[]
        checkpoint_sha256=file_sha256(args.checkpoint)
        source_ids=[]
        for source_path in args.evaluation_config:
            source=read_config(source_path)
            rows=read_config(source['motion_metadata']);name=str(rows[0]['motion_id'])
            if name in source_ids:raise ValueError('Duplicate evaluation source: '+name)
            source_ids.append(name)
            cfg,_=evaluation_config(source,args.checkpoint,out/name)
            path=write_config(out/(name+'.json'),cfg)
            run_job(args.runtime,path,cfg,args.validation_envs,out/(name+'.log'),args.job_timeout)
            reports.append(read_config(Path(cfg['output'])/'evaluation.json'))
            if file_sha256(args.checkpoint)!=checkpoint_sha256:
                raise ValueError('Evaluation checkpoint changed during the independent scene runs')
        report=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=checkpoint_sha256,
                    source_ids=source_ids,summary=aggregate(reports),
                    per_source={source_id:result['summary'] for source_id,result in zip(source_ids,reports)})
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        print('SURFACE_EVALUATION_COMPLETE',json.dumps(report),flush=True)
        return
    if not args.train_config or not args.validation_config:
        raise ValueError('Cycles need --train-config and two --validation-config files')
    train=clean_config(read_config(args.train_config))
    if any(row['split']!='train' for row in read_config(train['motion_metadata'])):
        raise ValueError('Training metadata contains a held-out source')
    train.update(mode='train',resume=False,checkpoint=None)
    for key,value in dict(success_bonus=250.,stationary_weight=1.,power_weight=.01,
        force_strengths=[.45,.75],horizon=64,save_every=10,snapshot_every=args.interval).items():
        train.setdefault(key,value)
    validation=[read_config(path) for path in args.validation_config]
    ids={str(read_config(cfg['motion_metadata'])[0]['motion_id']) for cfg in validation}
    if len(validation)!=2 or ids!=VALIDATION_IDS:
        raise ValueError('Fixed val is exactly independent sources 014810 and 015330')
    checkpoint=args.checkpoint
    start=0
    initial_has_optimizers=False
    if checkpoint:
        from teacher_factory import read_checkpoint
        initial_state=read_checkpoint(checkpoint)
        start=initial_state['update']
        initial_has_optimizers=bool(initial_state.get('optimizers'))
    if args.total_updates<=start:raise ValueError('total-updates must exceed resume update')
    stages=[]
    for target in range(start+args.interval,args.total_updates+args.interval,args.interval):
        target=min(target,args.total_updates)
        stage=out/f'update_{target:04d}';stage.mkdir(exist_ok=True)
        cfg=dict(train,updates=target,resume=bool(checkpoint) and (initial_has_optimizers or bool(stages)),checkpoint=str(checkpoint) if checkpoint else None)
        train_path=write_config(stage/'train.json',cfg)
        # Each block saves its exact final update even if target is not a multiple
        # of interval; validation selects last.pt after checking its actual update.
        last=Path(train['output'])/'last.pt'
        eval_jobs=[]
        for source in validation:
            source_id=str(read_config(source['motion_metadata'])[0]['motion_id'])
            eval_cfg,_=evaluation_config(source,last,stage/source_id)
            path=write_config(stage/(source_id+'.json'),eval_cfg)
            eval_jobs.append(dict(config=path,cfg=eval_cfg,id=source_id))
        record=dict(target_update=target,train_config=train_path,validation=[j['config'] for j in eval_jobs],
                    train_envs=args.num_envs,validation_envs=args.validation_envs,
                    resume_episode_reset=cfg['resume'],validation_is_separate_process=True)
        stages.append(record)
        if args.action=='run':
            run_job(args.runtime,train_path,cfg,args.num_envs,stage/'train.log',args.job_timeout)
            from teacher_factory import read_checkpoint
            actual=read_checkpoint(last)['update']
            checkpoint_sha256=file_sha256(last)
            reports=[]
            for job in eval_jobs:
                run_job(args.runtime,job['config'],job['cfg'],args.validation_envs,stage/(job['id']+'.log'),args.job_timeout)
                reports.append(read_config(Path(job['cfg']['output'])/'evaluation.json'))
            if file_sha256(last)!=checkpoint_sha256:
                raise ValueError('Checkpoint changed during fixed validation')
            report=dict(update=actual,requested_update=target,checkpoint=str(last),
                        checkpoint_sha256=checkpoint_sha256,summary=aggregate(reports),
                        per_source={job['id']:result['summary'] for job,result in zip(eval_jobs,reports)})
            (stage/'validation_summary.json').write_text(json.dumps(report,indent=2))
            print('SURFACE_FIXED_VAL',json.dumps(report),flush=True)
            if actual!=target:
                record['stopped_before_target']=actual
                break  # A time-budget stop is not silently called a full stage.
        checkpoint=last
        if target==args.total_updates:break
    (out/'cycle_plan.json').write_text(json.dumps(dict(action=args.action,stages=stages,
        validation_ids=sorted(ids),cases_per_strength=16,train_source=args.train_config,
        initial_physical_check='Run initial mean validation before choosing action=run'),indent=2))
    print('SURFACE_CYCLES_'+('PREPARED' if args.action=='prepare' else 'COMPLETE'),str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=['prepare','run','evaluate'],default='prepare')
    parser.add_argument('--runtime',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--train-config')
    parser.add_argument('--validation-config',nargs='+')
    parser.add_argument('--evaluation-config',nargs='+')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output',required=True)
    parser.add_argument('--total-updates',type=int,default=50)
    parser.add_argument('--interval',type=int,default=50)
    parser.add_argument('--num-envs',type=int,default=128)
    parser.add_argument('--validation-envs',type=int,default=8)
    parser.add_argument('--job-timeout',type=int,default=2400)
    main(parser.parse_args())
