"""Small CPU comparison: saved-history teacher reconstruction vs EMA students.

Only existing episodes/checkpoints are read. No simulator, training or GPU runs.
Teacher reconstruction error includes float16 history loss and CPU/GPU arithmetic;
it is a numerical reference, not a proven lower bound for every learned model.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import random
import sys

import torch

from contracts import file_sha256, signature
from controller import load_model
from dataset import validate_episode


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def stage(delta):
    if delta < -15:return 'before_last15_source_steps'
    if delta < 0:return 'last15_source_steps'
    if delta < 15:return 'hold_first15_steps'
    return 'hold_later'


def errors(prediction,target,scale):
    difference=prediction-target
    return dict(rmse=difference.square().mean(-1).sqrt(),mae=difference.abs().mean(-1),
                max_abs=difference.abs().amax(-1),normalized_rmse=(difference/scale).square().mean(-1).sqrt())


def summarize(values,positions):
    return {name:dict(mean=float(value[positions].mean()),
                     p95=float(torch.quantile(value[positions],.95)),maximum=float(value[positions].max()))
            for name,value in values.items()}


def run(args):
    torch.set_num_threads(args.threads)
    roots=[Path(p).resolve() for p in args.data]
    if len(set(roots))!=len(roots):raise ValueError('Data directories must be distinct')
    catalog=defaultdict(lambda:defaultdict(list))
    files=sorted({path for root in roots for path in root.rglob('episode_*.pt')})
    for path in files:
        streams=[root for root in roots if path.is_relative_to(root)]
        if len(streams)!=1:raise ValueError('Data roots may not overlap')
        # mmap reads metadata without eagerly copying every unselected tensor.
        episode=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        catalog[str(episode['source']['motion_id'])][str(streams[0])].append(path)
    if not catalog:raise ValueError('No saved episodes found')
    rng=random.Random(args.seed)
    chosen=[]
    for source,streams in sorted(catalog.items()):
        candidates={key:list(paths) for key,paths in sorted(streams.items())}
        for paths in candidates.values():rng.shuffle(paths)
        for index in range(args.episodes_per_source):
            available=[key for key,paths in candidates.items() if paths]
            if not available:break
            stream=available[index % len(available)]
            chosen.append((candidates[stream].pop(),stream))
    entries=[];identities=set();contract=None;by_source=defaultdict(list)
    for path,stream in chosen:
        data=torch.load(path,map_location='cpu',weights_only=False)
        current,feature,action,prefix=validate_episode(data)
        identities.add(signature(current));contract=contract or current
        if len(identities)!=1:raise ValueError('Selected episodes mix actual teacher/prior/reference contracts')
        if feature.shape[-1]<440:raise ValueError('Unexpected surface feature footer')
        for key in ('base_normalized','prior_delta','prior_total'):
            if data[key].shape!=action.shape or data[key].dtype!=torch.float32:
                raise ValueError('Need aligned float32 saved '+key)
        start=int(data['record_start_control_step']);source_frames=int(data['source']['source_frames'])
        index=len(entries)
        entries.append(dict(path=path,stream=stream,data=data,feature=feature,action=action,
                            extended=torch.cat([prefix,feature]),start=start,source_frames=source_frames))
        for frame in range(len(feature)):
            wall=start+frame-source_frames
            # V5 footer: remaining, lag/2, last_total69, body geometry97;
            # surface appends terrain256 + foot16. This is current observed lag.
            lag_seconds=float(feature[frame,-439])*2
            reference_delta=wall-lag_seconds/args.control_dt
            by_source[str(data['source']['motion_id'])].append((index,frame,stage(reference_delta)))
    # Equal source budget, stratified over available reference-clock stage buckets.
    selected=[];source_names=sorted(by_source)
    for i,source in enumerate(source_names):
        budget=args.max_frames//len(source_names)+(i<args.max_frames%len(source_names))
        buckets=defaultdict(list)
        for value in by_source[source]:buckets[value[2]].append(value)
        for values in buckets.values():rng.shuffle(values)
        while budget>0:
            active=[key for key in sorted(buckets) if buckets[key]]
            if not active:break
            for key in active:
                if not budget:break
                selected.append(buckets[key].pop());budget-=1
    if not selected:raise ValueError('No diagnostic frames selected')
    selected.sort(key=lambda row:(row[0],row[1]))
    history=[];targets=[];bases=[];deltas=[];prior_totals=[];records=[]
    for index,frame,reference_stage in selected:
        entry=entries[index];data=entry['data'];control=entry['start']+frame
        history.append(entry['extended'][frame:frame+4].flatten())
        targets.append(entry['action'][frame]);bases.append(data['base_normalized'][frame])
        deltas.append(data['prior_delta'][frame]);prior_totals.append(data['prior_total'][frame])
        lag=float(entry['feature'][frame,-439])*2
        records.append(dict(episode=str(entry['path']),source=str(data['source']['motion_id']),stream=entry['stream'],
            family=data['source']['family'],bw=float(data['bw']),local_record_index=frame,
            record_start_control_step=entry['start'],control_step=control,source_frames=entry['source_frames'],
            wall_steps_from_source_end=control-entry['source_frames'],
            wall_stage=stage(control-entry['source_frames']),phase_lag_seconds_from_feature=lag,
            reference_steps_from_source_end_estimate=control-entry['source_frames']-lag/args.control_dt,
            reference_stage=reference_stage,min_encoded_foot_gap_m=float(entry['feature'][frame,-16:-12].min())))
    history,targets,bases,deltas,prior_totals=map(torch.stack,(history,targets,bases,deltas,prior_totals))
    if not bool((bases.abs()<=1).all()):raise ValueError('Saved base action is not normalized')
    runtime=Path(args.teacher_runtime).resolve()
    teacher_module=load_module(runtime/'warm_policy.py','offline_surface_teacher_policy')
    composition=load_module(runtime/'recovery_runtime.py','offline_surface_action_composition')
    checkpoint=Path(args.teacher_checkpoint or contract['teacher_checkpoint']).resolve()
    if file_sha256(checkpoint)!=contract['teacher_checkpoint_sha256']:
        raise ValueError('Requested teacher weights differ from labels')
    teacher_state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if teacher_state.get('policy_kind')!='terrain_prior_surface_correction_v1':
        raise ValueError('Need the actual surface correction teacher checkpoint')
    if teacher_state['feature_version']!=contract['feature_version']:
        raise ValueError('Teacher feature convention differs from saved labels')
    teacher=teacher_module.WarmResidualPolicy(**teacher_state['model_config']).eval().requires_grad_(False)
    teacher.load_state_dict(teacher_state['policy'],strict=True)
    if teacher.obs_dim!=history.shape[-1]:raise ValueError('Reconstructed history dimension differs from teacher')
    students={}
    for kind,path in [('bc',args.bc_checkpoint),('diffusion',args.diffusion_checkpoint)]:
        model,state=load_model(path,device='cpu',weights='ema')
        if state['kind']!=kind or signature(state)!=signature(contract):
            raise ValueError(kind+' checkpoint and selected label contracts differ')
        if model.obs_dim!=history.shape[-1]:raise ValueError(kind+' history dimension differs')
        students[kind]=(model,state)
    if not torch.allclose(students['bc'][0].action_mean,students['diffusion'][0].action_mean,atol=1e-8,rtol=1e-6) or not torch.allclose(students['bc'][0].action_std,students['diffusion'][0].action_std,atol=1e-8,rtol=1e-6):
        raise ValueError('BC/diffusion normalization buffers differ')
    scale=students['bc'][0].action_std
    predictions={name:[] for name in ['teacher_reconstructed','bc_ema',*[f'diffusion_ema_seed{seed}' for seed in args.diffusion_seeds]]}
    prior_errors=[]
    generators={seed:torch.Generator(device='cpu').manual_seed(seed) for seed in args.diffusion_seeds}
    with torch.no_grad():
        for begin in range(0,len(history),args.batch_size):
            end=begin+args.batch_size;x=history[begin:end]
            base_raw=bases[begin:end].atanh()
            _,reconstructed_prior,_=composition.compose_action(base_raw,deltas[begin:end])
            prior_errors.append((reconstructed_prior-prior_totals[begin:end]).abs())
            correction=teacher.correction(x)
            _,total,_=composition.compose_action(base_raw,deltas[begin:end]+correction)
            predictions['teacher_reconstructed'].append(total-prior_totals[begin:end])
            predictions['bc_ema'].append(students['bc'][0].sample(x)[:,0])
            for seed,generator in generators.items():
                predictions[f'diffusion_ema_seed{seed}'].append(students['diffusion'][0].sample(x,steps=args.inference_steps,generator=generator)[:,0])
    predictions={name:torch.cat(value) for name,value in predictions.items()}
    for name,value in list(predictions.items()):
        if name=='teacher_reconstructed':continue
        _,total,_=composition.compose_action(bases.atanh(),deltas+value)
        predictions[name+'_executed']=total-prior_totals
    metrics={name:errors(value,targets,scale) for name,value in predictions.items()}
    for name,values in metrics.items():
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):raise ValueError('Nonfinite '+name+' errors')
    groups=defaultdict(list)
    for i,record in enumerate(records):
        for axis in ('source','stream','wall_stage','reference_stage'):
            groups[(axis,str(record[axis]))].append(i)
        groups[('source_reference_stage',record['source']+'/'+record['reference_stage'])].append(i)
        for name,values in metrics.items():
            record[name]={metric:float(value[i]) for metric,value in values.items()}
    all_positions=torch.arange(len(records))
    summary={name:summarize(values,all_positions) for name,values in metrics.items()}
    by_group={}
    for (axis,key),positions in groups.items():
        by_group.setdefault(axis,{})[key]=dict(frames=len(positions),
            error={name:summarize(values,positions) for name,values in metrics.items()})
    return dict(status='CPU_OFFLINE_IMITATION_DIAGNOSTIC',device='cpu',threads=args.threads,
        frames=len(records),selected_episodes=len(entries),catalog_episodes=len(files),
        selection=dict(seed=args.seed,max_frames=args.max_frames,episodes_per_source=args.episodes_per_source,
            rule='Round-robin streams per source; equal source frame budget stratified over available estimated reference stages'),
        control_dt=args.control_dt,reference_stage_semantics='control_step - source_frames - observed_phase_lag/control_dt; estimate uses the verified surface feature footer, not a simulated reference query',
        teacher_checkpoint=str(checkpoint),teacher_checkpoint_sha256=file_sha256(checkpoint),
        teacher_numeric_reference='Teacher reconstructed from stored float16 history and saved same-state base/prior; error includes history quantization and CPU/GPU arithmetic, not a strict theoretical lower bound',
        saved_prior_reconstruction_max_abs_error=float(torch.cat(prior_errors).max()),
        student_weights='ema',inference_steps=args.inference_steps,diffusion_seeds=args.diffusion_seeds,
        student_checkpoints={kind:dict(path=str(Path(path).resolve()),sha256=file_sha256(path)) for kind,path in [('bc',args.bc_checkpoint),('diffusion',args.diffusion_checkpoint)]},
        contract_signature=json.loads(signature(contract)),summary=summary,by_group=by_group,frames_detail=records,
        selected_episode_files=[dict(path=str(entry['path']),sha256=file_sha256(entry['path'])) for entry in entries],
        implementation_sha256=file_sha256(__file__),teacher_runtime=str(runtime),
        teacher_source_sha256={name:file_sha256(runtime/name) for name in ['warm_policy.py','recovery_runtime.py']},
        optimizer_steps=0,simulator_runs=0,physical_success_claimed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',nargs='+',required=True)
    parser.add_argument('--bc-checkpoint',required=True)
    parser.add_argument('--diffusion-checkpoint',required=True)
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--teacher-runtime',default=str(Path(__file__).resolve().parent.parent/'scripts_teacher_planned'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-frames',type=int,default=1024)
    parser.add_argument('--episodes-per-source',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=109)
    parser.add_argument('--diffusion-seeds',nargs='+',type=int,default=[17])
    parser.add_argument('--inference-steps',type=int,default=8)
    parser.add_argument('--control-dt',type=float,default=1/30)
    args=parser.parse_args()
    if min(args.max_frames,args.episodes_per_source,args.batch_size,args.threads,args.inference_steps)<1 or args.control_dt<=0:
        parser.error('Positive frame/episode/batch/thread/step counts and control dt required')
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Use a new diagnostic report path')
    try:report=run(args)
    except Exception as error:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(dict(status='DIAGNOSTIC_ERROR',error_type=type(error).__name__,error=str(error)),indent=2)+'\n')
        raise
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],output=str(output),frames=report['frames'],summary=report['summary']),indent=2))
