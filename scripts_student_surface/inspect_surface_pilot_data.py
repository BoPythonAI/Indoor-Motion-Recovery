"""Read real surface pilot episodes on CPU; report facts without changing training.

All normalization and sampling functions are the existing student functions.
No simulator, policy inference, optimizer, CUDA, synthetic data or auto-fix runs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

from contracts import file_sha256, signature
from dataset import load_episodes, correction_statistics
from dataset_sampling import balanced_frame_weights


def distribution(values):
    values=values.float()
    return dict(count=len(values),minimum=float(values.min()),median=float(values.median()),
                mean=float(values.mean()),maximum=float(values.max()))


def mask_statistics(valid_lengths,weights,horizon):
    weights=weights.double();weights=weights/weights.sum()
    probabilities=[float(weights[valid_lengths==k].sum()) for k in range(1,horizon+1)]
    return dict(start_count=len(valid_lengths),
        starts_with_any_padding_count=int((valid_lengths<horizon).sum()),
        starts_with_any_padding_probability=float(weights[valid_lengths<horizon].sum()),
        expected_valid_actions=float((weights*valid_lengths).sum()),
        padded_slot_probability=float((weights*(horizon-valid_lengths)/horizon).sum()),
        valid_length_probability={str(k):probabilities[k-1] for k in range(1,horizon+1)},
        valid_probability_per_chunk_position=[float(weights[valid_lengths>k].sum()) for k in range(horizon)])


def clip_statistics(outside,normalized_abs,weights):
    weights=weights.double();weights=weights/weights.sum()
    per_frame=outside.any(-1)
    return dict(frames=len(outside),frames_with_any_normalized_abs_gt8=int(per_frame.sum()),
        elements_with_normalized_abs_gt8=int(outside.sum()),
        normalized_abs_max=float(normalized_abs.max()),
        frame_probability=float(weights[per_frame].sum()),
        element_probability=float((weights*outside.double().mean(-1)).sum()),
        probability_per_dimension=(weights[:,None]*outside.double()).sum(0).tolist(),
        normalization='Same global frame mean and std floor .005 as current train_student.py; no per-stream renormalization')


def inspect(args):
    torch.set_num_threads(args.threads)
    directories=[str(Path(p).resolve()) for p in args.data]
    if len(set(directories))!=len(directories):raise ValueError('Data directories must be distinct')
    dataset,files=load_episodes(directories,device='cpu')
    sampling=dataset.sampling
    if len(files)!=len(sampling):raise ValueError('Episode manifest and sampling metadata differ')
    indices=dataset.valid_indices
    actions=dataset.actions[indices]
    stats=correction_statistics(actions)
    mean=torch.tensor(stats['per_dimension_mean'],dtype=actions.dtype)
    scale=torch.tensor(stats['per_dimension_normalization_scale'],dtype=actions.dtype)
    normalized_abs=((actions-mean)/scale).abs()
    outside=normalized_abs>8
    weights,sampling_report=balanced_frame_weights(sampling)
    if len(weights)!=len(indices):raise ValueError('Balanced weights do not align with action starts')
    uniform=torch.ones(len(indices),dtype=torch.float64)
    valid_lengths=(dataset.end[indices]-indices).clamp(max=args.horizon)
    if (valid_lengths<1).any():raise ValueError('A supervised start has no first-step label')
    # Read only the appended current-state geometry, not the whole flattened history.
    gaps=dataset.features[indices,-16:-12]
    if gaps.shape!=(len(indices),4):raise ValueError('Missing appended four foot-gap feature slots')
    minimum_gap=gaps.amin(-1)
    axes={axis:{} for axis in ('stream','family','bw','direction')}
    joint={};stream_positions=defaultdict(list);episode_starts=[]
    offset=0
    for row,file in zip(sampling,files):
        frames=row['frames'];stop=offset+frames
        if frames!=file['frames']:raise ValueError('Episode frame count differs from the loaded manifest')
        episode_starts.append(offset)
        stream_positions[row['stream']].append((offset,stop))
        values=dict(stream=row['stream'],family=str(row['family']),bw=f"{row['bw']:.6g}",direction=str(row['direction']))
        increments=dict(episodes=1,frames=frames,
            frames_with_normalized_abs_gt8=int(outside[offset:stop].any(-1).sum()),
            starts_with_any_tail_padding=int((valid_lengths[offset:stop]<args.horizon).sum()))
        for axis,value in values.items():
            target=axes[axis].setdefault(value,{key:0 for key in increments})
            for key,value in increments.items():target[key]+=value
        key=(values['stream'],values['family'],values['bw'],values['direction'])
        target=joint.setdefault(key,dict(**values,**{k:0 for k in increments}))
        for name,value in increments.items():target[name]+=value
        offset=stop
    if offset!=len(indices):raise ValueError('Sampling counts do not cover every first-step target')
    by_stream={}
    for stream,ranges in stream_positions.items():
        positions=torch.cat([torch.arange(a,b) for a,b in ranges])
        by_stream[stream]=dict(
            counts=axes['stream'][stream],
            probability_mass_under_balanced=float(weights[positions].double().sum()/weights.double().sum()),
            tail_mask=dict(uniform=mask_statistics(valid_lengths[positions],uniform[positions],args.horizon),
                balanced_conditional_on_stream=mask_statistics(valid_lengths[positions],weights[positions],args.horizon)),
            normalized_gt8=dict(uniform=clip_statistics(outside[positions],normalized_abs[positions],uniform[positions]),
                balanced_conditional_on_stream=clip_statistics(outside[positions],normalized_abs[positions],weights[positions])),
            current_encoded_min_foot_gap=distribution(minimum_gap[positions]))
    first_positions=torch.tensor(episode_starts)
    first_indices=indices[first_positions]
    history,target,mask=dataset.chunk(first_indices,args.horizon)
    if not torch.equal(history.reshape(len(first_indices),4,-1)[:,-1],dataset.features[first_indices]):
        raise ValueError('Episode-first chunk history does not end at its labeled state')
    if not torch.equal(target[:,0],dataset.actions[first_indices]) or not bool(mask[:,0].all()):
        raise ValueError('Episode-first chunk target/mask is not aligned')
    runtime=Path(__file__).resolve().parent
    snapshot=runtime/'source_snapshot.json'
    contract=dataset.contract
    keys=('feature_version','action_target_version','teacher_identity','teacher_checkpoint',
          'teacher_checkpoint_sha256','frozen_prior','geometry_version','surface_mjcf_sha256',
          'surface_mjcf_path','stop_reference_version','motion_bank_sha256','motion_metadata_sha256')
    return dict(status='INSPECTED_REAL_EPISODE_FILES',device='cpu',threads=args.threads,
        input_directories=directories,horizon=args.horizon,
        totals=dict(episodes=dataset.episode_count,supervised_frames=len(indices),
            frames_including_history_prefix=len(dataset.features),features_per_frame=dataset.features.shape[-1],
            episode_length_frames=distribution(torch.tensor([row['frames'] for row in sampling]))),
        counts_by_axis=axes,counts_by_stream_family_bw_direction=list(joint.values()),
        missing_requested_streams=[p for p in directories if p not in stream_positions],
        correction_statistics=stats,
        normalized_gt8=dict(uniform=clip_statistics(outside,normalized_abs,uniform),
            balanced=clip_statistics(outside,normalized_abs,weights)),
        tail_mask=dict(definition='Padding means a chunk action position lies after the saved successful continuation end; first-step targets are always real.',
            uniform=mask_statistics(valid_lengths,uniform,args.horizon),
            balanced=mask_statistics(valid_lengths,weights,args.horizon)),
        by_stream=by_stream,balanced_sampling=sampling_report,
        first_step_labels=dict(scope='One real executed correction at every supervised chunk start; also checked the first start of each episode.',
            count=len(indices),episode_first_start_count=len(first_indices),
            episode_first_history_latest_matches_labeled_feature=True,
            episode_first_chunk_action_matches_saved_correction=True,
            all_first_step_masks_valid=True,
            dtype=str(actions.dtype),history_prefix_frames=3,
            loader_contract='Every episode was checked by existing validate_episode: train split, clean success, finite arrays, float32 totals and target, matching history prefix, executed_total-prior_total within atol 2e-6.',
            teacher_actions_recomputed=False,physical_alignment_inferred_from_reexecution=False),
        current_first_step_geometry=dict(contract=contract['geometry_version'],
            feature_slots='Last 16 values: four encoded SAT gaps then twelve heading-local directions',
            min_gap=distribution(minimum_gap),per_foot_gap=[distribution(gaps[:,j]) for j in range(4)],
            frames_min_gap_below_10cm=int((minimum_gap<.10).sum()),
            frames_min_gap_below_5cm=int((minimum_gap<.05).sum()),
            frames_min_gap_below_zero=int((minimum_gap<0).sum()),
            interpretation='These are stored, clamped [-.2,1] SAT observation features; they are not recomputed exact mesh distances or contact outcomes.'),
        recorded_contract_from_first_episode={key:contract.get(key) for key in keys},
        teacher_provenance_from_first_episode=contract.get('teacher_provenance'),
        shared_contract_signature=json.loads(signature(contract)),
        contract_validation='Existing load_episodes rejects mixed teacher SHA, prior SHA, bank/meta SHA, feature/action and geometry identities.',
        file_manifest=files,
        file_manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        inspection_sources={name:file_sha256(runtime/name) for name in
            ('inspect_surface_pilot_data.py','dataset.py','dataset_sampling.py','contracts.py','train_student.py','model.py')},
        existing_source_snapshot_sha256=file_sha256(snapshot) if snapshot.is_file() else None,
        automatic_changes=False,simulator_runs=0,optimizer_steps=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',nargs='+',required=True)
    parser.add_argument('--output',required=True,help='New JSON report path; input data are never written')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--horizon',type=int,default=6)
    args=parser.parse_args()
    if min(args.threads,args.horizon)<1:parser.error('Positive threads and horizon required')
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Use a new inspection JSON path')
    try:
        with torch.no_grad():report=inspect(args)
    except Exception as error:
        report=dict(status='DATA_INSPECTION_ERROR',error_type=type(error).__name__,error=str(error),
            input_directories=args.data,device='cpu',automatic_changes=False)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report),file=sys.stderr);raise SystemExit(1)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],output=str(output),totals=report['totals'],
        normalized_gt8_frame_probability=report['normalized_gt8']['uniform']['frame_probability'],
        tail_padding_probability=report['tail_mask']['balanced']['starts_with_any_padding_probability'])))


if __name__=='__main__':main()
