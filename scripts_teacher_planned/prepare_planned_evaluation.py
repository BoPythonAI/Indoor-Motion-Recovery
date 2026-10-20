"""Prepare one original-goal, 24-slot evaluation. Never starts the simulator."""
import argparse
import json
from pathlib import Path
import shlex
import torch

from motion_reference import file_sha256
from reference_goal_contract import PLAN_VERSION, validate_plan_status
from surface_contract import POLICY_KIND, FEATURE_VERSION


def prepare(args):
    source_path = Path(args.source_config).resolve()
    cfg = json.loads(source_path.read_text())
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    experimental = validate_plan_status(manifest)
    rows = json.loads(Path(cfg['motion_metadata']).read_text())
    if len(rows)!=1 or len(manifest['sources'])!=1 or str(rows[0]['motion_id'])!=manifest['sources'][0]['source_id']:
        raise ValueError('Use the original single-source config matching this manifest')
    if file_sha256(cfg['motion_metadata']) != manifest['original_metadata_sha256']:
        raise ValueError('Original metadata changed')
    if file_sha256(cfg['motion_bank']) != manifest['original_bank_sha256']:
        raise ValueError('Source config must still identify the original unplanned bank')
    original = Path(args.original_bank or manifest['original_bank']).resolve()
    planned = Path(args.planned_bank or manifest['planned_bank']).resolve()
    if file_sha256(original)!=manifest['original_bank_sha256'] or file_sha256(planned)!=manifest['planned_bank_sha256']:
        raise ValueError('Actual original/planned bank bytes do not match the manifest')
    default_checkpoint = Path(__file__).resolve().parent.parent/'checkpoints/teacher_surface_v1/train/update_0050.pt'
    checkpoint = Path(args.checkpoint or default_checkpoint).resolve()
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('policy_kind') != POLICY_KIND or state.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Use the selected surface teacher, with unchanged feature/geometry semantics')
    if args.checkpoint is None and state.get('update')!=50:
        raise ValueError('The default checkpoint must be the immutable surface50 snapshot')
    if state['config'].get('stop_reference_version') != cfg.get('stop_reference_version'):
        raise ValueError('Original freeze/terrain reference convention differs from the teacher')
    if state['config'].get('surface_mjcf_sha256') != manifest.get('mjcf_sha256'):
        raise ValueError('Planning collision geometry differs from the selected teacher')
    canonical = [dict(motion=0,bw=bw,direction=d,phase=p)
                 for bw in [0.,.45,.75] for d in range(4) for p in [.3,.6]]
    if cfg.get('cases') and cfg['cases'] != canonical:
        raise ValueError('This A/B uses the original canonical 24-slot order; source cases differ')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new A/B output directory')
    cfg.update(mode='evaluate',resume=False,teacher_kind='terrain_surface',method='terrain_surface',
        checkpoint=str(checkpoint),feature_version=FEATURE_VERSION,output=str(output),
        motion_bank=str(planned),motion_bank_sha256=manifest['planned_bank_sha256'],
        motion_metadata_sha256=manifest['original_metadata_sha256'],
        reference_planning_version=PLAN_VERSION,reference_plan_manifest=str(manifest_path),
        reference_plan_status=manifest['status'],reference_plan_experimental_export=experimental,
        reference_plan_geometry_qualified=manifest.get('geometry_qualified'),
        reference_plan_manifest_sha256=file_sha256(manifest_path),reference_original_bank=str(original),
        strengths=[0.,.45,.75],cases=canonical,record=bool(args.record),record_env=args.record_env)
    if not 0<=args.record_env<24:
        raise ValueError('record-env must be an original slot from 0 to 23')
    output.mkdir(parents=True)
    config_path = output/'planned_evaluation.json'
    config_path.write_text(json.dumps(cfg,indent=2)+'\n')
    runtime = Path(__file__).resolve().parent
    command = ['bash',str(runtime/'launch.sh'),str(config_path),'24',str(planned),cfg['tracker_checkpoint']]
    provenance = dict(source_config=str(source_path),source_config_sha256=file_sha256(source_path),
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),checkpoint_update=state['update'],
        original_bank_sha256=manifest['original_bank_sha256'],planned_bank_sha256=manifest['planned_bank_sha256'],
        reference_plan_manifest_sha256=file_sha256(manifest_path),num_envs=24,cases=canonical,
        reference_plan_status=manifest['status'],experimental_export=experimental,
        geometry_qualified=manifest.get('geometry_qualified'),manifest_verbatim=manifest,
        execution='NOT_RUN',command=command)
    (output/'preparation.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'command.txt').write_text(shlex.join(command)+'\n')
    print(json.dumps(provenance,indent=2))
    return cfg,provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--original-bank');parser.add_argument('--planned-bank')
    parser.add_argument('--record',action='store_true');parser.add_argument('--record-env',type=int,default=0)
    prepare(parser.parse_args())
