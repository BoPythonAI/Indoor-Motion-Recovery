"""Prepare the original V5 flat baseline on the same planned reference; never run it."""
import argparse
import copy
import json
from pathlib import Path
import shlex

from motion_reference import file_sha256, resolve_motion_reference
from reference_goal_contract import PLAN_VERSION, validate_plan_status


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scenes(config):
    specs=config.get('scene_specs') or [dict(scene_dir=config.get('scene_dir'),forbidden_id=config.get('forbidden_id'))]
    if any(not row.get('scene_dir') or row.get('forbidden_id') is None for row in specs):
        raise ValueError('Both inputs need explicit scene/forbidden instance bindings')
    return [(str(Path(row['scene_dir']).resolve()),int(row['forbidden_id'])) for row in specs]


def prepare(args):
    source=Path(args.source_baseline_config).resolve()
    planned_path=Path(args.planned_evaluation_config).resolve()
    base,planned=read(source),read(planned_path)
    if base.get('method')!='base' or base.get('mode')!='evaluate':
        raise ValueError('Source must be the original evaluate/method=base config')
    if base.get('feature_version','v3_terrain_reference')!='v3_terrain_reference':
        raise ValueError('Source must use the V5 terrain-reference flat baseline')
    if planned.get('mode')!='evaluate' or planned.get('reference_planning_version')!=PLAN_VERSION:
        raise ValueError('Select the corresponding prepared planned evaluation config')
    if scenes(base)!=scenes(planned):
        raise ValueError('Original baseline and planned evaluation scene/forbidden bindings differ')
    if base.get('seed',17)!=planned.get('seed',17):
        raise ValueError('Preserve the same original baseline/paired-evaluation seed')
    reference=resolve_motion_reference(planned)
    if file_sha256(base['motion_metadata'])!=reference['motion_metadata_sha256']:
        raise ValueError('Original baseline and planned evaluation metadata differ')
    rows=read(reference['motion_metadata'])
    if len(rows)!=1:
        raise ValueError('Prepare one independent source at a time, with 24 original slots')
    if base.get('stop_reference_version',reference['stop_reference_version'])!=reference['stop_reference_version']:
        raise ValueError('Freeze/reference convention changed')
    manifest_path=Path(planned['reference_plan_manifest']).resolve()
    manifest=read(manifest_path)
    experimental=validate_plan_status(manifest)
    if manifest.get('plan_version')!=PLAN_VERSION:
        raise ValueError('Unrecognized planning manifest version')
    manifest_sha=file_sha256(manifest_path)
    if planned.get('reference_plan_manifest_sha256',manifest_sha)!=manifest_sha:
        raise ValueError('Planning manifest changed after preparing the teacher evaluation')
    if (reference['motion_bank_sha256']!=manifest['planned_bank_sha256'] or
            reference['motion_metadata_sha256']!=manifest['original_metadata_sha256']):
        raise ValueError('Actual planned bank/metadata differs from the shared planning manifest')
    if len(manifest['sources'])!=1 or str(manifest['sources'][0]['source_id'])!=str(rows[0]['motion_id']):
        raise ValueError('Planning source differs from the original baseline source')
    original=Path(planned.get('reference_original_bank',manifest['original_bank'])).resolve()
    if file_sha256(original)!=manifest['original_bank_sha256']:
        raise ValueError('Original reference bank changed')
    if base.get('motion_bank') and file_sha256(base['motion_bank'])!=manifest['original_bank_sha256']:
        raise ValueError('Source baseline did not use the original unplanned bank')
    flat=Path(planned['tracker_checkpoint']).resolve();flat_sha=file_sha256(flat)
    if base.get('tracker_checkpoint') and file_sha256(base['tracker_checkpoint'])!=flat_sha:
        raise ValueError('Frozen flat tracker differs from the original baseline')
    mjcf=Path(planned['surface_mjcf_path']).resolve();mjcf_sha=file_sha256(mjcf)
    if mjcf_sha!=planned['surface_mjcf_sha256'] or mjcf_sha!=manifest['mjcf_sha256']:
        raise ValueError('Actual MJCF differs from the shared planning geometry')
    canonical=[dict(motion=0,bw=bw,direction=d,phase=p) for bw in [0.,.45,.75] for d in range(4) for p in [.3,.6]]
    for label,config in [('Original baseline',base),('Planned evaluation',planned)]:
        if config.get('cases') and config['cases']!=canonical:
            raise ValueError(label+' does not use the original canonical 24-slot order')
        if config.get('strengths',[0.,.45,.75])!=[0.,.45,.75]:
            raise ValueError(label+' has a different strength matrix')
    runtime=Path(__file__).resolve().parent
    if not (runtime/'launch.sh').is_file():raise ValueError('Missing selected planned runtime launch.sh')
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Use a new baseline output directory')
    # Start from the baseline. No teacher PPO/force/curriculum fields are merged.
    cfg=copy.deepcopy(base)
    removed=[]
    for key in ('checkpoint','resume','student_checkpoint'):
        if key in cfg:removed.append(key);cfg.pop(key)
    cfg.update(reference,mode='evaluate',method='base',feature_version='v3_terrain_reference',
        output=str(output),cases=canonical,strengths=[0.,.45,.75],
        tracker_checkpoint=str(flat),surface_mjcf_path=str(mjcf),surface_mjcf_sha256=mjcf_sha,
        reference_planning_version=PLAN_VERSION,reference_plan_manifest=str(manifest_path),
        reference_plan_manifest_sha256=manifest_sha,reference_original_bank=str(original),
        reference_original_bank_sha256=manifest['original_bank_sha256'],
        reference_planned_bank_sha256=manifest['planned_bank_sha256'],
        reference_plan_status=manifest['status'],reference_plan_experimental_export=experimental,
        reference_plan_geometry_qualified=manifest.get('geometry_qualified'))
    if 'geometry_version' in planned:cfg['geometry_version']=planned['geometry_version']
    output.mkdir(parents=True)
    config_path=output/'planned_baseline.json'
    config_path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    command=['bash',str(runtime/'launch.sh'),str(config_path),'24',reference['motion_bank'],str(flat)]
    changed={key:dict(before=base.get(key),after=value) for key,value in cfg.items() if key not in base or base[key]!=value}
    provenance=dict(execution='NOT_RUN',method='base',source_id=str(rows[0]['motion_id']),
        source_baseline_config=str(source),source_baseline_config_sha256=file_sha256(source),
        planned_evaluation_config=str(planned_path),planned_evaluation_config_sha256=file_sha256(planned_path),
        config=str(config_path),config_sha256=file_sha256(config_path),
        tracker_checkpoint=str(flat),tracker_checkpoint_sha256=flat_sha,
        original_flat_checkpoint_hash_comparable=bool(base.get('tracker_checkpoint')),
        surface_mjcf_sha256=mjcf_sha,reference_plan_manifest_sha256=manifest_sha,
        original_bank_sha256=manifest['original_bank_sha256'],planned_bank_sha256=manifest['planned_bank_sha256'],
        reference_plan_status=manifest['status'],experimental_export=experimental,
        geometry_qualified=manifest.get('geometry_qualified'),manifest_verbatim=manifest,
        num_envs=24,cases=canonical,seed=cfg.get('seed',17),
        removed_baseline_keys=removed,changed_baseline_fields=changed,
        controller='Original V5 evaluate(task, None); zero residual with original flat tracker',
        initialization='Original V5 task reset and unused residual-policy construction retained by dispatch',
        command=command)
    (output/'baseline_preparation.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    (output/'command.txt').write_text(shlex.join(command)+'\n',encoding='utf-8')
    print(json.dumps(provenance,indent=2))
    return cfg,provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-baseline-config',required=True)
    parser.add_argument('--planned-evaluation-config',required=True)
    parser.add_argument('--output',required=True)
    prepare(parser.parse_args())
