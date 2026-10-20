"""Explicit warm100 -> surface initialization. Writes weights/config, never trains."""
import argparse
import copy
import json
from pathlib import Path
import torch

from motion_reference import file_sha256
from surface_contract import (OLD_FEATURE_VERSION, FEATURE_VERSION, POLICY_KIND, GEOMETRY_VERSION,
                              PROXIMITY_REWARD, EXTRA_PER_FRAME, HISTORY_FRAMES)
from warm_policy import WarmResidualPolicy


def insert_surface_columns(weight, frame_dim):
    if weight.shape[1]!=HISTORY_FRAMES*frame_dim:
        raise ValueError('Original first-layer weight is not a four-frame history')
    expanded=weight.new_zeros((weight.shape[0],HISTORY_FRAMES*(frame_dim+EXTRA_PER_FRAME)))
    for frame in range(HISTORY_FRAMES):
        expanded[:,frame*(frame_dim+EXTRA_PER_FRAME):frame*(frame_dim+EXTRA_PER_FRAME)+frame_dim]=\
            weight[:,frame*frame_dim:(frame+1)*frame_dim]
    return expanded


@torch.no_grad()
def check_initial_equivalence(old_state,new_state,seed=817):
    """Use arbitrary extra observations and nonzero priors; preserve the caller RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        old=WarmResidualPolicy(**old_state['model_config']).cpu().eval()
        new=WarmResidualPolicy(**new_state['model_config']).cpu().eval()
        old.load_state_dict(old_state['policy']);new.load_state_dict(new_state['policy'])
        d=old.obs_dim//HISTORY_FRAMES
        history=torch.randn(32,HISTORY_FRAMES,d)
        extra=torch.randn(32,HISTORY_FRAMES,EXTRA_PER_FRAME)*3
        expanded=torch.cat([history,extra],-1).flatten(1)
        prior=torch.randn(32,69)*.2
        a,v=old(history.flatten(1),prior)
        b,w=new(expanded,prior)
        mean_delta=float((a.mean-b.mean).abs().max())
        value_delta=float((v-w).abs().max())
        torch.testing.assert_close(a.mean,b.mean,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(v,w,atol=1e-5,rtol=1e-6)
        if not torch.equal(a.stddev,b.stddev):raise ValueError('Migration changed exploration scale')
        return dict(max_mean_abs_error=mean_delta,max_value_abs_error=value_delta,
                    mean_bitwise_equal=torch.equal(a.mean,b.mean),value_bitwise_equal=torch.equal(v,w),
                    arbitrary_extra_features=True,original_frame_dim=d,new_frame_dim=d+EXTRA_PER_FRAME,
                    comparison='float32 CPU, including nonzero random prior and all four history frames')


def migrate(args):
    source=Path(args.checkpoint).resolve();output=Path(args.output).resolve();mjcf=Path(args.mjcf).resolve()
    if output.exists():raise ValueError('Use a new migration checkpoint path')
    old=torch.load(source,map_location='cpu',weights_only=False)
    if old.get('policy_kind')!='terrain_prior_correction_v1' or old.get('feature_version')!=OLD_FEATURE_VERSION:
        raise ValueError('Migration requires the original warm correction feature/checkpoint contract')
    if int(old.get('update',-1))!=100:
        raise ValueError('This evidence-based branch explicitly initializes from warm100')
    old_dim=old['model_config']['obs_dim']
    if old_dim%HISTORY_FRAMES:raise ValueError('Original obs_dim is not a four-frame history')
    new=copy.deepcopy(old)
    new.pop('_loaded_checkpoint_path',None);new.pop('_loaded_checkpoint_sha256',None)
    for name in ('actor.0.weight','critic.0.weight'):
        new['policy'][name]=insert_surface_columns(old['policy'][name],old_dim//HISTORY_FRAMES)
    new['model_config']['obs_dim']=old_dim+HISTORY_FRAMES*EXTRA_PER_FRAME
    migration=dict(version='warm100_zero_surface_columns_v1',source_checkpoint=str(source),
        source_checkpoint_sha256=file_sha256(source),source_update=100,source_policy_kind=old['policy_kind'],
        source_feature_version=OLD_FEATURE_VERSION,feature_version=FEATURE_VERSION,
        inherited_weights='All except actor/critic first input layers; those retain original per-frame columns plus 16 zero columns per frame',
        optimizer_reset=True,new_update=0,geometry_version=GEOMETRY_VERSION,
        surface_mjcf_path=str(mjcf),surface_mjcf_sha256=file_sha256(mjcf))
    config=new['config']
    for key in ('teacher_provenance','teacher_identity','checkpoint','resume_episode_reset'):
        config.pop(key,None)
    config.update(mode='train',teacher_kind='terrain_surface',method='terrain_surface',
        feature_version=FEATURE_VERSION,geometry_version=GEOMETRY_VERSION,
        surface_mjcf_path=str(mjcf),surface_mjcf_sha256=migration['surface_mjcf_sha256'],
        proximity_reward=PROXIMITY_REWARD,migration=migration,resume=False,updates=50,
        output=str(Path(args.training_output).resolve() if args.training_output else output.parent/'train'))
    new.update(policy_kind=POLICY_KIND,feature_version=FEATURE_VERSION,update=0,optimizers=None,
        training_seconds=0.,migration=migration,geometry_version=GEOMETRY_VERSION,
        surface_mjcf_sha256=migration['surface_mjcf_sha256'])
    check=check_initial_equivalence(old,new)
    migration['initial_equivalence']=check
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp');torch.save(new,temporary);temporary.replace(output)
    train_config=copy.deepcopy(config);train_config['checkpoint']=str(output)
    config_path=output.with_name(output.stem+'_train.json')
    config_path.write_text(json.dumps(train_config,indent=2)+'\n',encoding='utf-8')
    report=dict(migration,output_checkpoint=str(output),output_checkpoint_sha256=file_sha256(output),
                training_config=str(config_path),status='MIGRATED_NOT_TRAINED')
    report_path=output.with_name(output.stem+'_migration.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('SURFACE_MIGRATION_COMPLETE',json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mjcf',required=True,help='Actual original robot collision MJCF used by the simulator')
    parser.add_argument('--training-output')
    migrate(parser.parse_args())
