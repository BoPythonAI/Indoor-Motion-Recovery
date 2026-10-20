"""Small CPU migration check using learned nonzero actor/critic weights."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import torch
from warm_policy import WarmResidualPolicy
from migrate_warm_checkpoint import migrate
from surface_contract import OLD_FEATURE_VERSION, FEATURE_VERSION, POLICY_KIND


def run():
    torch.set_num_threads(2);torch.manual_seed(13)
    with tempfile.TemporaryDirectory(prefix='surface_migration_') as directory:
        root=Path(directory)
        model=WarmResidualPolicy(4*37,width=32)
        torch.nn.init.normal_(model.actor[-1].weight,std=.1)
        torch.nn.init.normal_(model.critic[-1].weight,std=2.)
        model.critic[-1].bias.data.fill_(100.)
        state=dict(policy_kind='terrain_prior_correction_v1',feature_version=OLD_FEATURE_VERSION,
            model_config=model.model_config,policy=model.state_dict(),config={'feature_version':OLD_FEATURE_VERSION},
            update=100,optimizers={'synthetic':'old moments'},frozen_prior={'flat_sha256':'a'*64,'terrain_sha256':'b'*64})
        source=root/'warm100.pt';torch.save(state,source)
        xml=root/'robot.xml';xml.write_text('<mujoco/>')
        output=root/'surface_initial.pt'
        report=migrate(argparse.Namespace(checkpoint=str(source),output=str(output),mjcf=str(xml),training_output=None))
        loaded=torch.load(output,map_location='cpu',weights_only=False)
        assert loaded['update']==0 and loaded['optimizers'] is None
        assert loaded['feature_version']==FEATURE_VERSION and loaded['policy_kind']==POLICY_KIND
        assert loaded['frozen_prior']==state['frozen_prior']
        assert loaded['migration']['source_update']==100
        assert loaded['migration']['source_checkpoint_sha256']!=report['output_checkpoint_sha256']
        for key,value in state['policy'].items():
            if key in ('actor.0.weight','critic.0.weight'):
                old=value.reshape(len(value),4,37)
                new=loaded['policy'][key].reshape(len(value),4,53)
                assert torch.equal(old,new[:,:,:37]) and torch.count_nonzero(new[:,:,37:])==0
            else:assert torch.equal(value,loaded['policy'][key])
        config=json.loads(Path(report['training_config']).read_text())
        assert config['checkpoint']==str(output) and not config['resume']
        assert config['teacher_kind']=='terrain_surface' and config['output']!=state['config'].get('output')
        print('SURFACE_MIGRATION_CPU_PASSED',json.dumps(report['initial_equivalence']))


if __name__=='__main__':run()
