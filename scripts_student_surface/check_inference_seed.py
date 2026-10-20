"""CPU checks: sampling-seed override preserves legacy and global RNG behavior."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import controller
from model import ChunkPolicy


class Task:
    device='cpu'
    n=2
    feature_version='synthetic_seed_fixture'
    agent=None
    _pending_prior=None
    def set_prior(self, value):self._pending_prior=value


class Prior:
    def current(self):return SimpleNamespace(delta=torch.zeros(2,69))


def run():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    model=ChunkPolicy(16,'diffusion',horizon=2)
    state=dict(obs_dim=16,kind='diffusion',horizon=2,model=model.state_dict(),ema_model=model.state_dict())
    history=torch.arange(32,dtype=torch.float32).reshape(2,16)/32
    fake_prior_module=ModuleType('warm_prior')
    fake_prior_module.FrozenTerrainPrior=lambda *args:Prior()
    with tempfile.TemporaryDirectory(prefix='private_inference_seed_') as temp:
        checkpoint=Path(temp)/'student.pt';torch.save(state,checkpoint)
        base=dict(student_checkpoint=str(checkpoint),rollin_student_checkpoint=str(checkpoint),
                  method='diffusion',rollin_method='diffusion',student_weights='ema',seed=17,
                  execute_steps=1,rollin_execute_steps=1,inference_steps=2,rollin_inference_steps=2)

        def trial(overrides,rollin=False):
            cfg=dict(base,**overrides);before_cfg=copy.deepcopy(cfg)
            torch.manual_seed(991)
            before=torch.get_rng_state().clone()
            with patch.object(controller,'validate_checkpoint',lambda *args:None), \
                 patch.object(controller,'validate_runtime',lambda *args:None), \
                 patch.dict(sys.modules,{'warm_prior':fake_prior_module}):
                actor=controller.make_student(Task(),cfg,prior=Prior(),checkpoint_key=(
                    'rollin_student_checkpoint' if rollin else 'student_checkpoint'))
            after_load=torch.get_rng_state().clone()
            # Legacy load_model constructs then restores the network and thus
            # consumes initialization RNG. Its behavior must remain identical
            # across private seeds; do not suppress it and change old physics RNG.
            assert not torch.equal(before,after_load)
            actions=[]
            for _ in range(2):
                actions.append(actor(history)[0].mean.clone())
                assert torch.equal(torch.get_rng_state(),after_load)
            assert cfg==before_cfg and cfg['seed']==17
            return torch.stack(actions),after_load,actor.inference_seed

        legacy=trial({})
        explicit=trial({'inference_seed':17})
        different=trial({'inference_seed':23})
        alternate=trial({'inference_seed':42})
        fallback=trial({'inference_seed':None})
        rollin=trial({'inference_seed':42,'rollin_seed':23},True)
        rollin_default=trial({'inference_seed':42},True)
        assert torch.equal(legacy[0],explicit[0]) and torch.equal(legacy[0],fallback[0])
        assert not torch.equal(legacy[0],different[0]) and not torch.equal(different[0],alternate[0])
        assert torch.equal(rollin[0],different[0]) and torch.equal(rollin_default[0],legacy[0])
        assert all(torch.equal(legacy[1],x[1]) for x in (explicit,different,alternate,fallback,rollin,rollin_default))
        assert [legacy[2],different[2],alternate[2],rollin[2],rollin_default[2]]==[17,23,42,23,17]

    # Reuse the existing configuration fixture; no physics/model inference is
    # launched by prepare(). It also checks immutable cases and task criteria.
    surface=Path(__file__).resolve().parent.parent/'scripts_teacher_surface'
    sys.path.append(str(surface))
    import check_surface_student_eval as config_check
    original_prepare=config_check.prepare
    prepared=[]
    def checked_prepare(args):
        args.inference_seed=23
        manifest=original_prepare(args)
        cfg=json.loads(Path(manifest['config']).read_text())
        assert cfg['seed']==cfg['physics_seed']==manifest['physics_seed']==17
        assert cfg['inference_seed']==manifest['inference_seed']==23
        prepared.append(cfg)
        return manifest
    with patch.object(config_check,'prepare',checked_prepare), contextlib.redirect_stdout(io.StringIO()):
        config_check.run()
    assert len(prepared)==2
    print(json.dumps(dict(default_matches_legacy_bitwise=True,explicit_17_matches_legacy=True,
        private_seeds_23_42_change_diffusion_samples=True,global_rng_identical_across_seeds=True,
        inference_consumes_no_global_rng=True,legacy_model_load_rng_consumption_preserved=True,
        rollin_seed_separate_with_cfg_seed_fallback=True,physics_seed_and_cases_preserved=True,
        preparation_records_both_seeds=True,simulator_runs=0),indent=2))


if __name__=='__main__':run()
