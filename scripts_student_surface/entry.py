"""Original inference entry with the warm task and correction student adapter."""
import json
import os
import runpy
import sys
from pathlib import Path
from warm_dependency import install

warm_root = install()
from motion_reference import actual_motion_argument, resolve_motion_reference
config = json.loads(Path(os.environ['V4_CONFIG']).read_text())
verified = resolve_motion_reference(config, actual_motion_argument(sys.argv[1:]), require_version=True)
indices = [i for i, value in enumerate(sys.argv) if value == '--checkpoint']
if len(indices) != 1 or indices[0] + 1 >= len(sys.argv):
    raise ValueError('Bind exactly one actual frozen flat tracker checkpoint')
if Path(sys.argv[indices[0] + 1]).resolve() != Path(config['tracker_checkpoint']).resolve():
    raise ValueError('Actual --checkpoint and tracker_checkpoint provenance differ')
os.environ['V4_VERIFIED_MOTION_REFERENCE'] = json.dumps(verified)
vendor = Path(os.environ.get('V4_VENDOR', '/root/autodl-tmp/ProtoMotions'))
sys.path.insert(2, str(vendor))
from protomotions.utils.simulator_imports import import_simulator_before_torch
import_simulator_before_torch('isaaclab')
from lightning.fabric.strategies import SingleDeviceStrategy
import protomotions.utils.fabric_config as fc
Original = fc.FabricConfig


class SingleGPU(Original):
    def __post_init__(self):
        super().__post_init__()
        self.strategy = SingleDeviceStrategy(device='cuda:0')


fc.FabricConfig = SingleGPU
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
import protomotions.utils.component_builder as cb
build_original = cb.build_all_components


def build_with_contacts(**kwargs):
    from scene_adapter import install as scene_install
    from scene_heightmap import install as heightmap_install
    scene_install(); heightmap_install()
    robot = kwargs['robot_config']
    robot.contact_bodies = list(robot.kinematic_info.body_names)
    return build_original(**kwargs)


cb.build_all_components = build_with_contacts


def execute(self, **kwargs):
    from student_runtime import run
    return run(self.agent)


BaseEvaluator.simple_test_policy = execute
runpy.run_path(str(vendor / 'protomotions/inference_agent.py'), run_name='__main__')
