"""Run V4 in the original, unchanged ProtoMotions runtime."""
import os
import runpy
import sys
from pathlib import Path
import json
from motion_reference import actual_motion_argument, resolve_motion_reference

entry_config=json.loads(Path(os.environ['V4_CONFIG']).read_text())
verified=resolve_motion_reference(entry_config,actual_motion_argument(sys.argv[1:]),require_version=False)
os.environ['V4_VERIFIED_MOTION_REFERENCE']=json.dumps(verified)

VENDOR = Path(os.environ.get('V4_VENDOR', '/root/autodl-tmp/ProtoMotions'))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    from scene_adapter import install
    install()
    from scene_heightmap import install as install_scene_heightmap
    install_scene_heightmap()
    robot = kwargs['robot_config']
    robot.contact_bodies = list(robot.kinematic_info.body_names)
    return build_original(**kwargs)
cb.build_all_components = build_with_contacts
def execute(self, **kwargs):
    from recovery_runtime import run
    return run(self.agent)
BaseEvaluator.simple_test_policy = execute
runpy.run_path(str(VENDOR / 'protomotions/inference_agent.py'), run_name='__main__')
