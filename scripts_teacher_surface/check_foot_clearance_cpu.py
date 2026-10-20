"""CPU geometry checks, no simulator/teacher/student changes or GPU work."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from foot_clearance import FootClearance, quaternion_matrix_xyzw, FOOT_NAMES

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal_reference_probe import box_aabb_margin


def main():
    torch.manual_seed(211)
    with tempfile.TemporaryDirectory() as directory:
        xml = Path(directory)/'foot.xml'
        xml.write_text('<mujoco><worldbody>'+''.join(
            f'<body name="{name}"><geom type="box" size=".1 .05 .04" pos=".03 0 0" quat="1 0 0 0"/></body>'
            for name in FOOT_NAMES)+'</worldbody></mujoco>')
        names = ['Pelvis', *FOOT_NAMES]
        module = FootClearance(xml,names)
        pos = torch.zeros(2,5,3); quat = torch.zeros(2,5,4);quat[...,3]=1
        center = torch.tensor([[.33,0,0],[.33,0,0]])
        half = torch.tensor([[.1,.2,.2],[.1,.2,.2]])
        result = module(pos,quat,center,half)
        torch.testing.assert_close(result.gaps,torch.full((2,4),.1),atol=1e-7,rtol=0)
        torch.testing.assert_close(result.centers_world,torch.tensor([.03,0,0])[None,None].expand(2,4,3))
        torch.testing.assert_close(result.directions_world,torch.tensor([1.,0,0])[None,None].expand(2,4,3))
        shift = torch.tensor([[20.,-12.,4.],[-19.,4.,-2.]])
        translated = module(pos+shift[:,None],quat,center+shift,half)
        torch.testing.assert_close(translated.gaps,result.gaps,atol=2e-6,rtol=0)
        torch.testing.assert_close(translated.centers_world,result.centers_world+shift[:,None])
        # Root yaw pi/2: world +X should be local -Y.
        root_quat = torch.tensor([[0.,0.,2**-.5,2**-.5]]).expand(2,4)
        feature = module.features(result,root_quat)
        assert feature.shape == (2,16)
        torch.testing.assert_close(feature[:,4:].reshape(2,4,3),torch.tensor([0.,-1.,0.])[None,None].expand(2,4,3),atol=2e-7,rtol=0)
        distances=torch.full((2,24),.3);distances[1,0]=.02
        torch.testing.assert_close(module.proximity(distances,result),torch.tensor([.1,.02]),atol=1e-7,rtol=0)
        # Coincident centers and parallel SAT axes must remain finite.
        overlap=module(pos,quat,result.centers_world[:,0],half)
        assert (overlap.gaps<0).all()
        assert torch.isfinite(overlap.gaps).all() and torch.isfinite(module.features(overlap,root_quat)).all()
        torch.testing.assert_close(module.proximity(distances,overlap),torch.zeros(2))
        # Random rotations and translations: SAT separation must not overstate
        # the independent exact OBB/AABB Euclidean distance.
        n=24
        pos=torch.randn(n,5,3)*.15
        quat=torch.randn(n,5,4);quat=quat/quat.norm(dim=-1,keepdim=True)
        center=torch.randn(n,3)*.4;half=torch.rand(n,3)*.15+.03
        result=module(pos,quat,center,half)
        signs=torch.tensor(list(itertools.product([-1.,1.],repeat=3)))
        body_rot=quaternion_matrix_xyzw(quat[:,module.foot_ids])
        collider_rot=body_rot@module.local_rotations[None]
        corners=torch.matmul(collider_rot[:,:,None],(signs[None,None]*module.local_halves[None,:,None])[...,None]).squeeze(-1)+result.centers_world[:,:,None]
        upper_error=-np.inf;separated=0
        for env in range(n):
            bounds=torch.stack([center[env]-half[env],center[env]+half[env]]).numpy()
            for foot in range(4):
                exact=box_aabb_margin(corners[env,foot].numpy().astype(float),bounds.astype(float))
                error=float(result.gaps[env,foot])-exact
                upper_error=max(upper_error,error)
                assert error <= 2e-6,(env,foot,error)
                separated+=int(exact>0)
        assert torch.isfinite(result.gaps).all() and torch.isfinite(result.centers_world).all()
        torch.testing.assert_close(result.directions_world.norm(dim=-1),torch.ones(n,4),atol=2e-6,rtol=0)
        projection=(result.directions_world*(center[:,None]-result.centers_world)).sum(-1)
        assert projection.min()>=-1e-7
        # Local MJCF geom rotation and offset both rotate with the body pose.
        xml.write_text('<mujoco><worldbody>'+''.join(
            f'<body name="{name}"><geom type="box" size=".1 .05 .04" pos=".03 0 0" quat=".7071067812 0 0 .7071067812"/></body>'
            for name in FOOT_NAMES)+'</worldbody></mujoco>')
        rotated_module=FootClearance(xml,names)
        p=torch.zeros(1,5,3);q=torch.zeros(1,5,4);q[...,3]=1
        q[:,1:]=torch.tensor([0.,0.,2**-.5,2**-.5])
        rotated=rotated_module(p,q,torch.tensor([[.4,0,0]]),torch.tensor([[.1,.2,.2]]))
        torch.testing.assert_close(rotated.centers_world,torch.tensor([0.,.03,0.])[None,None].expand(1,4,3),atol=1e-7,rtol=0)
        torch.testing.assert_close(rotated.gaps,torch.full((1,4),.2),atol=2e-7,rtol=0)
    print(json.dumps(dict(known_gap_m=.1,translation_invariant=True,geom_pose_and_body_rotation_checked=True,
        feature_shape=[2,16],heading_direction_checked=True,proximity_min_contract_checked=True,
        sat_vs_independent_exact_cases=n*4,separated_cases=separated,
        largest_sat_minus_exact_m=upper_error,finite_parallel_and_overlap_cases=True,
        simulator_runs=0,gpu_runs=0),indent=2))


if __name__=='__main__':main()
