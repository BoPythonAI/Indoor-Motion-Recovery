"""Current-state foot-box proximity to the forbidden object's conservative AABB.

The maximum signed gap over 15 normalized SAT axes is a separating-distance
LOWER BOUND, not exact Euclidean distance and not distance to the object mesh.
Negative/zero means box/AABB overlap. No mesh, goal, contact or success rule is
changed. All runtime operations are vectorized torch and support CUDA tensors.
"""
from typing import NamedTuple
import xml.etree.ElementTree as ET
import torch
from torch import nn

FOOT_NAMES = ('L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe')


class FootClearanceResult(NamedTuple):
    gaps: torch.Tensor                 # [N,4], maximum normalized signed SAT gap, meters
    directions_world: torch.Tensor    # [N,4,3], selected axis oriented toward the obstacle
    centers_world: torch.Tensor       # [N,4,3], true collision geom centers, not body origins


def quaternion_matrix_xyzw(quaternion):
    """Quaternion -> local-to-world matrix, supporting arbitrary leading axes."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class FootClearance(nn.Module):
    def __init__(self, mjcf_path, body_names):
        super().__init__()
        names = list(body_names)
        root = ET.parse(mjcf_path).getroot()
        centers, halves, rotations = [], [], []
        for name in FOOT_NAMES:
            body = next((node for node in root.findall('.//body') if node.attrib.get('name') == name), None)
            if body is None or name not in names:
                raise ValueError(f'Missing required original foot body: {name}')
            geoms = body.findall('geom')
            if len(geoms) != 1 or geoms[0].attrib.get('type') != 'box':
                raise ValueError(f'{name}: expected the original single-box collision geometry')
            attributes = geoms[0].attrib
            if any(key in attributes for key in ('euler', 'axisangle', 'xyaxes', 'zaxis', 'fromto')):
                raise ValueError(f'{name}: use the original explicit quaternion box geometry')
            values = lambda key, default: torch.tensor([float(value) for value in attributes.get(key, default).split()], dtype=torch.float32)
            center, half = values('pos', '0 0 0'), values('size', '')
            quaternion = values('quat', '1 0 0 0')
            if center.shape != (3,) or half.shape != (3,) or quaternion.shape != (4,) or (half <= 0).any():
                raise ValueError(f'{name}: invalid original collision-box dimensions/pose')
            centers.append(center); halves.append(half)
            rotations.append(quaternion_matrix_xyzw(quaternion[[1, 2, 3, 0]]))
        self.body_names = names
        self.register_buffer('foot_ids', torch.tensor([names.index(name) for name in FOOT_NAMES], dtype=torch.long))
        self.register_buffer('local_centers', torch.stack(centers))
        self.register_buffer('local_halves', torch.stack(halves))
        self.register_buffer('local_rotations', torch.stack(rotations))
        self.register_buffer('world_axes', torch.eye(3))

    def forward(self, body_pos, body_quat_xyzw, forbidden_center, forbidden_half):
        if body_pos.ndim != 3 or body_pos.shape[1:] != (len(self.body_names), 3):
            raise ValueError('Expected current body_pos[N,body_count,3] in declared body order')
        n = body_pos.shape[0]
        if body_quat_xyzw.shape != (n, len(self.body_names), 4) or forbidden_center.shape != (n, 3) or forbidden_half.shape != (n, 3):
            raise ValueError('Expected body quaternion[N,B,4] and world AABB center/half[N,3]')
        rotation = quaternion_matrix_xyzw(body_quat_xyzw[:, self.foot_ids])
        centers = body_pos[:, self.foot_ids] + torch.matmul(rotation, self.local_centers[None, :, :, None]).squeeze(-1)
        box_rotation = torch.matmul(rotation, self.local_rotations[None])
        # Matrix columns are OBB axes; SAT stores each candidate direction as a row.
        box_axes = box_rotation.transpose(-1, -2)
        world = self.world_axes[None, None].expand(n, 4, 3, 3)
        cross = torch.cross(world[:, :, :, None].expand(-1, -1, -1, 3, -1),
                            box_axes[:, :, None].expand(-1, -1, 3, -1, -1), dim=-1)
        axes = torch.cat([world, box_axes, cross.reshape(n, 4, 9, 3)], dim=2)
        length = axes.norm(dim=-1, keepdim=True)
        valid = length.squeeze(-1) > 1e-7
        axes = axes / length.clamp_min(1e-7)
        toward = forbidden_center[:, None] - centers
        projection = (axes * toward[:, :, None]).sum(-1)
        body_radius = (torch.matmul(axes, box_rotation).abs() * self.local_halves[None, :, None]).sum(-1)
        object_radius = (axes.abs() * forbidden_half[:, None, None]).sum(-1)
        signed_gaps = projection.abs() - body_radius - object_radius
        signed_gaps = signed_gaps.masked_fill(~valid, -torch.inf)
        gaps, selected = signed_gaps.max(-1)
        chosen_axes = axes.gather(2, selected[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)
        chosen_projection = projection.gather(2, selected[:, :, None]).squeeze(2)
        # Coincident-center ties choose the positive axis; the result stays finite.
        sign = torch.where(chosen_projection < 0, -torch.ones_like(gaps), torch.ones_like(gaps))
        return FootClearanceResult(gaps, chosen_axes * sign[:, :, None], centers)

    @staticmethod
    def features(result, root_quat_xyzw):
        """16 extra heading-local features: four clipped meter gaps then 12 dirs."""
        if root_quat_xyzw.shape != (len(result.gaps), 4):
            raise ValueError('Expected root quaternion[N,4] in XYZW order')
        heading_rotation = quaternion_matrix_xyzw(root_quat_xyzw)
        yaw = torch.atan2(heading_rotation[:, 1, 0], heading_rotation[:, 0, 0])
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        direction = result.directions_world
        local = torch.stack([cosine*direction[..., 0]+sine*direction[..., 1],
                            -sine*direction[..., 0]+cosine*direction[..., 1], direction[..., 2]], -1)
        return torch.cat([result.gaps.clamp(-.2, 1.), local.flatten(1)], -1)

    @staticmethod
    def proximity(body_point_distances, result):
        """New reward prox; caller keeps its existing coefficient 2 and scale .12."""
        if body_point_distances.ndim != 2 or len(body_point_distances) != len(result.gaps):
            raise ValueError('Expected the unchanged current body point distances[N,B]')
        return torch.minimum(body_point_distances.amin(-1), result.gaps.clamp_min(0).amin(-1))
