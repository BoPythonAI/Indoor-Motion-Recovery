"""Explicit feature, geometry and checkpoint semantics for the surface branch."""
OLD_FEATURE_VERSION = 'v3_terrain_reference__terrain_correction_v1'
FEATURE_VERSION = OLD_FEATURE_VERSION + '__foot_clearance_v1'
POLICY_KIND = 'terrain_prior_surface_correction_v1'
GEOMETRY_VERSION = 'mjcf_foot_obb_forbidden_aabb_sat15_v1'
PROXIMITY_REWARD = '-2 * exp(-min(bodypoint_gap, max(foot_sat_gap, 0)) / 0.12)'
EXTRA_PER_FRAME = 16
HISTORY_FRAMES = 4
