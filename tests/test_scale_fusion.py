"""
tests/test_scale_fusion.py

Unit tests for Channel 2 — Metric Scale Fusion.

Tests cover:
    - All 3 anchors present → correct weighted average
    - 1 or 2 anchors are None → remaining anchors renormalized
    - All anchors None → raises ValueError
    - Out-of-range anchors are excluded
    - Fused scale stays within valid range
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.sensors.depth_pipeline import MetricDepthPipeline, SCALE_VALID_MIN, SCALE_VALID_MAX


@pytest.fixture
def pipeline():
    """Create pipeline without loading the depth model (no GPU needed for scale tests)."""
    p = MetricDepthPipeline(camera_height_m=1.2, device="cpu")
    # Skip model loading for unit tests
    return p


class TestScaleFusion:

    def test_all_three_anchors_correct_weighted_average(self, pipeline):
        """With all 3 anchors, result should equal weighted average with full weights."""
        s_imu, s_ground, s_motion = 10.0, 8.0, 6.0
        result = pipeline.fuse_scales(s_imu, s_ground, s_motion)
        expected = 0.5 * s_imu + 0.3 * s_ground + 0.2 * s_motion
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_imu_only(self, pipeline):
        """Only IMU anchor valid → result equals imu_scale."""
        result = pipeline.fuse_scales(s_imu=5.0, s_ground=None, s_motion=None)
        assert abs(result - 5.0) < 1e-6

    def test_ground_only(self, pipeline):
        """Only ground anchor valid → result equals ground_scale."""
        result = pipeline.fuse_scales(s_imu=None, s_ground=7.0, s_motion=None)
        assert abs(result - 7.0) < 1e-6

    def test_motion_only(self, pipeline):
        """Only motion anchor valid → result equals motion_scale."""
        result = pipeline.fuse_scales(s_imu=None, s_ground=None, s_motion=4.5)
        assert abs(result - 4.5) < 1e-6

    def test_two_anchors_renormalized(self, pipeline):
        """Two anchors → weights renormalized to sum to 1."""
        s_imu, s_ground = 10.0, 8.0
        result = pipeline.fuse_scales(s_imu=s_imu, s_ground=s_ground, s_motion=None)
        # Weights: imu=0.5, ground=0.3 → total=0.8 → renorm: imu=0.625, ground=0.375
        total_w = 0.5 + 0.3
        expected = (0.5 / total_w) * s_imu + (0.3 / total_w) * s_ground
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_all_none_raises_value_error(self, pipeline):
        """All anchors returning None → ValueError."""
        with pytest.raises(ValueError, match="All scale anchors failed"):
            pipeline.fuse_scales(s_imu=None, s_ground=None, s_motion=None)

    def test_out_of_range_excluded(self, pipeline):
        """Scale values outside [0.1, 100] should be excluded like None."""
        # s_imu is wildly out of range → should be excluded
        result = pipeline.fuse_scales(s_imu=1000.0, s_ground=5.0, s_motion=None)
        assert abs(result - 5.0) < 1e-6

    def test_result_in_valid_range(self, pipeline):
        """Fused scale should always be within SCALE_VALID_MIN/MAX."""
        for _ in range(20):
            s_imu    = float(np.random.uniform(0.5, 50.0))
            s_ground = float(np.random.uniform(0.5, 50.0))
            s_motion = float(np.random.uniform(0.5, 50.0))
            result = pipeline.fuse_scales(s_imu, s_ground, s_motion)
            assert SCALE_VALID_MIN < result < SCALE_VALID_MAX

    def test_equal_scales_returns_same(self, pipeline):
        """If all anchors give the same value, fusion returns that value."""
        scale = 3.0
        result = pipeline.fuse_scales(scale, scale, scale)
        assert abs(result - scale) < 1e-6


class TestGroundPlaneScaleAnchor:

    def test_uniform_median_depth(self, pipeline):
        """Known uniform depth map → scale = camera_height / median."""
        h, w = 480, 640
        depth_map = np.full((h, w), 0.4, dtype=np.float32)
        scale = pipeline._recover_scale_ground_plane(depth_map, h, w)
        expected = pipeline.camera_height / 0.4
        assert scale is not None
        assert abs(scale - expected) < 0.01

    def test_degenerate_zero_depth_returns_none(self, pipeline):
        """Depth map of zeros → cannot recover scale."""
        depth_map = np.zeros((480, 640), dtype=np.float32)
        scale = pipeline._recover_scale_ground_plane(depth_map, 480, 640)
        assert scale is None
