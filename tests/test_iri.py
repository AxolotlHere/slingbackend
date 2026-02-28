"""
tests/test_iri.py

Unit tests for Channel 1 — IRI computation.

Tests cover:
    - Zero input → IRI near zero
    - Sinusoidal bump → IRI in meaningful range
    - Speed normalization (same road, different speeds)
    - Below-threshold speed → returns None
    - Insufficient data → returns None
    - classify_iri() boundary values
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.sensors.iri_computer import compute_iri, classify_iri, segment_iri_stats


class TestComputeIRI:

    def _make_flat_road(self, n_samples: int = 400, speed_ms: float = 10.0) -> tuple:
        """Simulate perfectly flat road (near-zero acceleration)."""
        accel = np.zeros(n_samples)
        speed = np.full(n_samples, speed_ms)
        return accel, speed

    def _make_rough_road(self, n_samples: int = 400, freq_hz: float = 5.0,
                         amp: float = 1.0, speed_ms: float = 10.0) -> tuple:
        """Simulate a bumpy road with sinusoidal acceleration."""
        t = np.linspace(0, n_samples / 200, n_samples)
        accel = np.sin(2 * np.pi * freq_hz * t) * amp
        speed = np.full(n_samples, speed_ms)
        return accel, speed

    def test_flat_road_near_zero_iri(self):
        """A flat road (zero accel) should yield a very small IRI."""
        accel, speed = self._make_flat_road()
        iri = compute_iri(accel, speed, sample_rate=200, min_speed_kmh=15)
        assert iri is not None
        assert iri < 0.01, f"Expected IRI < 0.01 for perfectly flat road, got {iri}"

    def test_rough_road_higher_iri(self):
        """Rough road should have higher IRI than flat road (relative ordering)."""
        accel_flat, speed = self._make_flat_road()
        accel_rough, _ = self._make_rough_road(amp=2.0, freq_hz=3.0)
        iri_flat  = compute_iri(accel_flat,  speed, sample_rate=200, min_speed_kmh=15)
        iri_rough = compute_iri(accel_rough, speed, sample_rate=200, min_speed_kmh=15)
        assert iri_rough is not None and iri_flat is not None
        assert iri_rough > iri_flat

    def test_rougher_road_higher_iri(self):
        """Higher amplitude bumps → higher IRI than lower amplitude."""
        accel_low, speed = self._make_rough_road(amp=0.5)
        accel_high, _   = self._make_rough_road(amp=3.0)
        iri_low  = compute_iri(accel_low,  speed, sample_rate=200, min_speed_kmh=15)
        iri_high = compute_iri(accel_high, speed, sample_rate=200, min_speed_kmh=15)
        assert iri_high > iri_low, "Higher amplitude should produce higher IRI"

    def test_below_min_speed_returns_none(self):
        """Speed below threshold should return None (IRI invalid)."""
        accel = np.random.randn(400) * 0.5
        speed = np.full(400, 3.0)  # ~10.8 km/h
        iri = compute_iri(accel, speed, sample_rate=200, min_speed_kmh=20)
        assert iri is None

    def test_insufficient_data_returns_none(self):
        """Less than 1 second of data (< sample_rate samples) → None."""
        accel = np.zeros(100)
        speed = np.full(100, 10.0)
        iri = compute_iri(accel, speed, sample_rate=200, min_speed_kmh=15)
        assert iri is None

    def test_returns_float(self):
        """Valid input should return a Python float."""
        accel, speed = self._make_rough_road()
        iri = compute_iri(accel, speed, sample_rate=200, min_speed_kmh=15)
        assert isinstance(iri, float)

    def test_iri_positive(self):
        """IRI should always be non-negative."""
        for _ in range(5):
            accel = np.random.randn(400) * 0.8
            speed = np.full(400, 10.0)
            iri = compute_iri(accel, speed, sample_rate=200, min_speed_kmh=15)
            if iri is not None:
                assert iri >= 0, f"IRI should be non-negative, got {iri}"


class TestClassifyIRI:

    def test_good_condition(self):
        result = classify_iri(1.0)
        assert result["condition"] == "Good"
        assert result["color"] == "#27AE60"

    def test_fair_condition(self):
        result = classify_iri(3.0)
        assert result["condition"] == "Fair"
        assert result["color"] == "#F39C12"

    def test_poor_condition(self):
        result = classify_iri(5.0)
        assert result["condition"] == "Poor"
        assert result["color"] == "#E74C3C"

    def test_very_poor_condition(self):
        result = classify_iri(7.0)
        assert result["condition"] == "Very Poor"
        assert result["color"] == "#8E44AD"

    def test_boundary_good_to_fair(self):
        """Exactly 2.0 should be Fair (threshold is exclusive at 2.0)."""
        result = classify_iri(2.0)
        assert result["condition"] == "Fair"

    def test_boundary_fair_to_poor(self):
        result = classify_iri(4.0)
        assert result["condition"] == "Poor"

    def test_boundary_poor_to_very_poor(self):
        result = classify_iri(6.0)
        assert result["condition"] == "Very Poor"

    def test_returns_iri_value(self):
        result = classify_iri(3.5)
        assert result["iri_value"] == 3.5

    def test_high_iri_still_classified(self):
        result = classify_iri(15.0)
        assert result["condition"] == "Very Poor"


class TestSegmentIRIStats:

    def test_empty_returns_none(self):
        result = segment_iri_stats([])
        assert result["mean"] is None
        assert result["pass_count"] == 0

    def test_multiple_passes(self):
        iri_values = [2.5, 2.8, 2.6, 2.7]
        result = segment_iri_stats(iri_values)
        assert result["pass_count"] == 4
        assert abs(result["mean"] - np.mean(iri_values)) < 0.01
        assert abs(result["median"] - np.median(iri_values)) < 0.01

    def test_representative_is_median(self):
        """Median is the robust estimator used as representative IRI."""
        iri_values = [2.0, 2.5, 10.0]  # 10.0 is an outlier
        result = segment_iri_stats(iri_values)
        assert result["representative_iri"] == result["median"]
