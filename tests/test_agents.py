"""
tests/test_agents.py

Integration tests for the PULSE agent pipeline.

Tests (no GPU, no Ollama, no model loading required):
    - SensorFusionAgent.fuse() — all channels, conflict detection
    - DeteriorationOracle.predict_deterioration() — trajectory shape, economics
    - DevilsAdvocateAgent.review() — each rule fires on trigger condition
    - EconomicCascadeEngine.compute_cascade() — no Ollama (template fallback)
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Sensor Fusion Tests ────────────────────────────────────────────────────

class TestSensorFusionAgent:

    @pytest.fixture
    def agent(self):
        from backend.agents.sensor_fusion import SensorFusionAgent
        return SensorFusionAgent()

    def _make_segment(self, iri_value=3.0, visual_condition="Fair",
                      visual_surface="WBM", acoustic_surface="WBM",
                      rut_mm=12.0, pci=55):
        return {
            "segment_id": "test_seg_0001",
            "gps": {"lat": 12.9716, "lng": 77.5946, "heading": 180},
            "length_km": 0.1,
            "timestamp": 1700000000000,
            "avg_speed_kmh": 40,
            "avg_speed_ms": 11.1,
            "iri": {
                "iri_value": iri_value,
                "avg_speed_kmh": 40,
                "pass_count": 3,
            },
            "visual": {
                "overall_condition": visual_condition,
                "surface_type": visual_surface,
                "pci_estimate": pci,
                "distresses": [],
                "drainage_adequacy": "Adequate",
                "confidence": "High",
            },
            "depth_3d": {
                "rut_depth_mm": rut_mm,
                "severity": "Moderate",
                "confidence": "medium",
            },
            "acoustic": {
                "surface_type_acoustic": acoustic_surface,
                "confidence": 0.85,
            },
        }

    def test_fuse_returns_all_required_keys(self, agent):
        seg = self._make_segment()
        result = agent.fuse(seg)
        required = ["segment_id", "iri_value", "iri_condition", "rut_depth_mm",
                    "final_condition", "conflicts", "data_quality"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_iri_overrides_visual_on_conflict(self, agent):
        """IRI says Good but visual says Poor → final should be Good (IRI wins)."""
        seg = self._make_segment(iri_value=1.5, visual_condition="Poor")
        result = agent.fuse(seg)
        assert result["final_condition"] == "Good"
        # Conflict should be logged
        conflict_types = [c["type"] for c in result["conflicts"]]
        assert "iri_visual_mismatch" in conflict_types

    def test_no_conflict_when_aligned(self, agent):
        """IRI = Fair, visual = Fair → no condition conflict."""
        seg = self._make_segment(iri_value=3.0, visual_condition="Fair")
        result = agent.fuse(seg)
        condition_conflicts = [c for c in result["conflicts"]
                                if c["type"] == "iri_visual_mismatch"]
        assert len(condition_conflicts) == 0

    def test_surface_type_mismatch_flagged(self, agent):
        seg = self._make_segment(visual_surface="BC", acoustic_surface="Granular")
        result = agent.fuse(seg)
        types = [c["type"] for c in result["conflicts"]]
        assert "surface_type_mismatch" in types

    def test_data_quality_high_with_four_channels(self, agent):
        seg = self._make_segment()
        result = agent.fuse(seg)
        assert result["data_quality"] == "High"

    def test_data_quality_low_with_one_channel(self, agent):
        seg = self._make_segment()
        seg["visual"] = {"overall_condition": "Unknown"}
        seg["depth_3d"] = {}
        seg["acoustic"] = {}
        result = agent.fuse(seg)
        assert result["data_quality"] == "Low"


# ── Deterioration Oracle Tests ─────────────────────────────────────────────

class TestDeteriorationOracle:

    @pytest.fixture
    def oracle(self):
        from backend.agents.deterioration_oracle import DeteriorationOracle
        return DeteriorationOracle()

    def test_trajectory_correct_length(self, oracle):
        result = oracle.predict_deterioration(
            current_iri=3.0, surface_type="BC",
            aadt=500, rainfall_mm_year=1200, years=5
        )
        assert len(result["trajectory"]) == 6  # years 0..5 inclusive

    def test_trajectory_iri_increases_over_time(self, oracle):
        result = oracle.predict_deterioration(
            current_iri=3.0, surface_type="WBM",
            aadt=500, rainfall_mm_year=1200, years=5
        )
        iris = [t["iri"] for t in result["trajectory"]]
        assert iris == sorted(iris), "IRI should monotonically increase over time"

    def test_current_iri_is_year_zero(self, oracle):
        iri = 4.5
        result = oracle.predict_deterioration(
            current_iri=iri, surface_type="BC",
            aadt=200, rainfall_mm_year=800, years=5
        )
        assert result["trajectory"][0]["iri"] == iri

    def test_very_poor_road_fails_quickly(self, oracle):
        result = oracle.predict_deterioration(
            current_iri=5.8, surface_type="WBM",
            aadt=2000, rainfall_mm_year=2000, years=5
        )
        assert result["failure_year"] <= 2

    def test_good_road_does_not_fail_soon(self, oracle):
        result = oracle.predict_deterioration(
            current_iri=1.5, surface_type="BC",
            aadt=100, rainfall_mm_year=600, years=5
        )
        assert result["failure_year"] > 3

    def test_cost_now_less_than_cost_delayed(self, oracle):
        """Intervening now should always cost less than waiting until reconstruction."""
        result = oracle.predict_deterioration(
            current_iri=3.0, surface_type="WBM",
            aadt=500, rainfall_mm_year=1200, years=5
        )
        assert result["cost_now_lakh"] < result["cost_if_delayed_lakh"]

    def test_savings_positive(self, oracle):
        result = oracle.predict_deterioration(
            current_iri=3.5, surface_type="BC",
            aadt=300, rainfall_mm_year=1000, years=5
        )
        assert result["potential_savings_lakh"] > 0


# ── Devil's Advocate Tests ─────────────────────────────────────────────────

class TestDevilsAdvocateAgent:

    @pytest.fixture
    def agent(self):
        from backend.agents.devils_advocate import DevilsAdvocateAgent
        return DevilsAdvocateAgent()

    def test_no_issues_returns_high_confidence(self, agent):
        seg = {
            "iri_value": 3.0, "iri_condition": "Fair",
            "pci_estimate": 60,
            "avg_speed_kmh": 45, "pass_count": 5,
            "surface_type": "BC",
            "data_quality": "High",
            "rut_depth_mm": 8.0, "final_condition": "Fair",
        }
        result = agent.review(seg)
        assert result["final_confidence"] == "High"
        assert result["cleared_for_report"] is True

    def test_speed_too_low_flags_iri_invalid(self, agent):
        seg = {
            "iri_value": 3.0, "iri_condition": "Fair", "pci_estimate": 55,
            "avg_speed_kmh": 10, "pass_count": 5, "surface_type": "WBM",
            "data_quality": "High", "rut_depth_mm": 8.0, "final_condition": "Fair",
        }
        result = agent.review(seg)
        assert result["final_confidence"] == "Low — IRI Invalid"

    def test_extreme_iri_requires_human_review(self, agent):
        seg = {
            "iri_value": 15.0, "iri_condition": "Very Poor", "pci_estimate": 20,
            "avg_speed_kmh": 40, "pass_count": 3, "surface_type": "WBM",
            "data_quality": "High", "rut_depth_mm": 25.0, "final_condition": "Very Poor",
        }
        result = agent.review(seg)
        assert result["final_confidence"] == "Requires Human Review"
        assert result["cleared_for_report"] is False

    def test_single_pass_downgrades_confidence(self, agent):
        seg = {
            "iri_value": 4.0, "iri_condition": "Poor", "pci_estimate": 40,
            "avg_speed_kmh": 35, "pass_count": 1, "surface_type": "BC",
            "data_quality": "High", "rut_depth_mm": 15.0, "final_condition": "Poor",
        }
        result = agent.review(seg)
        assert result["final_confidence"] == "Medium — Verify Before Acting"

    def test_iri_visual_conflict_triggers(self, agent):
        seg = {
            "iri_value": 1.5, "iri_condition": "Good", "pci_estimate": 30,
            "avg_speed_kmh": 40, "pass_count": 4, "surface_type": "BC",
            "data_quality": "High", "rut_depth_mm": 5.0, "final_condition": "Good",
        }
        result = agent.review(seg)
        rule_ids = [c["rule_id"] for c in result["devils_advocate_challenges"]]
        assert "iri_visual_conflict" in rule_ids

    def test_challenge_count_matches_list(self, agent):
        """challenge_count should equal len(devils_advocate_challenges)."""
        seg = {
            "iri_value": 3.0, "iri_condition": "Fair", "pci_estimate": 55,
            "avg_speed_kmh": 8, "pass_count": 1, "surface_type": "WBM",
            "data_quality": "Low", "rut_depth_mm": 5.0, "final_condition": "Fair",
        }
        result = agent.review(seg)
        assert result["challenge_count"] == len(result["devils_advocate_challenges"])


# ── Economic Cascade Tests ─────────────────────────────────────────────────

class TestEconomicCascadeEngine:

    @pytest.fixture
    def engine(self):
        from backend.agents.economic_cascade import EconomicCascadeEngine
        # Force offline mode — no Gemini required
        e = EconomicCascadeEngine()
        e._llm_available = False  # Disable for unit test
        return e

    def _make_segment(self, iri=5.0):
        return {
            "segment_id": "test_seg_econ",
            "iri_value":  iri,
            "length_km":  0.5,
            "surface_type": "WBM",
        }

    def _make_osm_context(self):
        return {
            "schools": [
                {"name": "Govt School", "distance_km": 1.5, "student_count": 300},
            ],
            "health_facilities": [
                {"name": "PHC", "distance_km": 5.0},
            ],
            "agricultural_land_ha": 80,
        }

    def test_returns_required_keys(self, engine):
        seg = self._make_segment()
        result = engine.compute_cascade(seg, self._make_osm_context(), population=600)
        required = ["iri", "annual_voc_cost_lakh", "agricultural_loss_annual_lakh",
                    "total_annual_economic_loss_lakh", "schools_affected",
                    "ambulance_delay_minutes", "narrative"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_no_loss_on_good_road(self, engine):
        """On a Good road (IRI = 1.5), economic impact should be near zero."""
        seg = self._make_segment(iri=1.5)
        result = engine.compute_cascade(seg, self._make_osm_context())
        assert result["annual_voc_cost_lakh"] < 1.0

    def test_higher_iri_higher_loss(self, engine):
        """Higher IRI → higher economic loss."""
        seg_good = self._make_segment(iri=2.0)
        seg_bad  = self._make_segment(iri=6.0)
        ctx = self._make_osm_context()
        result_good = engine.compute_cascade(seg_good, ctx)
        result_bad  = engine.compute_cascade(seg_bad,  ctx)
        assert result_bad["total_annual_economic_loss_lakh"] > result_good["total_annual_economic_loss_lakh"]

    def test_narrative_is_string(self, engine):
        seg = self._make_segment()
        result = engine.compute_cascade(seg, self._make_osm_context())
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 20

    def test_no_iri_returns_error(self, engine):
        seg = {"segment_id": "x", "length_km": 0.1}  # No iri_value
        result = engine.compute_cascade(seg, {})
        assert "error" in result
