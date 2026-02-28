"""
backend/agents/deterioration_oracle.py

Agent 4 — Deterioration Oracle

Predicts road deterioration trajectory using an India-calibrated HDM-4 model.
Answers: "When will this road fail? What does intervention timing cost?"

Model basis: HDM-4 simplified IRI progression model calibrated with
CRRI (Central Road Research Institute) India data and MORT&H parameters.

Cost data: PMGSY/MoRTH Schedule of Rates 2024 (₹/km).
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── India-specific HDM-4 calibration (CRRI / MORT&H data) ─────────────────
INDIA_HDM4_PARAMS = {
    "BC": {
        "a0":  1.6,     # Initial IRI at construction (m/km)
        "kge": 0.025,   # Environmental cracking coefficient
        "kgp": 0.35,    # Pavement strength / traffic deterioration factor
    },
    "WBM": {
        "a0":  3.5,
        "kge": 0.045,
        "kgp": 0.55,
    },
    "Granular": {
        "a0":  4.0,
        "kge": 0.065,
        "kgp": 0.75,
    },
    "Concrete": {   # Rigid pavement degrades slower
        "a0":  1.2,
        "kge": 0.015,
        "kgp": 0.20,
    },
    "Unknown": {    # Default to WBM (conservative)
        "a0":  3.5,
        "kge": 0.045,
        "kgp": 0.55,
    },
}

# PMGSY unit costs 2024 (₹/km) — MoRTH Schedule of Rates
INTERVENTION_COSTS = {
    "Routine":       {"BC": 300_000,    "WBM": 150_000,   "Granular": 80_000,   "Concrete": 200_000},
    "Preventive":    {"BC": 1_200_000,  "WBM": 600_000,   "Granular": 300_000,  "Concrete": 800_000},
    "Rehabilitation":{"BC": 5_000_000,  "WBM": 3_000_000, "Granular": 1_500_000,"Concrete": 4_000_000},
    "Reconstruction":{"BC": 12_000_000, "WBM": 8_000_000, "Granular": 5_000_000,"Concrete": 10_000_000},
}

# IRI → recommended intervention mapping (IRC:SP:20)
IRI_INTERVENTION_MAP = [
    (2.0, "Routine"),
    (4.0, "Preventive"),
    (6.0, "Rehabilitation"),
    (float("inf"), "Reconstruction"),
]


def _classify_iri_intervention(iri: float) -> str:
    for threshold, intervention in IRI_INTERVENTION_MAP:
        if iri < threshold:
            return intervention
    return "Reconstruction"


class DeteriorationOracle:
    """
    Predicts 5-year IRI trajectory and computes intervention economics.
    """

    def predict_deterioration(
        self,
        current_iri: float,
        surface_type: str,
        aadt: int,
        rainfall_mm_year: int,
        years: int = 5,
        length_km: float = 1.0,
    ) -> dict:
        """
        HDM-4 simplified IRI progression: IRI(t) = IRI_0 × exp(a × t)

        Args:
            current_iri:       Current IRI measurement (m/km).
            surface_type:      "BC" | "WBM" | "Granular" | "Concrete".
            aadt:              Annual Average Daily Traffic (vehicles/day).
            rainfall_mm_year:  Annual rainfall (mm) — India district data.
            years:             Prediction horizon in years (default 5).
            length_km:         Road segment length (km) for cost scaling.

        Returns:
            Dict with trajectory, failure year, intervention recommendation,
            cost comparison, and decision urgency.
        """
        # ── HDM-4 parameters ──────────────────────────────────────────────
        params = INDIA_HDM4_PARAMS.get(surface_type, INDIA_HDM4_PARAMS["Unknown"])

        # Traffic factor: log-linear growth with AADT
        traffic_factor = np.log1p(aadt / 10_000) * 0.15

        # Climate factor: rainfall-driven cracking
        climate_factor = (rainfall_mm_year / 1_000.0) * 0.08

        # Exponential growth rate
        a = params["kge"] * climate_factor + params["kgp"] * traffic_factor

        # Guard against degenerate inputs
        a = max(a, 0.001)

        # ── 5-year trajectory ─────────────────────────────────────────────
        from backend.sensors.iri_computer import classify_iri

        trajectory = []
        for year in range(years + 1):
            predicted_iri = current_iri * np.exp(a * year)
            trajectory.append({
                "year":      year,
                "iri":       round(float(predicted_iri), 2),
                "condition": classify_iri(float(predicted_iri))["condition"],
            })

        # ── Failure year (IRI > 6.0 = "Very Poor") ───────────────────────
        failure_year = next(
            (t["year"] for t in trajectory if t["iri"] > 6.0),
            years + 1,
        )

        # ── Economic analysis ─────────────────────────────────────────────
        current_intervention = _classify_iri_intervention(current_iri)
        surface_key = surface_type if surface_type in INTERVENTION_COSTS["Routine"] else "WBM"

        cost_now_per_km    = INTERVENTION_COSTS[current_intervention].get(surface_key, 3_000_000)
        cost_delayed_per_km = INTERVENTION_COSTS["Reconstruction"].get(surface_key, 8_000_000)

        cost_now    = cost_now_per_km * length_km
        cost_delayed = cost_delayed_per_km * length_km
        savings      = cost_delayed - cost_now

        # ── Decision urgency ──────────────────────────────────────────────
        if failure_year <= 1:
            urgency = "IMMEDIATE"
        elif failure_year <= 2:
            urgency = "HIGH"
        elif failure_year <= 4:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # ── Growth rate for display ───────────────────────────────────────
        iri_year1 = trajectory[1]["iri"] if len(trajectory) > 1 else current_iri
        annual_deterioration = round(iri_year1 - current_iri, 3)

        return {
            "current_iri":              current_iri,
            "surface_type":             surface_type,
            "trajectory":               trajectory,
            "failure_year":             failure_year,
            "weeks_to_failure":         failure_year * 52,
            "annual_deterioration_rate":annual_deterioration,
            "recommended_intervention": current_intervention,
            "cost_now_lakh":            round(cost_now / 100_000, 1),
            "cost_if_delayed_lakh":     round(cost_delayed / 100_000, 1),
            "potential_savings_lakh":   round(savings / 100_000, 1),
            "decision_urgency":         urgency,
            "hdm4_growth_rate":         round(a, 5),
            "inputs": {
                "aadt":              aadt,
                "rainfall_mm_year":  rainfall_mm_year,
                "length_km":         length_km,
            },
        }

    def batch_predict(self, segments: list[dict], **defaults) -> list[dict]:
        """
        Run deterioration prediction across a list of segments.

        Each segment dict should have: iri_value, surface_type.
        Additional kwargs (aadt, rainfall_mm_year, length_km) applied as defaults
        if not present in individual segment dicts.
        """
        results = []
        for seg in segments:
            iri = seg.get("iri_value")
            if iri is None:
                continue
            result = self.predict_deterioration(
                current_iri=iri,
                surface_type=seg.get("surface_type", "WBM"),
                aadt=seg.get("aadt", defaults.get("aadt", 500)),
                rainfall_mm_year=seg.get("rainfall_mm_year", defaults.get("rainfall_mm_year", 1200)),
                length_km=seg.get("length_km", defaults.get("length_km", 0.1)),
            )
            result["segment_id"] = seg.get("segment_id")
            results.append(result)
        return results
