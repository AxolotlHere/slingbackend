"""
backend/agents/sensor_fusion.py

Agent 0 — Sensor Fusion

Combines all 5 sensor channels into a unified per-segment condition object.
Applies conflict detection with deterministic resolution rules:
    1. IRI (accelerometer physics) overrides visual condition if they conflict
    2. 3D rut depth overrides visual rut estimate
    3. Acoustic surface type never overrides visual — flagged for reference
    4. All conflicts logged for Devil's Advocate review

Returns the canonical fused segment dict consumed by all downstream agents.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Canonical condition ordering for comparison
CONDITION_RANK = {"Good": 0, "Fair": 1, "Poor": 2, "Very Poor": 3, "Unknown": -1}


def _condition_rank(condition: str) -> int:
    return CONDITION_RANK.get(condition, -1)


class SensorFusionAgent:
    """
    Merges IMU (IRI), visual (Qwen2.5-VL), depth-3D (rut), and acoustic
    channels into a single unified segment assessment dict.
    """

    def fuse(self, segment_data: dict) -> dict:
        """
        Fuse all channel outputs for a single 100m road segment.

        Args:
            segment_data: dict with keys:
                segment_id (str)
                gps        (dict: lat, lng, heading)
                iri        (dict: iri_value, avg_speed_kmh, pass_count)
                visual     (dict: overall_condition, pci_estimate, surface_type, distresses, ...)
                depth_3d   (dict: rut_depth_mm, severity, confidence)
                acoustic   (dict: surface_type_acoustic, confidence)
                length_km  (float)

        Returns:
            Unified segment dict for downstream agents.
        """
        iri_data    = segment_data.get("iri", {}) or {}
        visual_data = segment_data.get("visual", {}) or {}
        depth_data  = segment_data.get("depth_3d", {}) or {}
        acoustic    = segment_data.get("acoustic", {}) or {}

        # ── Condition from each channel ────────────────────────────────────
        from backend.sensors.iri_computer import classify_iri  # local import to avoid circular

        iri_value = iri_data.get("iri_value")
        iri_classification = classify_iri(iri_value) if iri_value is not None else {}
        iri_condition = iri_classification.get("condition", "Unknown")

        visual_condition = visual_data.get("overall_condition", "Unknown")
        visual_surface   = visual_data.get("surface_type", "Unknown")
        acoustic_surface = acoustic.get("surface_type_acoustic", "Unknown")

        # ── Conflict Detection ─────────────────────────────────────────────
        conflicts = []

        # Conflict 1: IRI vs visual condition disagreement
        if (
            iri_condition != "Unknown"
            and visual_condition not in ("Unknown", None)
            and iri_condition != visual_condition
        ):
            conflicts.append({
                "type": "iri_visual_mismatch",
                "iri_says":    iri_condition,
                "visual_says": visual_condition,
                "resolution":  "Trust IRI — physics overrides visual approximation.",
                "final":       iri_condition,
            })
            logger.info(
                f"Segment {segment_data.get('segment_id')}: "
                f"IRI={iri_condition} vs Visual={visual_condition} — forcing IRI."
            )

        # Conflict 2: Visual vs acoustic surface type disagreement
        if (
            visual_surface not in ("Unknown", None)
            and acoustic_surface not in ("Unknown", None)
            and visual_surface != acoustic_surface
        ):
            conflicts.append({
                "type":        "surface_type_mismatch",
                "visual_says": visual_surface,
                "acoustic_says": acoustic_surface,
                "resolution":  "Flagged for Devil's Advocate review. Visual takes precedence.",
                "final":       f"{visual_surface} (visual) / {acoustic_surface} (acoustic)",
            })

        # Conflict 3: 3D rut says Severe but visual says Good/Fair
        rut_mm = depth_data.get("rut_depth_mm")
        if rut_mm is not None and rut_mm > 20 and _condition_rank(visual_condition) <= 1:
            conflicts.append({
                "type": "rut_visual_mismatch",
                "rut_depth_mm": rut_mm,
                "visual_condition": visual_condition,
                "resolution":  "3D rut depth indicates structural deformation not visible from surface texture.",
                "final":       "Downgrade confidence — visual assessment may underestimate severity.",
            })

        # ── Final canonical condition: IRI is ground truth ─────────────────
        final_condition = iri_condition if iri_condition != "Unknown" else visual_condition

        # ── Data quality ───────────────────────────────────────────────────
        data_quality = self._assess_data_quality(iri_data, visual_data, depth_data, acoustic)

        # ── Build unified segment dict ──────────────────────────────────────
        fused = {
            "segment_id":    segment_data.get("segment_id"),
            "gps":           segment_data.get("gps", {}),
            "length_km":     segment_data.get("length_km", 0.1),
            "timestamp":     segment_data.get("timestamp"),

            # IRI channel
            "iri_value":     iri_value,
            "iri_condition": iri_condition,
            "iri_color":     iri_classification.get("color", "#AAAAAA"),
            "avg_speed_kmh": iri_data.get("avg_speed_kmh"),
            "pass_count":    iri_data.get("pass_count", 1),

            # Visual channel
            "pci_estimate":  visual_data.get("pci_estimate"),
            "surface_type":  visual_surface,
            "distresses":    visual_data.get("distresses", []),
            "drainage_adequacy": visual_data.get("drainage_adequacy"),
            "visual_confidence": visual_data.get("confidence"),

            # 3D depth channel
            "rut_depth_mm":  rut_mm,
            "rut_severity":  depth_data.get("severity"),
            "rut_confidence":depth_data.get("confidence"),

            # Acoustic channel
            "acoustic_surface": acoustic_surface,
            "acoustic_confidence": acoustic.get("confidence", 0.0),

            # Synthesis
            "final_condition": final_condition,
            "conflicts":       conflicts,
            "data_quality":    data_quality,
        }

        return fused

    def _assess_data_quality(self, iri, visual, depth_3d, acoustic) -> str:
        """
        Rate overall data quality based on how many channels contributed successfully.
        """
        channels_ok = sum([
            iri.get("iri_value") is not None,
            visual.get("overall_condition") not in (None, "Unknown"),
            depth_3d.get("rut_depth_mm") is not None,
            acoustic.get("surface_type_acoustic") not in (None, "Unknown"),
        ])

        if channels_ok >= 3:
            return "High"
        if channels_ok == 2:
            return "Medium"
        return "Low"
