"""
backend/agents/devils_advocate.py

Agent 6 — Devil's Advocate

Challenges every road assessment finding before it enters any report.
This is the trust-building agent: engineers trust the system because
they see it argue with itself. It elevates PULSE from "impressive AI"
to "trustworthy engineering tool."

Rules are deterministic (no LLM) — fast, auditable, reproducible.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)


# ── Challenge Rule Definitions ─────────────────────────────────────────────
# Each rule: id, description, predicate (segment → bool), challenge text, action

def _rule_iri_visual_conflict(seg: dict) -> bool:
    """IRI says Good but visual PCI suggests structural damage."""
    return (
        seg.get("iri_condition") == "Good"
        and seg.get("pci_estimate") is not None
        and seg.get("pci_estimate") < 40
    )


def _rule_low_speed(seg: dict) -> bool:
    """Speed too low — IRI algorithm unreliable below 20 km/h."""
    speed = seg.get("avg_speed_kmh")
    return speed is not None and speed < 20


def _rule_single_pass(seg: dict) -> bool:
    """Only 1–2 passes recorded — insufficient for statistical IRI confidence."""
    return seg.get("pass_count", 0) < 3


def _rule_extreme_iri(seg: dict) -> bool:
    """IRI > 12 m/km is anomalously high — possible bump/pothole hit/malfunction."""
    iri = seg.get("iri_value")
    return iri is not None and iri > 12.0


def _rule_low_data_quality(seg: dict) -> bool:
    """Only 1 channel contributed — assessment lacks multi-channel corroboration."""
    return seg.get("data_quality") == "Low"


def _rule_high_speed_iri_low(seg: dict) -> bool:
    """IRI looks suspiciously good on a known bad surface type."""
    return (
        seg.get("surface_type") in ("Granular", "WBM")
        and seg.get("iri_value") is not None
        and seg.get("iri_value") < 1.5
    )


def _rule_rut_visual_mismatch(seg: dict) -> bool:
    """3D rut says Severe but visual says Good or Fair."""
    rut_mm = seg.get("rut_depth_mm")
    condition = seg.get("final_condition", "Unknown")
    return (
        rut_mm is not None
        and rut_mm > 20
        and condition in ("Good", "Fair")
    )


CHALLENGE_RULES = [
    {
        "id":        "iri_visual_conflict",
        "check":     _rule_iri_visual_conflict,
        "challenge": (
            "IRI reports Good condition (< 2.0 m/km) but visual PCI is below 40, "
            "suggesting structural cracking or surface failure invisible to the accelerometer. "
            "IRI measures dynamic roughness only — a freshly patched road can read Good "
            "while hiding underlying fatigue cracking. "
            "Recommend: visual inspection before approving Good rating."
        ),
        "action": "DOWNGRADE_CONFIDENCE",
    },
    {
        "id":        "speed_too_low",
        "check":     _rule_low_speed,
        "challenge": (
            "Average speed during this segment was below 20 km/h. "
            "The quarter-car IRI model requires minimum 20 km/h for valid results. "
            "Low-speed acceleration data contains disproportionate steering/braking artefacts. "
            "This IRI reading should be treated as unreliable."
        ),
        "action": "FLAG_IRI_INVALID",
    },
    {
        "id":        "single_pass",
        "check":     _rule_single_pass,
        "challenge": (
            "Only 1–2 drive passes recorded for this segment. "
            "IRI confidence requires a minimum of 3 independent passes for statistical validity "
            "(variance between passes reveals measurement noise vs road roughness). "
            "Current reading is preliminary — treat as indicative, not confirmatory."
        ),
        "action": "DOWNGRADE_CONFIDENCE",
    },
    {
        "id":        "extreme_iri",
        "check":     _rule_extreme_iri,
        "challenge": (
            "IRI > 12 m/km is anomalously high and warrants verification. "
            "Possible causes: vehicle struck a speed bump or large pothole causing "
            "brief axle bounce (spike, not sustained roughness), sensor malfunction, "
            "or loose/vibrating phone mount. "
            "Cross-check against video footage at this GPS timestamp before reporting."
        ),
        "action": "REQUEST_HUMAN_REVIEW",
    },
    {
        "id":        "low_data_quality",
        "check":     _rule_low_data_quality,
        "challenge": (
            "Only a single sensor channel contributed to this assessment. "
            "PULSE's strength is multi-channel corroboration — one-channel assessments "
            "have no cross-validation. The assessment may be correct but cannot be "
            "independently verified from within the system."
        ),
        "action": "DOWNGRADE_CONFIDENCE",
    },
    {
        "id":        "granular_iri_suspiciously_low",
        "check":     _rule_high_speed_iri_low,
        "challenge": (
            "Surface type is Granular or WBM but IRI reads < 1.5 m/km. "
            "Granular and WBM roads rarely achieve IRI < 1.5 under traffic. "
            "Possible causes: vehicle was very slow (normalisation overcompensates), "
            "road is newly bladed, or surface type classification is incorrect. "
            "Verify surface type via visual channel."
        ),
        "action": "DOWNGRADE_CONFIDENCE",
    },
    {
        "id":        "rut_visual_mismatch",
        "check":     _rule_rut_visual_mismatch,
        "challenge": (
            "3D point cloud shows rut depth > 20mm (Severe) but visual assessment "
            "rates condition as Good or Fair. "
            "Possible causes: rutting channels are narrow and may appear minor on camera, "
            "lighting conditions obscured surface deformation, or depth scale calibration "
            "produced an overestimate. "
            "Physical measurement with a straightedge recommended."
        ),
        "action": "DOWNGRADE_CONFIDENCE",
    },
]

# Action priority (higher = more serious)
ACTION_PRIORITY = {
    "REQUEST_HUMAN_REVIEW": 3,
    "FLAG_IRI_INVALID":     2,
    "DOWNGRADE_CONFIDENCE": 1,
}


class DevilsAdvocateAgent:
    """
    Runs all challenge rules against a fused segment assessment.
    Returns the segment annotated with challenges and a final confidence level.
    """

    def review(self, segment: dict) -> dict:
        """
        Run all challenge rules.

        Args:
            segment: Fused segment dict from SensorFusionAgent.

        Returns:
            Annotated segment with: devils_advocate_challenges, final_confidence,
            cleared_for_report, highest_action.
        """
        challenges = []
        actions    = []

        for rule in CHALLENGE_RULES:
            try:
                if rule["check"](segment):
                    challenges.append({
                        "rule_id":   rule["id"],
                        "challenge": rule["challenge"],
                        "action":    rule["action"],
                    })
                    actions.append(rule["action"])
                    logger.info(
                        f"Devil's Advocate — rule '{rule['id']}' triggered on "
                        f"segment {segment.get('segment_id')}: action={rule['action']}"
                    )
            except Exception as exc:
                logger.debug(f"Rule '{rule['id']}' evaluation error: {exc}")

        # ── Determine highest-severity action ─────────────────────────────
        if actions:
            highest_action = max(actions, key=lambda a: ACTION_PRIORITY.get(a, 0))
        else:
            highest_action = None

        # ── Final confidence string ────────────────────────────────────────
        if highest_action == "REQUEST_HUMAN_REVIEW":
            final_confidence = "Requires Human Review"
        elif highest_action == "FLAG_IRI_INVALID":
            final_confidence = "Low — IRI Invalid"
        elif highest_action == "DOWNGRADE_CONFIDENCE":
            final_confidence = "Medium — Verify Before Acting"
        else:
            final_confidence = "High"

        # ── Cleared for report? ────────────────────────────────────────────
        # Segments requiring human review are held — others proceed
        cleared = highest_action != "REQUEST_HUMAN_REVIEW"

        segment_out = dict(segment)  # Shallow copy to avoid mutating input
        segment_out["devils_advocate_challenges"] = challenges
        segment_out["final_confidence"]           = final_confidence
        segment_out["cleared_for_report"]         = cleared
        segment_out["challenge_count"]            = len(challenges)
        segment_out["highest_action"]             = highest_action

        return segment_out
