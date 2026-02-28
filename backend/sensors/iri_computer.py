"""
backend/sensors/iri_computer.py

Channel 1: Accelerometer → International Roughness Index (IRI)

Physics:
    IRI is the World Bank's standard road roughness metric (m/km).
    Defined as accumulated suspension travel of a quarter-car model
    traversing the road at 80 km/h per km.

    Reference: Sayers (1986) quarter-car model.
    Validated accuracy: ±0.3 IRI units against laser profilometer
    (Douangphachanh & Oneyama, 2014).
"""

import numpy as np
from scipy import signal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quarter-car model parameters — Sayers (1986)
# ---------------------------------------------------------------------------
QC_PARAMS = {
    "k1": 653.0,    # Spring constant — sprung mass (N/m·kg⁻¹)
    "k2": 63.3,     # Spring constant — unsprung mass
    "c1": 6.0,      # Damping — sprung mass
    "c2": 0.01,     # Damping — unsprung mass
    "m1": 0.15,     # Mass ratio — sprung to total
    "m2": 1.0,      # Mass ratio — unsprung
}

# IRI classification thresholds — IRC:SP:20 (Indian Rural Roads Manual)
IRI_THRESHOLDS = [
    {"max": 2.0, "condition": "Good",      "color": "#27AE60", "action": "Routine maintenance"},
    {"max": 4.0, "condition": "Fair",      "color": "#F39C12", "action": "Preventive treatment"},
    {"max": 6.0, "condition": "Poor",      "color": "#E74C3C", "action": "Rehabilitation"},
    {"max": float("inf"), "condition": "Very Poor", "color": "#8E44AD", "action": "Reconstruction"},
]


def compute_iri(
    accel_z: np.ndarray,
    gps_speed: np.ndarray,
    sample_rate: int = 200,
    min_speed_kmh: float = 20.0,
) -> Optional[float]:
    """
    Compute International Roughness Index from vertical accelerometer data.

    Args:
        accel_z:      Vertical acceleration array (m/s²) at sample_rate Hz.
                      Positive = upward. Raw sensor output — gravity NOT removed yet.
        gps_speed:    Vehicle speed array (m/s), same length as accel_z.
                      Interpolated to match accelerometer timestamps.
        sample_rate:  Accelerometer sampling rate in Hz (default 200).
        min_speed_kmh: Minimum speed for valid IRI. Returns None if median
                       speed is below this threshold.

    Returns:
        IRI value (m/km), or None if data is invalid (too slow, too short).
    """
    if len(accel_z) < sample_rate:
        logger.warning("IRI computation: insufficient data (< 1 second)")
        return None

    # Ensure arrays are the same length
    min_len = min(len(accel_z), len(gps_speed))
    accel_z = accel_z[:min_len].astype(np.float64)
    gps_speed = gps_speed[:min_len].astype(np.float64)

    # Speed check
    median_speed_kmh = float(np.median(gps_speed)) * 3.6
    if median_speed_kmh < min_speed_kmh:
        logger.warning(
            f"IRI invalid: median speed {median_speed_kmh:.1f} km/h < {min_speed_kmh} km/h"
        )
        return None

    # Step 1: Remove gravity (DC offset) — high-pass Butterworth filter
    # Cutoff at 0.5 Hz removes the 9.81 m/s² gravity component
    nyquist = sample_rate / 2.0
    cutoff_hz = 0.5
    b, a = signal.butter(4, cutoff_hz / nyquist, btype="high")
    accel_filtered = signal.filtfilt(b, a, accel_z)

    # Step 2: Speed normalisation
    # IRI is defined at 80 km/h; normalise for actual travel speed
    speed_80_ms = 80.0 / 3.6  # 22.22 m/s
    speed_factor = np.clip(gps_speed / speed_80_ms, 0.3, 2.0)
    accel_normalised = accel_filtered / speed_factor

    # Step 3: Quarter-car model numerical integration (Euler forward)
    k1, k2 = QC_PARAMS["k1"], QC_PARAMS["k2"]
    c1, c2 = QC_PARAMS["c1"], QC_PARAMS["c2"]
    m1, m2 = QC_PARAMS["m1"], QC_PARAMS["m2"]

    dt = 1.0 / sample_rate
    n = len(accel_normalised)

    # State: [z1, dz1, z2, dz2]
    # z1 = sprung mass displacement, z2 = unsprung mass displacement
    state = np.zeros(4, dtype=np.float64)
    suspension_travel = 0.0

    for i in range(n):
        u = accel_normalised[i]  # road profile input (m/s²)

        dz1 = state[1]
        dz2 = state[3]
        rel_disp = state[0] - state[2]
        rel_vel = state[1] - state[3]

        ddz1 = -k1 / m1 * rel_disp - c1 / m1 * rel_vel
        ddz2 = (
            k1 / m2 * rel_disp
            + c1 / m2 * rel_vel
            - k2 / m2 * state[2]
            - c2 / m2 * dz2
            + u
        )

        state[0] += dz1 * dt
        state[1] += ddz1 * dt
        state[2] += dz2 * dt
        state[3] += ddz2 * dt

        suspension_travel += abs(rel_disp) * dt

    # Step 4: IRI = suspension travel / distance (m/km)
    distance_km = float(np.trapz(gps_speed, dx=dt)) / 1000.0
    if distance_km < 0.001:
        logger.warning("IRI computation: near-zero distance travelled")
        return None

    iri = suspension_travel / distance_km
    return float(iri)


def classify_iri(iri: float) -> dict:
    """
    Classify IRI value per IRC:SP:20 Indian Rural Roads Manual.

    Args:
        iri: IRI value in m/km.

    Returns:
        dict with keys: condition, color (hex), action, iri_value.
    """
    for threshold in IRI_THRESHOLDS:
        if iri < threshold["max"]:
            return {
                "iri_value": round(iri, 2),
                "condition": threshold["condition"],
                "color": threshold["color"],
                "action": threshold["action"],
            }
    # Fallback (should never reach here)
    return {
        "iri_value": round(iri, 2),
        "condition": "Very Poor",
        "color": "#8E44AD",
        "action": "Reconstruction",
    }


def segment_iri_stats(iri_values: list[float]) -> dict:
    """
    Aggregate statistics for multiple IRI readings on a segment.

    Args:
        iri_values: List of IRI readings from multiple passes.

    Returns:
        dict with mean, median, std, min, max, pass_count.
    """
    if not iri_values:
        return {"mean": None, "median": None, "std": None,
                "min": None, "max": None, "pass_count": 0}

    arr = np.array(iri_values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "pass_count": len(iri_values),
        "representative_iri": float(np.median(arr)),  # Median is the robust estimate
    }
