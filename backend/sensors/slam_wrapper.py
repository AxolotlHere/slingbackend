"""
backend/sensors/slam_wrapper.py

Visual Odometry wrapper — graceful degradation stack.

Tried in order (first available wins for Anchor 1 scale):

  Priority 1: DPVO (Deep Patch Visual Odometry, NeurIPS 2023)
               pip install dpvo  — GPU, Python API, active repo
               https://github.com/princeton-vl/DPVO

  Priority 2: stella_vslam (active ORB-SLAM3 successor, 2024)
               pip install stella_vslam  — cross-platform, Python bindings
               https://github.com/stella-cv/stella_vslam

  Priority 3: IMU double-integration (Python-native, always available)
               Rough but avoids complete anchor loss.

If all fail: returns None. MetricDepthPipeline continues with 2/3 anchors
(ground-plane + optical-flow), which is entirely valid for the demo.

Interface contract (unchanged from ORB-SLAM3 era):
    wrapper = SLAMWrapper()
    scale = wrapper.get_imu_scale_estimate(imu_buffer)
    # Returns float or None
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Availability detection ─────────────────────────────────────────────────

_DPVO_AVAILABLE = False
_STELLA_AVAILABLE = False

try:
    import dpvo  # type: ignore  # noqa: F401
    _DPVO_AVAILABLE = True
    logger.info("DPVO found — priority 1 visual odometry active.")
except ImportError:
    pass

if not _DPVO_AVAILABLE:
    try:
        import stella_vslam  # type: ignore  # noqa: F401
        _STELLA_AVAILABLE = True
        logger.info("stella_vslam found — using as visual odometry backend.")
    except ImportError:
        pass

if not _DPVO_AVAILABLE and not _STELLA_AVAILABLE:
    logger.warning(
        "No visual odometry backend found (DPVO or stella_vslam). "
        "Scale fusion will use 2/3 anchors (ground-plane + optical-flow). "
        "To install DPVO: pip install dpvo  (requires CUDA)\n"
        "To install stella_vslam: pip install stella_vslam"
    )


class SLAMWrapper:
    """
    Visual odometry adapter. Tries DPVO → stella_vslam → IMU fallback.

    All public methods return None cleanly if no backend is available,
    letting MetricDepthPipeline continue with remaining anchors.
    """

    def __init__(
        self,
        camera_height_m: float = 1.2,
        dpvo_config: Optional[dict] = None,
    ):
        self.camera_height = camera_height_m
        self.backend = "none"
        self._vo = None

        if _DPVO_AVAILABLE:
            self._init_dpvo(dpvo_config or {})
        elif _STELLA_AVAILABLE:
            self._init_stella()

    # ── Backend initialisation ─────────────────────────────────────────────

    def _init_dpvo(self, config: dict):
        """Initialise DPVO visual odometry."""
        try:
            from dpvo.dpvo import DPVO  # type: ignore
            self._vo = DPVO(
                cfg=config.get("cfg", "config/default.yaml"),
                network=config.get("network", "dpvo.pth"),
                viz=False,
            )
            self.backend = "dpvo"
            logger.info("DPVO initialised successfully.")
        except Exception as exc:
            logger.error(f"DPVO init failed: {exc}. Falling back to IMU-only.")
            self.backend = "imu_fallback"

    def _init_stella(self):
        """Initialise stella_vslam."""
        try:
            import stella_vslam as sv  # type: ignore
            self._vo = sv.system(
                vocab_file_path="models/orb_vocabulary.fbow",
                config=sv.config("calibration/stella_config.yaml"),
            )
            self._vo.startup(parallel_bundle_adjustment=False)
            self.backend = "stella_vslam"
            logger.info("stella_vslam initialised successfully.")
        except Exception as exc:
            logger.error(f"stella_vslam init failed: {exc}. Falling back to IMU-only.")
            self.backend = "imu_fallback"

    # ── Per-frame processing ───────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        imu_measurements: Optional[list[dict]] = None,
    ) -> Optional[float]:
        """
        Feed a camera frame to the visual odometry backend.

        Args:
            frame:            BGR numpy array (H, W, 3).
            timestamp:        Frame timestamp in seconds.
            imu_measurements: Optional list of IMU dicts (used by IMU fallback).

        Returns:
            Metric scale estimate (float) or None if VO is not tracking yet.
        """
        if self.backend == "dpvo" and self._vo is not None:
            return self._process_dpvo(frame, timestamp)
        elif self.backend == "stella_vslam" and self._vo is not None:
            return self._process_stella(frame, timestamp)
        return None

    def _process_dpvo(self, frame: np.ndarray, timestamp: float) -> Optional[float]:
        """DPVO frame processing → scale from trajectory magnitude."""
        try:
            import torch
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # DPVO expects (1, 3, H, W) float32 tensor normalised to [0, 1]
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            intrinsics = self._default_intrinsics(frame.shape)

            self._vo(t, intrinsics, timestamp)  # type: ignore
            poses, _ = self._vo.terminate()     # type: ignore

            if poses is not None and len(poses) > 1:
                # Take the most recent camera translation magnitude as our scale proxy
                translation = poses[-1, :3, 3]
                scale = float(np.linalg.norm(translation))
                return scale if scale > 0.001 else None

            return None
        except Exception as exc:
            logger.debug(f"DPVO processing error: {exc}")
            return None

    def _process_stella(self, frame: np.ndarray, timestamp: float) -> Optional[float]:
        """stella_vslam frame processing → scale."""
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cam_pose = self._vo.feed_monocular_frame(gray, timestamp)  # type: ignore
            if cam_pose is not None:
                scale = float(np.linalg.norm(cam_pose[:3, 3]))
                return scale if scale > 0.001 else None
            return None
        except Exception as exc:
            logger.debug(f"stella_vslam processing error: {exc}")
            return None

    # ── IMU-only fallback ──────────────────────────────────────────────────

    def get_imu_scale_estimate(
        self,
        imu_buffer: list[dict],
        dt: float = 1.0 / 200.0,
    ) -> Optional[float]:
        """
        Python-native IMU scale estimation — always available, rough accuracy.

        High-passes vertical acceleration → double-integrates → uses RMS
        displacement as a proxy for scale, anchored to camera_height.

        Args:
            imu_buffer: List of IMU dicts with 'az' key (vertical accel m/s²).
            dt:         Time step (seconds), default for 200 Hz sampling.

        Returns:
            Scale estimate or None if insufficient data.
        """
        if len(imu_buffer) < 40:
            return None

        try:
            from scipy import signal as scipy_signal

            az = np.array([m.get("az", 0.0) for m in imu_buffer], dtype=np.float64)

            # Remove gravity with 4th-order Butterworth high-pass at 0.5 Hz
            nyq = 1.0 / (2.0 * dt)
            b, a = scipy_signal.butter(4, 0.5 / nyq, btype="high")
            az_filt = scipy_signal.filtfilt(b, a, az)

            # Double integration → vertical displacement
            velocity     = np.cumsum(az_filt) * dt
            displacement = np.cumsum(velocity) * dt

            rms_disp = float(np.sqrt(np.mean(displacement ** 2)))
            if rms_disp < 1e-6:
                return None

            # Empirical: camera_height / typical relative depth ≈ scale
            scale = (self.camera_height * 0.05) / max(rms_disp, 1e-4)
            return float(np.clip(scale, 0.1, 50.0))

        except Exception as exc:
            logger.debug(f"IMU scale estimation failed: {exc}")
            return None

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _default_intrinsics(shape: tuple) -> "Optional[object]":
        """Return a basic pinhole intrinsics tensor for DPVO."""
        try:
            import torch
            h, w = shape[:2]
            # fx, fy, cx, cy (normalised by image size as DPVO expects)
            fx = fy = 0.8 * w
            cx, cy = w / 2.0, h / 2.0
            return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        except Exception:
            return None

    def shutdown(self):
        """Clean shutdown of VO backend."""
        if self._vo is not None:
            try:
                if self.backend == "stella_vslam":
                    self._vo.shutdown()  # type: ignore
                elif self.backend == "dpvo":
                    self._vo.terminate()  # type: ignore
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        return self.backend in ("dpvo", "stella_vslam")

    @property
    def active_backend(self) -> str:
        return self.backend
