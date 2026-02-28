"""
backend/segment_manager.py

100-metre GPS road segment buffering.

Receives raw sensor packets from the WebSocket stream and accumulates
them until a 100m road segment has been driven. Then emits the buffered
data for agent processing.

Packet types (from PWA collector):
    IMU  : {type, timestamp, ax, ay, az, rx, ry, rz}
    GPS  : {type, timestamp, lat, lng, speed_ms, accuracy_m, heading}
    FRAME: {type, timestamp, data (base64 JPEG)}
    AUDIO: {type, timestamp, rms, sample_rate, raw_samples (optional)}
"""

import base64
import logging
import math
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SEGMENT_LENGTH_M = 100.0   # Target segment length
MIN_GPS_ACCURACY_M = 20.0  # Discard GPS fixes with accuracy > this


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate great-circle distance between two GPS points (metres).
    """
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class SegmentManager:
    """
    Buffers multi-sensor data into 100m road segments.

    Usage:
        mgr = SegmentManager(session_id="abc123")
        await mgr.ingest(packet)
        if mgr.segment_ready():
            segment = mgr.pop_segment()
            # process_segment(segment)
    """

    def __init__(
        self,
        session_id: str,
        segment_length_m: float = SEGMENT_LENGTH_M,
    ):
        self.session_id = session_id
        self.segment_length_m = segment_length_m

        # Live buffers
        self._imu_buffer:    list[dict] = []
        self._gps_buffer:    list[dict] = []
        self._frame_buffer:  list[dict] = []  # Base64 decoded frames
        self._audio_buffer:  list[dict] = []

        # Segment tracking
        self._segment_start_gps: Optional[dict] = None
        self._distance_accumulated_m: float = 0.0
        self._last_gps: Optional[dict] = None
        self._segment_index: int = 0

        # Completed segments queue
        self._ready_segments: deque[dict] = deque()

        # Session GPS track
        self._full_track: list[dict] = []

    async def ingest(self, packet: dict):
        """
        Route a sensor packet to the appropriate buffer.
        Checks if a 100m segment has been completed after each GPS update.

        Args:
            packet: Sensor packet dict with 'type' key.
        """
        packet_type = packet.get("type", "").upper()

        if packet_type == "IMU":
            self._imu_buffer.append(packet)

        elif packet_type == "GPS":
            if packet.get("accuracy_m", 0) <= MIN_GPS_ACCURACY_M:
                self._process_gps(packet)

        elif packet_type == "FRAME":
            # Decode base64 JPEG to numpy array on receipt
            decoded = self._decode_frame(packet)
            if decoded is not None:
                self._frame_buffer.append({
                    "timestamp": packet.get("timestamp"),
                    "frame":     decoded,
                })

        elif packet_type == "AUDIO":
            self._audio_buffer.append(packet)

    def _process_gps(self, gps_packet: dict):
        """Update distance tracking and trigger segment completion."""
        self._gps_buffer.append(gps_packet)
        self._full_track.append(gps_packet)

        if self._segment_start_gps is None:
            self._segment_start_gps = gps_packet
            self._last_gps = gps_packet
            return

        if self._last_gps is not None:
            try:
                dist = haversine_distance(
                    self._last_gps["lat"], self._last_gps["lng"],
                    gps_packet["lat"],    gps_packet["lng"],
                )
                self._distance_accumulated_m += dist
            except (KeyError, TypeError):
                pass

        self._last_gps = gps_packet

        if self._distance_accumulated_m >= self.segment_length_m:
            self._finalise_segment()

    def _finalise_segment(self):
        """Cut the current buffers into a completed segment dict."""
        if not self._gps_buffer:
            return

        # Midpoint GPS for segment geolocation
        mid_idx = len(self._gps_buffer) // 2
        mid_gps = self._gps_buffer[mid_idx]

        segment_id = (
            f"{mid_gps.get('lat', 0):.5f},{mid_gps.get('lng', 0):.5f}"
            f"_seg{self._segment_index:04d}"
        )

        # Average speed
        speeds = [g.get("speed_ms", 0) for g in self._gps_buffer if g.get("speed_ms") is not None]
        avg_speed_ms = float(np.mean(speeds)) if speeds else 0.0

        segment = {
            "segment_id":     segment_id,
            "session_id":     self.session_id,
            "index":          self._segment_index,
            "timestamp":      self._gps_buffer[0].get("timestamp", time.time() * 1000),
            "gps": {
                "lat":     mid_gps.get("lat"),
                "lng":     mid_gps.get("lng"),
                "heading": mid_gps.get("heading"),
            },
            "start_gps":      self._segment_start_gps,
            "end_gps":        self._last_gps,
            "length_km":      round(self._distance_accumulated_m / 1000.0, 4),
            "avg_speed_kmh":  round(avg_speed_ms * 3.6, 1),
            "avg_speed_ms":   round(avg_speed_ms, 2),

            # Raw sensor buffers
            "imu_buffer":     list(self._imu_buffer),
            "gps_buffer":     list(self._gps_buffer),
            "frames":         [f["frame"] for f in self._frame_buffer],
            "audio_buffer":   list(self._audio_buffer),
        }

        self._ready_segments.append(segment)
        self._segment_index += 1

        # Reset buffers for next segment
        self._imu_buffer.clear()
        self._gps_buffer.clear()
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        self._distance_accumulated_m = 0.0
        self._segment_start_gps = self._last_gps

        logger.info(
            f"Session {self.session_id}: segment {self._segment_index - 1} complete "
            f"({segment['length_km']:.3f} km, {avg_speed_ms * 3.6:.1f} km/h)"
        )

    def segment_ready(self) -> bool:
        """Returns True if at least one completed segment is available."""
        return len(self._ready_segments) > 0

    def pop_segment(self) -> Optional[dict]:
        """
        Retrieve the oldest completed segment from the queue.
        Returns None if no segment is ready.
        """
        if self._ready_segments:
            return self._ready_segments.popleft()
        return None

    def finalise_session(self) -> Optional[dict]:
        """
        Force-finalise any remaining buffered data at end of session.
        Returns the final partial segment (may be < 100m).
        """
        if self._gps_buffer:
            self._finalise_segment()
            return self.pop_segment()
        return None

    @property
    def total_segments(self) -> int:
        return self._segment_index

    @property
    def session_track(self) -> list[dict]:
        """Full GPS track for the session (all points, not just segment midpoints)."""
        return self._full_track

    @staticmethod
    def _decode_frame(packet: dict) -> Optional[np.ndarray]:
        """Decode base64 JPEG frame data to BGR numpy array."""
        try:
            import cv2
            data_url = packet.get("data", "")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            raw = base64.b64decode(data_url)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as exc:
            logger.debug(f"Frame decode failed: {exc}")
            return None
