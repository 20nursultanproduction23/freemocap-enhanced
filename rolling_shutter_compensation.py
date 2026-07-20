"""
Rolling Shutter Compensation — corrects skew artifacts from non-global-shutter cameras.

When a camera uses a rolling shutter (line-by-line readout), fast movements cause
diagonal skew because different rows are exposed at different times.

This module estimates and compensates for rolling shutter distortion in 3D skeleton
data by adjusting keypoint positions based on their vertical position and the
camera's readout characteristics.

Usage:
    compensator = RollingShutterCompensator(readout_time_ms=33.0, fps=25.0)
    corrected = compensator.compensate(skeleton_3d, frame_idx=0)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RollingShutterParams:
    """Camera rolling shutter parameters."""
    readout_time_ms: float = 33.0
    fps: float = 25.0
    sensor_height_px: int = 1080
    frame_time_ms: float = 40.0

    @property
    def readout_ratio(self) -> float:
        """Fraction of frame time used for readout."""
        return self.readout_time_ms / self.frame_time_ms

    @property
    def row_delay_factor(self) -> float:
        """Per-row time delay (normalized)."""
        return self.readout_time_ms / max(1, self.sensor_height_px)


@dataclass
class DistortionEstimate:
    """Estimated rolling shutter distortion for a frame."""
    frame_idx: int
    max_skew_px: float
    affected_rows_pct: float
    severity: str


class RollingShutterCompensator:
    """
    Compensates for rolling shutter distortion in 3D skeleton data.

    The compensation works by estimating the time offset for each keypoint
    based on its vertical position in the camera frame, then interpolating
    the skeleton position to what it would be at the frame center time.
    """

    def __init__(self, readout_time_ms: float = 33.0, fps: float = 25.0,
                 sensor_height_px: int = 1080, compensation_strength: float = 1.0):
        """
        Args:
            readout_time_ms: Camera sensor readout time in milliseconds.
                             Common values: 33ms (1/30s), 16.7ms (1/60s), 8.3ms (1/120s)
            fps: Recording frame rate
            sensor_height_px: Camera sensor height in pixels
            compensation_strength: 0.0 = no compensation, 1.0 = full, >1.0 = overcompensate
        """
        self.params = RollingShutterParams(
            readout_time_ms=readout_time_ms,
            fps=fps,
            sensor_height_px=sensor_height_px,
            frame_time_ms=1000.0 / fps,
        )
        self.compensation_strength = compensation_strength

    def compensate_frame(self, keypoints_3d: np.ndarray,
                         keypoints_2d_y: Optional[np.ndarray] = None,
                         velocities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compensate rolling shutter distortion for a single frame.

        Args:
            keypoints_3d: (N, 3) array of 3D keypoints
            keypoints_2d_y: (N,) array of 2D Y coordinates in camera frame.
                           If None, assumes all keypoints at sensor center (no compensation).
            velocities: (N, 3) array of 3D velocities (mm/frame).
                       If None, estimated from keypoint position assumptions.

        Returns:
            (N, 3) compensated 3D keypoints
        """
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return keypoints_3d

        corrected = keypoints_3d.copy()
        n_keypoints = len(keypoints_3d)

        if velocities is None:
            velocities = np.zeros_like(keypoints_3d)

        if keypoints_2d_y is None:
            keypoints_2d_y = np.full(n_keypoints, self.params.sensor_height_px / 2.0)

        for i in range(n_keypoints):
            if np.any(np.isnan(keypoints_3d[i])):
                continue

            row_fraction = keypoints_2d_y[i] / max(1, self.params.sensor_height_px)
            time_offset_s = (row_fraction - 0.5) * self.params.readout_time_ms / 1000.0

            velocity = velocities[i] if i < len(velocities) else np.zeros(3)
            displacement = velocity * time_offset_s * self.compensation_strength

            corrected[i] = keypoints_3d[i] + displacement

        return corrected

    def compensate_sequence(self, skeleton_sequence: np.ndarray,
                            y_positions: Optional[np.ndarray] = None,
                            velocity_sequence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compensate rolling shutter for an entire sequence.

        Args:
            skeleton_sequence: (T, N, 3) skeleton frames
            y_positions: (T, N) 2D Y positions per keypoint per frame.
                        If None, uses center of sensor.
            velocity_sequence: (T, N, 3) velocities. If None, computed from sequence.

        Returns:
            (T, N, 3) compensated skeleton sequence
        """
        if skeleton_sequence is None or len(skeleton_sequence) == 0:
            return skeleton_sequence

        T, N, _ = skeleton_sequence.shape
        compensated = np.full_like(skeleton_sequence, np.nan)

        if velocity_sequence is None:
            velocity_sequence = self._estimate_velocities(skeleton_sequence)

        for t in range(T):
            y_frame = y_positions[t] if y_positions is not None else None
            vel_frame = velocity_sequence[t] if velocity_sequence is not None else None

            compensated[t] = self.compensate_frame(
                skeleton_sequence[t],
                keypoints_2d_y=y_frame,
                velocities=vel_frame,
            )

        return compensated

    def _estimate_velocities(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Estimate keypoint velocities from position differences.

        Uses central differences where possible, forward/backward at edges.
        """
        T, N, _ = skeleton.shape
        velocities = np.zeros_like(skeleton)

        for t in range(T):
            if t == 0 and T > 1:
                diff = skeleton[1] - skeleton[0]
            elif t == T - 1 and T > 1:
                diff = skeleton[-1] - skeleton[-2]
            elif T > 2:
                diff = (skeleton[min(t + 1, T - 1)] - skeleton[max(t - 1, 0)]) / 2.0
            else:
                diff = np.zeros((N, 3))

            nan_mask = np.isnan(diff)
            diff[nan_mask] = 0.0
            velocities[t] = diff

        return velocities

    def estimate_distortion(self, skeleton_sequence: np.ndarray,
                            y_positions: Optional[np.ndarray] = None) -> List[DistortionEstimate]:
        """
        Estimate rolling shutter distortion severity per frame.

        Returns a list of DistortionEstimate for each frame with significant distortion.
        """
        if skeleton_sequence is None or len(skeleton_sequence) < 2:
            return []

        T, N, _ = skeleton_sequence.shape
        estimates = []

        velocity_sequence = self._estimate_velocities(skeleton_sequence)

        for t in range(T):
            frame_skeleton = skeleton_sequence[t]
            vel_frame = velocity_sequence[t]

            if np.all(np.isnan(frame_skeleton)):
                continue

            valid = ~np.isnan(frame_skeleton).all(axis=1)
            if not np.any(valid):
                continue

            speed = np.linalg.norm(vel_frame[valid], axis=1)
            max_speed = float(np.max(speed)) if len(speed) > 0 else 0.0

            readout_time_s = self.params.readout_time_ms / 1000.0
            max_skew_3d = max_speed * readout_time_s * self.compensation_strength

            y_vals = y_positions[t] if y_positions is not None else np.full(N, self.params.sensor_height_px / 2)
            valid_y = y_vals[valid]
            row_range = float(np.max(valid_y) - np.min(valid_y)) if len(valid_y) > 1 else 0
            affected_pct = 100.0 * row_range / self.params.sensor_height_px

            severity = "none"
            if max_skew_3d > 5.0:
                severity = "high"
            elif max_skew_3d > 1.0:
                severity = "moderate"
            elif max_skew_3d > 0.1:
                severity = "low"

            if severity != "none":
                estimates.append(DistortionEstimate(
                    frame_idx=t,
                    max_skew_px=round(max_skew_3d, 3),
                    affected_rows_pct=round(affected_pct, 1),
                    severity=severity,
                ))

        return estimates

    def get_params(self) -> dict:
        """Return current compensation parameters."""
        return {
            "readout_time_ms": self.params.readout_time_ms,
            "fps": self.params.fps,
            "sensor_height_px": self.params.sensor_height_px,
            "compensation_strength": self.compensation_strength,
            "readout_ratio": round(self.params.readout_ratio, 3),
            "row_delay_factor_us": round(self.params.row_delay_factor * 1000, 2),
        }


def classify_camera_shutter(readout_time_ms: float) -> dict:
    """Classify camera type based on readout time."""
    if readout_time_ms <= 1.0:
        return {
            "type": "global_shutter",
            "description": "Global shutter — no rolling shutter distortion",
            "compensation_needed": False,
        }
    elif readout_time_ms <= 5.0:
        return {
            "type": "fast_rolling",
            "description": f"Fast rolling shutter ({readout_time_ms}ms) — minimal distortion",
            "compensation_needed": False,
        }
    elif readout_time_ms <= 16.7:
        return {
            "type": "rolling_60fps",
            "description": f"Rolling shutter at 60fps equivalent ({readout_time_ms}ms) — slight distortion on fast motion",
            "compensation_needed": True,
        }
    elif readout_time_ms <= 33.3:
        return {
            "type": "rolling_30fps",
            "description": f"Rolling shutter at 30fps equivalent ({readout_time_ms}ms) — noticeable distortion on fast motion",
            "compensation_needed": True,
        }
    else:
        return {
            "type": "slow_rolling",
            "description": f"Slow rolling shutter ({readout_time_ms}ms) — significant distortion, compensation strongly recommended",
            "compensation_needed": True,
        }


COMMON_CAMERAS = {
    "raspberry_pi_v2": {"readout_ms": 33.0, "sensor_height": 1944, "notes": "No global shutter mode"},
    "raspberry_pi_v3": {"readout_ms": 29.0, "sensor_height": 2464, "notes": "Improved but still rolling"},
    "logitech_c920": {"readout_ms": 33.0, "sensor_height": 1080, "notes": "USB webcam, rolling shutter"},
    "logitech_c930": {"readout_ms": 33.0, "sensor_height": 1080, "notes": "Wider FOV, same readout"},
    "iphone_14": {"readout_ms": 8.3, "sensor_height": 2160, "notes": "Fast readout, near-global"},
    "sony_alpha6400": {"readout_ms": 1.0, "sensor_height": 4000, "notes": "Global shutter capable"},
    "blackmagic_pocket": {"readout_ms": 15.0, "sensor_height": 2160, "notes": "Cinema camera, moderate readout"},
    "flir_boson": {"readout_ms": 1.0, "sensor_height": 512, "notes": "IR camera, global shutter"},
}
