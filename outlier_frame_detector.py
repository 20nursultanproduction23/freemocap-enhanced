"""
Automatic Outlier Frame Flagging — finds problematic frames in 3D skeleton data.

Analyzes 3D skeleton sequences and flags frames with:
  - Sudden joint velocity spikes (teleporting joints)
  - Bone length violations (stretching/compressing)
  - High NaN ratio (missing data)
  - Symmetry violations (impossible poses)
  - Jitter bursts (noisy tracking)

Usage:
    detector = OutlierFrameDetector()
    flags = detector.analyze(skeleton_3d)
    for f in flags:
        print(f"Frame {f.frame_idx}: {f.severity} — {f.description}")
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OutlierFlag:
    """A flagged problematic frame."""
    frame_idx: int
    severity: str
    category: str
    description: str
    score: float
    affected_keypoints: List[int] = field(default_factory=list)


@dataclass
class AnalysisSummary:
    """Summary of outlier analysis."""
    total_frames: int
    total_flagged: int
    severity_counts: dict
    category_counts: dict
    worst_frames: List[int]
    flags: List[OutlierFlag]


BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 18), (15, 19),
    (11, 22), (22, 23), (23, 24), (24, 25), (25, 26), (25, 27), (25, 28),
    (11, 17), (17, 18), (18, 19), (17, 20), (20, 21),
]

BONE_NAMES = [
    "nose→left_eye", "left_eye→left_ear", "left_ear→left_shoulder", "left_shoulder→left_elbow",
    "nose→right_ear", "right_ear→right_eye", "right_eye→right_shoulder", "right_shoulder→right_elbow",
    "left_wrist→left_palm",
    "left_hip→right_hip", "right_hip→right_knee", "right_knee→right_ankle",
    "right_ankle→right_big_toe", "right_big_toe→right_small_toe", "right_ankle→right_heel", "right_heel→right_small_toe",
    "left_hip→left_chest", "left_chest→left_knee", "left_knee→left_ankle",
    "left_ankle→left_big_toe", "left_big_toe→left_small_toe",
    "left_hip→left_hip2", "left_hip2→left_heel", "left_heel→left_small_toe",
    "right_hip→right_hip2", "right_hip2→right_heel", "right_heel→right_small_toe",
    "left_hip→left_ankle2", "left_ankle2→left_heel", "left_heel→left_small_toe2",
]


class OutlierFrameDetector:
    """
    Detects and flags problematic frames in 3D skeleton sequences.

    Each detection category produces OutlierFlags with severity and description.
    """

    def __init__(self, velocity_threshold: float = 50.0,
                 bone_stretch_threshold: float = 0.5,
                 nan_ratio_threshold: float = 0.3,
                 jitter_threshold: float = 20.0,
                 symmetry_threshold: float = 0.4):
        """
        Args:
            velocity_threshold: Max allowed joint velocity (mm/frame) before flagging
            bone_stretch_threshold: Max allowed bone length deviation (fraction of median)
            nan_ratio_threshold: Max allowed NaN fraction per frame before flagging
            jitter_threshold: Max allowed frame-to-frame acceleration (mm/frame^2)
            symmetry_threshold: Max allowed left-right asymmetry ratio
        """
        self.velocity_threshold = velocity_threshold
        self.bone_stretch_threshold = bone_stretch_threshold
        self.nan_ratio_threshold = nan_ratio_threshold
        self.jitter_threshold = jitter_threshold
        self.symmetry_threshold = symmetry_threshold

    def analyze(self, skeleton: np.ndarray, fps: float = 25.0) -> AnalysisSummary:
        """
        Analyze a skeleton sequence for outlier frames.

        Args:
            skeleton: (T, N, 3) where T=frames, N=keypoints, 3=XYZ
            fps: Frame rate for velocity normalization

        Returns:
            AnalysisSummary with all flags
        """
        if skeleton is None or len(skeleton) < 2:
            return AnalysisSummary(
                total_frames=len(skeleton) if skeleton is not None else 0,
                total_flagged=0,
                severity_counts={},
                category_counts={},
                worst_frames=[],
                flags=[],
            )

        T, N, _ = skeleton.shape
        all_flags: List[OutlierFlag] = []

        all_flags.extend(self._check_nan_ratio(skeleton))
        all_flags.extend(self._check_velocity_spikes(skeleton, fps))
        all_flags.extend(self._check_bone_lengths(skeleton))
        all_flags.extend(self._check_jitter_bursts(skeleton))
        all_flags.extend(self._check_symmetry(skeleton))
        all_flags.extend(self._check_sudden_appearance(skeleton))

        severity_counts = {}
        category_counts = {}
        for f in all_flags:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
            category_counts[f.category] = category_counts.get(f.category, 0) + 1

        worst_frames = sorted(
            set(f.frame_idx for f in all_flags),
            key=lambda idx: sum(f.score for f in all_flags if f.frame_idx == idx),
            reverse=True,
        )[:10]

        return AnalysisSummary(
            total_frames=T,
            total_flagged=len(set(f.frame_idx for f in all_flags)),
            severity_counts=severity_counts,
            category_counts=category_counts,
            worst_frames=worst_frames,
            flags=all_flags,
        )

    def _check_nan_ratio(self, skeleton: np.ndarray) -> List[OutlierFlag]:
        """Flag frames with too many NaN keypoints."""
        T, N, _ = skeleton.shape
        flags = []

        for t in range(T):
            frame = skeleton[t]
            n_nan = int(np.isnan(frame).all(axis=1).sum())
            ratio = n_nan / N

            if ratio > self.nan_ratio_threshold:
                severity = "critical" if ratio > 0.8 else "high" if ratio > 0.5 else "moderate"
                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity=severity,
                    category="nan_ratio",
                    description=f"{n_nan}/{N} keypoints are NaN ({ratio:.0%})",
                    score=ratio * 10,
                ))

        return flags

    def _check_velocity_spikes(self, skeleton: np.ndarray, fps: float) -> List[OutlierFlag]:
        """Flag frames where joints teleport (sudden large displacement)."""
        T, N, _ = skeleton.shape
        flags = []

        for t in range(1, T):
            prev = skeleton[t - 1]
            curr = skeleton[t]

            velocity = curr - prev
            speed = np.linalg.norm(velocity, axis=1)

            valid_mask = ~(np.isnan(prev).any(axis=1) | np.isnan(curr).any(axis=1))
            valid_speed = speed[valid_mask]

            if len(valid_speed) == 0:
                continue

            max_speed = float(np.max(valid_speed))
            if max_speed > self.velocity_threshold:
                spike_kpts = np.where(valid_mask & (speed > self.velocity_threshold))[0]
                severity = "critical" if max_speed > self.velocity_threshold * 3 else "high"

                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity=severity,
                    category="velocity_spike",
                    description=f"Joint teleport: max speed {max_speed:.1f} mm/frame (threshold: {self.velocity_threshold})",
                    score=max_speed / self.velocity_threshold,
                    affected_keypoints=spike_kpts.tolist(),
                ))

        return flags

    def _check_bone_lengths(self, skeleton: np.ndarray) -> List[OutlierFlag]:
        """Flag frames where bone lengths deviate significantly from median."""
        T, N, _ = skeleton.shape
        flags = []

        bone_lengths = {}
        for start, end in BONE_CONNECTIONS:
            if start >= N or end >= N:
                continue

            lengths = []
            for t in range(T):
                p1, p2 = skeleton[t, start], skeleton[t, end]
                if not np.any(np.isnan(p1)) and not np.any(np.isnan(p2)):
                    lengths.append(np.linalg.norm(p2 - p1))

            if lengths:
                bone_lengths[(start, end)] = {
                    "median": float(np.median(lengths)),
                    "lengths": lengths,
                }

        for t in range(T):
            violations = 0
            total_bones = 0

            for (start, end), bone_data in bone_lengths.items():
                if start >= N or end >= N:
                    continue

                p1, p2 = skeleton[t, start], skeleton[t, end]
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue

                current_length = float(np.linalg.norm(p2 - p1))
                median_length = bone_data["median"]

                if median_length < 1e-6:
                    continue

                deviation = abs(current_length - median_length) / median_length
                total_bones += 1

                if deviation > self.bone_stretch_threshold:
                    violations += 1

            if total_bones > 0 and violations >= 2:
                ratio = violations / total_bones
                severity = "high" if ratio > 0.3 else "moderate"

                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity=severity,
                    category="bone_stretch",
                    description=f"{violations}/{total_bones} bones deviate >{self.bone_stretch_threshold:.0%} from median",
                    score=violations,
                ))

        return flags

    def _check_jitter_bursts(self, skeleton: np.ndarray) -> List[OutlierFlag]:
        """Flag frames with high acceleration (jitter bursts)."""
        T, N, _ = skeleton.shape
        flags = []

        if T < 3:
            return flags

        for t in range(1, T - 1):
            prev = skeleton[t - 1]
            curr = skeleton[t]
            next_f = skeleton[t + 1]

            accel = next_f - 2 * curr + prev
            accel_mag = np.linalg.norm(accel, axis=1)

            valid_mask = ~(np.isnan(prev).any(axis=1) | np.isnan(curr).any(axis=1) | np.isnan(next_f).any(axis=1))
            valid_accel = accel_mag[valid_mask]

            if len(valid_accel) == 0:
                continue

            max_accel = float(np.max(valid_accel))
            if max_accel > self.jitter_threshold:
                jitter_kpts = np.where(valid_mask & (accel_mag > self.jitter_threshold))[0]
                severity = "high" if max_accel > self.jitter_threshold * 3 else "moderate"

                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity=severity,
                    category="jitter_burst",
                    description=f"Acceleration spike: {max_accel:.1f} mm/frame² (threshold: {self.jitter_threshold})",
                    score=max_accel / self.jitter_threshold,
                    affected_keypoints=jitter_kpts.tolist(),
                ))

        return flags

    def _check_symmetry(self, skeleton: np.ndarray) -> List[OutlierFlag]:
        """Flag frames with impossible left-right asymmetry."""
        T, N, _ = skeleton.shape
        flags = []

        SYMMETRY_PAIRS = [
            (5, 6),    # left_shoulder, right_shoulder
            (7, 8),    # left_elbow, right_elbow
            (9, 10),   # left_wrist, right_wrist
            (11, 12),  # left_hip, right_hip
            (13, 14),  # left_knee, right_knee
            (15, 16),  # left_ankle, right_ankle
        ]

        for t in range(T):
            frame = skeleton[t]
            asymmetric_pairs = 0

            for left, right in SYMMETRY_PAIRS:
                if left >= N or right >= N:
                    continue

                lp = frame[left]
                rp = frame[right]

                if np.any(np.isnan(lp)) or np.any(np.isnan(rp)):
                    continue

                mid_x = (lp[0] + rp[0]) / 2
                left_dist = abs(lp[0] - mid_x)
                right_dist = abs(rp[0] - mid_x)

                if min(left_dist, right_dist) > 1e-6:
                    asymmetry = abs(left_dist - right_dist) / max(left_dist, right_dist)
                    if asymmetry > self.symmetry_threshold:
                        asymmetric_pairs += 1

            if asymmetric_pairs >= 2:
                severity = "moderate" if asymmetric_pairs <= 3 else "high"
                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity=severity,
                    category="symmetry_violation",
                    description=f"{asymmetric_pairs} left-right pairs are highly asymmetric",
                    score=asymmetric_pairs,
                ))

        return flags

    def _check_sudden_appearance(self, skeleton: np.ndarray) -> List[OutlierFlag]:
        """Flag frames where the skeleton suddenly appears from all-NaN."""
        T, N, _ = skeleton.shape
        flags = []

        for t in range(1, T):
            prev_valid = int(~np.isnan(skeleton[t - 1]).all(axis=1).sum() == 0)
            curr_valid_count = int(~np.isnan(skeleton[t]).all(axis=1).sum() == 0)
            prev_nan_count = int(np.isnan(skeleton[t - 1]).all(axis=1).sum())
            curr_nan_count = int(np.isnan(skeleton[t]).all(axis=1).sum())

            if prev_nan_count > N * 0.8 and curr_nan_count < N * 0.3:
                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity="moderate",
                    category="sudden_appearance",
                    description=f"Skeleton appeared suddenly: {prev_nan_count}→{curr_nan_count} NaN keypoints",
                    score=5.0,
                ))

            if prev_nan_count < N * 0.3 and curr_nan_count > N * 0.8:
                flags.append(OutlierFlag(
                    frame_idx=t,
                    severity="moderate",
                    category="sudden_disappearance",
                    description=f"Skeleton disappeared suddenly: {prev_nan_count}→{curr_nan_count} NaN keypoints",
                    score=5.0,
                ))

        return flags


def quick_summary(flags: List[OutlierFlag]) -> str:
    """Generate a quick text summary of outlier flags."""
    if not flags:
        return "No outlier frames detected."

    categories = {}
    for f in flags:
        categories[f.category] = categories.get(f.category, 0) + 1

    lines = [f"Total flags: {len(flags)}"]
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {count}")

    critical = [f for f in flags if f.severity == "critical"]
    if critical:
        lines.append(f"\nCritical frames: {sorted(set(f.frame_idx for f in critical))}")

    return "\n".join(lines)
