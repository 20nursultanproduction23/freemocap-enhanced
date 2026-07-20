"""
Detector Drift Monitor — tracks detection quality degradation over time.

Monitors per-frame detection metrics (confidence, completeness, consistency)
and warns when detector performance degrades within a session.

Causes of drift:
  - Lighting changes (clouds, sunset)
  - Camera auto-exposure/focus adjustments
  - Model fatigue with specific clothing/poses
  - Temperature-related sensor noise

Usage:
    monitor = DetectorDriftMonitor()
    for frame_idx in range(n_frames):
        monitor.record_frame(frame_idx, detections)
    report = monitor.get_report()
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Detection metrics for a single frame."""
    frame_idx: int
    n_detections: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    mean_bbox_area: float
    mean_keypoints_detected: float
    completeness_ratio: float
    timestamp_s: float


@dataclass
class DriftAlert:
    """Alert about detector degradation."""
    timestamp_s: float
    frame_idx: int
    metric: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    severity: str
    description: str


@dataclass
class DriftReport:
    """Full drift analysis report."""
    total_frames: int
    baseline_window: int
    alerts: List[DriftAlert]
    metrics_timeline: List[FrameMetrics]
    baseline_metrics: dict
    current_metrics: dict
    overall_drift_score: float
    summary: str


class DetectorDriftMonitor:
    """
    Monitors detection quality over time and detects drift.

    Uses a rolling baseline (first N frames) to establish "normal" detection quality,
    then compares subsequent frames against this baseline.
    """

    def __init__(self, baseline_frames: int = 30,
                 confidence_drift_threshold: float = 0.15,
                 completeness_drift_threshold: float = 0.20,
                 detection_rate_threshold: float = 0.30,
                 window_size: int = 50):
        """
        Args:
            baseline_frames: Number of initial frames used to establish baseline
            confidence_drift_threshold: Alert when mean confidence drops by this fraction
            completeness_drift_threshold: Alert when keypoint completeness drops by this fraction
            detection_rate_threshold: Alert when detection count drops by this fraction
            window_size: Rolling window size for trend analysis
        """
        self.baseline_frames = baseline_frames
        self.confidence_drift_threshold = confidence_drift_threshold
        self.completeness_drift_threshold = completeness_drift_threshold
        self.detection_rate_threshold = detection_rate_threshold
        self.window_size = window_size

        self._frame_metrics: List[FrameMetrics] = []
        self._baseline_computed = False
        self._baseline: dict = {}
        self._alerts: List[DriftAlert] = []

    def record_frame(self, frame_idx: int, detections: List[dict],
                     timestamp_s: Optional[float] = None) -> Optional[FrameMetrics]:
        """
        Record detection metrics for a frame.

        Args:
            frame_idx: Frame number
            detections: List of detection dicts with keys:
                       'bbox', 'confidence', 'keypoints' (optional)
            timestamp_s: Timestamp in seconds (default: frame_idx / 25)

        Returns:
            FrameMetrics for this frame (or None if insufficient data)
        """
        n_detections = len(detections)

        confidences = [d.get("confidence", 0.0) for d in detections]
        bboxes = [d.get("bbox", [0, 0, 100, 100]) for d in detections]

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        min_conf = float(np.min(confidences)) if confidences else 0.0
        max_conf = float(np.max(confidences)) if confidences else 0.0

        bbox_areas = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                bbox_areas.append(area)
        mean_bbox_area = float(np.mean(bbox_areas)) if bbox_areas else 0.0

        kpt_counts = []
        for d in detections:
            kpts = d.get("keypoints", None)
            if kpts is not None:
                kpt_array = np.array(kpts)
                if kpt_array.ndim >= 2:
                    valid = int(np.sum(~np.isnan(kpt_array).all(axis=-1)))
                    kpt_counts.append(valid)
        mean_kpts = float(np.mean(kpt_counts)) if kpt_counts else 17.0
        completeness = mean_kpts / 17.0

        ts = timestamp_s if timestamp_s is not None else frame_idx / 25.0

        metrics = FrameMetrics(
            frame_idx=frame_idx,
            n_detections=n_detections,
            mean_confidence=round(mean_conf, 4),
            min_confidence=round(min_conf, 4),
            max_confidence=round(max_conf, 4),
            mean_bbox_area=round(mean_bbox_area, 1),
            mean_keypoints_detected=round(mean_kpts, 1),
            completeness_ratio=round(completeness, 4),
            timestamp_s=round(ts, 4),
        )

        self._frame_metrics.append(metrics)

        if len(self._frame_metrics) == self.baseline_frames and not self._baseline_computed:
            self._compute_baseline()

        if self._baseline_computed:
            self._check_drift(metrics)

        return metrics

    def record_from_skeleton(self, frame_idx: int, skeleton_3d: np.ndarray,
                              confidence: Optional[np.ndarray] = None,
                              timestamp_s: Optional[float] = None) -> Optional[FrameMetrics]:
        """
        Record metrics from 3D skeleton data (more common in our pipeline).

        Args:
            skeleton_3d: (N, 3) array of 3D keypoints
            confidence: (N,) array of per-keypoint confidence
        """
        n_kpts = len(skeleton_3d) if skeleton_3d is not None else 0
        n_valid = int(np.sum(~np.isnan(skeleton_3d).all(axis=1))) if n_kpts > 0 else 0
        completeness = n_valid / max(1, n_kpts)

        mean_conf = float(np.mean(confidence)) if confidence is not None and len(confidence) > 0 else completeness

        detections = []
        if n_valid > 0:
            valid_kpts = skeleton_3d[~np.isnan(skeleton_3d).all(axis=1)]
            x_min, y_min = valid_kpts[:, :2].min(axis=0)
            x_max, y_max = valid_kpts[:, :2].max(axis=0)
            detections.append({
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "confidence": mean_conf,
                "keypoints": skeleton_3d,
            })

        return self.record_frame(frame_idx, detections, timestamp_s)

    def _compute_baseline(self):
        """Compute baseline metrics from initial frames."""
        if len(self._frame_metrics) < self.baseline_frames:
            return

        baseline = self._frame_metrics[:self.baseline_frames]
        confidences = [m.mean_confidence for m in baseline]
        completeness = [m.completeness_ratio for m in baseline]
        n_detections = [m.n_detections for m in baseline]
        bbox_areas = [m.mean_bbox_area for m in baseline if m.mean_bbox_area > 0]

        self._baseline = {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "mean_completeness": float(np.mean(completeness)),
            "std_completeness": float(np.std(completeness)),
            "mean_n_detections": float(np.mean(n_detections)),
            "std_n_detections": float(np.std(n_detections)),
            "mean_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else 0,
        }

        self._baseline_computed = True
        logger.info(f"Baseline computed from {self.baseline_frames} frames: "
                     f"conf={self._baseline['mean_confidence']:.3f}, "
                     f"compl={self._baseline['mean_completeness']:.3f}")

    def _check_drift(self, metrics: FrameMetrics):
        """Check current frame against baseline for drift."""
        if not self._baseline_computed:
            return

        baseline = self._baseline

        conf_drift = (baseline["mean_confidence"] - metrics.mean_confidence) / max(0.01, baseline["mean_confidence"])
        if conf_drift > self.confidence_drift_threshold:
            severity = "high" if conf_drift > self.confidence_drift_threshold * 2 else "moderate"
            self._alerts.append(DriftAlert(
                timestamp_s=metrics.timestamp_s,
                frame_idx=metrics.frame_idx,
                metric="confidence",
                current_value=metrics.mean_confidence,
                baseline_value=baseline["mean_confidence"],
                degradation_pct=round(conf_drift * 100, 1),
                severity=severity,
                description=f"Detection confidence dropped {conf_drift:.0%} from baseline "
                           f"({baseline['mean_confidence']:.3f} → {metrics.mean_confidence:.3f})",
            ))

        compl_drift = (baseline["mean_completeness"] - metrics.completeness_ratio) / max(0.01, baseline["mean_completeness"])
        if compl_drift > self.completeness_drift_threshold:
            severity = "high" if compl_drift > self.completeness_drift_threshold * 2 else "moderate"
            self._alerts.append(DriftAlert(
                timestamp_s=metrics.timestamp_s,
                frame_idx=metrics.frame_idx,
                metric="completeness",
                current_value=metrics.completeness_ratio,
                baseline_value=baseline["mean_completeness"],
                degradation_pct=round(compl_drift * 100, 1),
                severity=severity,
                description=f"Keypoint completeness dropped {compl_drift:.0%} from baseline "
                           f"({baseline['mean_completeness']:.3f} → {metrics.completeness_ratio:.3f})",
            ))

        if baseline["mean_n_detections"] > 0:
            det_drift = (baseline["mean_n_detections"] - metrics.n_detections) / baseline["mean_n_detections"]
            if det_drift > self.detection_rate_threshold:
                severity = "high" if det_drift > self.detection_rate_threshold * 2 else "moderate"
                self._alerts.append(DriftAlert(
                    timestamp_s=metrics.timestamp_s,
                    frame_idx=metrics.frame_idx,
                    metric="detection_count",
                    current_value=float(metrics.n_detections),
                    baseline_value=baseline["mean_n_detections"],
                    degradation_pct=round(det_drift * 100, 1),
                    severity=severity,
                    description=f"Detection count dropped {det_drift:.0%} from baseline "
                               f"({baseline['mean_n_detections']:.1f} → {metrics.n_detections})",
                ))

    def get_report(self) -> DriftReport:
        """Generate full drift analysis report."""
        if not self._frame_metrics:
            return DriftReport(
                total_frames=0, baseline_window=self.baseline_frames,
                alerts=[], metrics_timeline=[], baseline_metrics={},
                current_metrics={}, overall_drift_score=0.0,
                summary="No frames recorded.",
            )

        all_confs = [m.mean_confidence for m in self._frame_metrics]
        all_compl = [m.completeness_ratio for m in self._frame_metrics]

        current_window = self._frame_metrics[-self.window_size:]
        current_confs = [m.mean_confidence for m in current_window]
        current_compl = [m.completeness_ratio for m in current_window]

        current_metrics = {
            "mean_confidence": round(float(np.mean(current_confs)), 4),
            "mean_completeness": round(float(np.mean(current_compl)), 4),
            "n_frames": len(current_window),
        }

        high_alerts = [a for a in self._alerts if a.severity == "high"]
        moderate_alerts = [a for a in self._alerts if a.severity == "moderate"]

        drift_score = 0.0
        if self._baseline_computed:
            conf_change = abs(self._baseline["mean_confidence"] - current_metrics["mean_confidence"])
            compl_change = abs(self._baseline["mean_completeness"] - current_metrics["mean_completeness"])
            drift_score = round(conf_change * 50 + compl_change * 50, 3)

        if drift_score < 0.05:
            summary = f"Stable — no significant drift detected (score: {drift_score:.3f})"
        elif drift_score < 0.15:
            summary = f"Minor drift detected (score: {drift_score:.3f}) — {len(moderate_alerts)} moderate alerts"
        elif drift_score < 0.30:
            summary = f"Moderate drift detected (score: {drift_score:.3f}) — {len(high_alerts)} high alerts"
        else:
            summary = f"Significant drift detected (score: {drift_score:.3f}) — consider recalibrating or relighting"

        return DriftReport(
            total_frames=len(self._frame_metrics),
            baseline_window=self.baseline_frames,
            alerts=self._alerts,
            metrics_timeline=self._frame_metrics,
            baseline_metrics=self._baseline.copy() if self._baseline else {},
            current_metrics=current_metrics,
            overall_drift_score=drift_score,
            summary=summary,
        )

    def get_confidence_timeline(self) -> Tuple[List[float], List[float]]:
        """Return (timestamps, confidence values) for plotting."""
        timestamps = [m.timestamp_s for m in self._frame_metrics]
        confidences = [m.mean_confidence for m in self._frame_metrics]
        return timestamps, confidences

    def get_completeness_timeline(self) -> Tuple[List[float], List[float]]:
        """Return (timestamps, completeness values) for plotting."""
        timestamps = [m.timestamp_s for m in self._frame_metrics]
        completeness = [m.completeness_ratio for m in self._frame_metrics]
        return timestamps, completeness

    def reset(self):
        """Reset the monitor for a new session."""
        self._frame_metrics.clear()
        self._baseline.clear()
        self._baseline_computed = False
        self._alerts.clear()


def format_drift_report(report: DriftReport) -> str:
    """Format a human-readable drift report."""
    lines = [
        "=== Detector Drift Report ===",
        f"Total frames analyzed: {report.total_frames}",
        f"Baseline window: {report.baseline_window} frames",
        f"Drift score: {report.overall_drift_score:.3f}",
        f"Summary: {report.summary}",
        "",
    ]

    if report.baseline_metrics:
        lines.append("Baseline metrics:")
        for key, val in report.baseline_metrics.items():
            lines.append(f"  {key}: {val:.4f}")
        lines.append("")

    if report.current_metrics:
        lines.append("Current metrics (last window):")
        for key, val in report.current_metrics.items():
            lines.append(f"  {key}: {val}")
        lines.append("")

    if report.alerts:
        lines.append(f"Alerts ({len(report.alerts)} total):")
        for alert in report.alerts[-20:]:
            lines.append(
                f"  [{alert.severity.upper()}] frame {alert.frame_idx} "
                f"({alert.timestamp_s:.1f}s): {alert.description}"
            )
    else:
        lines.append("No drift alerts.")

    return "\n".join(lines)
