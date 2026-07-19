"""
Data loader for FreeMoCap recording output.

Reads 2D/3D numpy arrays produced by FreeMoCap's processing pipeline
and provides bounding boxes and metrics for the GUI screens.
"""
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_2D_FILE = "2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy"
DATA_3D_FILE = "3dData_numFrames_numTrackedPoints_spatialXYZ.npy"
REPROJ_ERROR_FILE = "3dData_numFrames_numTrackedPoints_reprojectionError.npy"

OUTPUT_DATA_DIR = "output_data"


def find_recording_data(recording_path: str) -> dict:
    """Find all output data files in a recording folder.

    Returns:
        dict with keys '2d', '3d', 'reprojection_error' -> Path or None
    """
    base = Path(recording_path)

    paths = {}

    for candidate in [
        base / OUTPUT_DATA_DIR / DATA_2D_FILE,
        base / DATA_2D_FILE,
    ]:
        if candidate.exists():
            paths["2d"] = candidate
            break

    for candidate in [
        base / OUTPUT_DATA_DIR / DATA_3D_FILE,
    ]:
        if candidate.exists():
            paths["3d"] = candidate
            break

    for candidate in [
        base / OUTPUT_DATA_DIR / REPROJ_ERROR_FILE,
    ]:
        if candidate.exists():
            paths["reprojection_error"] = candidate
            break

    return paths


def load_2d_data(recording_path: str) -> np.ndarray:
    """Load 2D detection data.

    Returns:
        array of shape (num_cameras, num_frames, num_keypoints, 2) or None
    """
    paths = find_recording_data(recording_path)
    if "2d" not in paths:
        logger.warning(f"No 2D data found in {recording_path}")
        return None
    try:
        data = np.load(str(paths["2d"]))
        logger.info(f"Loaded 2D data: shape={data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load 2D data: {e}")
        return None


def load_3d_data(recording_path: str) -> np.ndarray:
    """Load 3D triangulation data.

    Returns:
        array of shape (num_frames, num_keypoints, 3) or None
    """
    paths = find_recording_data(recording_path)
    if "3d" not in paths:
        logger.warning(f"No 3D data found in {recording_path}")
        return None
    try:
        data = np.load(str(paths["3d"]))
        logger.info(f"Loaded 3D data: shape={data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load 3D data: {e}")
        return None


def load_reprojection_error(recording_path: str) -> np.ndarray:
    """Load reprojection error data.

    Returns:
        array of shape (num_frames, num_keypoints) or None
    """
    paths = find_recording_data(recording_path)
    if "reprojection_error" not in paths:
        return None
    try:
        return np.load(str(paths["reprojection_error"]))
    except Exception as e:
        logger.error(f"Failed to load reprojection error: {e}")
        return None


def extract_bboxes_for_frame(two_d_data: np.ndarray, frame_idx: int,
                             scale_x: float = 1.0, scale_y: float = 1.0,
                             pad: int = 15) -> dict:
    """Extract bounding boxes from 2D keypoints for a single frame.

    Args:
        two_d_data: shape (num_cameras, num_frames, num_keypoints, 2)
        frame_idx: which frame
        scale_x, scale_y: scale factors (e.g. 0.5 if skellycam downscaled)
        pad: padding in pixels around bbox

    Returns:
        dict: {camera_index: (x1, y1, x2, y2)}
    """
    if two_d_data is None or frame_idx >= two_d_data.shape[1]:
        return {}

    bboxes = {}
    num_cameras = two_d_data.shape[0]

    for cam_idx in range(num_cameras):
        kps = two_d_data[cam_idx, frame_idx]  # (num_keypoints, 2)

        valid = np.isfinite(kps).all(axis=1) & (kps != 0).any(axis=1)
        if not valid.any():
            continue

        valid_kps = kps[valid]
        x1 = valid_kps[:, 0].min() * scale_x - pad
        y1 = valid_kps[:, 1].min() * scale_y - pad
        x2 = valid_kps[:, 0].max() * scale_x + pad
        y2 = valid_kps[:, 1].max() * scale_y + pad

        bboxes[cam_idx] = (max(0, x1), max(0, y1), x2, y2)

    return bboxes


def compute_frame_confidence(three_d_data: np.ndarray) -> np.ndarray:
    """Compute per-frame confidence score (0..1) based on keypoint visibility.

    Returns:
        array of shape (num_frames,) with values 0..1
    """
    if three_d_data is None or three_d_data.shape[0] == 0:
        return np.array([])

    num_frames = three_d_data.shape[0]
    num_kps = three_d_data.shape[1]
    confidences = np.zeros(num_frames)

    for i in range(num_frames):
        frame_data = three_d_data[i]  # (num_keypoints, 3)
        valid = np.isfinite(frame_data).all(axis=1) & (frame_data != 0).any(axis=1)
        confidences[i] = valid.sum() / num_kps

    return confidences


def compute_reprojection_quality(reproj_error: np.ndarray) -> np.ndarray:
    """Compute per-frame quality from reprojection error (lower is better).

    Returns:
        array of shape (num_frames,) with values 0..1 (1 = best)
    """
    if reproj_error is None or reproj_error.shape[0] == 0:
        return np.array([])

    mean_error = np.nanmean(reproj_error, axis=1)
    max_acceptable = 20.0
    quality = np.clip(1.0 - mean_error / max_acceptable, 0, 1)
    return quality


def compute_quality_metrics(three_d_data: np.ndarray,
                            reproj_error: np.ndarray = None) -> dict:
    """Compute overall recording quality metrics.

    Returns:
        dict with keys:
            total_frames, confident_frames, low_confidence_frames,
            tracking_gaps, quality_pct
    """
    if three_d_data is None or three_d_data.shape[0] == 0:
        return {
            "total_frames": 0,
            "confident_frames": 0,
            "low_confidence_frames": 0,
            "tracking_gaps": 0,
            "quality_pct": 0,
        }

    confidences = compute_frame_confidence(three_d_data)

    confident = int((confidences > 0.8).sum())
    low_conf = int((confidences < 0.5).sum())

    gaps = 0
    in_gap = False
    for c in confidences:
        if c < 0.3:
            if not in_gap:
                gaps += 1
                in_gap = True
        else:
            in_gap = False

    total = len(confidences)
    quality_pct = int(confident / total * 100) if total > 0 else 0

    return {
        "total_frames": total,
        "confident_frames": confident,
        "low_confidence_frames": low_conf,
        "tracking_gaps": gaps,
        "quality_pct": quality_pct,
    }


def detect_tracking_issues(three_d_data: np.ndarray,
                           reproj_error: np.ndarray = None) -> list:
    """Detect specific tracking issues for the diagnostics screen.

    Returns:
        list of dicts: {frame, timestamp, description, severity, type}
    """
    issues = []
    if three_d_data is None or three_d_data.shape[0] == 0:
        return issues

    confidences = compute_frame_confidence(three_d_data)

    in_gap = False
    gap_start = 0
    for i, c in enumerate(confidences):
        if c < 0.3:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                duration = i - gap_start
                issues.append({
                    "frame": gap_start,
                    "timestamp": f"{gap_start / 30:.1f}s",
                    "description": f"Tracking gap — {duration} frames with low visibility",
                    "severity": "warning",
                    "type": "tracking_gap",
                })
                in_gap = False

    if in_gap:
        duration = len(confidences) - gap_start
        issues.append({
            "frame": gap_start,
            "timestamp": f"{gap_start / 30:.1f}s",
            "description": f"Tracking gap — {duration} frames (to end)",
            "severity": "warning",
            "type": "tracking_gap",
        })

    for i, c in enumerate(confidences):
        if c < 0.2:
            issues.append({
                "frame": i,
                "timestamp": f"{i / 30:.1f}s",
                "description": f"Very low confidence ({c:.0%})",
                "severity": "error",
                "type": "low_confidence",
            })

    if reproj_error is not None and reproj_error.shape[0] > 0:
        mean_errors = np.nanmean(reproj_error, axis=1)
        high_error_frames = np.where(mean_errors > 15.0)[0]
        for i in high_error_frames[:5]:
            issues.append({
                "frame": int(i),
                "timestamp": f"{i / 30:.1f}s",
                "description": f"High reprojection error ({mean_errors[i]:.1f}mm)",
                "severity": "warning",
                "type": "reprojection_error",
            })

    issues.sort(key=lambda x: x["frame"])
    return issues


def export_to_csv(recording_path: str, output_path: str = None) -> str:
    """Export 3D skeleton data to CSV.

    Args:
        recording_path: path to FreeMoCap recording folder
        output_path: where to save CSV (default: recording_folder/export.csv)

    Returns:
        path to saved CSV file
    """
    import csv

    data_3d = load_3d_data(recording_path)
    if data_3d is None:
        raise FileNotFoundError(f"No 3D data found in {recording_path}")

    if output_path is None:
        output_path = str(Path(recording_path) / "export_3d_skeleton.csv")

    num_frames, num_kps, _ = data_3d.shape
    headers = ["frame"]
    for kp in range(num_kps):
        headers.extend([f"kp{kp}_x", f"kp{kp}_y", f"kp{kp}_z"])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for frame_idx in range(num_frames):
            row = [frame_idx]
            for kp in range(num_kps):
                x, y, z = data_3d[frame_idx, kp]
                row.extend([
                    f"{x:.4f}" if np.isfinite(x) else "",
                    f"{y:.4f}" if np.isfinite(y) else "",
                    f"{z:.4f}" if np.isfinite(z) else "",
                ])
            writer.writerow(row)

    logger.info(f"Exported {num_frames} frames to {output_path}")
    return output_path
