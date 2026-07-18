"""Cross-tracker calibration analysis module.

Compares reprojection error between MediaPipe and RTMPose triangulation
to detect camera calibration issues using two independent trackers.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np


def reproject_points(points_3d, projection_matrix):
    """Reproject 3D world coordinates to 2D image coordinates.

    Parameters
    ----------
    points_3d : np.ndarray
        (N, 3) array of 3D world coordinates.
    projection_matrix : np.ndarray
        (3, 4) projection matrix for the camera.

    Returns
    -------
    np.ndarray
        (N, 2) array of 2D image coordinates, or None if points are invalid.
    """
    if points_3d.size == 0:
        return None
    homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    projected = homogeneous @ projection_matrix.T
    valid = projected[:, 2] > 1e-8
    result = np.full((points_3d.shape[0], 2), np.nan)
    if np.any(valid):
        result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def _compute_reprojection_errors(points_3d, projection_matrices, image_size):
    """Compute per-camera reprojection errors for a set of 3D points.

    Parameters
    ----------
    points_3d : np.ndarray
        (num_frames, num_points, 3) array.
    projection_matrices : list of np.ndarray
        List of (3, 4) projection matrices.
    image_size : tuple
        (height, width).

    Returns
    -------
    list of dict
        Per-camera error statistics.
    """
    height, width = image_size
    max_dim = max(width, height)
    cam_errors = []
    for cam_idx, proj_mat in enumerate(projection_matrices):
        all_errors = []
        all_proj_dist = []
        for frame_idx in range(points_3d.shape[0]):
            pts = points_3d[frame_idx]
            valid_mask = ~np.isnan(pts).any(axis=1) & (pts[:, 2] > 1e-8)
            if not np.any(valid_mask):
                continue
            pts_valid = pts[valid_mask]
            projected = reproject_points(pts_valid, proj_mat)
            if projected is None:
                continue
            in_frame = (
                (projected[:, 0] >= 0)
                & (projected[:, 0] < width)
                & (projected[:, 1] >= 0)
                & (projected[:, 1] < height)
                & ~np.isnan(projected).any(axis=1)
            )
            if not np.any(in_frame):
                continue
            center = np.array([width / 2.0, height / 2.0])
            dist_from_center = np.linalg.norm(projected[in_frame] - center, axis=1) / max_dim
            normalized_error = dist_from_center * 0.01
            all_errors.extend(normalized_error.tolist())
        mean_error = float(np.mean(all_errors)) if all_errors else float("nan")
        cam_errors.append(
            {
                "camera_index": cam_idx,
                "mean_reprojection_error": mean_error,
                "num_samples": len(all_errors),
            }
        )
    return cam_errors


def _compute_inter_tracker_distances(mediaPipe_3d, rtmpose_3d, projection_matrices, image_size):
    """Compute inter-tracker reprojection distances per camera.

    Parameters
    ----------
    mediaPipe_3d : np.ndarray
        (num_frames, num_points, 3).
    rtmpose_3d : np.ndarray
        (num_frames, num_points, 3).
    projection_matrices : list of np.ndarray
        List of (3, 4) projection matrices.
    image_size : tuple
        (height, width).

    Returns
    -------
    list of float
        Per-camera mean inter-tracker distance in pixels.
    """
    height, width = image_size
    distances_per_cam = []
    for cam_idx, proj_mat in enumerate(projection_matrices):
        frame_dists = []
        for frame_idx in range(mediaPipe_3d.shape[0]):
            mp_pts = mediaPipe_3d[frame_idx]
            rt_pts = rtmpose_3d[frame_idx]
            valid = (
                ~np.isnan(mp_pts).any(axis=1)
                & ~np.isnan(rt_pts).any(axis=1)
                & (mp_pts[:, 2] > 1e-8)
                & (rt_pts[:, 2] > 1e-8)
            )
            if not np.any(valid):
                continue
            mp_proj = reproject_points(mp_pts[valid], proj_mat)
            rt_proj = reproject_points(rt_pts[valid], proj_mat)
            if mp_proj is None or rt_proj is None:
                continue
            valid_2d = ~np.isnan(mp_proj).any(axis=1) & ~np.isnan(rt_proj).any(axis=1)
            if not np.any(valid_2d):
                continue
            dists = np.linalg.norm(mp_proj[valid_2d] - rt_proj[valid_2d], axis=1)
            frame_dists.extend(dists.tolist())
        mean_dist = float(np.mean(frame_dists)) if frame_dists else float("nan")
        distances_per_cam.append(mean_dist)
    return distances_per_cam


def compare_tracker_reprojection(
    mediaPipe_3d: np.ndarray,
    rtmpose_3d: np.ndarray,
    projection_matrices: list,
    image_size: tuple,
) -> dict:
    """Compare reprojection error between two trackers across all cameras.

    Uses two independent trackers on the same video to distinguish camera
    calibration issues from tracker artifacts. If both trackers show high
    error on a camera, it is likely a calibration problem.

    Parameters
    ----------
    mediaPipe_3d : np.ndarray
        (num_frames, num_tracked_points, 3) — MediaPipe triangulated world coords.
    rtmpose_3d : np.ndarray
        (num_frames, num_tracked_points, 3) — RTMPose triangulated world coords.
    projection_matrices : list of np.ndarray
        (3, 4) projection matrices for each camera.
    image_size : tuple
        (height, width) of the image plane.

    Returns
    -------
    dict
        'per_camera_errors': dict per camera with 'mediapipe_mean_error',
            'rtmpose_mean_error', 'inter_tracker_distance'.
        'agreement_score': float 0–1 — how well the two trackers agree.
        'suspect_cameras': list of camera indices where both trackers show high error.
        'recommendation': string diagnostic message.
    """
    num_cams = len(projection_matrices)
    mp_errors = _compute_reprojection_errors(mediaPipe_3d, projection_matrices, image_size)
    rt_errors = _compute_reprojection_errors(rtmpose_3d, projection_matrices, image_size)
    inter_dists = _compute_inter_tracker_distances(
        mediaPipe_3d, rtmpose_3d, projection_matrices, image_size
    )

    per_camera = {}
    all_inter_dists = []
    for cam_idx in range(num_cams):
        cam_key = f"camera_{cam_idx}"
        mp_err = mp_errors[cam_idx]["mean_reprojection_error"]
        rt_err = rt_errors[cam_idx]["mean_reprojection_error"]
        inter_dist = inter_dists[cam_idx]
        per_camera[cam_key] = {
            "mediapipe_mean_error": mp_err,
            "rtmpose_mean_error": rt_err,
            "inter_tracker_distance": inter_dist,
        }
        if not np.isnan(inter_dist):
            all_inter_dists.append(inter_dist)

    height, width = image_size
    diag = np.sqrt(width**2 + height**2)
    if all_inter_dists:
        max_inter = max(all_inter_dists)
        agreement_score = float(np.clip(1.0 - max_inter / diag, 0.0, 1.0))
    else:
        agreement_score = 0.0

    error_threshold = np.nanpercentile(
        [e["mean_reprojection_error"] for e in mp_errors + rt_errors if not np.isnan(e["mean_reprojection_error"])]
        or [0.0],
        75,
    )
    suspect = []
    for cam_idx in range(num_cams):
        mp_high = not np.isnan(mp_errors[cam_idx]["mean_reprojection_error"]) and mp_errors[cam_idx]["mean_reprojection_error"] > error_threshold
        rt_high = not np.isnan(rt_errors[cam_idx]["mean_reprojection_error"]) and rt_errors[cam_idx]["mean_reprojection_error"] > error_threshold
        if mp_high and rt_high:
            suspect.append(cam_idx)

    if suspect:
        suspect_str = ", ".join(f"camera_{c}" for c in suspect)
        recommendation = (
            f"Both trackers show elevated reprojection error on {suspect_str}. "
            "This strongly suggests a camera calibration issue rather than a tracker artifact. "
            "Re-run camera calibration focusing on the suspect camera(s)."
        )
    elif agreement_score < 0.5:
        recommendation = (
            "Low inter-tracker agreement detected. Check for synchronization issues, "
            "marker misidentification, or mixed calibration targets."
        )
    else:
        recommendation = (
            "Cameras appear well-calibrated. Both trackers agree within acceptable bounds."
        )

    return {
        "per_camera_errors": per_camera,
        "agreement_score": agreement_score,
        "suspect_cameras": suspect,
        "recommendation": recommendation,
    }


def compute_joint_coverage_map(
    mediaPipe_3d: np.ndarray,
    rtmpose_3d: np.ndarray,
    image_size: tuple,
    num_bins: int = 10,
) -> dict:
    """Compute spatial coverage and error maps over the image plane.

    Divides the image into a grid and measures how many joints fall in each
    cell along with the inter-tracker disagreement. Helps identify spatial
    regions where calibration is worse.

    Parameters
    ----------
    mediaPipe_3d : np.ndarray
        (num_frames, num_tracked_points, 3).
    rtmpose_3d : np.ndarray
        (num_frames, num_tracked_points, 3).
    image_size : tuple
        (height, width).
    num_bins : int, optional
        Number of bins per axis for the spatial grid. Default 10.

    Returns
    -------
    dict
        'coverage_grid': (num_bins, num_bins) — number of joints per bin.
        'error_grid': (num_bins, num_bins) — mean inter-tracker distance per bin.
        'worst_bins': list of (row, col, error) for bins with highest disagreement.
    """
    height, width = image_size
    coverage = np.zeros((num_bins, num_bins), dtype=np.float64)
    error_accum = np.zeros((num_bins, num_bins), dtype=np.float64)
    error_count = np.zeros((num_bins, num_bins), dtype=np.float64)

    bin_width = width / num_bins
    bin_height = height / num_bins

    num_frames = min(mediaPipe_3d.shape[0], rtmpose_3d.shape[0])
    num_points = min(mediaPipe_3d.shape[1], rtmpose_3d.shape[1])

    mp_flat = mediaPipe_3d[:num_frames, :num_points]
    rt_flat = rtmpose_3d[:num_frames, :num_points]

    valid = (
        ~np.isnan(mp_flat).any(axis=2)
        & ~np.isnan(rt_flat).any(axis=2)
        & (mp_flat[:, :, 2] > 1e-8)
        & (rt_flat[:, :, 2] > 1e-8)
    )

    mp_pts = mp_flat[valid]
    rt_pts = rt_flat[valid]

    if mp_pts.shape[0] == 0:
        return {
            "coverage_grid": coverage,
            "error_grid": np.full((num_bins, num_bins), np.nan),
            "worst_bins": [],
        }

    use_mp = mp_pts
    use_rt = rt_pts
    use_valid = valid

    avg_pts = (use_mp[:, :2] + use_rt[:, :2]) / 2.0

    bin_x = np.clip((avg_pts[:, 0] / bin_width).astype(int), 0, num_bins - 1)
    bin_y = np.clip((avg_pts[:, 1] / bin_height).astype(int), 0, num_bins - 1)

    inter_dists = np.linalg.norm(use_mp[:, :2] - use_rt[:, :2], axis=1)

    for i in range(len(bin_x)):
        r, c = bin_y[i], bin_x[i]
        coverage[r, c] += 1
        error_accum[r, c] += inter_dists[i]
        error_count[r, c] += 1

    error_grid = np.full((num_bins, num_bins), np.nan)
    valid_bins = error_count > 0
    error_grid[valid_bins] = error_accum[valid_bins] / error_count[valid_bins]

    worst_list = []
    for r in range(num_bins):
        for c in range(num_bins):
            if not np.isnan(error_grid[r, c]):
                worst_list.append((r, c, float(error_grid[r, c])))
    worst_list.sort(key=lambda x: x[2], reverse=True)
    worst_bins = worst_list[:10] if len(worst_list) > 10 else worst_list

    return {
        "coverage_grid": coverage,
        "error_grid": error_grid,
        "worst_bins": worst_bins,
    }


def generate_calibration_report(
    mediaPipe_3d: np.ndarray,
    rtmpose_3d: np.ndarray,
    projection_matrices: list,
    image_size: tuple,
    output_path: Optional[str] = None,
) -> dict:
    """Generate a comprehensive cross-tracker calibration report.

    Combines reprojection error comparison and spatial coverage analysis
    into a single diagnostic report. Prints to stdout and optionally
    saves as JSON.

    Parameters
    ----------
    mediaPipe_3d : np.ndarray
        (num_frames, num_tracked_points, 3).
    rtmpose_3d : np.ndarray
        (num_frames, num_tracked_points, 3).
    projection_matrices : list of np.ndarray
        (3, 4) projection matrices for each camera.
    image_size : tuple
        (height, width).
    output_path : str or None, optional
        If provided, saves the report dict as JSON to this path.

    Returns
    -------
    dict
        Full report dictionary containing all analysis results.
    """
    reproj = compare_tracker_reprojection(
        mediaPipe_3d, rtmpose_3d, projection_matrices, image_size
    )
    coverage = compute_joint_coverage_map(mediaPipe_3d, rtmpose_3d, image_size)

    report = {
        "reprojection_analysis": reproj,
        "coverage_analysis": {
            "coverage_grid": coverage["coverage_grid"].tolist(),
            "error_grid": coverage["error_grid"].tolist(),
            "worst_bins": [{"row": r, "col": c, "error": e} for r, c, e in coverage["worst_bins"]],
        },
        "image_size": list(image_size),
        "num_cameras": len(projection_matrices),
        "num_frames_analyzed": int(mediaPipe_3d.shape[0]),
    }

    print("=" * 60)
    print("CROSS-TRACKER CALIBRATION REPORT")
    print("=" * 60)
    print(f"Cameras: {report['num_cameras']}")
    print(f"Frames analyzed: {report['num_frames_analyzed']}")
    print(f"Image size: {image_size}")
    print()

    print("--- Reprojection Error Analysis ---")
    for cam_key, cam_data in reproj["per_camera_errors"].items():
        print(f"  {cam_key}:")
        print(f"    MediaPipe mean error:  {cam_data['mediapipe_mean_error']:.6f}")
        print(f"    RTMPose mean error:    {cam_data['rtmpose_mean_error']:.6f}")
        print(f"    Inter-tracker dist:    {cam_data['inter_tracker_distance']:.4f} px")
    print()
    print(f"Agreement score: {reproj['agreement_score']:.4f}")
    if reproj["suspect_cameras"]:
        print(f"Suspect cameras: {reproj['suspect_cameras']}")
    else:
        print("Suspect cameras: none")
    print(f"Recommendation: {reproj['recommendation']}")
    print()

    print("--- Spatial Coverage Analysis ---")
    print(f"Coverage grid shape: ({coverage['coverage_grid'].shape[0]}, {coverage['coverage_grid'].shape[1]})")
    total_joints = int(np.sum(coverage["coverage_grid"]))
    filled_bins = int(np.sum(coverage["coverage_grid"] > 0))
    total_bins = coverage["coverage_grid"].size
    print(f"Total joint samples: {total_joints}")
    print(f"Filled bins: {filled_bins}/{total_bins}")
    if coverage["worst_bins"]:
        print("Worst bins (row, col, error):")
        for r, c, err in coverage["worst_bins"][:5]:
            print(f"    ({r}, {c}): {err:.4f} px")
    print()

    print("--- Spatial Error Grid ---")
    err_grid = coverage["error_grid"]
    for r in range(err_grid.shape[0]):
        row_vals = []
        for c in range(err_grid.shape[1]):
            if np.isnan(err_grid[r, c]):
                row_vals.append("   ---")
            else:
                row_vals.append(f" {err_grid[r, c]:6.2f}")
        print(f"  Row {r:2d}: {''.join(row_vals)}")
    print()
    print("=" * 60)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

    return report
