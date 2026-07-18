import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Dict, Any
import os
import pickle
import glob


def dlt_triangulate_single_point(
    points_2d: NDArray[np.floating],
    projection_matrices: List[NDArray[np.floating]],
    weights: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.float64]:
    """
    Triangulate a single 3D point from multiple 2D observations using DLT.

    Constructs a system of linear equations from each camera's 2D observation
    and solves via SVD. Each camera provides two rows in the equation matrix:
    one from the u-coordinate and one from the v-coordinate.

    Args:
        points_2d: (numCameras, 2) array of 2D pixel coordinates.
        projection_matrices: List of numCameras, each a (3, 4) projection matrix.
        weights: (numCameras,) optional per-camera weights. If provided, rows
            in the DLT system are scaled by the corresponding weight.

    Returns:
        (3,) array of triangulated world coordinates. Returns NaN vector if
        the system is degenerate or the homogeneous coordinate is near zero.
    """
    num_cams = len(projection_matrices)
    rows = []

    for cam_idx in range(num_cams):
        u = float(points_2d[cam_idx, 0])
        v = float(points_2d[cam_idx, 1])
        P = projection_matrices[cam_idx].astype(np.float64)

        w = 1.0 if weights is None else float(weights[cam_idx])

        row_u = w * (u * P[2] - P[0])
        row_v = w * (v * P[2] - P[1])
        rows.append(row_u)
        rows.append(row_v)

    A = np.vstack(rows)

    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    X_homog = Vt[-1]

    if abs(X_homog[3]) < 1e-10:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    return (X_homog[:3] / X_homog[3]).astype(np.float64)


def reproject_to_2d(
    points_3d: NDArray[np.floating],
    projection_matrix: NDArray[np.floating],
) -> NDArray[np.float64]:
    """
    Project 3D world coordinates back to 2D pixel coordinates.

    Multiplies each 3D point (in homogeneous coordinates) by the projection
    matrix, then performs perspective division.

    Args:
        points_3d: (N, 3) array of 3D world coordinates.
        projection_matrix: (3, 4) camera projection matrix.

    Returns:
        (N, 2) array of 2D pixel coordinates.
    """
    P = projection_matrix.astype(np.float64)
    pts = points_3d.astype(np.float64)

    N = pts.shape[0]
    pts_homog = np.ones((N, 4), dtype=np.float64)
    pts_homog[:, :3] = pts

    projected = (P @ pts_homog.T).T

    z = projected[:, 2:3]
    z = np.where(np.abs(z) < 1e-10, 1e-10, z)

    return (projected[:, :2] / z).astype(np.float64)


def compute_reprojection_error(
    points_3d: NDArray[np.floating],
    points_2d_observed: List[NDArray[np.floating]],
    projection_matrices: List[NDArray[np.floating]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute reprojection error for triangulated 3D points.

    Reprojects each 3D point to 2D for every camera and computes the
    Euclidean distance between the reprojected and observed 2D positions.

    Args:
        points_3d: (numFrames, numTrackedPoints, 3) triangulated world coords.
        points_2d_observed: List of numCameras, each a (numFrames, numTrackedPoints, 2)
            array of observed 2D pixel coordinates.
        projection_matrices: List of numCameras, each a (3, 4) projection matrix.

    Returns:
        Tuple of:
            per_camera_mean_error: (numCameras,) mean reprojection error per camera.
            per_point_errors: (numFrames, numTrackedPoints) mean reprojection error
                across all cameras for each point.
    """
    num_frames = points_3d.shape[0]
    num_points = points_3d.shape[1]
    num_cams = len(projection_matrices)

    all_errors = np.zeros((num_cams, num_frames, num_points), dtype=np.float64)

    for cam_idx in range(num_cams):
        P = projection_matrices[cam_idx].astype(np.float64)

        for f in range(num_frames):
            reprojected = reproject_to_2d(points_3d[f], P)
            observed = points_2d_observed[cam_idx][f].astype(np.float64)
            diff = reprojected - observed
            all_errors[cam_idx, f] = np.sqrt(np.sum(diff ** 2, axis=1))

    per_camera_mean = np.mean(all_errors, axis=(1, 2))
    per_point = np.mean(all_errors, axis=0)

    return per_camera_mean.astype(np.float64), per_point.astype(np.float64)


def triangulate_rtmpose(
    rtmpose_per_camera: List[NDArray[np.floating]],
    camera_matrices: List[NDArray[np.floating]],
    projection_matrices: List[NDArray[np.floating]],
    min_cams: int = 2,
    confidence_scores: Optional[List[NDArray[np.floating]]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Triangulate RTMPose 2D detections into 3D world coordinates using DLT.

    For each frame and each tracked point, collects 2D observations from
    cameras that have valid (non-NaN) detections, builds the DLT system,
    and solves via SVD. Optionally weights observations by detector confidence.

    Args:
        rtmpose_per_camera: List of numCameras, each a (numFrames, numTrackedPoints, 2)
            array of 2D pixel coordinates from RTMPose.
        camera_matrices: List of numCameras, each a (3, 3) intrinsic matrix.
        projection_matrices: List of numCameras, each a (3, 4) projection matrix.
        min_cams: Minimum number of cameras required for triangulation (default 2).
        confidence_scores: Optional list of numCameras, each a (numFrames, numTrackedPoints)
            array of confidence scores per detection.

    Returns:
        Tuple of:
            world_coords: (numFrames, numTrackedPoints, 3) triangulated world coordinates.
            reprojection_errors: (numFrames, numTrackedPoints) per-point reprojection error.
    """
    num_cams = len(rtmpose_per_camera)
    num_frames = rtmpose_per_camera[0].shape[0]
    num_points = rtmpose_per_camera[0].shape[1]

    world_coords = np.full((num_frames, num_points, 3), np.nan, dtype=np.float64)
    reprojection_errors = np.full((num_frames, num_points), np.nan, dtype=np.float64)

    for f in range(num_frames):
        for k in range(num_points):
            valid_cam_indices = []
            pts_2d = []
            cam_weights = []

            for cam_idx in range(num_cams):
                pt = rtmpose_per_camera[cam_idx][f, k]
                if not np.any(np.isnan(pt)):
                    valid_cam_indices.append(cam_idx)
                    pts_2d.append(pt)
                    if confidence_scores is not None:
                        cam_weights.append(float(confidence_scores[cam_idx][f, k]))
                    else:
                        cam_weights.append(1.0)

            if len(valid_cam_indices) < min_cams:
                continue

            pts_2d_arr = np.array(pts_2d, dtype=np.float64)
            weights_arr = np.array(cam_weights, dtype=np.float64)
            valid_proj = [projection_matrices[i] for i in valid_cam_indices]

            pt_3d = dlt_triangulate_single_point(pts_2d_arr, valid_proj, weights_arr)

            if not np.any(np.isnan(pt_3d)):
                world_coords[f, k] = pt_3d

                reproj = reproject_to_2d(pt_3d.reshape(1, 3), projection_matrices[valid_cam_indices[0]])
                observed = pts_2d_arr[0:1]
                diff = reproj - observed
                reprojection_errors[f, k] = np.sqrt(np.sum(diff ** 2))

    return world_coords, reprojection_errors


def triangulate_with_distortion_weighting(
    rtmpose_per_camera: List[NDArray[np.floating]],
    camera_matrices: List[NDArray[np.floating]],
    projection_matrices: List[NDArray[np.floating]],
    image_sizes: List[Tuple[int, int]],
    power: float = 2.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Triangulate RTMPose detections with distortion-based camera weighting.

    Same as triangulate_rtmpose but weights each camera's contribution by
    distance from image center. Wide-angle lens edges have worse calibration
    quality, so detections near frame borders receive lower weight.

    Args:
        rtmpose_per_camera: List of numCameras, each a (numFrames, numTrackedPoints, 2)
            array of 2D pixel coordinates.
        camera_matrices: List of numCameras, each a (3, 3) intrinsic matrix.
        projection_matrices: List of numCameras, each a (3, 4) projection matrix.
        image_sizes: List of numCameras, each a (height, width) tuple.
        power: Exponent controlling edge penalty steepness (default 2.0).

    Returns:
        Tuple of:
            world_coords: (numFrames, numTrackedPoints, 3) triangulated world coordinates.
            reprojection_errors: (numFrames, numTrackedPoints) per-point reprojection error.
    """
    from distortion_weighting import compute_distortion_weights

    num_cams = len(rtmpose_per_camera)
    num_frames = rtmpose_per_camera[0].shape[0]
    num_points = rtmpose_per_camera[0].shape[1]

    world_coords = np.full((num_frames, num_points, 3), np.nan, dtype=np.float64)
    reprojection_errors = np.full((num_frames, num_points), np.nan, dtype=np.float64)

    distortion_weights = []
    for cam_idx in range(num_cams):
        w = compute_distortion_weights(
            rtmpose_per_camera[cam_idx].astype(np.float64),
            image_sizes[cam_idx],
            power=power,
        )
        distortion_weights.append(w)

    for f in range(num_frames):
        for k in range(num_points):
            valid_cam_indices = []
            pts_2d = []
            cam_weights = []

            for cam_idx in range(num_cams):
                pt = rtmpose_per_camera[cam_idx][f, k]
                if not np.any(np.isnan(pt)):
                    valid_cam_indices.append(cam_idx)
                    pts_2d.append(pt)
                    cam_weights.append(float(distortion_weights[cam_idx][f, k]))

            if len(valid_cam_indices) < 2:
                continue

            pts_2d_arr = np.array(pts_2d, dtype=np.float64)
            weights_arr = np.array(cam_weights, dtype=np.float64)
            valid_proj = [projection_matrices[i] for i in valid_cam_indices]

            pt_3d = dlt_triangulate_single_point(pts_2d_arr, valid_proj, weights_arr)

            if not np.any(np.isnan(pt_3d)):
                world_coords[f, k] = pt_3d

                reproj = reproject_to_2d(pt_3d.reshape(1, 3), projection_matrices[valid_cam_indices[0]])
                observed = pts_2d_arr[0:1]
                diff = reproj - observed
                reprojection_errors[f, k] = np.sqrt(np.sum(diff ** 2))

    return world_coords, reprojection_errors


def load_calibration_from_freemocap(
    recording_folder: str,
) -> Optional[Dict[str, Any]]:
    """
    Load camera calibration matrices from a FreeMoCap recording folder.

    Searches for calibration data in common FreeMoCap output formats:
    camera_calibration.toml, calibration.pkl, calibration_data.npy, and
    other known patterns.

    Args:
        recording_folder: Path to the FreeMoCap recording session folder.

    Returns:
        Dictionary with keys:
            'camera_matrices': list of (3, 3) intrinsic matrices, or None.
            'projection_matrices': list of (3, 4) projection matrices, or None.
            'camera_names': list of camera name strings, or None.
            'image_sizes': list of (height, width) tuples, or None.
        Returns None if no calibration data is found.
    """
    if not os.path.isdir(recording_folder):
        return None

    toml_path = os.path.join(recording_folder, "camera_calibration.toml")
    if os.path.exists(toml_path):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        camera_matrices = []
        projection_matrices = []
        camera_names = []
        image_sizes = []

        if "cameras" in data:
            for cam_name, cam_data in data["cameras"].items():
                camera_names.append(cam_name)
                if "camera_matrix" in cam_data:
                    camera_matrices.append(np.array(cam_data["camera_matrix"], dtype=np.float64))
                if "projection_matrix" in cam_data:
                    projection_matrices.append(np.array(cam_data["projection_matrix"], dtype=np.float64))
                if "image_size" in cam_data:
                    image_sizes.append(tuple(cam_data["image_size"]))

        if camera_matrices or projection_matrices:
            return {
                "camera_matrices": camera_matrices or None,
                "projection_matrices": projection_matrices or None,
                "camera_names": camera_names or None,
                "image_sizes": image_sizes or None,
            }

    pkl_patterns = ["calibration.pkl", "camera_calibration.pkl", "calibration_data.pkl"]
    for pattern in pkl_patterns:
        pkl_path = os.path.join(recording_folder, pattern)
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                if isinstance(data, dict):
                    return {
                        "camera_matrices": data.get("camera_matrices") or data.get("intrinsic_matrices"),
                        "projection_matrices": data.get("projection_matrices"),
                        "camera_names": data.get("camera_names"),
                        "image_sizes": data.get("image_sizes"),
                    }
                elif isinstance(data, (list, tuple)):
                    return {
                        "camera_matrices": None,
                        "projection_matrices": data if all(
                            np.array(p).shape == (3, 4) for p in data
                        ) else None,
                        "camera_names": None,
                        "image_sizes": None,
                    }
            except Exception:
                continue

    npy_patterns = [
        "calibration_data.npy",
        "camera_intrinsics.npy",
        "camera_matrices.npy",
        "projection_matrices.npy",
    ]

    camera_matrices = None
    projection_matrices = None

    for pattern in npy_patterns:
        npy_path = os.path.join(recording_folder, pattern)
        if os.path.exists(npy_path):
            arr = np.load(npy_path, allow_pickle=True)
            if arr.dtype == object or arr.ndim >= 2:
                if pattern in ("camera_intrinsics.npy", "camera_matrices.npy"):
                    camera_matrices = [np.array(c, dtype=np.float64) for c in arr] if arr.ndim == 3 else [arr]
                elif pattern == "projection_matrices.npy":
                    projection_matrices = [np.array(p, dtype=np.float64) for p in arr] if arr.ndim == 3 else [arr]
                elif pattern == "calibration_data.npy":
                    if arr.ndim == 3 and arr.shape[1:] == (3, 4):
                        projection_matrices = [np.array(p, dtype=np.float64) for p in arr]
                    elif arr.ndim == 3 and arr.shape[1:] == (3, 3):
                        camera_matrices = [np.array(c, dtype=np.float64) for c in arr]

    if camera_matrices is not None or projection_matrices is not None:
        return {
            "camera_matrices": camera_matrices,
            "projection_matrices": projection_matrices,
            "camera_names": None,
            "image_sizes": None,
        }

    json_patterns = ["camera_calibration.json", "calibration.json"]
    for pattern in json_patterns:
        json_path = os.path.join(recording_folder, pattern)
        if os.path.exists(json_path):
            import json
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                if isinstance(data, dict) and ("cameras" in data or "camera_matrices" in data):
                    camera_matrices = []
                    projection_matrices = []
                    camera_names = []
                    image_sizes = []

                    cams = data.get("cameras", {})
                    if isinstance(cams, dict):
                        for cam_name, cam_data in cams.items():
                            camera_names.append(cam_name)
                            if "camera_matrix" in cam_data:
                                camera_matrices.append(np.array(cam_data["camera_matrix"], dtype=np.float64))
                            if "projection_matrix" in cam_data:
                                projection_matrices.append(np.array(cam_data["projection_matrix"], dtype=np.float64))
                            if "image_size" in cam_data:
                                image_sizes.append(tuple(cam_data["image_size"]))
                    elif "camera_matrices" in data:
                        camera_matrices = [np.array(c, dtype=np.float64) for c in data["camera_matrices"]]
                        projection_matrices = [np.array(p, dtype=np.float64) for p in data.get("projection_matrices", [])]
                        camera_names = data.get("camera_names", [])
                        image_sizes = [tuple(s) for s in data.get("image_sizes", [])]

                    if camera_matrices or projection_matrices:
                        return {
                            "camera_matrices": camera_matrices or None,
                            "projection_matrices": projection_matrices or None,
                            "camera_names": camera_names or None,
                            "image_sizes": image_sizes or None,
                        }
            except Exception:
                continue

    subdir_names = ["calibration", "calibration_data", "camera_calibration"]
    for subdir_name in subdir_names:
        subdir = os.path.join(recording_folder, subdir_name)
        if os.path.isdir(subdir):
            result = load_calibration_from_freemocap(subdir)
            if result is not None:
                return result

    return None
