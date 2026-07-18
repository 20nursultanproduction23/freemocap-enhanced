import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Union


def compute_distortion_weights(
    keypoints_2d: NDArray[np.floating],
    image_size: tuple[int, int],
    power: float = 2.0,
) -> NDArray[np.floating]:
    """
    Compute per-keypoint distortion weights based on distance from image center.

    Wide-angle lenses cause barrel distortion that is worst at frame edges.
    This function penalizes keypoints near edges by assigning lower weights
    using the formula: w = 1.0 / (1.0 + (d/max_d)^power), where d is the
    Euclidean distance from the image center and max_d is the half-diagonal.

    Args:
        keypoints_2d: (numFrames, numKeypoints, 2) array of pixel coordinates.
        image_size: (height, width) tuple of the image dimensions in pixels.
        power: Exponent controlling how aggressively edge keypoints are penalized.
            Higher values produce a sharper falloff near the edges.

    Returns:
        (numFrames, numKeypoints) array of weights in the [0, 1] range.
        Center keypoints receive weights near 1.0; edge keypoints approach 0.0.
    """
    height, width = image_size
    center_x = width / 2.0
    center_y = height / 2.0
    half_diagonal = np.sqrt(center_x**2 + center_y**2)

    dx = keypoints_2d[..., 0] - center_x
    dy = keypoints_2d[..., 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)

    normalized_distances = distances / half_diagonal
    weights = 1.0 / (1.0 + np.power(normalized_distances, power))

    return weights.astype(np.float64)


def weighted_triangulate(
    points_2d_multi_cam: List[NDArray[np.floating]],
    weights_multi_cam: List[NDArray[np.floating]],
    projection_matrices: List[NDArray[np.floating]],
) -> NDArray[np.float64]:
    """
    Perform weighted Direct Linear Transform (DLT) triangulation.

    Reconstructs 3D points from multiple camera views by solving the
    weighted system AX = 0 via SVD. Each camera's contribution is scaled
    by its distortion weight, so reliable (center) detections dominate
    the triangulation while edge detections are down-weighted.

    Args:
        points_2d_multi_cam: List of length numCams, each element is
            (numFrames, numKeypoints, 2) pixel coordinates for that camera.
        weights_multi_cam: List of length numCams, each element is
            (numFrames, numKeypoints) distortion weights for that camera.
        projection_matrices: List of length numCams, each element is a
            (3, 4) camera projection matrix.

    Returns:
        (numFrames, numKeypoints, 3) array of triangulated 3D coordinates.
        Points that cannot be triangulated (e.g., degenerate geometry) are NaN.
    """
    num_cams = len(points_2d_multi_cam)
    num_frames = points_2d_multi_cam[0].shape[0]
    num_keypoints = points_2d_multi_cam[0].shape[1]

    result = np.full((num_frames, num_keypoints, 3), np.nan, dtype=np.float64)

    for f in range(num_frames):
        for k in range(num_keypoints):
            rows = []
            for cam_idx in range(num_cams):
                u = float(points_2d_multi_cam[cam_idx][f, k, 0])
                v = float(points_2d_multi_cam[cam_idx][f, k, 1])
                w = float(weights_multi_cam[cam_idx][f, k])
                P = projection_matrices[cam_idx].astype(np.float64)

                row_u = w * (u * P[2] - P[0])
                row_v = w * (v * P[2] - P[1])
                rows.append(row_u)
                rows.append(row_v)

            A = np.vstack(rows)

            _, s, Vt = np.linalg.svd(A, full_matrices=True)
            X_homog = Vt[-1]

            if abs(X_homog[3]) > 1e-10:
                result[f, k] = X_homog[:3] / X_homog[3]

    return result


def analyze_edge_reliability(
    keypoints_2d: NDArray[np.floating],
    image_size: tuple[int, int],
    min_weight: float = 0.3,
) -> Dict[str, Union[NDArray[np.floating], List[int], NDArray[np.bool_]]]:
    """
    Analyze how consistently each keypoint falls in the unreliable edge zone.

    Keypoints near frame edges suffer from residual distortion even after
    undistortion. This function identifies which keypoints are persistently
    in the edge zone across frames and builds a per-frame reliability mask.

    Edge zone is defined as: distance from center > 70% of the half-diagonal.
    Reliable zone additionally requires the distortion weight >= min_weight.

    Args:
        keypoints_2d: (numFrames, numKeypoints, 2) pixel coordinates.
        image_size: (height, width) tuple of the image dimensions in pixels.
        min_weight: Minimum distortion weight for a keypoint to be considered
            in the reliable zone. Points with weight below this are unreliable
            even if they are not strictly in the geometric edge zone.

    Returns:
        Dictionary containing:
            'edge_fraction_per_keypoint': (numKeypoints,) float array —
                fraction of frames where each keypoint is in the edge zone.
            'edge_keypoints': list of int — indices of keypoints that are
                in the edge zone for more than 50% of frames.
            'reliable_zone_mask': (numFrames, numKeypoints) boolean array —
                True where the keypoint is in the reliable zone (not in edge
                zone AND distortion weight >= min_weight).
    """
    height, width = image_size
    center_x = width / 2.0
    center_y = height / 2.0
    half_diagonal = np.sqrt(center_x**2 + center_y**2)
    edge_threshold = 0.7 * half_diagonal

    dx = keypoints_2d[..., 0] - center_x
    dy = keypoints_2d[..., 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)

    in_edge_zone = distances > edge_threshold

    weights = compute_distortion_weights(keypoints_2d, image_size)

    edge_fraction_per_keypoint = np.mean(in_edge_zone, axis=0)

    edge_keypoints = np.where(edge_fraction_per_keypoint > 0.5)[0].tolist()

    reliable_zone_mask = (~in_edge_zone) & (weights >= min_weight)

    return {
        "edge_fraction_per_keypoint": edge_fraction_per_keypoint,
        "edge_keypoints": edge_keypoints,
        "reliable_zone_mask": reliable_zone_mask,
    }
