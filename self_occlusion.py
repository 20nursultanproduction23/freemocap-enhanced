import numpy as np
from typing import Any, Dict, List, Tuple


def _normalize_to_unit(
    keypoints: np.ndarray, image_size: Tuple[int, int]
) -> np.ndarray:
    """Normalize pixel coordinates to [0, 1] range for cross-camera comparison."""
    h, w = image_size
    normalized = keypoints.astype(np.float64).copy()
    normalized[..., 0] /= w
    normalized[..., 1] /= h
    return normalized


def _compute_pairwise_disagreement(
    normalized_cameras: List[np.ndarray],
    confidence_masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """For each keypoint, compute per-camera average distance to other cameras
    and the maximum pairwise L2 distance across cameras.

    Args:
        normalized_cameras: list of (numFrames, numKps, 3) arrays in [0,1] space.
        confidence_masks: list of (numFrames, numKps) boolean arrays.

    Returns:
        per_camera_scores: (numFrames, numCameras, numKps) average distance from
            each camera to all other cameras that also have confident detections.
        max_disagreement: (numFrames, numKps) maximum pairwise L2 distance.
    """
    num_cameras = len(normalized_cameras)
    num_frames = normalized_cameras[0].shape[0]
    num_kps = normalized_cameras[0].shape[1]

    coords = np.stack(normalized_cameras, axis=0)
    masks = np.stack(confidence_masks, axis=0)

    per_camera_scores = np.zeros((num_cameras, num_frames, num_kps))
    pair_count = np.zeros((num_cameras, num_frames, num_kps))

    for i in range(num_cameras):
        for j in range(num_cameras):
            if i == j:
                continue
            valid = masks[i] & masks[j]
            diff = coords[i] - coords[j]
            l2 = np.sqrt(np.sum(diff ** 2, axis=-1))
            per_camera_scores[i] += l2 * valid
            pair_count[i] += valid

    pair_count = np.maximum(pair_count, 1.0)
    per_camera_scores /= pair_count

    per_camera_scores = np.transpose(per_camera_scores, (1, 0, 2))

    max_disagreement = np.max(per_camera_scores, axis=1)

    return per_camera_scores, max_disagreement


def _compute_baseline_disagreement(
    disagreement_history: np.ndarray, window: int = 30
) -> np.ndarray:
    """Rolling mean baseline to adapt to different body positions over time.

    Args:
        disagreement_history: (numFrames, numKps) disagreement values.
        window: number of frames in the rolling window.

    Returns:
        baseline: (numFrames, numKps) rolling mean disagreement.
    """
    num_frames = disagreement_history.shape[0]
    baseline = np.zeros_like(disagreement_history)
    half = window // 2

    for i in range(num_frames):
        start = max(0, i - half)
        end = min(num_frames, i + half + 1)
        baseline[i] = np.mean(disagreement_history[start:end], axis=0)

    return baseline


def detect_self_occlusion(
    cameras_data: List[Dict[str, np.ndarray]],
    image_sizes: List[Tuple[int, int]],
    confidence_threshold: float = 0.3,
    disagreement_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Detect self-occlusion via cross-camera consistency checking.

    When a body part blocks another from a camera's view, MediaPipe may output
    confident but incorrect detections. This function compares the same keypoints
    across multiple cameras. Normal parallax produces a baseline level of
    disagreement. Self-occlusion causes spikes in disagreement for specific
    keypoints in specific cameras, which this function flags.

    Args:
        cameras_data: list of dicts, one per camera. Each dict has keys from
            RTMPose segments (body, face, left_hand, right_hand). Each value
            has shape (numFrames, numKeypoints, 4) where [:, :, :3] are
            coordinates and [:, :, 3] are confidence scores.
        image_sizes: list of (height, width) tuples, one per camera.
        confidence_threshold: minimum confidence to treat a detection as valid.
            Low-confidence detections are excluded from disagreement computation
            rather than counted as disagreements.
        disagreement_threshold: multiplier above the rolling baseline required
            to flag a keypoint as a suspected self-occlusion. A value of 2.0
            means the disagreement must be at least twice the recent average.

    Returns:
        dict with keys:
            'per_frame_scores': ndarray of shape
                (numFrames, numCameras, totalKeypoints) giving the average
                distance each camera's detection has from the other cameras'
                detections for each keypoint at each frame.
            'flagged_frames': sorted list of frame indices where at least one
                keypoint in at least one camera is flagged.
            'flagged_keypoints': list of (frame_idx, camera_idx, keypoint_idx)
                tuples identifying each flagged detection. keypoints are indexed
                across all segments in order: body, face, left_hand, right_hand.
            'report': dict with summary statistics including total flagged
                counts, mean/max disagreement values, and per-segment info.
    """
    segment_order = ["body", "face", "left_hand", "right_hand"]
    num_cameras = len(cameras_data)
    num_frames = cameras_data[0][segment_order[0]].shape[0]

    segment_info: List[Dict[str, Any]] = []
    all_per_camera: List[np.ndarray] = []
    keypoint_offset = 0

    for segment in segment_order:
        seg_arrays = [cam.get(segment) for cam in cameras_data]
        if seg_arrays[0] is None:
            continue

        num_kps = seg_arrays[0].shape[1]
        normalized: List[np.ndarray] = []
        masks: List[np.ndarray] = []

        for cam_idx, arr in enumerate(seg_arrays):
            if arr is None:
                normalized.append(np.zeros((num_frames, num_kps, 3)))
                masks.append(np.zeros((num_frames, num_kps), dtype=bool))
                continue

            norm = _normalize_to_unit(arr[:, :, :3], image_sizes[cam_idx])
            conf = arr[:, :, 3] > confidence_threshold
            normalized.append(norm)
            masks.append(conf)

        per_cam, _ = _compute_pairwise_disagreement(normalized, masks)
        all_per_camera.append(per_cam)

        segment_info.append(
            {
                "name": segment,
                "num_keypoints": num_kps,
                "offset": keypoint_offset,
            }
        )
        keypoint_offset += num_kps

    total_kps = keypoint_offset

    if not segment_info:
        return {
            "per_frame_scores": np.zeros((num_frames, num_cameras, 0)),
            "flagged_frames": [],
            "flagged_keypoints": [],
            "report": {
                "total_flagged": 0,
                "flagged_frames_count": 0,
                "segments_processed": 0,
                "num_cameras": num_cameras,
                "num_frames": num_frames,
            },
        }

    per_frame_scores = np.zeros((num_frames, num_cameras, total_kps))
    for i, info in enumerate(segment_info):
        off = info["offset"]
        nk = info["num_keypoints"]
        per_frame_scores[:, :, off : off + nk] = all_per_camera[i]

    baseline = np.zeros_like(per_frame_scores)
    for c in range(num_cameras):
        baseline[:, c, :] = _compute_baseline_disagreement(
            per_frame_scores[:, c, :]
        )

    safe_baseline = np.maximum(baseline, 1e-10)
    spike_ratio = per_frame_scores / safe_baseline
    flagged_mask = spike_ratio > disagreement_threshold
    flagged_mask &= per_frame_scores > 0

    flagged_frames = sorted(set(np.nonzero(flagged_mask.any(axis=1))[0].tolist()))

    flagged_raw = np.argwhere(flagged_mask)
    flagged_keypoints = [
        (int(r[0]), int(r[1]), int(r[2])) for r in flagged_raw
    ]

    nonzero_scores = per_frame_scores[per_frame_scores > 0]
    report: Dict[str, Any] = {
        "total_flagged": len(flagged_keypoints),
        "flagged_frames_count": len(flagged_frames),
        "segments_processed": len(segment_info),
        "segment_details": {
            info["name"]: {"num_keypoints": info["num_keypoints"]}
            for info in segment_info
        },
        "num_cameras": num_cameras,
        "num_frames": num_frames,
        "mean_disagreement": float(np.mean(nonzero_scores))
        if nonzero_scores.size > 0
        else 0.0,
        "max_disagreement": float(np.max(nonzero_scores))
        if nonzero_scores.size > 0
        else 0.0,
        "mean_baseline": float(np.mean(baseline)),
        "mean_spike_ratio": float(np.mean(spike_ratio[flagged_mask]))
        if np.any(flagged_mask)
        else 0.0,
    }

    return {
        "per_frame_scores": per_frame_scores,
        "flagged_frames": flagged_frames,
        "flagged_keypoints": flagged_keypoints,
        "report": report,
    }
