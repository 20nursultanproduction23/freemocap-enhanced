"""
Stage 7, Parts B+C — ArUco Marker Fallback

Detects ArUco markers in raw camera frames and binds them to skeleton
detections. When a marker is found near a skeleton's chest/hip center,
it provides a hard override for person_id — bypassing geometric association
from Stages 2/3.

Part B — Marker detection in frames:
  Independent pass using cv2.aruco.ArucoDetector (separate from RTMDet/RTMPose).
  Output per frame per camera: list of {marker_id, corners, center_2d, detection_confidence}.

Part C — Marker-to-skeleton binding:
  For each detected marker, find the nearest skeleton center (midpoint of
  left_shoulder + right_shoulder or left_hip + right_hip) on the same camera
  in the same frame. If within distance threshold → hard override person_id_temp.

Part D — Integration with identity_tracker.py:
  Marker-anchored frames become "anchor frames" where the temporal tracker
  uses marker-based ID as ground truth instead of geometric association.

Distance threshold justification:
  The chest/hip center of a person is typically 150-200mm from a marker
  attached to their chest. In pixel space at typical FreeMoCap resolutions
  (1280x720, camera ~2m away), this translates to ~50-100px. We use 120px
  as a conservative threshold that accounts for:
    - Marker placement variation (chest vs back vs shoulder)
    - Camera angle and perspective distortion
    - Slight misalignment between marker center and body center
  This is analogous to the MAX_KEYPOINT_DISTANCE=100px threshold in
  temporal_tracker.py (Stage 3), but slightly larger because marker
  placement is less precisely controlled than body keypoint estimation.

Usage:
    from marker_fallback import ArucoMarkerDetector, detect_and_bind

    detector = ArucoMarkerDetector(config_path="actor_marker_map.yaml")
    marker_detections = detector.detect_frame(frame_image, cam_idx=0, frame_idx=0)
    overrides = detect_and_bind(marker_detections, per_camera_detections, frame_idx=0)
    # overrides: {local_actor_id: marker_id, ...} — use as anchor in temporal tracker
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os
import yaml

from multiperson_detector import PersonDetection


# ── Constants ──────────────────────────────────────────────────────────

# Marker detection parameters
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}

# Distance threshold for marker-to-skeleton binding (pixels).
# See module docstring for justification.
MARKER_SKELETON_BIND_THRESHOLD_PX = 120.0

# Body keypoint indices for center computation (COCO 17)
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class MarkerDetection:
    """Single ArUco marker detection in one frame from one camera."""
    marker_id: int
    corners: np.ndarray        # (4, 2) corner pixel coordinates
    center_2d: np.ndarray      # (2,) center point [x, y]
    detection_confidence: float  # 0-1, based on marker contrast/sharpness
    cam_idx: int
    frame_idx: int


@dataclass
class MarkerBinding:
    """Result of binding a marker to a skeleton detection."""
    marker_id: int
    actor_id: int              # local actor_id from detection
    cam_idx: int
    frame_idx: int
    distance_px: float         # distance from marker center to body center
    bound: bool                # whether binding was successful


# ── Part B: Marker Detection ───────────────────────────────────────────

class ArucoMarkerDetector:
    """Detect ArUco markers in raw camera frames.

    Args:
        config_path: path to actor_marker_map.yaml
        marker_length: physical marker length in meters (for pose estimation if needed)
        detector_params: optional dict of ArucoDetector parameters
    """

    def __init__(
        self,
        config_path: str = "tools/actor_marker_map.yaml",
        marker_length: float = 0.05,
        detector_params: Optional[dict] = None,
    ):
        self.config = self._load_config(config_path)
        self.marker_length = marker_length

        dict_name = self.config.get("dictionary", "DICT_4X4_50")
        dict_type = ARUCO_DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)

        params = cv2.aruco.DetectorParameters()
        if detector_params:
            for k, v in detector_params.items():
                if hasattr(params, k):
                    setattr(params, k, v)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

        self.actor_to_marker = self.config.get("actors", {})
        self.marker_to_actor = {v: k for k, v in self.actor_to_marker.items()}

    @staticmethod
    def _load_config(config_path):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {"dictionary": "DICT_4X4_50", "marker_size_mm": 50, "actors": {"actor_0": 0, "actor_1": 1}}

    def detect_frame(
        self, frame_image: np.ndarray, cam_idx: int = 0, frame_idx: int = 0
    ) -> List[MarkerDetection]:
        """Detect ArUco markers in a single frame.

        Args:
            frame_image: BGR image (H, W, 3) or grayscale (H, W)
            cam_idx: camera index
            frame_idx: frame index

        Returns:
            List of MarkerDetection for this frame/camera
        """
        if frame_image is None or frame_image.size == 0:
            return []

        if len(frame_image.shape) == 3:
            gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_image

        corners, ids, rejected = self.detector.detectMarkers(gray)

        detections = []
        if ids is not None and len(ids) > 0:
            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])
                marker_corners = corners[i].reshape(4, 2)
                center = marker_corners.mean(axis=0)

                confidence = self._estimate_confidence(gray, marker_corners)

                detections.append(MarkerDetection(
                    marker_id=marker_id,
                    corners=marker_corners,
                    center_2d=center,
                    detection_confidence=confidence,
                    cam_idx=cam_idx,
                    frame_idx=frame_idx,
                ))

        return detections

    def detect_video(
        self,
        video_path: str,
        cam_idx: int = 0,
        frame_range: Optional[Tuple[int, int]] = None,
    ) -> List[List[MarkerDetection]]:
        """Detect markers across all frames of a video file.

        Args:
            video_path: path to video file
            cam_idx: camera index
            frame_range: optional (start, end) frame indices

        Returns:
            List of lists: [frame_idx] -> List[MarkerDetection]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        all_detections = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_range is not None:
                if frame_idx < frame_range[0]:
                    frame_idx += 1
                    continue
                if frame_idx >= frame_range[1]:
                    break

            dets = self.detect_frame(frame, cam_idx, frame_idx)
            all_detections.append(dets)
            frame_idx += 1

        cap.release()
        return all_detections

    @staticmethod
    def _estimate_confidence(gray, corners):
        """Estimate marker detection confidence from local contrast."""
        cx, cy = int(corners[:, 0].mean()), int(corners[:, 1].mean())
        h, w = gray.shape
        r = 20
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 <= x1 or y2 <= y1:
            return 0.5

        patch = gray[y1:y2, x1:x2].astype(np.float64)
        if patch.size == 0:
            return 0.5

        std_val = np.std(patch)
        confidence = min(1.0, std_val / 60.0)
        return float(confidence)


# ── Part C: Marker-to-Skeleton Binding ─────────────────────────────────

def _compute_body_center(keypoints_133: np.ndarray) -> Optional[np.ndarray]:
    """Compute the center of the torso from RTMPose keypoints.

    Uses midpoint of shoulders as primary, falls back to midpoint of hips.

    Args:
        keypoints_133: (133, 3) array [x, y, score]

    Returns:
        (2,) center point [x, y] or None if insufficient keypoints
    """
    ls = keypoints_133[LEFT_SHOULDER_IDX]
    rs = keypoints_133[RIGHT_SHOULDER_IDX]

    if ls[2] > 0.3 and rs[2] > 0.3:
        return np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])

    lh = keypoints_133[LEFT_HIP_IDX]
    rh = keypoints_133[RIGHT_HIP_IDX]

    if lh[2] > 0.3 and rh[2] > 0.3:
        return np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])

    return None


def bind_markers_to_skeletons(
    marker_detections: List[MarkerDetection],
    skeleton_detections: List[PersonDetection],
    threshold_px: float = MARKER_SKELETON_BIND_THRESHOLD_PX,
) -> List[MarkerBinding]:
    """Bind detected markers to the nearest skeleton on the same camera/frame.

    Args:
        marker_detections: markers found on this camera in this frame
        skeleton_detections: skeleton detections from RTMPose on same camera/frame
        threshold_px: maximum distance (pixels) for valid binding

    Returns:
        List of MarkerBinding (one per marker detection)
    """
    bindings = []

    for marker in marker_detections:
        best_binding = MarkerBinding(
            marker_id=marker.marker_id,
            actor_id=-1,
            cam_idx=marker.cam_idx,
            frame_idx=marker.frame_idx,
            distance_px=float("inf"),
            bound=False,
        )

        for det in skeleton_detections:
            body_center = _compute_body_center(det.keypoints_133)
            if body_center is None:
                continue

            dist = float(np.linalg.norm(marker.center_2d - body_center))

            if dist < best_binding.distance_px:
                best_binding.distance_px = dist
                best_binding.actor_id = det.person_id

        if best_binding.distance_px <= threshold_px and best_binding.actor_id >= 0:
            best_binding.bound = True

        bindings.append(best_binding)

    return bindings


def detect_and_bind(
    per_camera_marker_detections: List[List[MarkerDetection]],
    per_camera_skeleton_detections: List[List[PersonDetection]],
    frame_idx: int,
    actor_marker_map: Optional[Dict[str, int]] = None,
) -> Dict[int, int]:
    """Detect markers and bind to skeletons for one frame across all cameras.

    This is the main entry point called by the pipeline.

    Args:
        per_camera_marker_detections: [cam_idx] -> List[MarkerDetection] for this frame
        per_camera_skeleton_detections: [cam_idx] -> List[PersonDetection] for this frame
        frame_idx: current frame index
        actor_marker_map: optional {actor_name: marker_id} mapping

    Returns:
        Dict mapping marker_id -> local_actor_id (the override mapping).
        Empty dict if no valid bindings found.
    """
    all_bindings = []

    for cam_idx in range(len(per_camera_marker_detections)):
        markers = per_camera_marker_detections[cam_idx]
        skeletons = per_camera_skeleton_detections[cam_idx] if cam_idx < len(per_camera_skeleton_detections) else []

        bindings = bind_markers_to_skeletons(markers, skeletons)
        all_bindings.extend(bindings)

    marker_votes = {}
    for binding in all_bindings:
        if binding.bound:
            mid = binding.marker_id
            if mid not in marker_votes:
                marker_votes[mid] = {}
            aid = binding.actor_id
            marker_votes[mid][aid] = marker_votes[mid].get(aid, 0) + 1

    overrides = {}
    for marker_id, actor_votes in marker_votes.items():
        best_actor = max(actor_votes, key=actor_votes.get)
        best_count = actor_votes[best_actor]
        total_votes = sum(actor_votes.values())

        if best_count >= 2 or (total_votes == 1 and best_count == 1):
            overrides[marker_id] = best_actor

    return overrides
