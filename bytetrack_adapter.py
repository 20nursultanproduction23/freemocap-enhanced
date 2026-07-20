"""
ByteTrack Adapter — Drop-in replacement for TemporalTracker (Stage 3)

Implements a lightweight ByteTrack-style multi-object tracker using:
  - Kalman filter per track for state estimation (bbox: x, y, w, h + velocities)
  - Two-stage Hungarian matching: high-confidence first, then low-confidence
  - IoU-based cost matrix (instead of keypoint distance)
  - Track lifecycle management with configurable lost-frame tolerance

ByteTrack's key insight: low-confidence detections (occluded/partial actors)
are still matched to existing tracks after high-confidence matching, reducing
ID switches in crowded or occluded scenes.

Input / Output is identical to TemporalTracker:
  Input:  List[FrameAssociation] from cross_view_association
  Output: List[TemporalAssociation] from temporal_tracker

Preserves the marker-anchor override feature from TemporalTracker.

Usage:
    from bytetrack_adapter import ByteTrackTracker

    tracker = ByteTrackTracker(fps=30.0)
    results = tracker.track(frame_associations, per_camera_per_frame_detections)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKF
except ImportError:
    _FilterPyKF = None

from cross_view_association import FrameAssociation
from temporal_tracker import TemporalAssociation


# ── Configuration ──────────────────────────────────────────────────────

HIGH_CONF_THRESHOLD = 0.5
LOW_CONF_THRESHOLD = 0.1

HIGH_CONF_IOU_THRESHOLD = 0.3
LOW_CONF_IOU_THRESHOLD = 0.5

MAX_AGE = 30
MAX_COCO = 10
MIN_HITS = 3

Q_WEIGHT = 0.01
Q_VEL = 0.01
R_WEIGHT = 10.0


# ── Kalman Filter ──────────────────────────────────────────────────────

class KalmanFilter:
    """8-state Kalman filter for bounding box tracking.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement:  [cx, cy, w, h]

    Falls back to pure-numpy if filterpy is not installed.
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self._use_filterpy = False

        if _FilterPyKF is not None:
            try:
                self._kf = _FilterPyKF(dim_x=8, dim_z=4)
                self._init_filterpy()
                self._use_filterpy = True
                return
            except Exception:
                pass

        self._init_numpy()

    def _init_filterpy(self):
        kf = self._kf
        kf.F = np.eye(8)
        kf.F[0, 4] = self.dt
        kf.F[1, 5] = self.dt
        kf.F[2, 6] = self.dt
        kf.F[3, 7] = self.dt

        kf.H = np.zeros((4, 8))
        kf.H[:4, :4] = np.eye(4)

        kf.Q = np.eye(8)
        kf.Q[:4, :4] *= Q_WEIGHT
        kf.Q[4:, 4:] *= Q_VEL

        kf.R = np.eye(4) * R_WEIGHT
        kf.x = np.zeros((8, 1))
        kf.P = np.eye(8) * 10.0

    def _init_numpy(self):
        self._F = np.eye(8)
        self._F[0, 4] = self.dt
        self._F[1, 5] = self.dt
        self._F[2, 6] = self.dt
        self._F[3, 7] = self.dt

        self._H = np.zeros((4, 8))
        self._H[:4, :4] = np.eye(4)

        self._Q = np.eye(8)
        self._Q[:4, :4] *= Q_WEIGHT
        self._Q[4:, 4:] *= Q_VEL

        self._R = np.eye(4) * R_WEIGHT

        self._x = np.zeros((8, 1))
        self._P = np.eye(8) * 10.0

    def init_state(self, bbox_xyxy: np.ndarray):
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        w = max(bbox_xyxy[2] - bbox_xyxy[0], 1.0)
        h = max(bbox_xyxy[3] - bbox_xyxy[1], 1.0)
        z = np.array([[cx], [cy], [w], [h]])

        if self._use_filterpy:
            self._kf.x[:4] = z
            self._kf.x[4:] = 0
            self._kf.P = np.eye(8) * 10.0
        else:
            self._x[:4] = z
            self._x[4:] = 0
            self._P = np.eye(8) * 10.0

    def predict(self):
        if self._use_filterpy:
            self._kf.predict()
        else:
            self._x = self._F @ self._x
            self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, bbox_xyxy: np.ndarray):
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
        w = max(bbox_xyxy[2] - bbox_xyxy[0], 1.0)
        h = max(bbox_xyxy[3] - bbox_xyxy[1], 1.0)
        z = np.array([[cx], [cy], [w], [h]])

        if self._use_filterpy:
            self._kf.update(z)
        else:
            S = self._H @ self._P @ self._H.T + self._R
            S_inv = np.linalg.inv(S)
            K = self._P @ self._H.T @ S_inv
            y = z - self._H @ self._x
            self._x = self._x + K @ y
            I_KH = np.eye(8) - K @ self._H
            self._P = I_KH @ self._P

    @property
    def state_xyxy(self) -> np.ndarray:
        if self._use_filterpy:
            s = self._kf.x.flatten()
        else:
            s = self._x.flatten()
        cx, cy, w, h = s[0], s[1], max(s[2], 1.0), max(s[3], 1.0)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


# ── IoU computation ────────────────────────────────────────────────────

def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_cost_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute (1 - IoU) cost matrix between two sets of boxes."""
    N_a, N_b = boxes_a.shape[0], boxes_b.shape[0]
    cost = np.ones((N_a, N_b), dtype=np.float64)
    for i in range(N_a):
        for j in range(N_b):
            cost[i, j] = 1.0 - _compute_iou(boxes_a[i], boxes_b[j])
    return cost


# ── Internal track state ──────────────────────────────────────────────

class _Track:
    __slots__ = (
        "track_id", "kf", "confidence",
        "age", "hits", "time_since_update",
        "is_lost", "is_confirmed", "feature_history",
    )

    def __init__(self, track_id: int, bbox: np.ndarray, confidence: float, dt: float):
        self.track_id = track_id
        self.kf = KalmanFilter(dt=dt)
        self.kf.init_state(bbox)
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.is_lost = False
        self.is_confirmed = False
        self.feature_history: List[Optional[np.ndarray]] = []

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: np.ndarray, confidence: float):
        self.kf.update(bbox)
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.is_lost = False
        if self.hits >= MIN_HITS:
            self.is_confirmed = True

    @property
    def state_xyxy(self) -> np.ndarray:
        return self.kf.state_xyxy


# ── Hungarian matching ─────────────────────────────────────────────────

def _hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if cost_matrix.size == 0:
        return (
            [],
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_ind, col_ind):
        if r < cost_matrix.shape[0] and c < cost_matrix.shape[1]:
            if cost_matrix[r, c] <= threshold:
                matches.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)

    unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


# ── Feature extraction helpers ────────────────────────────────────────

def _extract_actor_bbox(
    frame_assoc: FrameAssociation,
    actor_id: int,
    per_camera_detections: List = None,
    reference_camera: int = 0,
) -> Optional[np.ndarray]:
    if per_camera_detections is None:
        return None
    members = frame_assoc.actor_memberships.get(actor_id, [])
    for cam_idx, person_id in members:
        if cam_idx == reference_camera:
            return per_camera_detections[cam_idx][person_id].bbox
    for cam_idx, person_id in members:
        return per_camera_detections[cam_idx][person_id].bbox
    return None


def _extract_actor_confidence(
    frame_assoc: FrameAssociation,
    actor_id: int,
    per_camera_detections: List = None,
    reference_camera: int = 0,
) -> float:
    if per_camera_detections is None:
        return 0.5
    members = frame_assoc.actor_memberships.get(actor_id, [])
    if not members:
        return 0.0

    for cam_idx, person_id in members:
        if cam_idx == reference_camera:
            det = per_camera_detections[cam_idx][person_id]
            body = det.keypoints_133[:17]
            scores = body[body[:, 2] > 0.3, 2]
            if len(scores) >= 2:
                return float(np.mean(scores))

    for cam_idx, person_id in members:
        det = per_camera_detections[cam_idx][person_id]
        body = det.keypoints_133[:17]
        scores = body[body[:, 2] > 0.3, 2]
        if len(scores) >= 2:
            return float(np.mean(scores))
    return 0.3


def _extract_actor_keypoints_normalized(
    frame_assoc: FrameAssociation,
    actor_id: int,
    per_camera_detections: List = None,
    reference_camera: int = 0,
) -> Optional[np.ndarray]:
    if per_camera_detections is None:
        return None
    members = frame_assoc.actor_memberships.get(actor_id, [])
    body_kpts = None
    for cam_idx, person_id in members:
        if cam_idx == reference_camera:
            body_kpts = per_camera_detections[cam_idx][person_id].keypoints_133[:17]
            break
    if body_kpts is None and members:
        cam_idx, person_id = members[0]
        body_kpts = per_camera_detections[cam_idx][person_id].keypoints_133[:17]
    if body_kpts is None:
        return None
    visible = body_kpts[:, 2] > 0.3
    if np.sum(visible) < 2:
        return None
    return body_kpts[visible, :2].copy()


# ── Main ByteTrack Tracker ────────────────────────────────────────────

class ByteTrackTracker:
    """ByteTrack-style multi-object tracker.

    Drop-in replacement for TemporalTracker. Uses Kalman filter state
    estimation and two-stage Hungarian matching (high-confidence first,
    then low-confidence).

    Args:
        fps: video frame rate (used to set Kalman dt)
        high_conf_threshold: confidence above which detections are "high"
        low_conf_threshold: confidence below which detections are ignored
        high_conf_iou_threshold: min IoU to match high-conf detection to track
        low_conf_iou_threshold: min IoU to match low-conf detection to track
        max_age: frames without detection before track removal
        max_coco: consecutive frames lost before removal
        min_hits: matched frames before a track is published
        reference_camera: camera index to extract features from
    """

    def __init__(
        self,
        fps: float = 30.0,
        high_conf_threshold: float = HIGH_CONF_THRESHOLD,
        low_conf_threshold: float = LOW_CONF_THRESHOLD,
        high_conf_iou_threshold: float = HIGH_CONF_IOU_THRESHOLD,
        low_conf_iou_threshold: float = LOW_CONF_IOU_THRESHOLD,
        max_age: int = MAX_AGE,
        max_coco: int = MAX_COCO,
        min_hits: int = MIN_HITS,
        reference_camera: int = 0,
        **kwargs,
    ):
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.high_conf_iou_threshold = high_conf_iou_threshold
        self.low_conf_iou_threshold = low_conf_iou_threshold
        self.max_age = max_age
        self.max_coco = max_coco
        self.min_hits = min_hits
        self.reference_camera = reference_camera

        self._tracks: List[_Track] = []
        self._next_track_id = 0

        # marker_id -> persistent_global_id
        self._marker_anchor_map: Dict[int, int] = {}

    def _create_track(self, bbox: np.ndarray, confidence: float) -> _Track:
        track = _Track(self._next_track_id, bbox, confidence, self.dt)
        self._next_track_id += 1
        self._tracks.append(track)
        return track

    def _gc_tracks(self):
        """Remove tracks that have exceeded max_age."""
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

    def track_frame(
        self,
        frame_idx: int,
        frame_assoc: FrameAssociation,
        per_camera_detections: List = None,
        marker_overrides: Optional[Dict[int, int]] = None,
    ) -> TemporalAssociation:
        """Track actors for one frame.

        Args:
            frame_idx: frame index
            frame_assoc: Stage 2 result for this frame
            per_camera_detections: List[List[PersonDetection]] for this frame
            marker_overrides: optional {marker_id: local_actor_id}

        Returns:
            TemporalAssociation
        """
        local_actor_ids = list(frame_assoc.actor_memberships.keys())
        num_local = len(local_actor_ids)

        if num_local == 0:
            return TemporalAssociation(
                frame_idx=frame_idx, num_persistent_actors=0,
                local_to_persistent={}, persistent_to_local={}, active_tracks=[],
            )

        # Extract per-actor (local_id, bbox, confidence)
        det_info: Dict[int, Tuple[np.ndarray, float]] = {}
        for local_id in local_actor_ids:
            bbox = _extract_actor_bbox(
                frame_assoc, local_id, per_camera_detections, self.reference_camera
            )
            if bbox is None:
                bbox = np.array([0.0, 0.0, 1.0, 1.0])
            conf = _extract_actor_confidence(
                frame_assoc, local_id, per_camera_detections, self.reference_camera
            )
            kpts = _extract_actor_keypoints_normalized(
                frame_assoc, local_id, per_camera_detections, self.reference_camera
            )
            det_info[local_id] = (bbox, conf, kpts)

        # ── First frame: create tracks for everyone ───────────────────
        if frame_idx == 0:
            local_to_persistent: Dict[int, int] = {}
            for local_id in local_actor_ids:
                bbox, conf, kpts = det_info[local_id]
                track = self._create_track(bbox, conf)
                track.feature_history.append(kpts)
                local_to_persistent[local_id] = track.track_id

            if marker_overrides:
                for marker_id, local_id in marker_overrides.items():
                    if local_id in local_to_persistent:
                        self._marker_anchor_map[marker_id] = local_to_persistent[local_id]

            active_ids = [t.track_id for t in self._tracks if not t.is_lost]
            return TemporalAssociation(
                frame_idx=frame_idx, num_persistent_actors=num_local,
                local_to_persistent=local_to_persistent,
                persistent_to_local={v: k for k, v in local_to_persistent.items()},
                active_tracks=active_ids,
            )

        # ── Predict all active tracks ─────────────────────────────────
        active_tracks = [t for t in self._tracks if t.age <= self.max_age]
        for track in active_tracks:
            track.predict()

        # ── Marker-anchor overrides (hard assignments) ────────────────
        local_to_persistent = {}
        anchored_local_ids: set = set()
        anchored_track_indices: set = set()

        if marker_overrides:
            for marker_id, local_actor_id in marker_overrides.items():
                if local_actor_id not in local_actor_ids:
                    continue

                bbox, conf, kpts = det_info[local_actor_id]

                if marker_id in self._marker_anchor_map:
                    target_gid = self._marker_anchor_map[marker_id]
                    target_track = None
                    target_idx = None
                    for idx, t in enumerate(active_tracks):
                        if t.track_id == target_gid:
                            target_track = t
                            target_idx = idx
                            break

                    if target_track is not None and target_idx is not None:
                        target_track.update(bbox, 1.0)
                        target_track.feature_history.append(kpts)
                        local_to_persistent[local_actor_id] = target_gid
                        anchored_local_ids.add(local_actor_id)
                        anchored_track_indices.add(target_idx)
                else:
                    track = self._create_track(bbox, 1.0)
                    track.feature_history.append(kpts)
                    self._marker_anchor_map[marker_id] = track.track_id
                    local_to_persistent[local_actor_id] = track.track_id
                    anchored_local_ids.add(local_actor_id)

        # ── Build detection arrays for non-anchored actors ────────────
        free_local_ids = [lid for lid in local_actor_ids if lid not in anchored_local_ids]

        if not free_local_ids:
            # All actors anchored
            self._gc_tracks()
            active_ids = [t.track_id for t in self._tracks if not t.is_lost]
            return TemporalAssociation(
                frame_idx=frame_idx,
                num_persistent_actors=len(active_ids),
                local_to_persistent=local_to_persistent,
                persistent_to_local={v: k for k, v in local_to_persistent.items()},
                active_tracks=active_ids,
            )

        free_bboxes = np.array([det_info[lid][0] for lid in free_local_ids])
        free_confs = np.array([det_info[lid][1] for lid in free_local_ids])

        # Free tracks = active tracks not anchored
        free_tracks = [
            t for i, t in enumerate(active_tracks)
            if i not in anchored_track_indices
        ]

        if not free_tracks:
            # No existing tracks — create new ones for all free detections
            for lid in free_local_ids:
                bbox, conf, kpts = det_info[lid]
                track = self._create_track(bbox, conf)
                track.feature_history.append(kpts)
                local_to_persistent[lid] = track.track_id

            self._gc_tracks()
            active_ids = [t.track_id for t in self._tracks if not t.is_lost]
            return TemporalAssociation(
                frame_idx=frame_idx,
                num_persistent_actors=len(active_ids),
                local_to_persistent=local_to_persistent,
                persistent_to_local={v: k for k, v in local_to_persistent.items()},
                active_tracks=active_ids,
            )

        free_track_bboxes = np.array([t.state_xyxy for t in free_tracks])
        N_tracks = len(free_tracks)
        N_dets = len(free_local_ids)

        # ── Stage 1: High-confidence matching ─────────────────────────
        high_mask = free_confs >= self.high_conf_threshold
        high_indices = np.where(high_mask)[0]

        matched_track_idx_set: set = set()
        matched_det_idx_set: set = set()
        matches: List[Tuple[int, int]] = []  # (free_track_idx, free_det_idx)

        if len(high_indices) > 0:
            high_cost = _iou_cost_matrix(free_track_bboxes, free_bboxes[high_indices])
            h_matches, h_unmatched_tracks, _ = _hungarian_match(
                high_cost, 1.0 - self.high_conf_iou_threshold
            )
            for t_local, d_local in h_matches:
                t_global = t_local
                d_global = high_indices[d_local]
                matches.append((t_global, d_global))
                matched_track_idx_set.add(t_global)
                matched_det_idx_set.add(d_global)

        # ── Stage 2: Low-confidence matching (remaining tracks) ───────
        low_mask = (free_confs >= self.low_conf_threshold) & (free_confs < self.high_conf_threshold)
        low_indices = np.where(low_mask)[0]
        remaining_track_indices = [i for i in range(N_tracks) if i not in matched_track_idx_set]

        if remaining_track_indices and len(low_indices) > 0:
            rem_bboxes = free_track_bboxes[remaining_track_indices]
            low_cost = _iou_cost_matrix(rem_bboxes, free_bboxes[low_indices])
            l_matches, _, _ = _hungarian_match(
                low_cost, 1.0 - self.low_conf_iou_threshold
            )
            for t_local, d_local in l_matches:
                t_global = remaining_track_indices[t_local]
                d_global = low_indices[d_local]
                matches.append((t_global, d_global))
                matched_track_idx_set.add(t_global)
                matched_det_idx_set.add(d_global)

        # ── Apply matches ─────────────────────────────────────────────
        for t_idx, d_idx in matches:
            track = free_tracks[t_idx]
            local_id = free_local_ids[d_idx]
            bbox, conf, kpts = det_info[local_id]
            track.update(bbox, conf)
            track.feature_history.append(kpts)
            local_to_persistent[local_id] = track.track_id

        # ── Mark unmatched tracks as lost ─────────────────────────────
        for i, track in enumerate(free_tracks):
            if i not in matched_track_idx_set:
                if track.time_since_update > self.max_coco:
                    track.is_lost = True

        # ── Create new tracks for unmatched detections ────────────────
        for d_idx in range(N_dets):
            if d_idx not in matched_det_idx_set:
                local_id = free_local_ids[d_idx]
                bbox, conf, kpts = det_info[local_id]
                track = self._create_track(bbox, conf)
                track.feature_history.append(kpts)
                local_to_persistent[local_id] = track.track_id

        # ── Garbage-collect and build output ──────────────────────────
        self._gc_tracks()
        active_ids = [t.track_id for t in self._tracks if not t.is_lost]

        return TemporalAssociation(
            frame_idx=frame_idx,
            num_persistent_actors=len(active_ids),
            local_to_persistent=local_to_persistent,
            persistent_to_local={v: k for k, v in local_to_persistent.items()},
            active_tracks=active_ids,
        )

    def track(
        self,
        frame_associations: List[FrameAssociation],
        per_camera_per_frame_detections: Optional[List[List[List]]] = None,
        marker_anchors: Optional[List[Dict[int, int]]] = None,
        **kwargs,
    ) -> List[TemporalAssociation]:
        """Track actors across all frames.

        Identical signature to TemporalTracker.track().

        Args:
            frame_associations: list of FrameAssociation, one per frame
            per_camera_per_frame_detections: optional nested list.
                Supports: [camera][frame] or [frame][camera]
            marker_anchors: optional per-frame marker overrides.
                List of dicts: [frame_idx] -> {marker_id: local_actor_id}.
        """
        results = []

        cam_frame_format = True
        if per_camera_per_frame_detections is not None and len(per_camera_per_frame_detections) > 0:
            first = per_camera_per_frame_detections[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
                if len(per_camera_per_frame_detections) == len(frame_associations):
                    cam_frame_format = False

        for frame_idx, frame_assoc in enumerate(frame_associations):
            per_cam_dets = None
            if per_camera_per_frame_detections is not None:
                if cam_frame_format:
                    num_cams = len(per_camera_per_frame_detections)
                    per_cam_dets = [
                        per_camera_per_frame_detections[cam][frame_idx]
                        for cam in range(num_cams)
                    ]
                else:
                    per_cam_dets = per_camera_per_frame_detections[frame_idx]

            frame_marker_overrides = None
            if marker_anchors and frame_idx < len(marker_anchors):
                frame_marker_overrides = marker_anchors[frame_idx]

            result = self.track_frame(
                frame_idx, frame_assoc, per_cam_dets, frame_marker_overrides
            )
            results.append(result)

        return results

    def get_track_summary(self) -> dict:
        active = sum(1 for t in self._tracks if not t.is_lost)
        lost = sum(1 for t in self._tracks if t.is_lost)
        confirmed = sum(1 for t in self._tracks if t.is_confirmed)
        durations = [t.hits for t in self._tracks if t.hits > 0]
        return {
            "total_tracks_created": len(self._tracks),
            "active_tracks": active,
            "lost_tracks": lost,
            "confirmed_tracks": confirmed,
            "track_durations": durations,
            "mean_duration": float(np.mean(durations)) if durations else 0.0,
            "max_duration": max(durations) if durations else 0,
        }


# ── Convenience function ───────────────────────────────────────────────

def run_bytetrack(
    frame_associations: List[FrameAssociation],
    per_camera_per_frame_detections: Optional[List[List[List]]] = None,
    marker_anchors: Optional[List[Dict[int, int]]] = None,
    **kwargs,
) -> Tuple[List[TemporalAssociation], ByteTrackTracker]:
    """Drop-in replacement for run_temporal_tracking().

    Args:
        frame_associations: output from CrossViewAssociator.associate_video()
        per_camera_per_frame_detections: optional detections for feature extraction
        marker_anchors: optional per-frame marker overrides
        **kwargs: passed to ByteTrackTracker constructor

    Returns:
        (temporal_results, tracker) tuple
    """
    tracker = ByteTrackTracker(**kwargs)
    results = tracker.track(
        frame_associations, per_camera_per_frame_detections, marker_anchors
    )
    return results, tracker
