"""
Temporal Tracker — Stage 3 of multi-actor pipeline

Links per-frame global actor IDs across time to maintain consistent identity.

Input:
  List[FrameAssociation] from Stage 2 (one per frame)
  Each FrameAssociation contains:
    - detection_to_actor: {(cam_idx, person_id) -> actor_id}
    - actor_memberships: {actor_id: [(cam_idx, person_id), ...]}
    - num_actors: int

Output:
  List[TemporalAssociation] — one per frame, each mapping local_actor_id → persistent_global_id

Algorithm:
  For each consecutive frame pair (t, t+1):
    1. For each actor in frame t and t+1, extract a feature vector:
       - Body keypoints from reference camera (camera 0), or best-confidence camera
       - Bounding box IoU as secondary signal
    2. Build a cost matrix between actors in t vs t+1
    3. Hungarian algorithm finds optimal assignment
    4. Update persistent global track IDs

  Track management:
    - Each track has a persistent global_id (never reused within a session)
    - Tracks are created when a new actor appears
    - Tracks are marked "lost" if not seen for max_lost_frames
    - Lost tracks can be re-identified if they reappear within search window
    - Tracks are "terminated" if lost for too long

Usage:
    tracker = TemporalTracker(num_cameras=3)
    temporal_results = tracker.track(associator.associate_video(all_detections))
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment

from cross_view_association import FrameAssociation


# ── Configuration ──────────────────────────────────────────────────────

# Body keypoint indices (COCO 17) for temporal matching
# Same set used by cross_view_association for consistency
BODY_KEYPOINT_INDICES = [0, 5, 6, 11, 12]  # nose, L/R shoulder, L/R hip
BODY_CONF_THRESHOLD = 0.3  # minimum keypoint confidence to include

# Cost thresholds
MAX_KEYPOINT_DISTANCE = 100.0  # max L2 pixel distance to consider a match
IOU_WEIGHT = 0.3  # weight of IoU in combined cost (0 = keypoint only)
KEYPOINT_WEIGHT = 0.7  # weight of keypoint distance

# Track management
MAX_LOST_FRAMES = 10  # frames before a track is terminated
SEARCH_WINDOW = 30  # frames to search for re-identification of lost tracks


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class TrackState:
    """State of a single actor track across frames."""
    global_id: int  # persistent ID, never reused
    last_seen_frame: int  # last frame where this actor was matched
    feature_history: List[np.ndarray] = field(default_factory=list)
    bbox_history: List[Optional[np.ndarray]] = field(default_factory=list)
    is_lost: bool = False
    lost_frame: Optional[int] = None  # frame where track was first lost


@dataclass
class TemporalAssociation:
    """Temporal association result for one frame."""
    frame_idx: int
    num_persistent_actors: int
    # Maps local_frame_actor_id (from FrameAssociation) -> persistent_global_id
    local_to_persistent: Dict[int, int]
    # Reverse mapping: persistent_global_id -> local_frame_actor_id
    persistent_to_local: Dict[int, int]
    # Which tracks are currently active
    active_tracks: List[int]  # list of persistent_global_ids


# ── Feature extraction ─────────────────────────────────────────────────

def _extract_actor_keypoints(
    frame_assoc: FrameAssociation,
    actor_id: int,
    per_camera_detections: List = None,
    reference_camera: int = 0,
) -> Optional[np.ndarray]:
    """Extract body keypoints for an actor from the reference camera.

    Falls back to other cameras if reference camera has no detection for this actor.

    Args:
        frame_assoc: FrameAssociation for this frame
        actor_id: local actor ID in this frame
        per_camera_detections: List[List[PersonDetection]] for this frame (optional)
        reference_camera: camera index to prefer

    Returns:
        (K, 3) array of [x, y, score] for K visible body keypoints,
        or None if no keypoints found
    """
    if per_camera_detections is None:
        return None

    members = frame_assoc.actor_memberships.get(actor_id, [])
    if not members:
        return None

    # Try reference camera first
    for cam_idx, person_id in members:
        if cam_idx == reference_camera:
            det = per_camera_detections[cam_idx][person_id]
            body = det.keypoints_133[:17]  # first 17 = body
            visible = body[:, 2] > BODY_CONF_THRESHOLD
            if np.sum(visible) >= 2:
                return body[visible]

    # Fallback: any camera with enough keypoints
    for cam_idx, person_id in members:
        det = per_camera_detections[cam_idx][person_id]
        body = det.keypoints_133[:17]
        visible = body[:, 2] > BODY_CONF_THRESHOLD
        if np.sum(visible) >= 2:
            return body[visible]

    return None


def _extract_actor_bbox(
    frame_assoc: FrameAssociation,
    actor_id: int,
    per_camera_detections: List = None,
    reference_camera: int = 0,
) -> Optional[np.ndarray]:
    """Extract bounding box for an actor from the reference camera."""
    if per_camera_detections is None:
        return None

    members = frame_assoc.actor_memberships.get(actor_id, [])
    for cam_idx, person_id in members:
        if cam_idx == reference_camera:
            return per_camera_detections[cam_idx][person_id].bbox

    # Fallback
    for cam_idx, person_id in members:
        return per_camera_detections[cam_idx][person_id].bbox

    return None


def _normalize_keypoints(
    keypoints: np.ndarray, bbox: Optional[np.ndarray] = None
) -> np.ndarray:
    """Normalize keypoints to [0, 1] range relative to bounding box or image.

    If bbox is provided, normalize relative to bbox center and size.
    This makes the representation scale-invariant.

    Returns:
        Normalized (N, 2) array of [x, y] coordinates
    """
    pts = keypoints[:, :2].copy()

    if bbox is not None:
        # Normalize relative to bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = max(bbox[2] - bbox[0], 1.0)
        h = max(bbox[3] - bbox[1], 1.0)
        pts[:, 0] = (pts[:, 0] - cx) / w
        pts[:, 1] = (pts[:, 1] - cy) / h
    else:
        # Fallback: just return raw coordinates (less ideal)
        pass

    return pts


def _keypoint_distance(
    kpts_a: np.ndarray, kpts_b: np.ndarray
) -> float:
    """Compute normalized L2 distance between two keypoint sets.

    Both inputs should be (N, 2) normalized coordinates.
    Uses minimum distance assignment (Hungarian) to handle different N.

    Returns:
        Mean distance (lower = more similar)
    """
    N_a, N_b = kpts_a.shape[0], kpts_b.shape[0]

    if N_a == 0 or N_b == 0:
        return float('inf')

    # Build pairwise distance matrix
    dist_matrix = np.zeros((N_a, N_b))
    for i in range(N_a):
        for j in range(N_b):
            dist_matrix[i, j] = np.linalg.norm(kpts_a[i] - kpts_b[j])

    # Use Hungarian to find minimum assignment (only if all costs are finite)
    if not np.all(np.isfinite(dist_matrix)):
        return float('inf')

    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    if len(row_ind) == 0:
        return float('inf')

    return float(np.mean(dist_matrix[row_ind, col_ind]))


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ── Core tracker ───────────────────────────────────────────────────────

class TemporalTracker:
    """Link per-frame actor IDs across time for consistent identity.

    Args:
        max_lost_frames: frames before a track is terminated
        search_window: frames to search for re-identification
        max_keypoint_distance: max normalized keypoint distance for matching
        iou_weight: weight of IoU in combined cost
        keypoint_weight: weight of keypoint distance in combined cost
        reference_camera: camera to use as primary view for features
    """

    def __init__(
        self,
        max_lost_frames: int = MAX_LOST_FRAMES,
        search_window: int = SEARCH_WINDOW,
        max_keypoint_distance: float = MAX_KEYPOINT_DISTANCE,
        iou_weight: float = IOU_WEIGHT,
        keypoint_weight: float = KEYPOINT_WEIGHT,
        reference_camera: int = 0,
    ):
        self.max_lost_frames = max_lost_frames
        self.search_window = search_window
        self.max_keypoint_distance = max_keypoint_distance
        self.iou_weight = iou_weight
        self.keypoint_weight = keypoint_weight
        self.reference_camera = reference_camera

        self._tracks: List[TrackState] = []
        self._next_global_id = 0

        # Marker anchor mapping (Stage 7, Part D):
        # marker_id -> persistent_global_id. Once set, this mapping is stable
        # across frames and overrides geometric association.
        self._marker_anchor_map: Dict[int, int] = {}

    def _create_track(self, frame_idx: int) -> int:
        """Create a new track and return its global_id."""
        global_id = self._next_global_id
        self._next_global_id += 1
        self._tracks.append(TrackState(
            global_id=global_id,
            last_seen_frame=frame_idx,
        ))
        return global_id

    def _get_active_tracks(self, current_frame: int) -> List[TrackState]:
        """Get tracks that are active (not lost, or lost within search window)."""
        active = []
        for track in self._tracks:
            if track.is_lost:
                if (current_frame - track.lost_frame) <= self.search_window:
                    active.append(track)
            else:
                active.append(track)
        return active

    def _compute_match_cost(
        self,
        track: TrackState,
        actor_id: int,
        frame_assoc: FrameAssociation,
        per_camera_detections: List,
    ) -> float:
        """Compute cost of matching a track to an actor in the current frame.

        Lower cost = better match.
        """
        # Get current actor's keypoints
        cur_keypoints = _extract_actor_keypoints(
            frame_assoc, actor_id, per_camera_detections, self.reference_camera
        )
        cur_bbox = _extract_actor_bbox(
            frame_assoc, actor_id, per_camera_detections, self.reference_camera
        )

        if cur_keypoints is None or len(track.feature_history) == 0:
            return float('inf')

        # Normalize current keypoints
        cur_normalized = _normalize_keypoints(cur_keypoints, cur_bbox)

        # Compare to last known feature of the track
        last_feature = track.feature_history[-1]
        last_bbox = track.bbox_history[-1] if track.bbox_history else None

        # Keypoint distance
        kp_dist = _keypoint_distance(last_feature, cur_normalized)

        # IoU
        iou = 0.0
        if last_bbox is not None and cur_bbox is not None:
            iou = _compute_iou(last_bbox, cur_bbox)

        # Combined cost: keypoint distance penalized, IoU rewarded
        # cost = keypoint_weight * kp_dist - iou_weight * iou
        # (lower is better)
        cost = self.keypoint_weight * kp_dist - self.iou_weight * iou

        return cost

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
            marker_overrides: optional {marker_id: local_actor_id} from marker_fallback.
                When provided, these act as hard anchor assignments — the tracker
                trusts marker-based IDs over geometric association for these actors.
                Other actors (without markers) are still matched normally.

        Returns:
            TemporalAssociation mapping local actor IDs to persistent global IDs
        """
        local_actor_ids = list(frame_assoc.actor_memberships.keys())
        num_local = len(local_actor_ids)

        if num_local == 0:
            return TemporalAssociation(
                frame_idx=frame_idx,
                num_persistent_actors=0,
                local_to_persistent={},
                persistent_to_local={},
                active_tracks=[],
            )

        # First frame: create tracks for all actors
        if frame_idx == 0:
            local_to_persistent = {}
            for local_id in local_actor_ids:
                gid = self._create_track(frame_idx)
                local_to_persistent[local_id] = gid

                # Store feature
                kpts = _extract_actor_keypoints(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                bbox = _extract_actor_bbox(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                if kpts is not None:
                    self._tracks[-1].feature_history.append(_normalize_keypoints(kpts, bbox))
                self._tracks[-1].bbox_history.append(bbox)

            # Stage 7 Part D: register marker anchors on first frame
            if marker_overrides:
                for marker_id, local_id in marker_overrides.items():
                    if local_id in local_to_persistent:
                        self._marker_anchor_map[marker_id] = local_to_persistent[local_id]

            persistent_to_local = {v: k for k, v in local_to_persistent.items()}
            return TemporalAssociation(
                frame_idx=frame_idx,
                num_persistent_actors=num_local,
                local_to_persistent=local_to_persistent,
                persistent_to_local=persistent_to_local,
                active_tracks=list(local_to_persistent.values()),
            )

        # Subsequent frames: match existing tracks to local actors
        active_tracks = self._get_active_tracks(frame_idx)

        if not active_tracks:
            # No active tracks — create new ones for all actors
            local_to_persistent = {}
            for local_id in local_actor_ids:
                gid = self._create_track(frame_idx)
                local_to_persistent[local_id] = gid

                kpts = _extract_actor_keypoints(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                bbox = _extract_actor_bbox(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                if kpts is not None:
                    self._tracks[-1].feature_history.append(_normalize_keypoints(kpts, bbox))
                self._tracks[-1].bbox_history.append(bbox)

            persistent_to_local = {v: k for k, v in local_to_persistent.items()}
            return TemporalAssociation(
                frame_idx=frame_idx,
                num_persistent_actors=num_local,
                local_to_persistent=local_to_persistent,
                persistent_to_local=persistent_to_local,
                active_tracks=list(local_to_persistent.values()),
            )

        # Build cost matrix: rows = active tracks, cols = local actors
        N_tracks = len(active_tracks)
        N_actors = num_local

        cost_matrix = np.full((N_tracks, N_actors), np.inf)

        for i, track in enumerate(active_tracks):
            for j, local_id in enumerate(local_actor_ids):
                cost = self._compute_match_cost(
                    track, local_id, frame_assoc, per_camera_detections
                )
                cost_matrix[i, j] = cost

        # ── Stage 7, Part D: Marker-anchor override ────────────────────
        # If marker_overrides are provided, directly assign anchored actors
        # to their persistent tracks, bypassing Hungarian for those actors.
        # This is the second exception to "don't modify existing code"
        # (first: floor_plane.py shared_floor_data in Stage 5).
        # Modification is minimal: optional parameter, no signature change.
        anchored_actors = set()
        anchored_tracks = set()
        local_to_persistent = {}

        if marker_overrides:
            for marker_id, local_actor_id in marker_overrides.items():
                if local_actor_id not in local_actor_ids:
                    continue

                local_idx = local_actor_ids.index(local_actor_id)

                # Find or create persistent track for this marker
                if marker_id in self._marker_anchor_map:
                    target_gid = self._marker_anchor_map[marker_id]
                    # Find the track object
                    target_track = None
                    for t in self._tracks:
                        if t.global_id == target_gid:
                            target_track = t
                            break

                    if target_track is not None:
                        # Find this track in active_tracks
                        for i, t in enumerate(active_tracks):
                            if t.global_id == target_gid:
                                # Directly assign: cost = 0
                                cost_matrix[i, local_idx] = 0.0
                                anchored_actors.add(local_idx)
                                anchored_tracks.add(i)
                                break
                else:
                    # First time seeing this marker — create new track
                    gid = self._create_track(frame_idx)
                    self._marker_anchor_map[marker_id] = gid
                    local_to_persistent[local_actor_id] = gid
                    anchored_actors.add(local_idx)

                    kpts = _extract_actor_keypoints(
                        frame_assoc, local_actor_id, per_camera_detections, self.reference_camera
                    )
                    bbox = _extract_actor_bbox(
                        frame_assoc, local_actor_id, per_camera_detections, self.reference_camera
                    )
                    if kpts is not None:
                        self._tracks[-1].feature_history.append(_normalize_keypoints(kpts, bbox))
                    self._tracks[-1].bbox_history.append(bbox)

        # Hungarian assignment
        # Replace inf with large penalty for solver
        finite_mask = np.isfinite(cost_matrix)
        if not finite_mask.any():
            # No feasible matches — create new tracks for all
            local_to_persistent = {}
            for local_id in local_actor_ids:
                gid = self._create_track(frame_idx)
                local_to_persistent[local_id] = gid

                kpts = _extract_actor_keypoints(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                bbox = _extract_actor_bbox(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                if kpts is not None:
                    self._tracks[-1].feature_history.append(_normalize_keypoints(kpts, bbox))
                self._tracks[-1].bbox_history.append(bbox)

            persistent_to_local = {v: k for k, v in local_to_persistent.items()}
            return TemporalAssociation(
                frame_idx=frame_idx,
                num_persistent_actors=num_local,
                local_to_persistent=local_to_persistent,
                persistent_to_local=persistent_to_local,
                active_tracks=list(local_to_persistent.values()),
            )

        max_finite = cost_matrix[finite_mask].max() if finite_mask.any() else 1.0
        penalty = max(max_finite * 10, self.max_keypoint_distance * 2)
        cost_for_solver = np.where(finite_mask, cost_matrix, penalty)

        # Pad to square
        max_dim = max(N_tracks, N_actors)
        padded = np.full((max_dim, max_dim), penalty)
        padded[:N_tracks, :N_actors] = cost_for_solver

        row_ind, col_ind = linear_sum_assignment(padded)

        # Process assignments
        # (local_to_persistent may already have anchored entries from above)
        matched_tracks = set(anchored_tracks)
        matched_actors = set(anchored_actors)

        for r, c in zip(row_ind, col_ind):
            if r < N_tracks and c < N_actors:
                if c in anchored_actors:
                    continue  # already assigned by marker anchor
                if cost_matrix[r, c] < penalty:
                    # Valid match
                    track = active_tracks[r]
                    local_id = local_actor_ids[c]
                    local_to_persistent[local_id] = track.global_id
                    matched_tracks.add(r)
                    matched_actors.add(c)

                    # Update track
                    track.is_lost = False
                    track.lost_frame = None
                    track.last_seen_frame = frame_idx

                    kpts = _extract_actor_keypoints(
                        frame_assoc, local_id, per_camera_detections, self.reference_camera
                    )
                    bbox = _extract_actor_bbox(
                        frame_assoc, local_id, per_camera_detections, self.reference_camera
                    )
                    if kpts is not None:
                        track.feature_history.append(_normalize_keypoints(kpts, bbox))
                    track.bbox_history.append(bbox)

        # Also update anchored tracks (feature/bbox history)
        for local_idx in anchored_actors:
            local_id = local_actor_ids[local_idx]
            if local_id in local_to_persistent:
                gid = local_to_persistent[local_id]
                for t in self._tracks:
                    if t.global_id == gid:
                        t.is_lost = False
                        t.lost_frame = None
                        t.last_seen_frame = frame_idx
                        kpts = _extract_actor_keypoints(
                            frame_assoc, local_id, per_camera_detections, self.reference_camera
                        )
                        bbox = _extract_actor_bbox(
                            frame_assoc, local_id, per_camera_detections, self.reference_camera
                        )
                        if kpts is not None:
                            t.feature_history.append(_normalize_keypoints(kpts, bbox))
                        t.bbox_history.append(bbox)
                        break

        # Mark unmatched tracks as lost
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks and not track.is_lost:
                track.is_lost = True
                track.lost_frame = frame_idx

        # Terminate tracks that have been lost too long
        for track in self._tracks:
            if track.is_lost and track.lost_frame is not None:
                if (frame_idx - track.lost_frame) > self.max_lost_frames:
                    # Remove from active consideration (but don't delete, for history)
                    pass

        # Create new tracks for unmatched actors
        for c, local_id in enumerate(local_actor_ids):
            if c not in matched_actors:
                gid = self._create_track(frame_idx)
                local_to_persistent[local_id] = gid

                kpts = _extract_actor_keypoints(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                bbox = _extract_actor_bbox(
                    frame_assoc, local_id, per_camera_detections, self.reference_camera
                )
                if kpts is not None:
                    self._tracks[-1].feature_history.append(_normalize_keypoints(kpts, bbox))
                self._tracks[-1].bbox_history.append(bbox)

        persistent_to_local = {v: k for k, v in local_to_persistent.items()}
        active_ids = [t.global_id for t in self._tracks if not t.is_lost]

        return TemporalAssociation(
            frame_idx=frame_idx,
            num_persistent_actors=len(active_ids),
            local_to_persistent=local_to_persistent,
            persistent_to_local=persistent_to_local,
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

        Args:
            frame_associations: list of FrameAssociation, one per frame
            per_camera_per_frame_detections: optional nested list for feature extraction.
                Supports both formats:
                  - [camera][frame] -> List[PersonDetection]
                  - [frame][camera] -> List[PersonDetection]
            marker_anchors: optional per-frame marker overrides.
                List of dicts: [frame_idx] -> {marker_id: local_actor_id}.
                None or empty list means no marker-based anchoring.

        Returns:
            List of TemporalAssociation, one per frame
        """
        results = []

        # Auto-detect format: [camera][frame] vs [frame][camera]
        # If first element's first element is a list of lists, it's [frame][camera]
        cam_frame_format = True
        if per_camera_per_frame_detections is not None and len(per_camera_per_frame_detections) > 0:
            first = per_camera_per_frame_detections[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
                # Check if it looks like [frame][camera] (first element has same
                # length as number of frame_associations) vs [camera][frame]
                if len(per_camera_per_frame_detections) == len(frame_associations):
                    cam_frame_format = False  # likely [frame][camera]

        for frame_idx, frame_assoc in enumerate(frame_associations):
            # Extract per-frame detections if available
            per_cam_dets = None
            if per_camera_per_frame_detections is not None:
                if cam_frame_format:
                    # [camera][frame] format
                    num_cams = len(per_camera_per_frame_detections)
                    per_cam_dets = [
                        per_camera_per_frame_detections[cam][frame_idx]
                        for cam in range(num_cams)
                    ]
                else:
                    # [frame][camera] format
                    per_cam_dets = per_camera_per_frame_detections[frame_idx]

            # Extract per-frame marker overrides if available
            frame_marker_overrides = None
            if marker_anchors and frame_idx < len(marker_anchors):
                frame_marker_overrides = marker_anchors[frame_idx]

            result = self.track_frame(
                frame_idx, frame_assoc, per_cam_dets, frame_marker_overrides
            )
            results.append(result)

        return results

    def get_track_summary(self) -> dict:
        """Get summary of all tracks created during tracking."""
        active = sum(1 for t in self._tracks if not t.is_lost)
        lost = sum(1 for t in self._tracks if t.is_lost)

        track_durations = []
        for t in self._tracks:
            if t.feature_history:
                track_durations.append(len(t.feature_history))

        return {
            "total_tracks_created": len(self._tracks),
            "active_tracks": active,
            "lost_tracks": lost,
            "track_durations": track_durations,
            "mean_duration": float(np.mean(track_durations)) if track_durations else 0.0,
            "max_duration": max(track_durations) if track_durations else 0,
        }


# ── Convenience function ───────────────────────────────────────────────

def run_temporal_tracking(
    frame_associations: List[FrameAssociation],
    per_camera_per_frame_detections: Optional[List[List[List]]] = None,
    marker_anchors: Optional[List[Dict[int, int]]] = None,
    **kwargs,
) -> Tuple[List[TemporalAssociation], TemporalTracker]:
    """Run temporal tracking on Stage 2 results.

    Args:
        frame_associations: output from CrossViewAssociator.associate_video()
        per_camera_per_frame_detections: optional detections for feature extraction
        marker_anchors: optional per-frame marker overrides for anchor frames
        **kwargs: passed to TemporalTracker constructor

    Returns:
        (temporal_results, tracker) tuple
    """
    tracker = TemporalTracker(**kwargs)
    results = tracker.track(
        frame_associations, per_camera_per_frame_detections, marker_anchors
    )
    return results, tracker
