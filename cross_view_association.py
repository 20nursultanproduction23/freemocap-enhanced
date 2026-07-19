"""
Cross-View Association — Stage 2 of multi-actor pipeline

Associates person detections across multiple cameras using epipolar geometry
and the Hungarian algorithm.

Given per-camera detections from Stage 1 (MultiPersonDetector), this module
determines which person_id in camera 0 corresponds to which person_id in
camera 1, 2, ..., N-1. The output is a global actor ID for each detection.

Approach:
  1. Load calibration data (camera intrinsics + extrinsics)
  2. For each camera pair, compute the fundamental matrix F
  3. For each pair of detected persons across cameras, compute mean epipolar
     distance across their visible body keypoints
  4. Use scipy.optimize.linear_sum_assignment (Hungarian) to find the
     optimal assignment that minimizes total epipolar distance
  5. Merge assignments across all camera pairs to build global actor IDs
     using union-find (disjoint set) data structure

Key points for association:
  Body keypoints (17 COCO keypoints) are used because they are visible
  from most viewing angles. Keypoints with confidence > 0.3 are included.
  The median epipolar distance across visible keypoints is used per person pair.

Input:
  - Per-camera List[PersonDetection] from Stage 1
  - Calibration data (CameraGroup from anipose, or raw matrices)

Output:
  - FrameAssociation with global actor IDs for each detection

Usage:
    associator = CrossViewAssociator.from_camera_group(camera_group)
    frame_result = associator.associate_frame(per_camera_detections)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import cv2
from scipy.optimize import linear_sum_assignment

from multiperson_detector import PersonDetection


# ── Body keypoint indices (COCO 17) used for epipolar matching ────────
# These are the most reliable keypoints visible from multiple angles:
#   0=nose, 5=left_shoulder, 6=right_shoulder,
#   11=left_hip, 12=right_hip
# Other body keypoints can be noisy or occluded.
BODY_KEYPOINT_INDICES_FOR_MATCHING = [0, 5, 6, 11, 12]
MIN_KEYPOINTS_FOR_MATCHING = 2  # minimum visible keypoints to compute distance
EPIPOLAR_CONFIDENCE_THRESHOLD = 0.3  # minimum keypoint confidence to use


# ── Union-Find for merging assignments ────────────────────────────────
class _UnionFind:
    """Disjoint set (union-find) data structure for merging actor IDs."""

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


# ── Data structures ───────────────────────────────────────────────────
@dataclass
class CameraPairAssignment:
    """Result of associating detections between two cameras for one frame."""
    camera_a_idx: int
    camera_b_idx: int
    assignments: Dict[int, int]  # person_id_a -> person_id_b (-1 = unmatched)
    cost_matrix: np.ndarray      # (N_a, N_b) raw epipolar distances
    total_cost: float            # sum of matched costs
    matched: bool                # whether assignment was successful


@dataclass
class FrameAssociation:
    """Global association for one frame across all cameras."""
    frame_idx: int
    num_cameras: int
    num_actors: int
    # (camera_idx, person_id) -> global_actor_id
    detection_to_actor: Dict[Tuple[int, int], int]
    # Per camera-pair results (for diagnostics)
    pair_assignments: List[CameraPairAssignment] = field(default_factory=list)
    # For each actor: list of (camera_idx, person_id) contributing to it
    actor_memberships: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


# ── Core class ────────────────────────────────────────────────────────
class CrossViewAssociator:
    """Associate person detections across cameras using epipolar geometry.

    Uses the fundamental matrix derived from camera calibration to compute
    epipolar distances between detected person pairs across camera views.
    The Hungarian algorithm finds the optimal assignment.

    Args:
        camera_matrices: list of (3, 3) intrinsic matrices, one per camera
        extrinsic_matrices: list of (3, 4) extrinsic matrices [R|t], one per camera
        image_sizes: list of (width, height) tuples, one per camera
        max_epipolar_distance: maximum median epipolar distance (pixels) to accept
            an association. Pairs with higher distance are considered unmatched.
    """

    def __init__(
        self,
        camera_matrices: List[np.ndarray],
        extrinsic_matrices: List[np.ndarray],
        image_sizes: List[Tuple[int, int]],
        max_epipolar_distance: float = 50.0,
    ):
        if len(camera_matrices) != len(extrinsic_matrices):
            raise ValueError("camera_matrices and extrinsic_matrices must have same length")

        self.num_cameras = len(camera_matrices)
        self.camera_matrices = camera_matrices
        self.extrinsic_matrices = extrinsic_matrices
        self.image_sizes = image_sizes
        self.max_epipolar_distance = max_epipolar_distance

        # Precompute fundamental matrices for all camera pairs
        self._fundamental_matrices: Dict[Tuple[int, int], np.ndarray] = {}
        self._precompute_fundamentals()

    @classmethod
    def from_camera_group(cls, camera_group, max_epipolar_distance: float = 50.0):
        """Create associator from anipose CameraGroup object.

        Args:
            camera_group: aniposelib CameraGroup with calibrated cameras
            max_epipolar_distance: max epipolar distance threshold (pixels)
        """
        camera_matrices = []
        extrinsic_matrices = []
        image_sizes = []

        for cam in camera_group.cameras:
            camera_matrices.append(cam.matrix.copy())
            extrinsic_matrices.append(cam.get_extrinsics_mat()[:3].copy())
            image_sizes.append(cam.size)

        return cls(camera_matrices, extrinsic_matrices, image_sizes, max_epipolar_distance)

    def _precompute_fundamentals(self):
        """Precompute fundamental matrix for every camera pair."""
        for i in range(self.num_cameras):
            for j in range(i + 1, self.num_cameras):
                F = self._compute_fundamental_matrix(i, j)
                self._fundamental_matrices[(i, j)] = F
                self._fundamental_matrices[(j, i)] = F.T

    def _compute_fundamental_matrix(self, cam_a: int, cam_b: int) -> np.ndarray:
        """Compute fundamental matrix F from camera A to camera B.

        F maps a point in camera A to an epipolar line in camera B:
            l_B = F * p_A

        Given calibrated cameras with K, R, t (world-to-camera):
            F = K_B^{-T} * E * K_A^{-1}
            E = [t_rel]_x * R_rel

        Where R_rel, t_rel are the relative rotation/translation from A to B.
        """
        K_A = self.camera_matrices[cam_a]
        K_B = self.camera_matrices[cam_b]
        E_A = self.extrinsic_matrices[cam_a]  # (3, 4) = [R_A | t_A]
        E_B = self.extrinsic_matrices[cam_b]  # (3, 4) = [R_B | t_B]

        R_A = E_A[:, :3]
        t_A = E_A[:, 3]
        R_B = E_B[:, :3]
        t_B = E_B[:, 3]

        # Relative rotation and translation from camera A to camera B
        R_rel = R_B @ R_A.T
        t_rel = t_B - R_rel @ t_A

        # Essential matrix: E = [t_rel]_x * R_rel
        t_rel_skew = np.array([
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0],
        ])
        E = t_rel_skew @ R_rel

        # Fundamental matrix: F = K_B^{-T} * E * K_A^{-1}
        F = np.linalg.inv(K_B).T @ E @ np.linalg.inv(K_A)

        # Normalize F so that ||F|| = 1
        norm = np.linalg.norm(F)
        if norm > 0:
            F /= norm

        return F

    def _get_visible_keypoints(
        self, detection: PersonDetection, cam_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract visible body keypoints for a detection.

        Returns:
            points: (N, 2) array of pixel coordinates [x, y]
            scores: (N,) array of confidence scores
        """
        # Body keypoints are indices 0:16 in the 133-keypoint format
        body_kpts = detection.keypoints_133[:17]  # (17, 3) [x, y, score]

        visible_mask = body_kpts[:, 2] > EPIPOLAR_CONFIDENCE_THRESHOLD
        # Also filter to the specific matching keypoints
        matching_mask = np.zeros(17, dtype=bool)
        for idx in BODY_KEYPOINT_INDICES_FOR_MATCHING:
            if idx < 17:
                matching_mask[idx] = True

        combined_mask = visible_mask & matching_mask

        points = body_kpts[combined_mask, :2]
        scores = body_kpts[combined_mask, 2]

        return points, scores

    def _compute_epipolar_distance(
        self, points_a: np.ndarray, F_ab: np.ndarray, points_b: np.ndarray
    ) -> np.ndarray:
        """Compute point-to-epipolar-line distance.

        For each point pair (p_a, p_b), computes the distance from p_b
        to the epipolar line defined by F_ab * p_a.

        Args:
            points_a: (N, 2) points in camera A
            F_ab: (3, 3) fundamental matrix A -> B
            points_b: (N, 2) corresponding points in camera B

        Returns:
            distances: (N,) epipolar distances in pixels
        """
        # Convert to homogeneous coordinates
        N = points_a.shape[0]
        p_a_h = np.column_stack([points_a, np.ones(N)])  # (N, 3)
        p_b_h = np.column_stack([points_b, np.ones(N)])  # (N, 3)

        # Epipolar lines in camera B: l = F * p_a, shape (N, 3)
        lines_b = p_a_h @ F_ab.T  # (N, 3)

        # Point-to-line distance: |p_b^T * l| / sqrt(l[0]^2 + l[1]^2)
        numerator = np.abs(np.sum(p_b_h * lines_b, axis=1))
        denominator = np.sqrt(lines_b[:, 0]**2 + lines_b[:, 1]**2)

        # Avoid division by zero
        distances = np.where(denominator > 1e-10, numerator / denominator, np.inf)

        return distances

    def compute_pair_cost_matrix(
        self,
        detections_a: List[PersonDetection],
        detections_b: List[PersonDetection],
        cam_a: int,
        cam_b: int,
    ) -> np.ndarray:
        """Compute epipolar distance cost matrix between two cameras.

        For each person pair (i in camera A, j in camera B), computes the
        median epipolar distance across visible shared body keypoints.

        Args:
            detections_a: person detections from camera A
            detections_b: person detections from camera B
            cam_a: camera A index
            cam_b: camera B index

        Returns:
            cost_matrix: (N_a, N_b) array of median epipolar distances
                Higher values = less likely to be the same person
        """
        N_a = len(detections_a)
        N_b = len(detections_b)

        if N_a == 0 or N_b == 0:
            return np.full((max(N_a, 1), max(N_b, 1)), np.inf)

        # Get fundamental matrix
        key = (min(cam_a, cam_b), max(cam_a, cam_b))
        F_ab = self._fundamental_matrices[key]
        if cam_a > cam_b:
            F_ab = F_ab.T  # Transpose if order is reversed

        cost_matrix = np.full((N_a, N_b), np.inf)

        for i, det_a in enumerate(detections_a):
            pts_a, scores_a = self._get_visible_keypoints(det_a, cam_a)
            if len(pts_a) < MIN_KEYPOINTS_FOR_MATCHING:
                continue

            for j, det_b in enumerate(detections_b):
                pts_b, scores_b = self._get_visible_keypoints(det_b, cam_b)
                if len(pts_b) < MIN_KEYPOINTS_FOR_MATCHING:
                    continue

                # Find common keypoint indices between the two detections
                # (both must have the same body keypoint visible)
                # For simplicity, use all visible keypoints from both sides
                # and match by index in the body keypoint set

                # Build a set of keypoint indices visible in both
                # (use the matching keypoint indices for robustness)
                distances = []
                for kp_idx in BODY_KEYPOINT_INDICES_FOR_MATCHING:
                    if kp_idx >= len(det_a.keypoints_133) or kp_idx >= len(det_b.keypoints_133):
                        continue
                    conf_a = det_a.keypoints_133[kp_idx, 2]
                    conf_b = det_b.keypoints_133[kp_idx, 2]
                    if conf_a < EPIPOLAR_CONFIDENCE_THRESHOLD or conf_b < EPIPOLAR_CONFIDENCE_THRESHOLD:
                        continue

                    pt_a = det_a.keypoints_133[kp_idx, :2].reshape(1, 2)
                    pt_b = det_b.keypoints_133[kp_idx, :2].reshape(1, 2)
                    dist = self._compute_epipolar_distance(pt_a, F_ab, pt_b)
                    distances.append(dist[0])

                if len(distances) >= MIN_KEYPOINTS_FOR_MATCHING:
                    # Use median distance (robust to outlier keypoints)
                    cost_matrix[i, j] = float(np.median(distances))

        return cost_matrix

    def associate_frame(
        self,
        per_camera_detections: List[List[PersonDetection]],
        frame_idx: int = 0,
    ) -> FrameAssociation:
        """Associate detections across all cameras for one frame.

        Args:
            per_camera_detections: list of List[PersonDetection], one per camera.
                Length must equal num_cameras.
            frame_idx: frame index (for labeling in the result)

        Returns:
            FrameAssociation with global actor IDs
        """
        if len(per_camera_detections) != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} camera detection lists, "
                f"got {len(per_camera_detections)}"
            )

        # Step 1: Compute pair-wise assignments using Hungarian algorithm
        pair_assignments = []
        for i in range(self.num_cameras):
            for j in range(i + 1, self.num_cameras):
                dets_i = per_camera_detections[i]
                dets_j = per_camera_detections[j]

                if len(dets_i) == 0 or len(dets_j) == 0:
                    continue

                cost_matrix = self.compute_pair_cost_matrix(dets_i, dets_j, i, j)
                assignment = self._hungarian_assign(cost_matrix, i, j)
                pair_assignments.append(assignment)

        # Step 2: Merge assignments using union-find to get global actor IDs
        # Create a mapping from (camera_idx, person_id) to a flat index
        flat_index = {}
        reverse_flat = {}
        idx = 0
        for cam_idx, dets in enumerate(per_camera_detections):
            for det in dets:
                flat_index[(cam_idx, det.person_id)] = idx
                reverse_flat[idx] = (cam_idx, det.person_id)
                idx += 1

        if idx == 0:
            return FrameAssociation(
                frame_idx=frame_idx,
                num_cameras=self.num_cameras,
                num_actors=0,
                detection_to_actor={},
                pair_assignments=pair_assignments,
                actor_memberships={},
            )

        uf = _UnionFind(idx)

        for pa in pair_assignments:
            for pid_a, pid_b in pa.assignments.items():
                if pid_b == -1:
                    continue  # unmatched
                key_a = (pa.camera_a_idx, pid_a)
                key_b = (pa.camera_b_idx, pid_b)
                if key_a in flat_index and key_b in flat_index:
                    uf.union(flat_index[key_a], flat_index[key_b])

        # Step 3: Assign global actor IDs based on union-find groups
        # Each root becomes a global actor ID
        root_to_actor = {}
        detection_to_actor = {}
        actor_memberships = {}
        next_actor_id = 0

        for flat_idx in range(idx):
            root = uf.find(flat_idx)
            if root not in root_to_actor:
                root_to_actor[root] = next_actor_id
                next_actor_id += 1
            actor_id = root_to_actor[root]
            cam_per, pid_per = reverse_flat[flat_idx]
            detection_to_actor[(cam_per, pid_per)] = actor_id

            if actor_id not in actor_memberships:
                actor_memberships[actor_id] = []
            actor_memberships[actor_id].append((cam_per, pid_per))

        return FrameAssociation(
            frame_idx=frame_idx,
            num_cameras=self.num_cameras,
            num_actors=next_actor_id,
            detection_to_actor=detection_to_actor,
            pair_assignments=pair_assignments,
            actor_memberships=actor_memberships,
        )

    def _hungarian_assign(
        self,
        cost_matrix: np.ndarray,
        cam_a: int,
        cam_b: int,
    ) -> CameraPairAssignment:
        """Run Hungarian algorithm on cost matrix.

        If the cost matrix has infinite entries, handles gracefully:
        - For all-infinite rows/columns, leave unmatched
        """
        N_a, N_b = cost_matrix.shape

        # Replace inf with a large but finite value for the solver
        finite_mask = np.isfinite(cost_matrix)
        if not finite_mask.any():
            # No valid pairs at all
            return CameraPairAssignment(
                camera_a_idx=cam_a,
                camera_b_idx=cam_b,
                assignments={i: -1 for i in range(N_a)},
                cost_matrix=cost_matrix,
                total_cost=0.0,
                matched=False,
            )

        # Use a large penalty for inf entries
        max_finite = cost_matrix[finite_mask].max() if finite_mask.any() else 1.0
        penalty = max(max_finite * 100, self.max_epipolar_distance * 100)
        cost_for_solver = np.where(finite_mask, cost_matrix, penalty)

        # Ensure square matrix for Hungarian
        max_dim = max(N_a, N_b)
        padded = np.full((max_dim, max_dim), penalty)
        padded[:N_a, :N_b] = cost_for_solver

        row_ind, col_ind = linear_sum_assignment(padded)

        # Build assignment dict (only valid matches within bounds and below threshold)
        assignments = {}
        total_cost = 0.0
        for i in range(N_a):
            assignments[i] = -1  # default: unmatched

        for r, c in zip(row_ind, col_ind):
            if r < N_a and c < N_b:
                if cost_matrix[r, c] <= self.max_epipolar_distance:
                    assignments[r] = c
                    total_cost += cost_matrix[r, c]

        matched = any(v != -1 for v in assignments.values())

        return CameraPairAssignment(
            camera_a_idx=cam_a,
            camera_b_idx=cam_b,
            assignments=assignments,
            cost_matrix=cost_matrix,
            total_cost=total_cost,
            matched=matched,
        )

    def associate_video(
        self,
        per_camera_per_frame_detections: List[List[List[PersonDetection]]],
    ) -> List[FrameAssociation]:
        """Associate detections across all cameras for all frames.

        Args:
            per_camera_per_frame_detections: list of per-camera detections.
                Outer list: cameras (length = num_cameras)
                Middle list: frames
                Inner list: PersonDetections for that frame

        Returns:
            List of FrameAssociation, one per frame
        """
        if len(per_camera_per_frame_detections) != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, "
                f"got {len(per_camera_per_frame_detections)}"
            )

        num_frames = len(per_camera_per_frame_detections[0])
        results = []

        for frame_idx in range(num_frames):
            frame_dets = [
                per_camera_per_frame_detections[cam][frame_idx]
                for cam in range(self.num_cameras)
            ]
            result = self.associate_frame(frame_dets, frame_idx)
            results.append(result)

        return results
