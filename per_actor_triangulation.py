"""
Per-Actor Triangulation — Stage 4 of multi-actor pipeline

Triangulates 2D detections into 3D skeletons separately for each actor,
using the temporal tracking output from Stage 3 to maintain identity.

Data flow:
  Stage 1: MultiPersonDetector → per-camera per-frame List[PersonDetection]
  Stage 2: CrossViewAssociator → per-frame FrameAssociation (global actor IDs)
  Stage 3: TemporalTracker → per-frame TemporalAssociation (persistent IDs)
  Stage 4: THIS MODULE → per-actor 3D skeleton arrays

Input:
  - per_camera_per_frame_detections: [camera][frame] → List[PersonDetection]
  - frame_associations: [frame] → FrameAssociation
  - temporal_associations: [frame] → TemporalAssociation
  - calibration: dict with camera_matrices, extrinsic_matrices

Output:
  Dict[persistent_actor_id, dict]:
    - "skeleton_3d": (numFrames, numKeypoints, 3) array
    - "reprojection_errors": (numFrames, numKeypoints) array
    - "num_visible_frames": int
    - "mean_reprojection_error": float

Usage:
    from per_actor_triangulation import PerActorTriangulator
    from calibration_loader import load_calibration_toml

    calib = load_calibration_toml("calibration.toml")
    triangulator = PerActorTriangulator(calib)
    actors = triangulator.triangulate_all(
        per_camera_per_frame_detections,
        frame_associations,
        temporal_associations,
    )
    for actor_id, data in actors.items():
        print(f"Actor {actor_id}: {data['num_visible_frames']} frames, "
              f"reprojection error {data['mean_reprojection_error']:.4f}")
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from cross_view_association import FrameAssociation
from temporal_tracker import TemporalAssociation
from multiperson_detector import PersonDetection
from rtmpose_triangulation import dlt_triangulate_single_point, reproject_to_2d


# Number of keypoints in RTMPose wholebody format
NUM_KEYPOINTS = 133


@dataclass
class ActorTriangulationResult:
    """3D triangulation result for a single actor."""
    persistent_actor_id: int
    num_frames: int
    skeleton_3d: np.ndarray          # (numFrames, 133, 3) — 3D coordinates
    reprojection_errors: np.ndarray   # (numFrames, 133) — per-keypoint error
    visible_frames: np.ndarray        # (numFrames,) — boolean mask
    mean_reprojection_error: float
    mean_visible_kpts_per_frame: float


class PerActorTriangulator:
    """Triangulate 3D skeletons separately for each actor.

    Uses DLT (Direct Linear Transform) triangulation with confidence-weighted
    camera observations. For each actor in each frame, collects 2D keypoints
    from all cameras where that actor is detected, then triangulates to 3D.

    Args:
        calibration: dict from load_calibration_toml() with:
            camera_matrices: List[(3,3)] intrinsic K matrices
            extrinsic_matrices: List[(3,4)] [R|t] matrices
            image_sizes: List[(width, height)] tuples
        min_cams: minimum cameras for triangulation (default 2)
    """

    def __init__(
        self,
        calibration: dict,
        min_cams: int = 2,
    ):
        self.camera_matrices = calibration["camera_matrices"]
        self.extrinsic_matrices = calibration["extrinsic_matrices"]
        self.image_sizes = calibration["image_sizes"]
        self.num_cameras = len(self.camera_matrices)
        self.min_cams = min_cams

        # Build projection matrices: P = K @ [R|t]
        self.projection_matrices = []
        for i in range(self.num_cameras):
            K = self.camera_matrices[i]
            E = self.extrinsic_matrices[i]  # (3, 4) = [R|t]
            P = K @ E  # (3, 4) projection matrix
            self.projection_matrices.append(P)

    def _collect_actor_keypoints(
        self,
        frame_idx: int,
        local_actor_id: int,
        frame_assoc: FrameAssociation,
        per_camera_detections: List[List[PersonDetection]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect 2D keypoints for one actor across all cameras.

        For cameras where the actor is detected, uses their keypoints.
        For cameras where the actor is NOT detected, fills with NaN.

        Args:
            frame_idx: frame index (for diagnostics)
            local_actor_id: actor ID in this frame's FrameAssociation
            frame_assoc: FrameAssociation for this frame
            per_camera_detections: List[List[PersonDetection]] per camera

        Returns:
            keypoints: (num_cameras, 133, 2) — 2D pixel coordinates (NaN if missing)
            scores: (num_cameras, 133) — confidence scores (0 if missing)
        """
        keypoints = np.full((self.num_cameras, NUM_KEYPOINTS, 2), np.nan, dtype=np.float64)
        scores = np.zeros((self.num_cameras, NUM_KEYPOINTS), dtype=np.float64)

        members = frame_assoc.actor_memberships.get(local_actor_id, [])

        for cam_idx, person_id in members:
            if cam_idx >= self.num_cameras:
                continue
            if person_id >= len(per_camera_detections[cam_idx]):
                continue

            det = per_camera_detections[cam_idx][person_id]
            kpts = det.keypoints_133  # (133, 3) — [x, y, score]

            keypoints[cam_idx, :, 0] = kpts[:, 0]  # x
            keypoints[cam_idx, :, 1] = kpts[:, 1]  # y
            scores[cam_idx, :] = kpts[:, 2]         # confidence

        return keypoints, scores

    def triangulate_actor_frame(
        self,
        local_actor_id: int,
        frame_idx: int,
        frame_assoc: FrameAssociation,
        per_camera_detections: List[List[PersonDetection]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Triangulate one actor in one frame.

        Args:
            local_actor_id: actor ID in this frame
            frame_idx: frame index
            frame_assoc: FrameAssociation for this frame
            per_camera_detections: per-camera detections

        Returns:
            skeleton_3d: (133, 3) triangulated 3D coordinates (NaN if insufficient cameras)
            reproj_errors: (133,) reprojection errors per keypoint
        """
        keypoints_2d, scores = self._collect_actor_keypoints(
            frame_idx, local_actor_id, frame_assoc, per_camera_detections
        )

        skeleton_3d = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
        reproj_errors = np.full(NUM_KEYPOINTS, np.nan, dtype=np.float64)

        for k in range(NUM_KEYPOINTS):
            valid_cam_indices = []
            pts_2d = []
            cam_weights = []

            for cam_idx in range(self.num_cameras):
                pt = keypoints_2d[cam_idx, k]
                if not np.any(np.isnan(pt)):
                    valid_cam_indices.append(cam_idx)
                    pts_2d.append(pt)
                    cam_weights.append(float(scores[cam_idx, k]))

            if len(valid_cam_indices) < self.min_cams:
                continue

            pts_2d_arr = np.array(pts_2d, dtype=np.float64)
            weights_arr = np.array(cam_weights, dtype=np.float64)
            valid_proj = [self.projection_matrices[i] for i in valid_cam_indices]

            pt_3d = dlt_triangulate_single_point(pts_2d_arr, valid_proj, weights_arr)

            if not np.any(np.isnan(pt_3d)):
                skeleton_3d[k] = pt_3d

                # Compute reprojection error
                reproj = reproject_to_2d(
                    pt_3d.reshape(1, 3),
                    self.projection_matrices[valid_cam_indices[0]]
                )
                diff = reproj - pts_2d_arr[0:1]
                reproj_errors[k] = float(np.sqrt(np.sum(diff ** 2)))

        return skeleton_3d, reproj_errors

    def triangulate_actor(
        self,
        persistent_actor_id: int,
        per_camera_per_frame_detections: List[List[List[PersonDetection]]],
        frame_associations: List[FrameAssociation],
        temporal_associations: List[TemporalAssociation],
    ) -> ActorTriangulationResult:
        """Triangulate one actor across all frames.

        Args:
            persistent_actor_id: the persistent global ID of the actor
            per_camera_per_frame_detections: [camera][frame] → List[PersonDetection]
            frame_associations: list of FrameAssociation, one per frame
            temporal_associations: list of TemporalAssociation, one per frame

        Returns:
            ActorTriangulationResult with 3D skeleton and error metrics
        """
        num_frames = len(frame_associations)
        skeleton_3d = np.full((num_frames, NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
        reproj_errors = np.full((num_frames, NUM_KEYPOINTS), np.nan, dtype=np.float64)
        visible_mask = np.zeros(num_frames, dtype=bool)

        for frame_idx in range(num_frames):
            ta = temporal_associations[frame_idx]
            fa = frame_associations[frame_idx]

            # Find local actor ID for this persistent ID in this frame
            local_id = ta.persistent_to_local.get(persistent_actor_id)
            if local_id is None:
                continue  # Actor not visible in this frame

            # Check that this local actor exists in the frame association
            if local_id not in fa.actor_memberships:
                continue

            visible_mask[frame_idx] = True

            # Collect per-camera detections for this frame
            per_cam_dets = [
                per_camera_per_frame_detections[cam][frame_idx]
                for cam in range(self.num_cameras)
            ]

            skel, errors = self.triangulate_actor_frame(
                local_id, frame_idx, fa, per_cam_dets
            )
            skeleton_3d[frame_idx] = skel
            reproj_errors[frame_idx] = errors

        # Compute metrics
        num_visible = int(np.sum(visible_mask))
        valid_errors = reproj_errors[visible_mask]
        valid_errors = valid_errors[np.isfinite(valid_errors)]
        mean_reproj = float(np.mean(valid_errors)) if len(valid_errors) > 0 else float('nan')

        # Mean visible keypoints per visible frame
        if num_visible > 0:
            kpt_counts = []
            for f in range(num_frames):
                if visible_mask[f]:
                    valid_kpts = np.sum(np.isfinite(skeleton_3d[f, :, 0]))
                    kpt_counts.append(valid_kpts)
            mean_kpts = float(np.mean(kpt_counts))
        else:
            mean_kpts = 0.0

        return ActorTriangulationResult(
            persistent_actor_id=persistent_actor_id,
            num_frames=num_frames,
            skeleton_3d=skeleton_3d,
            reprojection_errors=reproj_errors,
            visible_frames=visible_mask,
            mean_reprojection_error=mean_reproj,
            mean_visible_kpts_per_frame=mean_kpts,
        )

    def triangulate_all(
        self,
        per_camera_per_frame_detections: List[List[List[PersonDetection]]],
        frame_associations: List[FrameAssociation],
        temporal_associations: List[TemporalAssociation],
    ) -> Dict[int, ActorTriangulationResult]:
        """Triangulate all actors across all frames.

        Discovers all unique persistent actor IDs from temporal_associations,
        then triangulates each one separately.

        Args:
            per_camera_per_frame_detections: [camera][frame] → List[PersonDetection]
            frame_associations: list of FrameAssociation, one per frame
            temporal_associations: list of TemporalAssociation, one per frame

        Returns:
            Dict mapping persistent_actor_id → ActorTriangulationResult
        """
        # Discover all persistent actor IDs
        all_persistent_ids = set()
        for ta in temporal_associations:
            all_persistent_ids.update(ta.local_to_persistent.values())

        print(f"Triangulating {len(all_persistent_ids)} actor(s): {sorted(all_persistent_ids)}")

        results = {}
        for pid in sorted(all_persistent_ids):
            result = self.triangulate_actor(
                pid,
                per_camera_per_frame_detections,
                frame_associations,
                temporal_associations,
            )
            results[pid] = result

            visible_pct = 100.0 * np.sum(result.visible_frames) / max(result.num_frames, 1)
            print(f"  Actor {pid}: {np.sum(result.visible_frames)}/{result.num_frames} frames "
                  f"({visible_pct:.0f}%), "
                  f"reproj err={result.mean_reprojection_error:.2f}px, "
                  f"mean visible kpts={result.mean_visible_kpts_per_frame:.0f}")

        return results


def triangulate_multi_actor(
    per_camera_per_frame_detections: List[List[List[PersonDetection]]],
    frame_associations: List[FrameAssociation],
    temporal_associations: List[TemporalAssociation],
    calibration: dict,
    min_cams: int = 2,
) -> Dict[int, ActorTriangulationResult]:
    """Convenience function for multi-actor triangulation.

    Args:
        per_camera_per_frame_detections: [camera][frame] → List[PersonDetection]
        frame_associations: list of FrameAssociation, one per frame
        temporal_associations: list of TemporalAssociation, one per frame
        calibration: dict from load_calibration_toml()
        min_cams: minimum cameras for triangulation

    Returns:
        Dict mapping persistent_actor_id → ActorTriangulationResult
    """
    triangulator = PerActorTriangulator(calibration, min_cams=min_cams)
    return triangulator.triangulate_all(
        per_camera_per_frame_detections,
        frame_associations,
        temporal_associations,
    )
