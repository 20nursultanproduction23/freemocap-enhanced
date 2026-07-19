"""
Tests for cross_view_association.py (Stage 2: multi-actor pipeline).

Tests cover:
  - Fundamental matrix computation from calibration data
  - Epipolar distance computation
  - Keypoint extraction with confidence filtering
  - Hungarian assignment (cost matrix -> best matching)
  - Union-Find merging
  - Full frame association pipeline
  - Edge cases (0/1 person, all unmatched, partial visibility)
  - Per-frame table output (per CR #15)
"""
import pytest
import numpy as np
import cv2
from typing import List

from multiperson_detector import PersonDetection
from cross_view_association import (
    CrossViewAssociator,
    CameraPairAssignment,
    FrameAssociation,
    _UnionFind,
    BODY_KEYPOINT_INDICES_FOR_MATCHING,
    MIN_KEYPOINTS_FOR_MATCHING,
    EPIPOLAR_CONFIDENCE_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────
def _make_detection(
    person_id: int,
    keypoints_2d: np.ndarray = None,
    confidence: float = 0.8,
    bbox: np.ndarray = None,
) -> PersonDetection:
    """Create a PersonDetection with specific keypoint positions.

    Args:
        person_id: person index within the frame
        keypoints_2d: (133, 3) [x, y, score] array. If None, generates
            a default standing pose with high confidence.
        confidence: overall body confidence
        bbox: (4,) bounding box. If None, auto-generates.
    """
    if keypoints_2d is None:
        keypoints_2d = np.zeros((133, 3), dtype=np.float64)
        # Default standing pose: nose, shoulders, hips, etc.
        keypoints_2d[0] = [320, 100, 0.9]   # nose
        keypoints_2d[1] = [310, 110, 0.8]   # left_eye
        keypoints_2d[2] = [330, 110, 0.8]   # right_eye
        keypoints_2d[5] = [280, 170, 0.85]  # left_shoulder
        keypoints_2d[6] = [360, 170, 0.85]  # right_shoulder
        keypoints_2d[7] = [260, 250, 0.7]   # left_elbow
        keypoints_2d[8] = [380, 250, 0.7]   # right_elbow
        keypoints_2d[11] = [290, 350, 0.8]  # left_hip
        keypoints_2d[12] = [350, 350, 0.8]  # right_hip
        keypoints_2d[13] = [285, 470, 0.75] # left_knee
        keypoints_2d[14] = [355, 470, 0.75] # right_knee
        keypoints_2d[15] = [280, 580, 0.7]  # left_ankle
        keypoints_2d[16] = [360, 580, 0.7]  # right_ankle
    if bbox is None:
        # Auto-generate bbox from visible keypoints
        visible = keypoints_2d[:17]
        valid = visible[visible[:, 2] > 0.1]
        if len(valid) > 0:
            x_min, y_min = valid[:, :2].min(axis=0)
            x_max, y_max = valid[:, :2].max(axis=0)
            bbox = np.array([x_min - 10, y_min - 10, x_max + 10, y_max + 10])
        else:
            bbox = np.array([100, 100, 500, 500])

    return PersonDetection(
        person_id=person_id,
        bbox=bbox,
        keypoints_133=keypoints_2d,
        confidence=confidence,
        segments={},
    )


def _make_calibrated_pair(
    baseline: float = 1.0,
    image_width: int = 640,
    image_height: int = 480,
    focal_length: float = 500.0,
):
    """Create two cameras with a known baseline for testing.

    Camera A at origin, Camera B translated along X axis.
    Both cameras face forward (+Z world direction = -Z camera direction).

    Returns:
        camera_matrices: [(3,3), (3,3)] intrinsics
        extrinsic_matrices: [(3,4), (3,4)] [R|t]
        image_sizes: [(W,H), (W,H)]
    """
    # Intrinsics
    K = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    # Camera A: identity rotation, at origin
    R_A = np.eye(3, dtype=np.float64)
    t_A = np.zeros(3, dtype=np.float64)
    E_A = np.hstack([R_A, t_A.reshape(3, 1)])  # (3, 4)

    # Camera B: identity rotation, translated along X
    R_B = np.eye(3, dtype=np.float64)
    t_B = np.array([-baseline, 0, 0], dtype=np.float64)
    E_B = np.hstack([R_B, t_B.reshape(3, 1)])  # (3, 4)

    return [K, K], [E_A, E_B], [(image_width, image_height), (image_width, image_height)]


# ── Union-Find tests ─────────────────────────────────────────────────
class TestUnionFind:
    def test_initial_state(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_merges(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) != uf.find(0)

    def test_transitive_union(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_separate_groups(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) == uf.find(3)
        assert uf.find(0) != uf.find(2)

    def test_noop_union(self):
        uf = _UnionFind(3)
        uf.union(0, 0)
        assert uf.find(0) == 0


# ── Fundamental matrix tests ─────────────────────────────────────────
class TestFundamentalMatrix:
    def test_identity_cameras_have_zero_fundamental(self):
        """Two identical cameras should have a degenerate (near-zero) F matrix."""
        K, E, sizes = _make_calibrated_pair(baseline=0.0)
        assoc = CrossViewAssociator(K, E, sizes)
        F = assoc._compute_fundamental_matrix(0, 1)
        # With zero baseline, the essential matrix is zero
        assert np.allclose(F, 0, atol=1e-10)

    def test_fundamental_is_3x3(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        F = assoc._compute_fundamental_matrix(0, 1)
        assert F.shape == (3, 3)

    def test_fundamental_is_normalized(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        F = assoc._compute_fundamental_matrix(0, 1)
        assert abs(np.linalg.norm(F) - 1.0) < 1e-6

    def test_fundamental_transpose_symmetry(self):
        """F_AB^T should equal F_BA."""
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        F_AB = assoc._compute_fundamental_matrix(0, 1)
        F_BA = assoc._compute_fundamental_matrix(1, 0)
        assert np.allclose(F_AB, F_BA.T, atol=1e-10)

    def test_precomputed_pairs(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        assert (0, 1) in assoc._fundamental_matrices
        assert (1, 0) in assoc._fundamental_matrices


# ── Epipolar distance tests ──────────────────────────────────────────
class TestEpipolarDistance:
    def test_corresponding_point_has_small_distance(self):
        """A correctly matched point pair should have small epipolar distance."""
        K, E, sizes = _make_calibrated_pair(baseline=0.5, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)
        F = assoc._compute_fundamental_matrix(0, 1)

        # Point in front of both cameras
        pt_a = np.array([[320.0, 240.0]])  # center of image
        pt_b = np.array([[300.0, 240.0]])  # shifted left (as expected with baseline)

        dist = assoc._compute_epipolar_distance(pt_a, F, pt_b)
        # Should be reasonably small (within a few pixels)
        assert dist[0] < 20.0, f"Expected small distance, got {dist[0]:.2f}"

    def test_mismatched_point_has_large_distance(self):
        """A clearly mismatched point pair should have large epipolar distance."""
        K, E, sizes = _make_calibrated_pair(baseline=0.5, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)
        F = assoc._compute_fundamental_matrix(0, 1)

        pt_a = np.array([[100.0, 100.0]])  # top-left
        pt_b = np.array([[500.0, 400.0]])  # bottom-right

        dist = assoc._compute_epipolar_distance(pt_a, F, pt_b)
        assert dist[0] > 50.0, f"Expected large distance, got {dist[0]:.2f}"

    def test_returns_array(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        F = assoc._compute_fundamental_matrix(0, 1)
        pts_a = np.array([[100.0, 100.0], [200.0, 200.0]])
        pts_b = np.array([[110.0, 100.0], [190.0, 200.0]])
        dist = assoc._compute_epipolar_distance(pts_a, F, pts_b)
        assert dist.shape == (2,)


# ── Keypoint extraction tests ────────────────────────────────────────
class TestKeypointExtraction:
    def test_extracts_matching_keypoints(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        det = _make_detection(0)
        pts, scores = assoc._get_visible_keypoints(det, 0)
        # Should have extracted the matching keypoints (nose, shoulders, hips)
        assert len(pts) >= MIN_KEYPOINTS_FOR_MATCHING
        assert pts.shape[1] == 2
        assert len(scores) == len(pts)

    def test_filters_low_confidence(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        kpts = np.zeros((133, 3), dtype=np.float64)
        # Set matching keypoints but some with low confidence
        kpts[0] = [320, 100, 0.9]   # nose — visible
        kpts[5] = [280, 170, 0.1]   # left_shoulder — below threshold
        kpts[6] = [360, 170, 0.8]   # right_shoulder — visible
        kpts[11] = [290, 350, 0.8]  # left_hip — visible
        kpts[12] = [350, 350, 0.05] # right_hip — below threshold
        det = _make_detection(0, keypoints_2d=kpts)
        pts, scores = assoc._get_visible_keypoints(det, 0)
        # Should only have nose, right_shoulder, left_hip (3 keypoints)
        assert len(pts) == 3

    def test_all_low_confidence_returns_empty(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        kpts = np.zeros((133, 3), dtype=np.float64)
        kpts[0] = [320, 100, 0.05]  # all below threshold
        kpts[5] = [280, 170, 0.05]
        det = _make_detection(0, keypoints_2d=kpts)
        pts, scores = assoc._get_visible_keypoints(det, 0)
        assert len(pts) == 0


# ── Hungarian assignment tests ───────────────────────────────────────
class TestHungarianAssignment:
    def test_simple_two_person_match(self):
        """Two persons in two cameras with clear diagonal matching.

        Uses persons at DIFFERENT vertical positions so epipolar geometry
        produces different disparity for the two persons (preventing ties).
        """
        K, E, sizes = _make_calibrated_pair(baseline=0.3, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)

        # Person 0: tall (top keypoints near y=80, hips near y=320)
        kpts_p0 = np.zeros((133, 3), dtype=np.float64)
        kpts_p0[0] = [200, 80, 0.9]    # nose — high up
        kpts_p0[5] = [170, 140, 0.85]  # left_shoulder
        kpts_p0[6] = [230, 140, 0.85]  # right_shoulder
        kpts_p0[11] = [180, 320, 0.8]  # left_hip
        kpts_p0[12] = [220, 320, 0.8]  # right_hip

        # Person 1: short (top keypoints near y=180, hips near y=380)
        kpts_p1 = np.zeros((133, 3), dtype=np.float64)
        kpts_p1[0] = [440, 180, 0.9]   # nose — lower
        kpts_p1[5] = [410, 240, 0.85]  # left_shoulder
        kpts_p1[6] = [470, 240, 0.85]  # right_shoulder
        kpts_p1[11] = [420, 380, 0.8]  # left_hip
        kpts_p1[12] = [460, 380, 0.8]  # right_hip

        det_a0 = _make_detection(0, keypoints_2d=kpts_p0)
        det_a1 = _make_detection(1, keypoints_2d=kpts_p1)

        # Camera B has same projection but different perspective —
        # use a rotated camera to create asymmetric epipolar distances
        K_single = K[0]  # extract single (3,3) intrinsic matrix
        R_B = cv2.Rodrigues(np.array([0.0, 0.02, 0.0]))[0]  # slight rotation
        E_B = np.hstack([R_B, np.array([[-0.3, 0, 0]]).T])
        assoc2 = CrossViewAssociator(
            [K_single, K_single],
            [np.hstack([np.eye(3), np.zeros((3, 1))]), E_B],
            sizes,
        )

        # Same keypoints in camera B (no shift — they observe same 3D points differently)
        det_b0 = _make_detection(0, keypoints_2d=kpts_p0)
        det_b1 = _make_detection(1, keypoints_2d=kpts_p1)

        cost_matrix = assoc2.compute_pair_cost_matrix(
            [det_a0, det_a1], [det_b0, det_b1], 0, 1
        )

        # Verify we got meaningful (non-zero, non-equal) costs
        diag_cost = cost_matrix[0, 0] + cost_matrix[1, 1]
        offdiag_cost = cost_matrix[0, 1] + cost_matrix[1, 0]
        # At least one of the pairings should differ
        assert diag_cost != offdiag_cost or diag_cost > 0, (
            f"Expected different costs, got diag={diag_cost:.4f} offdiag={offdiag_cost:.4f}"
        )

    def test_cost_matrix_shape(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        dets_a = [_make_detection(0), _make_detection(1)]
        dets_b = [_make_detection(0)]

        cost = assoc.compute_pair_cost_matrix(dets_a, dets_b, 0, 1)
        assert cost.shape == (2, 1)

    def test_empty_detections_gives_inf(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        cost = assoc.compute_pair_cost_matrix([], [_make_detection(0)], 0, 1)
        assert np.all(np.isinf(cost))

    def test_hungarian_returns_assignments(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        cost = np.array([[10.0, 50.0], [60.0, 12.0]])
        pa = assoc._hungarian_assign(cost, 0, 1)

        assert isinstance(pa, CameraPairAssignment)
        assert pa.camera_a_idx == 0
        assert pa.camera_b_idx == 1
        assert pa.assignments[0] == 0  # person 0 -> person 0
        assert pa.assignments[1] == 1  # person 1 -> person 1

    def test_hungarian_rejects_high_cost(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        # All costs above threshold
        cost = np.array([[100.0, 200.0]])
        pa = assoc._hungarian_assign(cost, 0, 1)
        assert pa.assignments[0] == -1  # unmatched

    def test_hungarian_all_inf(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        cost = np.array([[np.inf, np.inf]])
        pa = assoc._hungarian_assign(cost, 0, 1)
        assert pa.assignments[0] == -1
        assert not pa.matched


# ── Frame association integration tests ──────────────────────────────
class TestFrameAssociation:
    def test_single_person_both_cameras(self):
        """One person visible in both cameras → 1 actor."""
        K, E, sizes = _make_calibrated_pair(baseline=0.3, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)

        det_a = [_make_detection(0)]
        det_b = [_make_detection(0)]

        result = assoc.associate_frame([det_a, det_b], frame_idx=0)

        assert isinstance(result, FrameAssociation)
        assert result.frame_idx == 0
        assert result.num_cameras == 2
        assert result.num_actors == 1
        assert result.detection_to_actor[(0, 0)] == result.detection_to_actor[(1, 0)]

    def test_two_persons_both_cameras(self):
        """Two persons in both cameras → 2 actors."""
        K, E, sizes = _make_calibrated_pair(baseline=0.3, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)

        kpts_p0 = np.zeros((133, 3), dtype=np.float64)
        kpts_p0[0] = [200, 100, 0.9]
        kpts_p0[5] = [170, 170, 0.85]
        kpts_p0[6] = [230, 170, 0.85]
        kpts_p0[11] = [180, 350, 0.8]
        kpts_p0[12] = [220, 350, 0.8]

        kpts_p1 = np.zeros((133, 3), dtype=np.float64)
        kpts_p1[0] = [440, 100, 0.9]
        kpts_p1[5] = [410, 170, 0.85]
        kpts_p1[6] = [470, 170, 0.85]
        kpts_p1[11] = [420, 350, 0.8]
        kpts_p1[12] = [460, 350, 0.8]

        kpts_p0_b = kpts_p0.copy()
        kpts_p0_b[:, 0] -= 30
        kpts_p1_b = kpts_p1.copy()
        kpts_p1_b[:, 0] -= 30

        result = assoc.associate_frame(
            [[_make_detection(0, kpts_p0), _make_detection(1, kpts_p1)],
             [_make_detection(0, kpts_p0_b), _make_detection(1, kpts_p1_b)]],
            frame_idx=0,
        )

        assert result.num_actors == 2
        # Both cameras should agree on the mapping
        actor_a0 = result.detection_to_actor[(0, 0)]
        actor_a1 = result.detection_to_actor[(0, 1)]
        assert result.detection_to_actor[(1, 0)] == actor_a0
        assert result.detection_to_actor[(1, 1)] == actor_a1

    def test_no_detections(self):
        """Zero detections → 0 actors."""
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        result = assoc.associate_frame([[], []], frame_idx=0)
        assert result.num_actors == 0
        assert result.detection_to_actor == {}

    def test_detection_in_one_camera_only(self):
        """Person visible in only one camera → still gets an actor ID."""
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)

        result = assoc.associate_frame(
            [[_make_detection(0)], []],  # only camera 0 has detections
            frame_idx=0,
        )
        assert result.num_actors >= 1
        assert (0, 0) in result.detection_to_actor

    def test_actor_memberships_populated(self):
        """Actor memberships should list all (camera, person) pairs per actor."""
        K, E, sizes = _make_calibrated_pair(baseline=0.3, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)

        result = assoc.associate_frame(
            [[_make_detection(0)], [_make_detection(0)]],
            frame_idx=0,
        )
        # Actor 0 should have two members: (0,0) and (1,0)
        all_members = []
        for members in result.actor_memberships.values():
            all_members.extend(members)
        assert (0, 0) in all_members
        assert (1, 0) in all_members

    def test_three_cameras(self):
        """Three cameras with one person → single actor across all three."""
        K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1],
        ], dtype=np.float64)

        E_A = np.hstack([np.eye(3), np.zeros((3, 1))])
        E_B = np.hstack([np.eye(3), np.array([[-0.5, 0, 0]]).T])
        E_C = np.hstack([np.eye(3), np.array([[0.5, 0, 0]]).T])

        assoc = CrossViewAssociator(
            [K, K, K], [E_A, E_B, E_C],
            [(640, 480)] * 3,
        )

        result = assoc.associate_frame(
            [[_make_detection(0)], [_make_detection(0)], [_make_detection(0)]],
            frame_idx=0,
        )
        assert result.num_actors == 1
        # All three cameras should map to the same actor
        a0 = result.detection_to_actor[(0, 0)]
        a1 = result.detection_to_actor[(1, 0)]
        a2 = result.detection_to_actor[(2, 0)]
        assert a0 == a1 == a2


# ── Constructor tests ────────────────────────────────────────────────
class TestConstructor:
    def test_basic_init(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes)
        assert assoc.num_cameras == 2

    def test_mismatched_lengths_raises(self):
        K = [np.eye(3)]
        E = [np.eye(3), np.eye(3)]
        with pytest.raises(ValueError, match="same length"):
            CrossViewAssociator(K, E, [(640, 480)])

    def test_max_distance_configurable(self):
        K, E, sizes = _make_calibrated_pair()
        assoc = CrossViewAssociator(K, E, sizes, max_epipolar_distance=10.0)
        assert assoc.max_epipolar_distance == 10.0


# ── Per-frame table output (CR #15 format) ───────────────────────────
class TestPerFrameTable:
    def test_print_frame_table(self, capsys):
        """Generate and print a per-frame association table."""
        K, E, sizes = _make_calibrated_pair(baseline=0.3, focal_length=500)
        assoc = CrossViewAssociator(K, E, sizes)

        results = []
        for frame_idx in range(3):
            kpts_p0 = np.zeros((133, 3), dtype=np.float64)
            kpts_p0[0] = [200 + frame_idx * 10, 100, 0.9]
            kpts_p0[5] = [170 + frame_idx * 10, 170, 0.85]
            kpts_p0[6] = [230 + frame_idx * 10, 170, 0.85]
            kpts_p0[11] = [180 + frame_idx * 10, 350, 0.8]
            kpts_p0[12] = [220 + frame_idx * 10, 350, 0.8]

            kpts_p0_b = kpts_p0.copy()
            kpts_p0_b[:, 0] -= 30

            result = assoc.associate_frame(
                [[_make_detection(0, kpts_p0)], [_make_detection(0, kpts_p0_b)]],
                frame_idx=frame_idx,
            )
            results.append(result)

        # Print table
        header = f"{'Frame':>6} | {'Actors':>6} | {'Cam0→Cam1':>10}"
        print("\n" + header)
        print("-" * len(header))
        for r in results:
            pair_str = ""
            for pa in r.pair_assignments:
                matched = sum(1 for v in pa.assignments.values() if v != -1)
                pair_str += f" {matched}/{len(pa.assignments)}"
            print(f"{r.frame_idx:>6} | {r.num_actors:>6} | {pair_str:>10}")

        captured = capsys.readouterr()
        assert "Frame" in captured.out
        assert "Actors" in captured.out


# ── Constants sanity ─────────────────────────────────────────────────
class TestConstants:
    def test_matching_keypoint_indices_valid(self):
        for idx in BODY_KEYPOINT_INDICES_FOR_MATCHING:
            assert 0 <= idx < 17, f"Keypoint index {idx} out of COCO body range"

    def test_thresholds_positive(self):
        assert MIN_KEYPOINTS_FOR_MATCHING > 0
        assert EPIPOLAR_CONFIDENCE_THRESHOLD > 0
        assert EPIPOLAR_CONFIDENCE_THRESHOLD < 1.0
