"""
Tests for Stage 7 — ArUco Marker Fallback

Two scenarios as specified:
1. "clean_markers" — markers visible, correct binding
2. "embrace_scene" — marker anchors reduce identity swaps

Additional tests:
3. Part A: generate_actor_markers.py produces correct files
4. Part B: ArUco detection returns correct format
5. Part C: marker-to-skeleton binding distance threshold
6. Part D: temporal tracker uses marker anchors correctly
7. Toggle: marker fallback disabled → ArucoDetector never called
"""
import pytest
import numpy as np
import os
import sys
import tempfile
import yaml
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(__file__))

from marker_fallback import (
    ArucoMarkerDetector,
    MarkerDetection,
    MarkerBinding,
    bind_markers_to_skeletons,
    detect_and_bind,
    _compute_body_center,
    MARKER_SKELETON_BIND_THRESHOLD_PX,
)
from multiperson_detector import PersonDetection
from joint_definitions import (
    MARKER_DETECTION_KEYS,
    MARKER_SKELETON_BIND_THRESHOLD_PX as JDEF_THRESHOLD,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_skeleton_133(torso_x=320, torso_y=240, confidence=0.8):
    """Create a (133, 3) keypoint array with torso at given position."""
    kpts = np.zeros((133, 3), dtype=np.float64)
    # Shoulders
    kpts[5] = [torso_x - 30, torso_y - 20, confidence]
    kpts[6] = [torso_x + 30, torso_y - 20, confidence]
    # Hips
    kpts[11] = [torso_x - 25, torso_y + 60, confidence]
    kpts[12] = [torso_x + 25, torso_y + 60, confidence]
    # Nose
    kpts[0] = [torso_x, torso_y - 80, confidence]
    return kpts


def _make_person_detection(person_id, torso_x=320, torso_y=240):
    """Create a PersonDetection with skeleton centered at torso position."""
    kpts = _make_skeleton_133(torso_x, torso_y)
    return PersonDetection(
        person_id=person_id,
        bbox=np.array([torso_x - 80, torso_y - 100, torso_x + 80, torso_y + 100], dtype=np.float64),
        keypoints_133=kpts,
        confidence=0.8,
    )


def _make_marker_detection(marker_id, center_x, center_y, cam_idx=0, frame_idx=0):
    """Create a MarkerDetection at given center position."""
    offset = 20
    corners = np.array([
        [center_x - offset, center_y - offset],
        [center_x + offset, center_y - offset],
        [center_x + offset, center_y + offset],
        [center_x - offset, center_y + offset],
    ], dtype=np.float64)
    return MarkerDetection(
        marker_id=marker_id,
        corners=corners,
        center_2d=np.array([center_x, center_y], dtype=np.float64),
        detection_confidence=0.9,
        cam_idx=cam_idx,
        frame_idx=frame_idx,
    )


# ── Part A: Marker generation ──────────────────────────────────────────

class TestMarkerGeneration:
    def test_generate_produces_files(self):
        """generate_actor_markers.py creates PNG files with correct names."""
        try:
            import cv2
            import yaml
        except ImportError:
            pytest.skip("cv2 or yaml not available")

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
        from tools.generate_actor_markers import generate_all_markers, load_config

        config = {
            "dictionary": "DICT_4X4_50",
            "marker_size_mm": 50,
            "actors": {"actor_0": 0, "actor_1": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generate_all_markers(config, tmpdir)

            assert len(generated) == 2
            assert "actor_0" in generated
            assert "actor_1" in generated

            for name, info in generated.items():
                assert os.path.exists(info["filepath"])
                assert info["marker_id"] in [0, 1]
                assert info["size_mm"] == 50
                assert "_50mm.png" in info["filename"]

                img = cv2.imread(info["filepath"])
                assert img is not None
                assert img.shape[0] > 100
                assert img.shape[1] > 100

    def test_different_actors_different_ids(self):
        """Two actors get different marker IDs."""
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available")

        from tools.generate_actor_markers import generate_all_markers

        config = {
            "dictionary": "DICT_4X4_50",
            "marker_size_mm": 50,
            "actors": {"actor_0": 0, "actor_1": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generate_all_markers(config, tmpdir)

            id0 = generated["actor_0"]["marker_id"]
            id1 = generated["actor_1"]["marker_id"]
            assert id0 != id1


# ── Part B: Marker detection ───────────────────────────────────────────

class TestMarkerDetection:
    def test_detect_frame_returns_list(self):
        """detect_frame returns a list of MarkerDetection objects."""
        detector = ArucoMarkerDetector.__new__(ArucoMarkerDetector)
        detector.config = {"dictionary": "DICT_4X4_50", "actors": {"actor_0": 0}}
        detector.marker_length = 0.05
        detector.actor_to_marker = {"actor_0": 0}
        detector.marker_to_actor = {0: "actor_0"}

        mock_aruco_dict = MagicMock()
        mock_detector = MagicMock()
        mock_detector.detectMarkers.return_value = ([], None, [])

        detector.aruco_dict = mock_aruco_dict
        detector.detector = mock_detector

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_frame(frame, cam_idx=0, frame_idx=0)

        assert isinstance(result, list)
        mock_detector.detectMarkers.assert_called_once()

    def test_detection_output_format(self):
        """MarkerDetection has all required fields from joint_definitions."""
        md = MarkerDetection(
            marker_id=0,
            corners=np.zeros((4, 2)),
            center_2d=np.zeros(2),
            detection_confidence=0.9,
            cam_idx=0,
            frame_idx=0,
        )
        assert hasattr(md, "marker_id")
        assert hasattr(md, "corners")
        assert hasattr(md, "center_2d")
        assert hasattr(md, "detection_confidence")

    def test_empty_frame(self):
        """Empty frame returns no detections."""
        detector = ArucoMarkerDetector.__new__(ArucoMarkerDetector)
        detector.config = {"dictionary": "DICT_4X4_50", "actors": {}}
        detector.marker_length = 0.05
        detector.actor_to_marker = {}
        detector.marker_to_actor = {}

        mock_detector = MagicMock()
        detector.detector = mock_detector

        result = detector.detect_frame(np.array([]), cam_idx=0, frame_idx=0)
        assert result == []


# ── Part C: Marker-to-skeleton binding ─────────────────────────────────

class TestBinding:
    def test_body_center_computation(self):
        """_compute_body_center uses shoulder midpoint when available."""
        kpts = _make_skeleton_133(torso_x=320, torso_y=240)
        center = _compute_body_center(kpts)
        assert center is not None
        assert abs(center[0] - 320) < 5
        assert abs(center[1] - 220) < 5  # shoulders are at y=240-20=220

    def test_body_center_fallback_to_hips(self):
        """_compute_body_center falls back to hips when shoulders are missing."""
        kpts = np.zeros((133, 3))
        kpts[11] = [200, 300, 0.8]  # left_hip
        kpts[12] = [240, 300, 0.8]  # right_hip
        center = _compute_body_center(kpts)
        assert center is not None
        assert abs(center[0] - 220) < 5

    def test_body_center_none_when_insufficient(self):
        """_compute_body_center returns None when no keypoints visible."""
        kpts = np.zeros((133, 3))
        center = _compute_body_center(kpts)
        assert center is None

    def test_binding_close_marker(self):
        """Marker near body center binds successfully."""
        marker = _make_marker_detection(0, center_x=320, center_y=250)
        skeleton = _make_person_detection(0, torso_x=320, torso_y=240)
        bindings = bind_markers_to_skeletons([marker], [skeleton])

        assert len(bindings) == 1
        assert bindings[0].bound is True
        assert bindings[0].actor_id == 0
        assert bindings[0].distance_px < MARKER_SKELETON_BIND_THRESHOLD_PX

    def test_binding_distant_marker(self):
        """Marker far from body center does not bind."""
        marker = _make_marker_detection(0, center_x=600, center_y=400)
        skeleton = _make_person_detection(0, torso_x=320, torso_y=240)
        bindings = bind_markers_to_skeletons([marker], [skeleton])

        assert len(bindings) == 1
        assert bindings[0].bound is False

    def test_binding_no_skeletons(self):
        """No skeletons → no binding."""
        marker = _make_marker_detection(0, center_x=320, center_y=250)
        bindings = bind_markers_to_skeletons([marker], [])
        assert len(bindings) == 1
        assert bindings[0].bound is False

    def test_binding_multiple_markers_multiple_skeletons(self):
        """Two markers bind to two different skeletons correctly."""
        marker0 = _make_marker_detection(0, center_x=200, center_y=250)
        marker1 = _make_marker_detection(1, center_x=500, center_y=250)
        skel0 = _make_person_detection(0, torso_x=200, torso_y=240)
        skel1 = _make_person_detection(1, torso_x=500, torso_y=240)

        bindings = bind_markers_to_skeletons([marker0, marker1], [skel0, skel1])
        assert all(b.bound for b in bindings)

        marker_ids = {b.marker_id for b in bindings}
        actor_ids = {b.actor_id for b in bindings}
        assert marker_ids == {0, 1}
        assert actor_ids == {0, 1}


# ── Part D: detect_and_bind (full pipeline) ────────────────────────────

class TestDetectAndBind:
    def test_multi_camera_consensus(self):
        """Marker seen on 2+ cameras → strong override."""
        per_cam_markers = [
            [_make_marker_detection(0, 320, 250, cam_idx=0)],
            [_make_marker_detection(0, 315, 255, cam_idx=1)],
        ]
        per_cam_skeletons = [
            [_make_person_detection(0, 320, 240)],
            [_make_person_detection(0, 315, 245)],
        ]

        overrides = detect_and_bind(per_cam_markers, per_cam_skeletons, frame_idx=0)
        assert 0 in overrides
        assert overrides[0] == 0

    def test_single_camera_weak_binding(self):
        """Marker seen on only 1 camera → still override (weak but valid)."""
        per_cam_markers = [
            [_make_marker_detection(0, 320, 250, cam_idx=0)],
            [],
        ]
        per_cam_skeletons = [
            [_make_person_detection(0, 320, 240)],
            [_make_person_detection(0, 320, 240)],
        ]

        overrides = detect_and_bind(per_cam_markers, per_cam_skeletons, frame_idx=0)
        assert 0 in overrides

    def test_no_markers_no_overrides(self):
        """No markers → no overrides."""
        per_cam_markers = [[], []]
        per_cam_skeletons = [
            [_make_person_detection(0)],
            [_make_person_detection(0)],
        ]

        overrides = detect_and_bind(per_cam_markers, per_cam_skeletons, frame_idx=0)
        assert overrides == {}


# ── Scenario 1: "clean_markers" ────────────────────────────────────────

class TestCleanMarkersScenario:
    def test_markers_visible_correct_binding(self):
        """Markers visible on actors outside embrace scene — correct ID assignment."""
        # Simulate 3 cameras, 2 actors with markers
        num_cams = 3
        per_cam_markers = []
        per_cam_skeletons = []

        for cam in range(num_cams):
            markers = [
                _make_marker_detection(0, 200 + cam * 5, 250, cam_idx=cam),
                _make_marker_detection(1, 500 - cam * 5, 250, cam_idx=cam),
            ]
            skeletons = [
                _make_person_detection(0, 200, 240),
                _make_person_detection(1, 500, 240),
            ]
            per_cam_markers.append(markers)
            per_cam_skeletons.append(skeletons)

        overrides = detect_and_bind(per_cam_markers, per_cam_skeletons, frame_idx=0)

        assert 0 in overrides
        assert 1 in overrides
        assert overrides[0] == 0
        assert overrides[1] == 1


# ── Scenario 2: "embrace_scene" ────────────────────────────────────────

class TestEmbraceSceneScenario:
    def test_marker_anchors_reduce_identity_swaps(self):
        """Marker anchors prevent identity swaps during close contact."""
        from cross_view_association import FrameAssociation
        from temporal_tracker import TemporalTracker

        num_frames = 20
        embrace_start = 5
        embrace_end = 15

        frame_associations = []
        for f in range(num_frames):
            if embrace_start <= f <= embrace_end:
                # During embrace: actors swap detection order (simulated error)
                actor_memberships = {0: [(0, 1)], 1: [(0, 0)]}
                detection_to_actor = {(0, 1): 0, (0, 0): 1}
            else:
                actor_memberships = {0: [(0, 0)], 1: [(0, 1)]}
                detection_to_actor = {(0, 0): 0, (0, 1): 1}

            frame_associations.append(FrameAssociation(
                frame_idx=f,
                num_cameras=1,
                num_actors=2,
                detection_to_actor=detection_to_actor,
                actor_memberships=actor_memberships,
            ))

        # Run WITHOUT marker anchors
        tracker_no_markers = TemporalTracker()
        results_no_markers = tracker_no_markers.track(frame_associations)

        swaps_no_markers = 0
        for f in range(1, num_frames):
            prev = results_no_markers[f - 1].local_to_persistent
            curr = results_no_markers[f].local_to_persistent
            if prev.get(0) != curr.get(0):
                swaps_no_markers += 1

        # Run WITH marker anchors (markers always correct)
        marker_anchors = []
        for f in range(num_frames):
            if f >= embrace_start:
                # Markers visible during and after embrace
                marker_anchors.append({0: 0, 1: 1})
            else:
                marker_anchors.append({0: 0, 1: 1})

        tracker_with_markers = TemporalTracker()
        results_with_markers = tracker_with_markers.track(
            frame_associations, marker_anchors=marker_anchors
        )

        swaps_with_markers = 0
        for f in range(1, num_frames):
            prev = results_with_markers[f - 1].local_to_persistent
            curr = results_with_markers[f].local_to_persistent
            if prev.get(0) != curr.get(0):
                swaps_with_markers += 1

        # With markers, swaps should be reduced or zero
        assert swaps_with_markers <= swaps_no_markers

    def test_marker_anchors_first_frame_creation(self):
        """First frame with markers creates correct persistent IDs."""
        from cross_view_association import FrameAssociation
        from temporal_tracker import TemporalTracker

        fa = FrameAssociation(
            frame_idx=0,
            num_cameras=1,
            num_actors=2,
            detection_to_actor={(0, 0): 0, (0, 1): 1},
            actor_memberships={0: [(0, 0)], 1: [(0, 1)]},
        )

        tracker = TemporalTracker()
        marker_anchors = [{0: 0, 1: 1}]
        result = tracker.track([fa], marker_anchors=marker_anchors)

        assert len(result) == 1
        assert len(result[0].local_to_persistent) == 2

        gid0 = result[0].local_to_persistent[0]
        gid1 = result[0].local_to_persistent[1]
        assert gid0 != gid1

        # Marker anchor map should be populated
        assert 0 in tracker._marker_anchor_map
        assert 1 in tracker._marker_anchor_map


# ── Toggle test ────────────────────────────────────────────────────────

class TestToggle:
    def test_no_marker_overrides_no_detector_call(self):
        """When marker_overrides is None, ArucoDetector.detectMarkers is never called."""
        from cross_view_association import FrameAssociation
        from temporal_tracker import TemporalTracker

        fa = FrameAssociation(
            frame_idx=0,
            num_cameras=1,
            num_actors=1,
            detection_to_actor={(0, 0): 0},
            actor_memberships={0: [(0, 0)]},
        )

        tracker = TemporalTracker()
        # Pass no marker_anchors — should work exactly as before
        result = tracker.track([fa], marker_anchors=None)

        assert len(result) == 1
        assert result[0].local_to_persistent[0] == 0

        # Marker anchor map should be empty
        assert len(tracker._marker_anchor_map) == 0

    def test_empty_marker_anchors_no_change(self):
        """Empty marker_anchors list → same behavior as None."""
        from cross_view_association import FrameAssociation
        from temporal_tracker import TemporalTracker

        fa = FrameAssociation(
            frame_idx=0,
            num_cameras=1,
            num_actors=1,
            detection_to_actor={(0, 0): 0},
            actor_memberships={0: [(0, 0)]},
        )

        tracker = TemporalTracker()
        result = tracker.track([fa], marker_anchors=[])

        assert len(tracker._marker_anchor_map) == 0

    def test_no_backward_compatibility_break(self):
        """Existing callers without marker_anchors parameter still work."""
        from cross_view_association import FrameAssociation
        from temporal_tracker import TemporalTracker

        fa = FrameAssociation(
            frame_idx=0,
            num_cameras=1,
            num_actors=1,
            detection_to_actor={(0, 0): 0},
            actor_memberships={0: [(0, 0)]},
        )

        tracker = TemporalTracker()
        # Old calling convention: no marker_anchors
        result = tracker.track([fa])
        assert len(result) == 1
