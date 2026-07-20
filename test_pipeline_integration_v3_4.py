"""
Pipeline v3.4 Integration Tests — verify all new modules are wired correctly.

Tests that each new module (rolling shutter, outlier frame detection,
bone length solver, drift monitor, RTS smoother) works when called
through run_full_pipeline().

Uses synthetic skeleton data — no real recordings needed.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_full_pipeline
from joint_definitions import NUM_TOTAL, NUM_BODY, NUM_RIGHT_HAND, NUM_LEFT_HAND, NUM_FACE


def _make_skeleton(num_frames=60, num_keypoints=None, fps=30.0, seed=42):
    """Create a clean synthetic skeleton for testing."""
    if num_keypoints is None:
        num_keypoints = NUM_TOTAL
    rng = np.random.RandomState(seed)
    base = np.cumsum(
        rng.randn(num_frames, num_keypoints, 3).astype(np.float64) * 0.5, axis=0
    )
    base += 500.0
    return base


def _make_skeleton_with_nans(num_frames=60, num_keypoints=None, fps=30.0, seed=42):
    """Create a skeleton with some NaN gaps for testing."""
    if num_keypoints is None:
        num_keypoints = NUM_TOTAL
    data = _make_skeleton(num_frames, num_keypoints, fps, seed)
    rng = np.random.RandomState(seed + 1)
    for i in range(min(num_keypoints, NUM_BODY)):
        nan_start = rng.randint(10, num_frames - 10)
        data[nan_start:nan_start + 3, i, :] = np.nan
    return data


def _make_skeleton_with_spikes(num_frames=60, num_keypoints=None, fps=30.0, seed=42):
    """Create a skeleton with velocity spikes for testing."""
    if num_keypoints is None:
        num_keypoints = NUM_TOTAL
    data = _make_skeleton(num_frames, num_keypoints, fps, seed)
    rng = np.random.RandomState(seed + 2)
    for _ in range(5):
        f = rng.randint(5, num_frames - 5)
        k = rng.randint(0, min(num_keypoints, NUM_BODY))
        data[f, k, :] += rng.randn(3) * 100.0
    return data


# ============================================================
# Test 1: Default pipeline (all new modules OFF, backward compat)
# ============================================================
class TestPipelineDefault:
    """Verify the pipeline works with all new modules disabled (backward compat)."""

    def test_01_clean_skeleton(self):
        data = _make_skeleton(num_frames=60)
        result, report = run_full_pipeline(data, fps=30.0)
        assert result.shape == data.shape
        assert report["pipeline_version"] == "3.4-integrated"
        assert "stages" in report

    def test_02_skeleton_with_nans(self):
        data = _make_skeleton_with_nans(num_frames=60)
        nan_before = int(np.sum(np.isnan(data)))
        result, report = run_full_pipeline(data, fps=30.0)
        nan_after = int(np.sum(np.isnan(result)))
        assert result.shape == data.shape
        assert report["nan_reduced"] >= 0

    def test_03_skeleton_with_spikes(self):
        data = _make_skeleton_with_spikes(num_frames=60)
        result, report = run_full_pipeline(data, fps=30.0)
        assert result.shape == data.shape

    def test_04_report_has_all_default_stages(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(data, fps=30.0)
        expected_stages = [
            "outlier_removal",
            "bone_outlier_detection",
            "gap_filling",
            "foot_gap_correction",
            "bone_enforcement_pre",
            "jitter_filter",
            "bone_enforcement_post",
            "wrist_consistency",
        ]
        for stage in expected_stages:
            assert stage in report["stages"], f"Missing stage: {stage}"


# ============================================================
# Test 2: Rolling shutter compensation (Stage 0)
# ============================================================
class TestRollingShutterIntegration:
    """Verify rolling shutter compensation runs in pipeline."""

    def test_01_rolling_shutter_enabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            rolling_shutter_readout_ms=33.0,
        )
        assert result.shape == data.shape
        assert "rolling_shutter" in report["stages"]
        assert report["stages"]["rolling_shutter"]["readout_time_ms"] == 33.0

    def test_02_rolling_shutter_different_readout(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            rolling_shutter_readout_ms=16.7,
        )
        assert "rolling_shutter" in report["stages"]
        assert report["stages"]["rolling_shutter"]["readout_time_ms"] == 16.7

    def test_03_rolling_shutter_disabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=False,
        )
        assert "rolling_shutter" not in report["stages"]


# ============================================================
# Test 3: Outlier frame detection (Stage 0.5)
# ============================================================
class TestOutlierFrameDetection:
    """Verify outlier frame detection runs in pipeline."""

    def test_01_outlier_detection_enabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_outlier_frame_detection=True,
        )
        assert "outlier_frame_detection" in report["stages"]
        assert "total_flagged" in report["stages"]["outlier_frame_detection"]
        assert "severity_counts" in report["stages"]["outlier_frame_detection"]

    def test_02_outlier_detection_with_spikes(self):
        data = _make_skeleton_with_spikes(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_outlier_frame_detection=True,
        )
        od = report["stages"]["outlier_frame_detection"]
        assert od["total_flagged"] > 0

    def test_03_outlier_detection_disabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_outlier_frame_detection=False,
        )
        assert "outlier_frame_detection" not in report["stages"]


# ============================================================
# Test 4: Bone length solver (Stages 4b & 6b)
# ============================================================
class TestBoneLengthSolver:
    """Verify bone length solver runs in pipeline."""

    def test_01_solver_enabled_default(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_bone_length_solver=True,
        )
        assert "bone_solver_pre" in report["stages"]
        assert "bone_solver_post" in report["stages"]
        assert "num_corrections" in report["stages"]["bone_solver_pre"]
        assert "num_corrections" in report["stages"]["bone_solver_post"]

    def test_02_solver_threshold_affects_corrections(self):
        data = _make_skeleton_with_spikes(num_frames=60, num_keypoints=None)
        result_loose, report_loose = run_full_pipeline(
            data.copy(), fps=30.0,
            use_bone_length_solver=True,
            bone_solver_threshold=0.50,
        )
        result_strict, report_strict = run_full_pipeline(
            data.copy(), fps=30.0,
            use_bone_length_solver=True,
            bone_solver_threshold=0.05,
        )
        assert report_strict["stages"]["bone_solver_pre"]["num_corrections"] >= \
               report_loose["stages"]["bone_solver_pre"]["num_corrections"]

    def test_03_solver_passes_affects_corrections(self):
        data = _make_skeleton_with_spikes(num_frames=60, num_keypoints=None)
        _, report_1pass = run_full_pipeline(
            data.copy(), fps=30.0,
            use_bone_length_solver=True,
            bone_solver_passes=1,
        )
        _, report_4pass = run_full_pipeline(
            data.copy(), fps=30.0,
            use_bone_length_solver=True,
            bone_solver_passes=4,
        )
        assert report_4pass["stages"]["bone_solver_post"]["num_corrections"] >= \
               report_1pass["stages"]["bone_solver_post"]["num_corrections"]

    def test_04_solver_disabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_bone_length_solver=False,
        )
        assert "bone_solver_pre" not in report["stages"]
        assert "bone_solver_post" not in report["stages"]


# ============================================================
# Test 5: Detector drift monitor (Stage 2.5)
# ============================================================
class TestDriftMonitor:
    """Verify detector drift monitor runs in pipeline."""

    def test_01_drift_monitor_enabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_drift_monitor=True,
        )
        assert "drift_monitor" in report["stages"]
        dm = report["stages"]["drift_monitor"]
        assert "overall_drift_score" in dm
        assert "num_alerts" in dm
        assert "summary" in dm

    def test_02_drift_monitor_disabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_drift_monitor=False,
        )
        assert "drift_monitor" not in report["stages"]

    def test_03_drift_monitor_with_degradation(self):
        rng = np.random.RandomState(99)
        data = _make_skeleton(num_frames=80, num_keypoints=None, seed=99)
        for f in range(60, 80):
            data[f] += rng.randn(*data[f].shape) * 5.0
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_drift_monitor=True,
        )
        assert "drift_monitor" in report["stages"]


# ============================================================
# Test 6: RTS smoother (Stage 8.5)
# ============================================================
class TestRTSSmoother:
    """Verify RTS smoother runs in pipeline."""

    def test_01_rts_enabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape
        assert "rts_smoother" in report["stages"]
        assert "nan_frames" in report["stages"]["rts_smoother"]

    def test_02_rts_reduces_jitter(self):
        data = _make_skeleton(num_frames=120, num_keypoints=None, seed=7)
        rng = np.random.RandomState(7)
        data_noisy = data + rng.randn(*data.shape) * 3.0
        result_no_rts, _ = run_full_pipeline(
            data_noisy.copy(), fps=30.0,
            use_rts_smoother=False,
        )
        result_rts, _ = run_full_pipeline(
            data_noisy.copy(), fps=30.0,
            use_rts_smoother=True,
        )
        diff_no_rts = np.nanmean(np.abs(np.diff(result_no_rts, axis=0)))
        diff_rts = np.nanmean(np.abs(np.diff(result_rts, axis=0)))
        assert diff_rts < diff_no_rts

    def test_03_rts_disabled(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rts_smoother=False,
        )
        assert "rts_smoother" not in report["stages"]

    def test_04_rts_custom_params(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rts_smoother=True,
            rts_process_noise=0.5,
            rts_observation_noise=20.0,
        )
        assert "rts_smoother" in report["stages"]


# ============================================================
# Test 7: ALL modules enabled simultaneously
# ============================================================
class TestAllModulesCombined:
    """Verify all new modules work together without conflicts."""

    def test_01_all_enabled_clean(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            use_outlier_frame_detection=True,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape
        new_stages = [
            "rolling_shutter",
            "outlier_frame_detection",
            "bone_solver_pre",
            "bone_solver_post",
            "drift_monitor",
            "rts_smoother",
        ]
        for stage in new_stages:
            assert stage in report["stages"], f"Missing stage: {stage}"

    def test_02_all_enabled_with_nans(self):
        data = _make_skeleton_with_nans(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            use_outlier_frame_detection=True,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape
        assert "nan_after" in report

    def test_03_all_enabled_with_spikes(self):
        data = _make_skeleton_with_spikes(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            use_outlier_frame_detection=True,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape

    def test_04_total_time_reported(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            use_outlier_frame_detection=True,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        assert "total_time_seconds" in report
        assert report["total_time_seconds"] > 0

    def test_05_pipeline_does_not_modify_input(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        data_orig = data.copy()
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_rolling_shutter_compensation=True,
            use_outlier_frame_detection=True,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        np.testing.assert_array_equal(data, data_orig)

    def test_06_minimum_foot_sliding_off(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_bone_length_solver=True,
            fix_foot_sliding=False,
        )
        assert result.shape == data.shape


# ============================================================
# Test 8: Edge cases
# ============================================================
class TestPipelineEdgeCases:
    """Edge cases for v3.4 pipeline."""

    def test_01_short_skeleton(self):
        data = _make_skeleton(num_frames=5, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape

    def test_02_all_nan_frame(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        data[30, :, :] = np.nan
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_bone_length_solver=True,
            use_drift_monitor=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape

    def test_03_high_fps(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=120.0,
            use_rolling_shutter_compensation=True,
            rolling_shutter_readout_ms=8.3,
            use_bone_length_solver=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape

    def test_04_many_keypoints(self):
        data = _make_skeleton(num_frames=60, num_keypoints=133)
        result, report = run_full_pipeline(
            data, fps=30.0,
            use_bone_length_solver=True,
            use_rts_smoother=True,
        )
        assert result.shape == data.shape

    def test_05_no_enforce_bones_no_solver(self):
        data = _make_skeleton(num_frames=60, num_keypoints=None)
        result, report = run_full_pipeline(
            data, fps=30.0,
            enforce_bones=False,
            use_bone_length_solver=False,
        )
        assert "bone_enforcement_pre" not in report["stages"]
        assert "bone_solver_pre" not in report["stages"]
        assert result.shape == data.shape
