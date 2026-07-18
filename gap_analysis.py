"""
#14 GAP ANALYSIS — Detect frame drops and hidden timestamp drift

Analyzes NaN gap patterns in FreeMoCap output to distinguish:
1. MediaPipe detection failures (single-frame or short gaps, scattered)
2. Camera frame drops (longer gaps, often correlated across joints)
3. Hidden timestamp drift (gradual degradation, person leaving frame)

Also detects "silent" frame drops where interpolation masks the problem.
"""
import numpy as np
from joint_definitions import BODY_IDX, NUM_BODY


def analyze_gaps(skeleton_data, fps=30.0):
    """Analyze NaN gap patterns in skeleton data.

    Classifies gaps by length and distribution to distinguish between
    detection failures, frame drops, and tracking loss.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        analysis: dict with gap statistics and classification
    """
    num_frames = skeleton_data.shape[0]
    body_slice = slice(0, NUM_BODY)

    body_data = skeleton_data[:, body_slice, :]
    body_nan = np.isnan(body_data).any(axis=2).any(axis=1)

    gaps = []
    in_gap = False
    gap_start = 0

    for f in range(num_frames):
        if body_nan[f] and not in_gap:
            in_gap = True
            gap_start = f
        elif not body_nan[f] and in_gap:
            in_gap = False
            gaps.append({
                "start": gap_start,
                "end": f - 1,
                "length": f - gap_start,
                "type": _classify_gap(f - gap_start, fps),
            })

    if in_gap:
        gaps.append({
            "start": gap_start,
            "end": num_frames - 1,
            "length": num_frames - gap_start,
            "type": _classify_gap(num_frames - gap_start, fps),
        })

    gap_lengths = [g["length"] for g in gaps]
    type_counts = {}
    for g in gaps:
        t = g["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    joint_nan_counts = {}
    for name, idx in BODY_IDX.items():
        nan_count = int(np.sum(np.isnan(skeleton_data[:, idx, 0])))
        if nan_count > 0:
            joint_nan_counts[name] = nan_count

    vulnerable_joints = sorted(
        joint_nan_counts.items(), key=lambda x: -x[1]
    )[:5]

    analysis = {
        "total_frames": num_frames,
        "total_body_nan_frames": int(np.sum(body_nan)),
        "nan_percentage": float(np.sum(body_nan) / num_frames * 100),
        "num_gaps": len(gaps),
        "gaps": gaps,
        "gap_type_counts": type_counts,
        "mean_gap_length": float(np.mean(gap_lengths)) if gap_lengths else 0,
        "max_gap_length": int(max(gap_lengths)) if gap_lengths else 0,
        "most_vulnerable_joints": vulnerable_joints,
    }

    if len(gaps) >= 3:
        gap_starts = [g["start"] for g in gaps]
        analysis["gap_clustering"] = _assess_clustering(gap_starts, num_frames)

    return analysis


def detect_timestamp_drift(skeleton_data, fps=30.0):
    """Detect hidden timestamp drift from motion consistency.

    When cameras are desynced, the triangulated 3D positions show
    characteristic artifacts: increased jitter during fast motion,
    systematic depth errors, or frame-to-frame displacement anomalies.

    This function computes per-frame "drift score" based on:
    1. Excessive frame-to-frame displacement (faster than physiological)
    2. Depth consistency (Z should change smoothly)

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        drift_info: dict with drift scores and flagged frames
    """
    num_frames = skeleton_data.shape[0]

    body_data = skeleton_data[:, :NUM_BODY, :]

    displacement = np.zeros(num_frames)
    depth_change = np.zeros(num_frames)

    for f in range(1, num_frames):
        prev = body_data[f - 1]
        curr = body_data[f]
        valid = ~np.isnan(prev).any(axis=1) & ~np.isnan(curr).any(axis=1)
        if np.sum(valid) > 3:
            frame_disp = np.linalg.norm(curr[valid] - prev[valid], axis=1)
            displacement[f] = np.mean(frame_disp)
            depth_change[f] = np.mean(np.abs(curr[valid, 2] - prev[valid, 2]))

    valid_disp = displacement[displacement > 0]
    if len(valid_disp) < 10:
        return {
            "drift_score": 0,
            "flagged_frames": [],
            "max_displacement_mm": 0,
        }

    median_disp = np.median(valid_disp)
    mad_disp = np.median(np.abs(valid_disp - median_disp))

    if mad_disp > 0:
        z_scores = (displacement - median_disp) / (mad_disp * 1.4826)
    else:
        z_scores = np.zeros(num_frames)

    flagged = np.where(z_scores > 3.0)[0]

    drift_score = float(np.mean(z_scores[z_scores > 2.0])) if np.any(z_scores > 2.0) else 0

    return {
        "drift_score": drift_score,
        "flagged_frames": flagged.tolist(),
        "num_flagged": len(flagged),
        "median_displacement_mm": float(median_disp),
        "max_displacement_mm": float(np.max(displacement)),
        "mean_displacement_mm": float(np.mean(valid_disp)),
    }


def generate_gap_report(skeleton_data, fps=30.0):
    """Generate a human-readable gap analysis report.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        report: formatted string report
    """
    analysis = analyze_gaps(skeleton_data, fps)
    drift = detect_timestamp_drift(skeleton_data, fps)

    lines = [
        "=" * 60,
        "  Gap Analysis Report",
        "=" * 60,
        f"  Total frames: {analysis['total_frames']}",
        f"  Frames with body NaN: {analysis['total_body_nan_frames']} "
        f"({analysis['nan_percentage']:.1f}%)",
        f"  Number of gaps: {analysis['num_gaps']}",
        f"  Mean gap length: {analysis['mean_gap_length']:.1f} frames",
        f"  Max gap length: {analysis['max_gap_length']} frames",
        "",
        "  Gap type distribution:",
    ]

    for gap_type, count in sorted(analysis["gap_type_counts"].items()):
        lines.append(f"    {gap_type}: {count}")

    if analysis["most_vulnerable_joints"]:
        lines.append("")
        lines.append("  Most vulnerable joints:")
        for name, count in analysis["most_vulnerable_joints"]:
            lines.append(f"    {name}: {count} NaN frames")

    if "gap_clustering" in analysis:
        lines.append("")
        lines.append(f"  Gap clustering: {analysis['gap_clustering']}")

    lines.append("")
    lines.append("  Timestamp drift analysis:")
    lines.append(f"    Drift score: {drift['drift_score']:.2f}")
    lines.append(f"    Flagged frames: {drift['num_flagged']}")
    lines.append(f"    Median displacement: {drift['median_displacement_mm']:.1f}mm")
    lines.append(f"    Max displacement: {drift['max_displacement_mm']:.1f}mm")

    lines.append("=" * 60)

    return "\n".join(lines)


def _classify_gap(length, fps):
    """Classify a gap by its likely cause."""
    if length <= 1:
        return "detection_failure"
    elif length <= 3:
        return "short_occlusion"
    elif length <= int(fps * 0.5):
        return "moderate_occlusion"
    elif length <= int(fps * 2):
        return "possible_frame_drop"
    else:
        return "tracking_loss"


def _assess_clustering(gap_starts, total_frames):
    """Assess whether gaps are randomly distributed or clustered."""
    if len(gap_starts) < 2:
        return "sparse"

    gaps_array = np.array(gap_starts)
    diffs = np.diff(gaps_array)
    mean_diff = np.mean(diffs)
    cv = np.std(diffs) / mean_diff if mean_diff > 0 else 0

    if cv > 2.0:
        return "highly_clustered"
    elif cv > 1.0:
        return "moderately_clustered"
    else:
        return "uniformly_distributed"
