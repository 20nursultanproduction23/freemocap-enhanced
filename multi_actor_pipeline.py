"""
Multi-Actor Pipeline — Stage 5 of multi-actor pipeline

Applies the existing v3.1 post-processing pipeline independently to each
actor's 3D skeleton, with shared floor plane estimation.

Key design:
  - Each actor gets its own independent v3.1 processing (filter, bones, etc.)
  - Floor plane is estimated from Actor 0 (or the actor with most data)
    and shared with all other actors — they stand on the same floor
  - All other stages (jitter, bones, wrist) are per-actor

Usage:
    from multi_actor_pipeline import process_multi_actor
    results = process_multi_actor(
        actor_results,  # from Stage 4
        fps=30.0,
        align_floor=True,
    )
"""
import numpy as np
from typing import Dict, List, Optional

from pipeline import run_full_pipeline
from per_actor_triangulation import ActorTriangulationResult
from floor_plane import estimate_ground_plane, align_to_ground


def process_multi_actor(
    actor_results: Dict[int, ActorTriangulationResult],
    fps: float = 30.0,
    align_floor: bool = True,
    target_height: Optional[float] = None,
    **pipeline_kwargs,
) -> Dict[int, dict]:
    """Process all actors through the v3.1 pipeline with shared floor.

    Args:
        actor_results: dict from Stage 4 — persistent_id → ActorTriangulationResult
        fps: video frame rate
        align_floor: whether to align skeletons to ground plane
        target_height: optional target height in mm for scaling
        **pipeline_kwargs: passed to run_full_pipeline (filter params, etc.)

    Returns:
        Dict mapping actor_id → {
            "processed_skeleton": (numFrames, 133, 3) array,
            "pipeline_report": dict from run_full_pipeline,
            "floor_info": dict with shared floor plane data (only for first actor),
        }
    """
    print("=" * 60)
    print("  Multi-Actor Pipeline (Stage 5)")
    print("=" * 60)
    print(f"  Actors to process: {sorted(actor_results.keys())}")

    # Step 1: Process Actor 0 (or first actor) with floor estimation
    actor_ids = sorted(actor_results.keys())
    if not actor_ids:
        print("  No actors to process!")
        return {}

    # Estimate floor from first actor (before full processing)
    # We need the raw skeleton for floor estimation
    shared_floor_normal = None
    shared_floor_point = None
    floor_estimated_from = None

    if align_floor:
        print("\n  Estimating shared floor plane...")
        for pid in actor_ids:
            res = actor_results[pid]
            visible_count = int(np.sum(res.visible_frames))
            if visible_count < 20:
                continue

            normal, point, info = estimate_ground_plane(res.skeleton_3d, fps=fps)
            contact_count = info.get("num_contact_points", 0)

            if contact_count >= 5:
                shared_floor_normal = normal
                shared_floor_point = point
                floor_estimated_from = pid
                print(f"    Floor estimated from Actor {pid}: "
                      f"normal=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}], "
                      f"contact points: {contact_count}")
                break

        if shared_floor_normal is None:
            print("    WARNING: Could not estimate floor from any actor, using default Y-up")
            shared_floor_normal = np.array([0.0, 1.0, 0.0])
            shared_floor_point = np.zeros(3)
            floor_estimated_from = "default"

    # Step 2: Process each actor independently
    results = {}
    for pid in actor_ids:
        res = actor_results[pid]
        print(f"\n{'='*60}")
        print(f"  Processing Actor {pid}")
        print(f"{'='*60}")

        skeleton = res.skeleton_3d.copy()  # (F, 133, 3)

        # Replace NaN with 0 for pipeline processing (pipeline expects no NaN)
        # Pipeline handles NaN internally via its own gap detection
        # Actually, pipeline expects the raw data — NaN values are handled by
        # outlier detection and gap filling stages

        # Run the full v3.1 pipeline
        processed, report = run_full_pipeline(
            skeleton,
            fps=fps,
            align_floor=False,  # We handle floor separately with shared data
            **pipeline_kwargs,
        )

        # Apply shared floor alignment
        if align_floor and shared_floor_normal is not None:
            print(f"\n  Applying shared floor from Actor {floor_estimated_from}...")
            processed, floor_info = align_to_ground(
                processed,
                ground_normal=shared_floor_normal,
                ground_point=shared_floor_point,
                target_height=target_height,
            )
            report["shared_floor"] = {
                "estimated_from_actor": floor_estimated_from,
                "normal": shared_floor_normal.tolist(),
                "point": shared_floor_point.tolist(),
            }
            if "scale_factor" in floor_info:
                print(f"    Scaled: {floor_info['original_height_mm']:.0f}mm → "
                      f"{target_height}mm (x{floor_info['scale_factor']:.3f})")
            if "rotation_angle_deg" in floor_info:
                print(f"    Rotated: {floor_info['rotation_angle_deg']:.1f}°")

        results[pid] = {
            "processed_skeleton": processed,
            "pipeline_report": report,
        }

    print(f"\n{'='*60}")
    print(f"  Multi-Actor Pipeline complete!")
    print(f"  Processed {len(results)} actor(s)")
    print(f"{'='*60}")

    return results


def get_actor_skeleton(
    multi_actor_results: Dict[int, dict],
    actor_id: int,
) -> Optional[np.ndarray]:
    """Get processed skeleton for a specific actor.

    Args:
        multi_actor_results: output from process_multi_actor()
        actor_id: persistent actor ID

    Returns:
        (numFrames, numTrackedPoints, 3) array or None
    """
    if actor_id in multi_actor_results:
        return multi_actor_results[actor_id]["processed_skeleton"]
    return None


def get_pipeline_summary(
    multi_actor_results: Dict[int, dict],
) -> dict:
    """Get summary statistics across all processed actors.

    Returns:
        dict with per-actor timing, NaN stats, etc.
    """
    summary = {
        "num_actors": len(multi_actor_results),
        "per_actor": {},
    }

    total_time = 0.0
    total_nan_before = 0
    total_nan_after = 0

    for pid, data in sorted(multi_actor_results.items()):
        report = data["pipeline_report"]
        actor_time = report.get("total_time_seconds", 0)
        nan_before = report.get("nan_before", 0)
        nan_after = report.get("nan_after", 0)

        summary["per_actor"][pid] = {
            "time_seconds": actor_time,
            "nan_before": nan_before,
            "nan_after": nan_after,
            "nan_reduced": nan_before - nan_after,
        }
        total_time += actor_time
        total_nan_before += nan_before
        total_nan_after += nan_after

    summary["total_time_seconds"] = total_time
    summary["total_nan_before"] = total_nan_before
    summary["total_nan_after"] = total_nan_after
    summary["total_nan_reduced"] = total_nan_before - total_nan_after

    return summary
