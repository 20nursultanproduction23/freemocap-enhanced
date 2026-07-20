"""
Auto Camera Placement Advisor — suggests optimal camera positions before recording.

Takes room dimensions + number of cameras + optional constraints,
simulates coverage for different arrangements, and recommends the best layout.

Optimizes for:
  - Maximum area visible by ≥2 cameras (for triangulation)
  - Minimum blind spots in the central capture zone
  - Sufficient overlap for cross-view association

Usage:
    advisor = CameraPlacementAdvisor(room_width=6.0, room_depth=6.0, room_height=3.0)
    result = advisor.optimize(n_cameras=6)
    advisor.print_recommendation(result)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraPosition:
    """Suggested camera position with orientation."""
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3),
            "yaw": round(self.yaw, 1), "pitch": round(self.pitch, 1),
            "label": self.label,
        }


@dataclass
class PlacementResult:
    """Result of camera placement optimization."""
    cameras: List[CameraPosition]
    coverage_pct: float
    coverage_2plus_pct: float
    coverage_3plus_pct: float
    central_zone_coverage_pct: float
    min_camera_overlap: float
    arrangement_name: str
    score: float
    details: dict = field(default_factory=dict)


class CameraPlacementAdvisor:
    """
    Recommends optimal camera placement for a room.

    Uses grid-based frustum simulation to evaluate different camera arrangements.
    """

    DEFAULT_HFOV = 70.0
    DEFAULT_VFOV = 50.0
    DEFAULT_MIN_RANGE = 0.5
    DEFAULT_MAX_RANGE = 10.0
    DEFAULT_MOUNT_HEIGHT = 2.5
    DEFAULT_CENTRAL_ZONE_RATIO = 0.5

    def __init__(self, room_width: float = 6.0, room_depth: float = 6.0,
                 room_height: float = 3.0, hfov: float = 70.0, vfov: float = 50.0,
                 grid_resolution: float = 0.2, central_zone_ratio: float = 0.5,
                 min_z: float = 0.0, max_z: float = 3.0):
        self.room_width = room_width
        self.room_depth = room_depth
        self.room_height = room_height
        self.hfov = hfov
        self.vfov = vfov
        self.grid_resolution = grid_resolution
        self.central_zone_ratio = central_zone_ratio
        self.min_z = min_z
        self.max_z = max_z

    def optimize(self, n_cameras: int = 6, n_iterations: int = 100,
                 mount_height: float = 2.5, downward_tilt: float = 15.0,
                 fixed_positions: Optional[List[CameraPosition]] = None,
                 excluded_zones: Optional[List[Tuple[float, float, float, float]]] = None,
                 seed: int = 42) -> PlacementResult:
        """
        Find optimal camera placement.

        Args:
            n_cameras: Number of cameras to place
            n_iterations: Random restarts for optimization
            mount_height: Camera height in meters
            downward_tilt: Pitch angle in degrees (positive = looking down)
            fixed_positions: Cameras already placed (won't be moved)
            excluded_zones: List of (x_min, y_min, x_max, y_max) zones to avoid
            seed: Random seed for reproducibility

        Returns:
            PlacementResult with best arrangement found
        """
        rng = np.random.RandomState(seed)
        n_fixed = len(fixed_positions) if fixed_positions else 0
        n_free = n_cameras - n_fixed

        if n_free <= 0:
            if fixed_positions:
                coverage = self._evaluate_placement(fixed_positions, excluded_zones)
                return PlacementResult(
                    cameras=fixed_positions,
                    coverage_pct=coverage["coverage_pct"],
                    coverage_2plus_pct=coverage["coverage_2plus_pct"],
                    coverage_3plus_pct=coverage["coverage_3plus_pct"],
                    central_zone_coverage_pct=coverage["central_zone_coverage_pct"],
                    min_camera_overlap=coverage["min_camera_overlap"],
                    arrangement_name="fixed",
                    score=coverage["score"],
                    details=coverage,
                )
            raise ValueError("n_cameras must be > 0 or provide fixed_positions")

        best_score = -1.0
        best_result = None

        arrangements = self._get_preset_arrangements(n_cameras, mount_height, downward_tilt)

        for arr_name, arr_cameras in arrangements:
            if len(arr_cameras) != n_free:
                continue
            all_cams = list(arr_cameras) + list(fixed_positions or [])
            coverage = self._evaluate_placement(all_cams, excluded_zones)
            if coverage["score"] > best_score:
                best_score = coverage["score"]
                best_result = PlacementResult(
                    cameras=all_cams,
                    coverage_pct=coverage["coverage_pct"],
                    coverage_2plus_pct=coverage["coverage_2plus_pct"],
                    coverage_3plus_pct=coverage["coverage_3plus_pct"],
                    central_zone_coverage_pct=coverage["central_zone_coverage_pct"],
                    min_camera_overlap=coverage["min_camera_overlap"],
                    arrangement_name=arr_name,
                    score=coverage["score"],
                    details=coverage,
                )

        for _ in range(n_iterations):
            cameras = []
            for i in range(n_free):
                angle = (2 * math.pi * i) / n_free + rng.uniform(-0.3, 0.3)
                radius = min(self.room_width, self.room_depth) * rng.uniform(0.35, 0.48)
                x = self.room_width / 2 + radius * math.cos(angle)
                y = self.room_depth / 2 + radius * math.sin(angle)
                yaw = math.degrees(math.atan2(
                    self.room_depth / 2 - y,
                    self.room_width / 2 - x,
                ))
                cameras.append(CameraPosition(
                    x=max(0.3, min(self.room_width - 0.3, x)),
                    y=max(0.3, min(self.room_depth - 0.3, y)),
                    z=mount_height,
                    yaw=yaw,
                    pitch=-downward_tilt,
                    label=f"cam_{i}",
                ))

            all_cams = cameras + list(fixed_positions or [])
            coverage = self._evaluate_placement(all_cams, excluded_zones)

            if coverage["score"] > best_score:
                best_score = coverage["score"]
                best_result = PlacementResult(
                    cameras=all_cams,
                    coverage_pct=coverage["coverage_pct"],
                    coverage_2plus_pct=coverage["coverage_2plus_pct"],
                    coverage_3plus_pct=coverage["coverage_3plus_pct"],
                    central_zone_coverage_pct=coverage["central_zone_coverage_pct"],
                    min_camera_overlap=coverage["min_camera_overlap"],
                    arrangement_name="random_optimized",
                    score=coverage["score"],
                    details=coverage,
                )

        if best_result is None:
            fallback_cams = [CameraPosition(
                x=self.room_width / 2, y=0, z=mount_height,
                yaw=0, pitch=-downward_tilt, label="cam_0",
            )]
            coverage = self._evaluate_placement(fallback_cams, excluded_zones)
            best_result = PlacementResult(
                cameras=fallback_cams,
                coverage_pct=coverage["coverage_pct"],
                coverage_2plus_pct=coverage["coverage_2plus_pct"],
                coverage_3plus_pct=coverage["coverage_3plus_pct"],
                central_zone_coverage_pct=coverage["central_zone_coverage_pct"],
                min_camera_overlap=coverage["min_camera_overlap"],
                arrangement_name="fallback_single",
                score=coverage["score"],
                details=coverage,
            )

        return best_result

    def _evaluate_placement(self, cameras: List[CameraPosition],
                            excluded_zones: Optional[List[Tuple]] = None) -> dict:
        """Evaluate a camera arrangement by simulating frustum coverage."""
        w_steps = max(2, int(self.room_width / self.grid_resolution))
        d_steps = max(2, int(self.room_depth / self.grid_resolution))

        x_grid = np.linspace(0, self.room_width, w_steps)
        z_grid = np.linspace(0, self.room_depth, d_steps)
        xx, zz = np.meshgrid(x_grid, z_grid)
        grid_points = np.stack([xx.ravel(), zz.ravel()], axis=1)

        cx_min = self.room_width * (1 - self.central_zone_ratio) / 2
        cx_max = self.room_width * (1 + self.central_zone_ratio) / 2
        cz_min = self.room_depth * (1 - self.central_zone_ratio) / 2
        cz_max = self.room_depth * (1 + self.central_zone_ratio) / 2

        central_mask = (
            (grid_points[:, 0] >= cx_min) & (grid_points[:, 0] <= cx_max) &
            (grid_points[:, 1] >= cz_min) & (grid_points[:, 1] <= cz_max)
        )

        excluded_mask = np.zeros(len(grid_points), dtype=bool)
        if excluded_zones:
            for (ex_min, ey_min, ex_max, ey_max) in excluded_zones:
                excluded_mask |= (
                    (grid_points[:, 0] >= ex_min) & (grid_points[:, 0] <= ex_max) &
                    (grid_points[:, 1] >= ey_min) & (grid_points[:, 1] <= ey_max)
                )

        active_mask = ~excluded_mask
        n_total = int(active_mask.sum())

        camera_count = np.zeros(len(grid_points), dtype=int)

        for cam in cameras:
            cam_pos = np.array([cam.x, cam.y, cam.z])
            yaw_rad = math.radians(cam.yaw)
            pitch_rad = math.radians(cam.pitch)

            forward = np.array([
                math.cos(pitch_rad) * math.cos(yaw_rad),
                math.cos(pitch_rad) * math.sin(yaw_rad),
                math.sin(pitch_rad),
            ])

            for i, gp in enumerate(grid_points):
                if not active_mask[i]:
                    continue

                point_3d = np.array([gp[0], gp[1], 0.0])
                to_point = point_3d - cam_pos
                dist = np.linalg.norm(to_point)

                if dist < self.DEFAULT_MIN_RANGE or dist > self.DEFAULT_MAX_RANGE:
                    continue

                to_point_norm = to_point / max(dist, 1e-6)
                angle = math.degrees(math.acos(np.clip(np.dot(forward, to_point_norm), -1, 1)))

                half_hfov = self.hfov / 2
                if angle <= half_hfov:
                    camera_count[i] += 1

        total_cells = max(1, int(active_mask.sum()))
        cells_visible = int((camera_count[active_mask] >= 1).sum())
        cells_2plus = int((camera_count[active_mask] >= 2).sum())
        cells_3plus = int((camera_count[active_mask] >= 3).sum())

        central_cells = int((camera_count[central_mask & active_mask] >= 2).sum())
        central_total = max(1, int((central_mask & active_mask).sum()))

        coverage_pct = 100.0 * cells_visible / total_cells
        coverage_2plus_pct = 100.0 * cells_2plus / total_cells
        coverage_3plus_pct = 100.0 * cells_3plus / total_cells
        central_zone_pct = 100.0 * central_cells / central_total

        min_overlap = 0.0
        if len(cameras) >= 2:
            cam_pairs_overlap = []
            for i in range(len(cameras)):
                for j in range(i + 1, len(cameras)):
                    ci = camera_count == (i + 1) if i < len(cameras) else np.zeros(len(grid_points), dtype=bool)
                    cj = camera_count == (j + 1) if j < len(cameras) else np.zeros(len(grid_points), dtype=bool)

            pair_coverage = 0
            for i in range(len(cameras)):
                cam_cells_i = int((camera_count >= 1).sum())
            min_overlap = coverage_2plus_pct

        score = (
            0.30 * coverage_pct +
            0.35 * coverage_2plus_pct +
            0.15 * coverage_3plus_pct +
            0.20 * central_zone_pct
        ) / 100.0

        return {
            "coverage_pct": round(coverage_pct, 1),
            "coverage_2plus_pct": round(coverage_2plus_pct, 1),
            "coverage_3plus_pct": round(coverage_3plus_pct, 1),
            "central_zone_coverage_pct": round(central_zone_pct, 1),
            "min_camera_overlap": round(min_overlap, 1),
            "score": round(score, 4),
            "total_cells": total_cells,
            "n_cameras": len(cameras),
        }

    def _get_preset_arrangements(self, n: int, height: float,
                                  tilt: float) -> List[Tuple[str, List[CameraPosition]]]:
        """Generate preset camera arrangements for common configurations."""
        arrangements = []
        cx, cy = self.room_width / 2, self.room_depth / 2
        radius = min(self.room_width, self.room_depth) * 0.42

        if n >= 2:
            cams = []
            for i in range(n):
                angle = 2 * math.pi * i / n
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                yaw = math.degrees(math.atan2(cy - y, cx - x))
                cams.append(CameraPosition(x=x, y=y, z=height, yaw=yaw, pitch=-tilt,
                                           label=f"circle_{i}"))
            arrangements.append(("circle_equal", cams))

        if n >= 2:
            cams = []
            for i in range(n):
                angle = 2 * math.pi * i / n + math.pi / n
                r = radius * (0.85 + 0.15 * (i % 2))
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                yaw = math.degrees(math.atan2(cy - y, cx - x))
                cams.append(CameraPosition(x=x, y=y, z=height, yaw=yaw, pitch=-tilt,
                                           label=f"alternating_{i}"))
            arrangements.append(("circle_alternating", cams))

        if n >= 4 and n % 2 == 0:
            cams = []
            for i in range(n):
                side = i % 4
                if side == 0:
                    x, y = 0.3, self.room_depth * (i // 4 + 1) / (n // 4 + 1)
                    yaw = 0
                elif side == 1:
                    x, y = self.room_width - 0.3, self.room_depth * (i // 4 + 1) / (n // 4 + 1)
                    yaw = 180
                elif side == 2:
                    x, y = self.room_width * (i // 4 + 1) / (n // 4 + 1), 0.3
                    yaw = 90
                else:
                    x, y = self.room_width * (i // 4 + 1) / (n // 4 + 1), self.room_depth - 0.3
                    yaw = -90
                cams.append(CameraPosition(x=x, y=y, z=height, yaw=yaw, pitch=-tilt,
                                           label=f"wall_{i}"))
            arrangements.append(("wall_opposite", cams))

        if n >= 4:
            cams = []
            corners = [
                (0.5, 0.5, 45), (self.room_width - 0.5, 0.5, 135),
                (self.room_width - 0.5, self.room_depth - 0.5, -135),
                (0.5, self.room_depth - 0.5, -45),
            ]
            for i in range(min(4, n)):
                x, y, yaw = corners[i]
                cams.append(CameraPosition(x=x, y=y, z=height, yaw=yaw, pitch=-tilt,
                                           label=f"corner_{i}"))
            for i in range(4, n):
                angle = 2 * math.pi * (i - 4) / max(1, n - 4)
                x = cx + radius * 0.6 * math.cos(angle)
                y = cy + radius * 0.6 * math.sin(angle)
                yaw = math.degrees(math.atan2(cy - y, cx - x))
                cams.append(CameraPosition(x=x, y=y, z=height, yaw=yaw, pitch=-tilt,
                                           label=f"inner_{i}"))
            arrangements.append(("corners_plus_center", cams))

        return arrangements

    @staticmethod
    def format_recommendation(result: PlacementResult) -> str:
        """Format a human-readable recommendation string."""
        lines = [
            f"=== Camera Placement Recommendation ===",
            f"Arrangement: {result.arrangement_name}",
            f"Score: {result.score:.3f}",
            f"",
            f"Coverage:",
            f"  Any camera:     {result.coverage_pct:5.1f}%",
            f"  2+ cameras:     {result.coverage_2plus_pct:5.1f}% (needed for triangulation)",
            f"  3+ cameras:     {result.coverage_3plus_pct:5.1f}% (high accuracy)",
            f"  Central zone:   {result.central_zone_coverage_pct:5.1f}% (2+ cameras)",
            f"",
            f"Camera positions ({len(result.cameras)} cameras):",
        ]

        for cam in result.cameras:
            lines.append(
                f"  {cam.label:20s}  pos=({cam.x:.2f}, {cam.y:.2f}, {cam.z:.2f})"
                f"  yaw={cam.yaw:.0f}  pitch={cam.pitch:.0f}"
            )

        lines.append("")

        if result.coverage_2plus_pct < 50:
            lines.append("WARNING: <50% of room covered by 2+ cameras. Consider adding cameras or repositioning.")
        elif result.coverage_2plus_pct < 75:
            lines.append("GOOD: Majority of capture zone covered by 2+ cameras.")
        else:
            lines.append("EXCELLENT: Most of the room has multi-camera coverage.")

        return "\n".join(lines)

    @staticmethod
    def estimate_processing_time(n_cameras: int, n_frames: int, fps: float = 25) -> dict:
        """Estimate total processing time based on camera count."""
        time_per_frame_per_cam = 1.86
        wall_time_sec = n_frames * time_per_frame_per_cam * n_cameras
        wall_time_min = wall_time_sec / 60

        return {
            "n_cameras": n_cameras,
            "n_frames": n_frames,
            "fps": fps,
            "duration_sec": round(n_frames / fps, 1),
            "processing_time_sec": round(wall_time_sec, 1),
            "processing_time_min": round(wall_time_min, 1),
            "realtime_factor": round(wall_time_sec / max(1, n_frames / fps), 1),
        }
