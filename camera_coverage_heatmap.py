"""
Camera Coverage Heatmap

Visualizes which parts of the capture space are covered by how many cameras
using calibration matrices (K, [R|t]) and image sizes. Projects a 3D grid
through each camera's projection matrix and checks if the projected point
falls within image bounds. Outputs a 2D top-down coverage heatmap.

Complements camera_coverage.py (frustum-based) with intrinsics-aware
projection that accounts for lens distortion implicitly via the matrix.
"""
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import csv
import os


class CameraCoverageHeatmap:
    """Compute and visualize multi-camera coverage of a 3D capture volume.

    Uses camera intrinsics (K), extrinsics ([R|t]), and image sizes to
    determine coverage. A 3D grid of world points is projected to each camera;
    points landing inside the image rectangle contribute to that camera's
    coverage count.

    Args:
        calibration: dict with keys 'camera_matrices' (list of (3,3)),
            'extrinsic_matrices' (list of (3,4)), 'image_sizes' (list of
            (w, h)). Matches the format returned by calibration_loader.
        bounds: optional dict with keys 'x', 'y', 'z', each a (min, max)
            tuple in meters. Default: -1.0 to 1.0 on all axes (2m cube).
        resolution: grid spacing in meters. Default 0.05 (5cm).
        projection_height: y-value of the horizontal slice to visualize.
            Default 1.0 (roughly pelvis height). Set to None to compute
            full 3D coverage collapsed to top-down (max over y).
    """

    def __init__(
        self,
        calibration: Dict[str, Any],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        resolution: float = 0.05,
        projection_height: Optional[float] = 1.0,
    ):
        self.camera_matrices = calibration["camera_matrices"]
        self.extrinsic_matrices = calibration["extrinsic_matrices"]
        self.image_sizes = calibration["image_sizes"]
        self.n_cameras = len(self.camera_matrices)

        if bounds is None:
            self.bounds = {"x": (-1.0, 1.0), "y": (0.0, 2.0), "z": (-1.0, 1.0)}
        else:
            self.bounds = bounds
        self.resolution = resolution
        self.projection_height = projection_height

        self._coverage = None
        self._grid_info = None

    def _build_projection_matrices(self) -> List[np.ndarray]:
        """Compute P = K @ [R|t] for each camera."""
        projections = []
        for K, ext in zip(self.camera_matrices, self.extrinsic_matrices):
            K = np.asarray(K, dtype=np.float64)
            ext = np.asarray(ext, dtype=np.float64)
            P = K @ ext
            projections.append(P)
        return projections

    def _build_grid(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Create 3D grid of world points.

        Returns:
            points: (N, 3) float64 array of grid vertices
            info: dict with axes lengths and cell_size
        """
        x_min, x_max = self.bounds["x"]
        y_min, y_max = self.bounds["y"]
        z_min, z_max = self.bounds["z"]

        xs = np.arange(x_min, x_max + self.resolution * 0.5, self.resolution)
        zs = np.arange(z_min, z_max + self.resolution * 0.5, self.resolution)

        if self.projection_height is not None:
            ys = np.array([self.projection_height])
        else:
            ys = np.arange(y_min, y_max + self.resolution * 0.5, self.resolution)

        nx, ny, nz = len(xs), len(ys), len(zs)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")

        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

        info = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "cell_size": self.resolution,
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max),
            "z_range": (z_min, z_max),
        }
        return points.astype(np.float64), info

    def compute(self) -> np.ndarray:
        """Compute the coverage heatmap.

        For each grid point, projects to every camera and checks if the
        projected pixel falls within image bounds (0..w, 0..h).

        Returns:
            coverage: (grid_z, grid_x) int32 array of camera counts.
                When projection_height is set, y-axis is collapsed.
        """
        proj_matrices = self._build_projection_matrices()
        grid_points, info = self._build_grid()

        n_total = grid_points.shape[0]
        nx, ny, nz = info["nx"], info["ny"], info["nz"]

        coverage_3d = np.zeros((nx, ny, nz), dtype=np.int32)

        ones = np.ones((n_total, 1), dtype=np.float64)
        pts_homog = np.hstack([grid_points, ones])

        for cam_idx in range(self.n_cameras):
            P = proj_matrices[cam_idx]
            w, h = self.image_sizes[cam_idx]

            projected = pts_homog @ P.T

            denom = projected[:, 2]
            valid = np.abs(denom) > 1e-10

            u = np.full(n_total, -1.0)
            v = np.full(n_total, -1.0)
            u[valid] = projected[valid, 0] / denom[valid]
            v[valid] = projected[valid, 1] / denom[valid]

            in_bounds = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)

            counts = in_bounds.astype(np.int32).reshape(nx, ny, nz)
            coverage_3d += counts

        if self.projection_height is not None:
            coverage_2d = coverage_3d[:, 0, :]
        else:
            coverage_2d = np.max(coverage_3d, axis=1)

        self._coverage = coverage_2d
        self._grid_info = info
        return coverage_2d

    def save(
        self,
        output_path: str,
        fmt: str = "auto",
        title: str = "Camera Coverage Heatmap",
    ) -> str:
        """Save the computed heatmap to file.

        Args:
            output_path: destination file path.
            fmt: 'png', 'csv', or 'auto' (infer from extension).
            title: title for PNG plots.

        Returns:
            Absolute path of the saved file.

        Raises:
            RuntimeError: if compute() has not been called.
            ValueError: if format is unsupported.
        """
        if self._coverage is None:
            raise RuntimeError("Call compute() before save().")

        if fmt == "auto":
            ext = os.path.splitext(output_path)[1].lower()
            if ext == ".csv":
                fmt = "csv"
            elif ext in (".png", ".jpg", ".jpeg", ".svg"):
                fmt = "png"
            else:
                fmt = "csv"

        if fmt == "csv":
            self._save_csv(output_path)
        elif fmt == "png":
            self._save_png(output_path, title)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return os.path.abspath(output_path)

    def _save_csv(self, path: str) -> None:
        cell = self._grid_info["cell_size"]
        x_min, x_max = self._grid_info["x_range"]
        z_min, z_max = self._grid_info["z_range"]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"# Camera Coverage Heatmap",
                f"# cell_size={cell:.4f}m",
                f"# x=[{x_min:.2f},{x_max:.2f}] z=[{z_min:.2f},{z_max:.2f}]",
                f"# shape=({self._coverage.shape[0]},{self._coverage.shape[1]})",
            ])
            for row in self._coverage:
                writer.writerow(row.tolist())

    def _save_png(self, path: str, title: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            self._save_csv(path.replace(".png", ".csv"))
            return

        cell = self._grid_info["cell_size"]
        x_min, x_max = self._grid_info["x_range"]
        z_min, z_max = self._grid_info["z_range"]
        nx, nz = self._coverage.shape

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = ["#1a1a2e", "#e94560", "#f5a623", "#0f3460", "#16213e"]
        blind_cmap = LinearSegmentedColormap.from_list(
            "coverage", ["#0d0d0d", "#e94560", "#f5a623", "#53d769", "#1b98e0"],
            N=256,
        )

        extent = [x_min, x_max, z_min, z_max]
        im = ax.imshow(
            self._coverage,
            origin="lower",
            extent=extent,
            cmap=blind_cmap,
            vmin=0,
            vmax=max(int(self._coverage.max()), 3),
            aspect="equal",
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Number of cameras", fontsize=11)
        ax.set_xlabel("X (meters)", fontsize=11)
        ax.set_ylabel("Z (meters)", fontsize=11)
        ax.set_title(f"{title}\n{nx}x{nz} grid, {cell*100:.1f}cm resolution", fontsize=13)

        blind_mask = self._coverage == 0
        if blind_mask.any():
            blind_y, blind_x = np.where(blind_mask)
            ax.scatter(
                x_min + blind_x * cell + cell / 2,
                z_min + blind_y * cell + cell / 2,
                c="red",
                s=2,
                alpha=0.6,
                label="Blind zones",
            )
            ax.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def text_summary(self) -> str:
        """Return a console-friendly summary of coverage percentages.

        Returns:
            Multi-line string with per-camera-count percentages.
        """
        if self._coverage is None:
            raise RuntimeError("Call compute() before text_summary().")

        total = self._coverage.size
        max_count = int(self._coverage.max())

        lines = [
            f"Camera Coverage Summary ({self.n_cameras} cameras)",
            f"  Grid: {self._coverage.shape[1]}x{self._coverage.shape[0]} "
            f"({self._grid_info['cell_size']*100:.1f}cm resolution)",
            f"  Volume: x={self._grid_info['x_range']}, "
            f"z={self._grid_info['z_range']}",
        ]

        for k in range(max_count + 1):
            count = int(np.sum(self._coverage == k))
            pct = 100.0 * count / total
            label = f"{k} camera{'s' if k != 1 else ''}"
            marker = " *** BLIND ZONE" if k == 0 else ""
            lines.append(f"  {label:>12}: {count:>6} cells ({pct:5.1f}%){marker}")

        covered = int(np.sum(self._coverage >= 2))
        pct_covered = 100.0 * covered / total
        lines.append(f"  {'2+ cameras':>12}: {covered:>6} cells ({pct_covered:.1f}%)")
        lines.append(f"  {'3+ cameras':>12}: {int(np.sum(self._coverage >= 3)):>6} cells "
                      f"({100.0*np.sum(self._coverage >= 3)/total:.1f}%)")

        return "\n".join(lines)

    @property
    def coverage(self) -> Optional[np.ndarray]:
        """The computed coverage array, or None if not yet computed."""
        return self._coverage

    @property
    def grid_info(self) -> Optional[Dict[str, Any]]:
        """Grid metadata, or None if not yet computed."""
        return self._grid_info


def compute_coverage(
    calibration: Dict[str, Any],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    resolution: float = 0.05,
    projection_height: Optional[float] = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience function: compute coverage and return results.

    Args:
        calibration: dict with camera_matrices, extrinsic_matrices, image_sizes.
        bounds: optional x/y/z min/max dict. Default 2m cube centered at origin.
        resolution: grid spacing in meters.
        projection_height: y-slice to visualize. None for max-over-y.

    Returns:
        Tuple of (coverage_array, grid_info_dict).
    """
    heatmap = CameraCoverageHeatmap(
        calibration=calibration,
        bounds=bounds,
        resolution=resolution,
        projection_height=projection_height,
    )
    coverage = heatmap.compute()
    return coverage, heatmap.grid_info
