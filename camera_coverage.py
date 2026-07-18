"""
#20 CAMERA OVERLAP ZONE VISUALIZATION

Triangulation requires >=2 cameras seeing the same point. With multiple cameras
pointed at the center, the real overlap zone is smaller than the room area.
Actors at the periphery may be seen by only 1-2 cameras, making triangulation
unreliable.

This module computes a 2D floor-plane coverage map showing how many cameras
see each point on the floor.
"""
import numpy as np


def _euler_to_direction(yaw, pitch, roll):
    """Convert Euler angles (degrees) to a unit direction vector.

    Yaw rotates around Y (right-hand rule), pitch around X, roll around Z.
    Returns the forward (look-at) direction of the camera.

    Args:
        yaw: rotation around vertical axis in degrees
        pitch: rotation around lateral axis in degrees
        roll: rotation around optical axis in degrees (unused for direction)

    Returns:
        (3,) unit vector pointing in the camera's view direction
    """
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    x = np.cos(pitch_rad) * np.sin(yaw_rad)
    y = np.sin(pitch_rad)
    z = np.cos(pitch_rad) * np.cos(yaw_rad)
    vec = np.array([x, y, z])
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def _is_in_frustum(point, cam_pos, cam_dir, fov_h, fov_v):
    """Check if a 3D point lies within a camera's view frustum.

    The frustum is defined by the camera position, look direction, and
    horizontal/vertical field-of-view angles. The check ignores distance
    (no far-plane clipping).

    Args:
        point: (3,) world-space point
        cam_pos: (3,) camera position
        cam_dir: (3,) unit vector of camera look direction
        fov_h: horizontal field of view in degrees
        fov_v: vertical field of view in degrees

    Returns:
        bool: True if the point is within the frustum
    """
    to_point = np.array(point) - np.array(cam_pos)
    dist = np.linalg.norm(to_point)
    if dist < 1e-12:
        return True
    to_point = to_point / dist

    cos_angle = np.dot(cam_dir, to_point)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    half_h = fov_h / 2.0
    half_v = fov_v / 2.0

    if angle > max(half_h, half_v):
        return False

    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(cam_dir, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])

    right = np.cross(cam_dir, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        return True
    right = right / right_norm
    cam_up = np.cross(right, cam_dir)
    cam_up_norm = np.linalg.norm(cam_up)
    if cam_up_norm < 1e-12:
        return True
    cam_up = cam_up / cam_up_norm

    horiz_angle = np.degrees(np.arctan2(np.dot(to_point, right), np.dot(to_point, cam_dir)))
    vert_angle = np.degrees(np.arctan2(np.dot(to_point, cam_up), np.dot(to_point, cam_dir)))

    return abs(horiz_angle) <= half_h and abs(vert_angle) <= half_v


def compute_coverage_map(
    camera_positions, camera_orientations, camera_fov_h, camera_fov_v,
    room_size, resolution=100
):
    """Compute a 2D coverage map showing how many cameras see each floor cell.

    Projects each floor-grid cell center up to a nominal height (1.0m, roughly
    pelvis height) and tests frustum visibility from each camera.

    Args:
        camera_positions: list of (x, y, z) camera positions in world meters
        camera_orientations: list of (yaw, pitch, roll) Euler angles in degrees
        camera_fov_h: list of horizontal FOV in degrees (one per camera, or scalar)
        camera_fov_v: list of vertical FOV in degrees (None = use same as horizontal)
        room_size: (width, depth) of the room in meters
        resolution: grid cells per meter (default 100)

    Returns:
        dict with keys:
            'coverage_grid': (height_cells, width_cells) uint8 array
            'grid_origin': (x_min, y_min) world coords of grid[0,0]
            'cell_size': meters per cell
            'min_coverage': int (reliable threshold = 3)
            'reliable_zone': (height_cells, width_cells) bool array
            'unreliable_cells': list of (row, col, count) for cells with 0-1 cameras
    """
    n_cameras = len(camera_positions)
    if isinstance(camera_fov_h, (int, float)):
        camera_fov_h = [float(camera_fov_h)] * n_cameras
    if camera_fov_v is None:
        camera_fov_v = camera_fov_h[:]
    elif isinstance(camera_fov_v, (int, float)):
        camera_fov_v = [float(camera_fov_v)] * n_cameras

    width, depth = room_size
    n_cells_x = int(round(width * resolution))
    n_cells_z = int(round(depth * resolution))
    cell_size = width / n_cells_x

    cam_dirs = []
    for orient in camera_orientations:
        yaw, pitch, roll = orient
        cam_dirs.append(_euler_to_direction(yaw, pitch, roll))

    x_min = -width / 2.0
    z_min = -depth / 2.0

    x_centers = np.linspace(x_min + cell_size / 2, x_min + width - cell_size / 2, n_cells_x)
    z_centers = np.linspace(z_min + cell_size / 2, z_min + depth - cell_size / 2, n_cells_z)

    coverage_grid = np.zeros((n_cells_z, n_cells_x), dtype=np.uint8)
    nominal_height = 1.0

    for cam_idx in range(n_cameras):
        pos = np.array(camera_positions[cam_idx])
        dir_vec = cam_dirs[cam_idx]
        fov_h = float(camera_fov_h[cam_idx])
        fov_v = float(camera_fov_v[cam_idx])

        for row in range(n_cells_z):
            for col in range(n_cells_x):
                point = np.array([x_centers[col], nominal_height, z_centers[row]])
                if _is_in_frustum(point, pos, dir_vec, fov_h, fov_v):
                    coverage_grid[row, col] += 1

    min_coverage = 3
    reliable_zone = coverage_grid >= min_coverage

    unreliable_cells = []
    for row in range(n_cells_z):
        for col in range(n_cells_x):
            count = int(coverage_grid[row, col])
            if count <= 1:
                unreliable_cells.append((row, col, count))

    return {
        'coverage_grid': coverage_grid,
        'grid_origin': (x_min, z_min),
        'cell_size': cell_size,
        'min_coverage': min_coverage,
        'reliable_zone': reliable_zone,
        'unreliable_cells': unreliable_cells,
    }


def compute_capture_quality_map(
    camera_positions, camera_orientations, camera_fov_h,
    camera_confidences, room_size, resolution=100
):
    """Extended coverage map weighted by average detection confidence per camera.

    Instead of counting cameras, each camera contributes its confidence score
    when it sees a cell. The resulting grid shows the average confidence of
    all cameras that cover that cell.

    Args:
        camera_positions: list of (x, y, z) camera positions
        camera_orientations: list of (yaw, pitch, roll) Euler angles
        camera_fov_h: list of horizontal FOV in degrees
        camera_confidences: list of float confidence scores per camera (0-1)
        room_size: (width, depth) in meters
        resolution: grid cells per meter

    Returns:
        dict with keys:
            'quality_grid': (height_cells, width_cells) float array — mean confidence
            'coverage_grid': (height_cells, width_cells) uint8 — camera count
            'grid_origin': (x_min, y_min)
            'cell_size': float
            'reliable_zone': (height_cells, width_cells) bool — >=3 cameras
    """
    base = compute_coverage_map(
        camera_positions, camera_orientations, camera_fov_h,
        None, room_size, resolution
    )
    coverage = base['coverage_grid']
    n_cells_z, n_cells_x = coverage.shape

    n_cameras = len(camera_positions)
    width, depth = room_size
    n_cells_x_full = int(round(width * resolution))
    n_cells_z_full = int(round(depth * resolution))
    cell_size = width / n_cells_x_full
    x_min = -width / 2.0
    z_min = -depth / 2.0

    x_centers = np.linspace(x_min + cell_size / 2, x_min + width - cell_size / 2, n_cells_x)
    z_centers = np.linspace(z_min + cell_size / 2, z_min + depth - cell_size / 2, n_cells_z)

    cam_dirs = [_euler_to_direction(*o) for o in camera_orientations]
    quality_sum = np.zeros_like(coverage, dtype=np.float64)
    nominal_height = 1.0

    for cam_idx in range(n_cameras):
        pos = np.array(camera_positions[cam_idx])
        dir_vec = cam_dirs[cam_idx]
        fov_h = float(camera_fov_h[cam_idx])
        conf = float(camera_confidences[cam_idx])

        for row in range(n_cells_z):
            for col in range(n_cells_x):
                point = np.array([x_centers[col], nominal_height, z_centers[row]])
                if _is_in_frustum(point, pos, dir_vec, fov_h, fov_h):
                    quality_sum[row, col] += conf

    quality_grid = np.where(coverage > 0, quality_sum / coverage.astype(np.float64), 0.0)

    return {
        'quality_grid': quality_grid,
        'coverage_grid': coverage,
        'grid_origin': base['grid_origin'],
        'cell_size': base['cell_size'],
        'reliable_zone': base['reliable_zone'],
    }


def generate_ascii_map(coverage_grid, cell_size):
    """Render a coverage grid as ASCII art for terminal display.

    Character mapping:
        ' ' — 0 cameras
        '.' — 1 camera
        ':' — 2 cameras
        '+' — 3 cameras
        '#' — 4 cameras
        '@' — 5+ cameras

    Args:
        coverage_grid: 2D numpy array of coverage counts
        cell_size: meters per cell (used only for the info line)

    Returns:
        str: multi-line ASCII map with dimensions in the header
    """
    chars = np.array([' ', '.', ':', '+', '#', '@'])

    n_cells_z, n_cells_x = coverage_grid.shape

    lines = [
        f"Coverage Map  ({n_cells_x} x {n_cells_z} cells, {cell_size:.3f}m/cell)",
        f"Legend: ' '=0  '.'=1  ':'=2  '+'=3  '#'=4  '@'=5+",
        "",
    ]

    for row in range(n_cells_z):
        line_chars = []
        for col in range(n_cells_x):
            val = int(coverage_grid[row, col])
            idx = min(val, 5)
            line_chars.append(chars[idx])
        lines.append(''.join(line_chars))

    return '\n'.join(lines)
