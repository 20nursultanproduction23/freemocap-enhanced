"""
Load camera calibration from FreeMoCap TOML format.

Converts Rodrigues rotation vectors to 3x3 rotation matrices,
and builds (3, 4) extrinsic matrices [R|t] for use with
CrossViewAssociator.
"""
import numpy as np
import cv2
from typing import List, Tuple


def load_calibration_toml(toml_path: str) -> dict:
    """Load FreeMoCap calibration TOML and return camera parameters.

    Args:
        toml_path: path to calibration TOML file

    Returns:
        dict with keys:
            camera_matrices: list of (3, 3) intrinsic K matrices
            extrinsic_matrices: list of (3, 4) [R|t] matrices
            image_sizes: list of (width, height) tuples
            names: list of camera name strings
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    camera_matrices = []
    extrinsic_matrices = []
    image_sizes = []
    names = []

    for key in sorted(data.keys()):
        if not key.startswith("cam_"):
            continue

        cam = data[key]
        names.append(cam.get("name", key))

        # Intrinsic matrix K (3x3)
        K = np.array(cam["matrix"], dtype=np.float64)

        # Image size: TOML stores [height, width], we need (width, height)
        size_raw = cam["size"]
        image_sizes.append((int(size_raw[1]), int(size_raw[0])))

        # Rotation: Rodrigues vector (3,) -> rotation matrix (3, 3)
        rvec = np.array(cam["rotation"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)

        # Translation vector (3,)
        t = np.array(cam["translation"], dtype=np.float64)

        # Extrinsic matrix [R|t] (3, 4)
        extrinsic = np.hstack([R, t.reshape(3, 1)])

        camera_matrices.append(K)
        extrinsic_matrices.append(extrinsic)

    return {
        "camera_matrices": camera_matrices,
        "extrinsic_matrices": extrinsic_matrices,
        "image_sizes": image_sizes,
        "names": names,
    }
