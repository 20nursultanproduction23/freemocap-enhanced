"""
Stage 7, Part A — ArUco Marker Generator

Generates printable ArUco markers for actor identification.
Each actor gets a unique marker ID from a YAML config file.

Dictionary: DICT_4X4_50 (4x4 white border, 4x4 data, 50 unique IDs)
Output: PNG files with physical size encoded in filename (e.g. actor0_50mm.png)

Usage:
    python tools/generate_actor_markers.py
    python tools/generate_actor_markers.py --config actor_marker_map.yaml --size-mm 50 --output-dir markers/
"""
import cv2
import numpy as np
import os
import sys
import argparse
import yaml


DEFAULT_CONFIG = {
    "dictionary": "DICT_4X4_50",
    "marker_size_mm": 50,
    "actors": {
        "actor_0": 0,
        "actor_1": 1,
    },
}

ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG


def save_config(config_path, config):
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def generate_marker_image(marker_id, dictionary_type, marker_size_px=400, border_bits=1):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

    border_px = int(marker_size_px * 0.1)
    canvas = np.ones(
        (marker_size_px + 2 * border_px, marker_size_px + 2 * border_px),
        dtype=np.uint8,
    ) * 255
    canvas[border_px : border_px + marker_size_px, border_px : border_px + marker_size_px] = marker_img

    return canvas


def generate_all_markers(config, output_dir):
    dictionary_name = config.get("dictionary", "DICT_4X4_50")
    dictionary_type = ARUCO_DICT_MAP.get(dictionary_name, cv2.aruco.DICT_4X4_50)
    size_mm = config.get("marker_size_mm", 50)
    actors = config.get("actors", {})

    os.makedirs(output_dir, exist_ok=True)

    generated = {}
    for actor_name, marker_id in actors.items():
        img = generate_marker_image(marker_id, dictionary_type)
        filename = f"{actor_name}_{size_mm}mm.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        generated[actor_name] = {
            "marker_id": marker_id,
            "filepath": filepath,
            "filename": filename,
            "size_mm": size_mm,
            "image_shape": img.shape,
        }

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate ArUco markers for actor identification")
    parser.add_argument(
        "--config", type=str, default="actor_marker_map.yaml",
        help="Path to actor-marker mapping YAML (default: actor_marker_map.yaml)",
    )
    parser.add_argument(
        "--size-mm", type=int, default=None,
        help="Physical marker size in mm (overrides config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="markers",
        help="Output directory for marker PNGs (default: markers/)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.size_mm is not None:
        config["marker_size_mm"] = args.size_mm

    if not os.path.exists(args.config):
        save_config(args.config, config)
        print(f"Created default config: {args.config}")

    generated = generate_all_markers(config, args.output_dir)

    for actor_name, info in generated.items():
        print(f"  {actor_name}: marker_id={info['marker_id']}, "
              f"size={info['size_mm']}mm, file={info['filepath']}, "
              f"shape={info['image_shape']}")

    return generated


if __name__ == "__main__":
    main()
