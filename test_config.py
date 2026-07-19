"""
Shared test configuration — absolute paths for test data.

Set environment variable FREEMOCAP_TEST_DATA to override the default location.
Default: <project_root>/test_data/freemocap_test_data
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TEST_DATA_ROOT = os.environ.get(
    "FREEMOCAP_TEST_DATA",
    os.path.join(_PROJECT_ROOT, "test_data", "freemocap_test_data"),
)

CALIB_PATH = os.path.join(
    TEST_DATA_ROOT, "freemocap_test_data_camera_calibration.toml"
)
VID_DIR = os.path.join(TEST_DATA_ROOT, "synchronized_videos")
ARUCO_CONFIG = os.path.join(_PROJECT_ROOT, "tools", "actor_marker_map.yaml")

TWO_PERSON_VIDEO = os.path.join(
    _PROJECT_ROOT, "test_data", "two_persons_v2", "two_people_talking.mp4"
)
