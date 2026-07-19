# FreeMoCap Enhanced v3.2 — Installation Guide

## Requirements

- **Python 3.11** (recommended)
- **Windows 10/11** (tested), Linux/macOS (untested but should work)
- **NVIDIA GPU** with CUDA support (recommended for RTMPose acceleration)
- **Blender** (optional, for 3D visualization)

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### 2. Install FreeMoCap

```bash
pip install freemocap
```

### 3. Install additional dependencies

```bash
pip install rtmlib onnxruntime opencv-contrib-python pyyaml
```

### 4. Clone / copy this repository

Copy the `freemocap_enhanced` folder into your working directory, or clone it:

```bash
git clone -b multi-actor-support <repo-url>
```

### 5. Run

```bash
cd freemocap_enhanced
python run_freemocap_enhanced.py
```

The application will launch with:
- Custom auto-resizing logo
- Original FreeMoCap welcome dialog
- Enhanced welcome screen with EN/RU language toggle
- Release notes popup

## What's Included

### Pipeline Improvements (v3.2)
- Jitter filter (OneEuro + Butterworth)
- Bone length enforcement (FK-BFS)
- Foot sliding detection/correction
- Floor plane alignment (RANSAC + PCA)
- Self-occlusion detection
- NaN gap interpolation
- Left/right confusion correction
- Real-time logging + CSV export

### Multi-Person Tracking (BETA)
- RTMDet + RTMPose for 2D detection (133 keypoints)
- Epipolar geometry + Hungarian cross-view association
- Temporal tracking with persistent global IDs
- Per-actor DLT triangulation
- Physical interaction validation
- ArUco marker fallback

### GUI Enhancements
- Mode selector: Single Actor / Dual Actor
- Bilingual interface (English / Russian)
- Custom home screen logo (auto-resize)
- Dual-actor settings with ArUco preview

## Running Tests

Place test data in `test_data/freemocap_test_data/` relative to the project root, or set the environment variable:

```bash
set FREEMOCAP_TEST_DATA=D:\path\to\test_data\freemocap_test_data
```

Then run:

```bash
python -m pytest test_full_pipeline_integration.py -v
```

## Test Data Structure

```
test_data/
  freemocap_test_data/
    synchronized_videos/
      camera_0.mp4
      camera_1.mp4
      camera_2.mp4
    freemocap_test_data_camera_calibration.toml
```

## Troubleshooting

- **CUDA errors**: Ensure `onnxruntime-gpu` is installed instead of `onnxruntime`
- **Import errors**: Make sure all dependencies are installed in the same virtual environment
- **Logo not showing**: The `FREEMOCAP-BETA.png` file must be in the same directory as `run_freemocap_enhanced.py`
