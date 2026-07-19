# FreeMoCap Enhanced v3.2

Post-processing improvements and multi-person tracking for [FreeMoCap](https://freemocap.org/) — open-source markerless motion capture.

> **Status: BETA / Early Development**
> See [Honest Status](#honest-status) below.

---

## What This Is

This is an **add-on layer** for FreeMoCap v1.8.x that adds:

### Pipeline v3.2 Improvements
- **Jitter filter** (OneEuro + Butterworth) — smooth motion without lag
- **Bone length enforcement** (FK-BFS) — eliminates "breathing bones" artifact
- **Foot sliding** detection and correction
- **Floor plane alignment** (RANSAC + PCA)
- **Self-occlusion detection** via cross-camera consistency
- **NaN gap interpolation** for brief tracking drops
- **Left/right confusion** correction
- **Real-time logging** and export to CSV

### Multi-Person Tracking (BETA)
- **RTMDet + RTMPose** for 2D detection (133 keypoints per person)
- **Epipolar geometry + Hungarian algorithm** for cross-view association
- **Temporal tracking** with persistent global IDs
- **Per-actor DLT triangulation**
- **Physical interaction validation** (interpenetration + contact detection)
- **ArUco marker fallback** for identity tracking during occlusions
- **ArUco marker generator** with PNG export

### GUI Enhancements
- **Mode selector**: Single Actor / Dual Actor
- **Bilingual interface** (English / Russian)
- **Dual-actor settings** with ArUco marker preview

---

## Honest Status

**I'm going to be completely honest with you.**

This project was developed and tested primarily using:
- **Stock videos** downloaded from Mixkit and Pexels (free stock footage)
- **Synthetic data** generated in code (artificial 2-person scenarios)
- **Neural network inference** (RTMPose via rtmlib) run on those pre-recorded videos
- **Unit tests** with mock data and synthetic detections

**I do NOT have a working multi-camera setup yet.** My cameras are currently on order and should arrive soon. This means:

- The **single-person pipeline** (v3.2 improvements) was tested on FreeMoCap's own recorded data and works. These improvements are solid.
- The **multi-person pipeline** was validated on stock two-person videos and synthetic data. The math checks out, the tests pass (102/102), and the logic is sound — but **I have not tested it end-to-end with real cameras on real people in a real room.**
- The **GUI** was tested via unit tests (screens open, buttons connect, signals fire) but not through a full recording session with actual hardware.
- The **ArUco marker system** works in tests — markers generate, detect, and bind to skeletons — but I haven't verified it with a real 3-camera recording of two people wearing markers.

### What Might Not Work (Honestly)

- Camera synchronization quirks with specific camera models
- Multi-person tracking with fast movements or complex choreography
- Edge cases in cross-view association (e.g., actors standing very close for extended periods)
- Real-world lighting conditions that differ from stock videos
- The GUI might have rough edges in dual-actor mode that tests didn't catch
- Floor plane estimation might need calibration tuning with real camera setups

### What Happens Next

**When my cameras arrive (expected soon):**

1. I will set up a real multi-camera capture environment
2. Record actual two-person sessions
3. Test the full pipeline end-to-end with real data
4. Fix whatever breaks (and something probably will)
5. Post updated versions with verified working configurations
6. Add real capture data examples and calibration files
7. Update documentation with actual setup photos and troubleshooting

**I would rather be honest about what works and what doesn't than pretend everything is perfect.** If you try this and something breaks — I'm sorry. Open an issue and I'll do my best to fix it once I have the hardware to reproduce the problem.

---

## Quick Start

### Requirements
- Python 3.11
- Windows 10/11 (tested), Linux/macOS (untested)
- NVIDIA GPU (recommended)
- FreeMoCap v1.8.x

### Install

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install freemocap
pip install rtmlib onnxruntime opencv-contrib-python pyyaml
```

### Run

```bash
git clone -b multi-actor-support https://github.com/20nursultanproduction23/freemocap-enhanced.git
cd freemocap-enhanced
python run_freemocap_enhanced.py
```

### Generate ArUco Markers

```bash
python tools/generate_actor_markers.py
```

### Run Tests

```bash
set FREEMOCAP_TEST_DATA=D:\path\to\test_data
python -m pytest test_full_pipeline_integration.py -v
```

See [INSTALL[ENG].md](INSTALL%5BENG%5D.md) for detailed installation instructions.

---

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALL[ENG].md](INSTALL%5BENG%5D.md) | Installation guide (English) |
| [INSTALL[RUS].md](INSTALL%5BRUS%5D.md) | Installation guide (Russian) |
| [SHOOTING_GUIDE_DUAL_ACTOR[ENG].md](SHOOTING_GUIDE_DUAL_ACTOR%5BENG%5D.md) | Two-actor recording guide (English) |
| [SHOOTING_GUIDE_DUAL_ACTOR[RUS].md](SHOOTING_GUIDE_DUAL_ACTOR%5BRUS%5D.md) | Two-actor recording guide (Russian) |
| [docs/changelog[ENG].md](docs/changelog%5BENG%5D.md) | Full changelog (English) |
| [docs/final_test_report[ENG].md](docs/final_test_report%5BENG%5D.md) | Test report: 102/102 tests passing |
| [docs/technical_report[ENG].docx](docs/technical_report%5BENG%5D.docx) | Technical report (English) |

---

## Project Structure

```
freemocap-enhanced/
├── run_freemocap_enhanced.py          Main launcher
├── requirements.txt                  Dependencies
│
├── multiperson_detector.py           RTMDet + RTMPose detection
├── cross_view_association.py         Epipolar + Hungarian association
├── calibration_loader.py             TOML calibration loader
├── temporal_tracker.py               Temporal tracking + marker override
├── per_actor_triangulation.py        DLT triangulation per actor
├── multi_actor_pipeline.py           v3.1 pipeline per actor
├── physical_interaction_validator.py Interpenetration + contact
├── marker_fallback.py                ArUco detection + binding
├── joint_definitions.py              Keypoint definitions
├── rtmpose_triangulation.py          DLT triangulation
├── pipeline.py                       Main v3.1 processing pipeline
├── alternative_tracker.py            Single-person RTMPose tracker
├── floor_plane.py                    Floor plane estimation
│
├── gui/
│   ├── screens/                      GUI screens (mode selector, settings)
│   └── i18n/                         Strings EN/RU + locale manager
│
├── tools/
│   ├── generate_actor_markers.py     ArUco marker generator
│   └── actor_marker_map.yaml         Actor-marker mapping config
│
├── docs/                             Changelogs, test reports, technical docs
│
├── test_config.py                    Shared test paths config
├── test_full_pipeline_integration.py Full pipeline integration test
├── test_*.py                         Unit and integration tests
│
├── INSTALL[ENG].md                   Installation guide
├── INSTALL[RUS].md                   Installation guide (Russian)
├── SHOOTING_GUIDE_DUAL_ACTOR[ENG].md Two-actor recording guide
└── SHOOTING_GUIDE_DUAL_ACTOR[RUS].md Two-actor recording guide (Russian)
```

---

## Credits

Built on top of [FreeMoCap](https://freemocap.org/) by Jon Matthis and the FreeMoCap community.

Detection: [RTMPose](https://github.com/open-mmlab/mmpose) via [rtmlib](https://github.com/harlanhong/rtmlib).

---

## License

This project extends FreeMoCap, which is open-source. Use at your own risk.
