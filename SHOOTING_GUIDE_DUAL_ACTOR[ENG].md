# Two-Actor Recording Guide — FreeMoCap Enhanced v3.2

Complete step-by-step guide for recording two people simultaneously with markerless motion capture.

---

## 1. ROOM PREPARATION

### 1.1 Space Requirements

- **Minimum area**: 3m x 3m clear space (ideally 4m x 4m)
- **Ceiling height**: at least 2.5m
- Remove all objects between cameras and the capture area (chairs, tables, boxes)
- **Floor**: uniform color, no reflective surfaces (mirror, glass, polished metal)
- **Walls**: avoid large mirrors or windows that cause reflections

### 1.2 Lighting

- **Even, diffuse lighting** from multiple sources — no harsh shadows
- Avoid direct sunlight (changes during recording)
- Avoid single overhead light (creates strong shadows under chin, arms)
- **Recommended**: 2-4 soft LED panels at different angles, or overhead fluorescent + side fill
- Light color temperature should be consistent (all warm or all cool, not mixed)

### 1.3 Camera Placement

```
                    [Camera 0 - Front]
                          |
                          |
                          v

[Camera 1 - Left] <--- [ ACTORS ] ---> [Camera 2 - Right]
```

- **Minimum 3 cameras** (more = better triangulation)
- Place cameras **at or slightly above eye level** (1.5-1.8m height)
- Cameras should have **overlapping fields of view** covering the entire capture area
- Angle cameras **15-30 degrees downward** toward the center of the capture area
- **Distance from actors**: 2-4 meters depending on lens focal length
- Ensure each camera can see both actors simultaneously

### 1.4 Camera Synchronization

- All cameras must start recording at the **exact same frame**
- Use hardware sync if available (trigger cable, genlock)
- If no hardware sync: start all cameras within **1-2 frames** of each other
- All cameras must run at the **same frame rate** (e.g., all at 30fps or all at 60fps)
- All cameras must have the **same resolution**

---

## 2. ACTOR PREPARATION

### 2.1 Clothing

| DO | DON'T |
|---|---|
| Tight-fitting, solid-color clothing | Loose/baggy clothing (hides body shape) |
| Long sleeves + long pants (helps tracking) | Shorts + tank top (skin confuses detector) |
| Matte fabric (cotton, polyester) | Shiny/reflective fabric (satin, leather) |
| Dark or bright solid colors | All-black or all-white (hard to distinguish from background) |
| Flat-soled shoes or bare feet | High heels, platform shoes |

### 2.2 ArUco Markers (IMPORTANT for 2-actor mode)

ArUco markers help the system tell **who is who** when actors cross paths or are briefly occluded.

**How to generate markers:**
```bash
cd freemocap_enhanced
python tools/generate_actor_markers.py
```
This creates printable marker images in `markers/` folder.

**How to wear markers:**
- Print markers at **A5 or A4 size**
- Tape or pin **one marker per actor** to the front of their torso (chest level)
- Marker should be **flat, not curved** — use cardboard backing if needed
- **Actor A**: marker ID 0 (usually top-left pattern)
- **Actor B**: marker ID 1 (usually top-right pattern)
- **Critical**: markers must be visible from **at least 2 cameras** at all times

```
    Actor A                    Actor B
  ┌──────────┐              ┌──────────┐
  │ ┌──────┐ │              │ ┌──────┐ │
  │ │ID: 0 │ │              │ │ID: 1 │ │
  │ └──────┘ │              │ └──────┘ │
  │  (chest) │              │  (chest) │
  └──────────┘              └──────────┘
```

### 2.3 Maximum Distance Between Actors

- Actors should stay within **1.5m of each other** for optimal cross-view association
- They can move up to **3m apart** but association quality may drop
- **Never turn backs to each other** for more than 2-3 seconds (markers lose visibility)

---

## 3. SOFTWARE SETUP

### 3.1 Start the Application

```bash
cd freemocap_enhanced
venv\Scripts\activate          # Activate virtual environment
python run_freemocap_enhanced.py
```

Wait for the application to load. You will see:
1. FreeMoCap main window
2. Original welcome dialog — click **Done**
3. Enhanced welcome screen — click close or toggle language (RU/EN)
4. Release notes (if first launch) — click **OK**

### 3.2 Select Dual Actor Mode

1. Click **"New Recording"** in the top menu (or use the toolbar)
2. **Mode Selector** screen appears
3. Select **"Dual Actor"** mode
4. Click **Next**

### 3.3 Configure Cameras

1. Switch to the **Camera View** tab
2. Click **"Detect Cameras"** — all connected cameras should appear
3. Verify you see **live feeds from ALL cameras**
4. Check that both actors are visible in **every camera feed**
5. If a camera is missing:
   - Check USB connection
   - Close other applications using the camera
   - Restart the application

### 3.4 Camera Calibration (FIRST TIME ONLY)

Calibration maps camera positions in 3D space. You only need to do this **once** unless cameras move.

**Calibration pattern**: Use a printed CharUco board (A3 size recommended)

Steps:
1. Print a CharUco board (7x10 squares minimum)
2. Hold the board **flat** in front of Camera 0
3. Slowly rotate it to different angles (±30° tilt, ±30° pan)
4. Capture **20-30 frames** per camera
5. Repeat for Camera 1, Camera 2, etc.
6. The system will compute camera matrices and extrinsics
7. Save calibration to `.toml` file

**Calibration quality checklist:**
- [ ] Each camera has at least 20 calibration frames
- [ ] Board is shown at varied angles (not just straight-on)
- [ ] Board fills at least 30% of the camera frame
- [ ] No motion blur in calibration frames (hold still!)
- [ ] Reprojection error < 2.0 pixels (shown in calibration output)

---

## 4. RECORDING SESSION

### 4.1 Pre-Recording Checklist

```
ROOM:
  [ ] Capture area clear of obstacles
  [ ] Lighting is even, no harsh shadows
  [ ] Floor is clean, uniform color

CAMERAS:
  [ ] All cameras connected and detected
  [ ] All cameras showing live feed
  [ ] Both actors visible in ALL cameras
  [ ] All cameras at same frame rate and resolution
  [ ] Calibration done (first time) or loaded

ACTORS:
  [ ] Clothing is tight-fitting, solid color
  [ ] ArUco markers attached to chests (if using)
  [ ] Markers visible from at least 2 cameras
  [ ] Actors practiced their choreography

SOFTWARE:
  [ ] FreeMoCap Enhanced running
  [ ] Dual Actor mode selected
  [ ] Recording folder selected
```

### 4.2 Starting the Recording

1. Open the **Camera View** tab
2. Both actors walk to their **starting positions** in the center of the capture area
3. Actors face **toward the main camera** (Camera 0)
4. Actor A stands on the **left**, Actor B on the **right** (from Camera 0's perspective)
5. Actors stand **1-1.5m apart**, side by side
6. Click the **red "Record" button**
7. **3-2-1 countdown** appears
8. Recording starts — you'll see frame counter incrementing

**IMPORTANT**: The first **2-3 seconds** of recording serve as a **reference pose** — actors should stand still in T-pose or A-pose for best results.

```
T-pose (recommended):        A-pose (alternative):
       |                            |
    ---+---                       \ | /
       |                           \|/
    ---+---                         |
       |                            |
      / \                          / \
```

### 4.3 During the Recording

**Actors should:**
- Move **slowly and deliberately** (avoid sudden jerky movements)
- Stay within the **overlapping camera view** (center area)
- Keep **arms away from torso** when possible (helps tracking)
- Avoid **turning backs** to cameras for long periods
- If they need to cross paths: **walk around each other** rather than through
- Keep movements within the **2m x 2m center zone** for best results

**Actors should NOT:**
- Run or make very fast movements
- Crouch or lie on the floor (except as part of choreography)
- Wear or carry objects that obscure the body (bags, large props)
- Touch or grab each other extensively (brief handshakes OK, prolonged grappling will confuse the tracker)

**Camera operator should:**
- Monitor all camera feeds for obstructions
- Watch the frame counter — a good recording is **10-60 seconds** for a single take
- If a camera feed goes black, **stop and re-check** the connection

### 4.4 Ending the Recording

1. Actors return to **starting position** (optional, for clean cut)
2. Click **"Stop"** button
3. Wait for the status bar to show **"Saving..."** — do NOT close the application
4. Wait for **"Recording saved"** confirmation
5. Recording is saved to the selected folder

### 4.5 Post-Recording Processing

After stopping the recording:

1. Switch to the **"Process"** tab
2. Select the recording you just made
3. Click **"Process Motion Capture Data"**
4. The pipeline runs:
   - Stage 1: Multi-person detection (RTMDet + RTMPose) — ~2-5 min
   - Stage 2: Cross-view association — ~10-30 sec
   - Stage 3: Temporal tracking — ~10-30 sec
   - Stage 4: Per-actor triangulation — ~30-60 sec
   - Stage 5: v3.1 post-processing — ~1-3 min
5. Wait for **"Processing complete"** message
6. Results appear in the **3D Viewer** tab

**Processing time depends on:**
- Number of frames (more = slower)
- Number of cameras (more = slower but more accurate)
- GPU speed (NVIDIA GPU significantly faster)
- CPU speed (for non-GPU stages)

Typical: **10-20 minutes** for a 30-second recording at 30fps with 3 cameras.

---

## 5. TROUBLESHOOTING

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Actors swap IDs mid-recording | Insufficient ArUco markers | Attach markers to chests, ensure visibility |
| One actor not detected | Occluded from most cameras | Reposition cameras or actors |
| Skeleton "jitters" violently | Camera sync issue | Check all cameras are at same FPS |
| Holes/gaps in skeleton | Actor occluded too long | Redo with actors staying more visible |
| "Breathing bones" artifact | Pipeline not applied | Re-run with v3.2 post-processing |
| Foot slides across floor | No foot locking | Enable foot sliding correction in pipeline |
| Two skeletons merge into one | Actors too close together | Increase distance to >1m, add ArUco markers |

### Recording Tips

- **Do 3-5 takes** of each choreography — pick the best one
- **Keep takes short**: 15-30 seconds is ideal for quality
- **Label your recordings** clearly in the folder name
- **Save calibration** after a good calibration session — reuse it
- **Test with 1 actor first** to verify cameras work, then switch to dual actor mode

---

## 6. QUICK REFERENCE CARD

```
┌─────────────────────────────────────────────┐
│         TWO-ACTOR RECORDING CHEATSHEET      │
├─────────────────────────────────────────────┤
│                                             │
│  1. Check room: clear, even lighting        │
│  2. Check cameras: all 3 detected, synced   │
│  3. Check actors: markers on, visible       │
│  4. Start FreeMoCap → New → Dual Actor      │
│  5. Actors in T-pose, 1-1.5m apart          │
│  6. Record → 3-2-1 → Stand still 3 sec      │
│  7. Perform choreography (stay in center)   │
│  8. Stop → Wait for "Saved"                 │
│  9. Process → Wait for completion           │
│  10. Review in 3D Viewer                    │
│                                             │
│  MAX TAKE LENGTH: 60 seconds               │
│  IDEAL TAKE LENGTH: 15-30 seconds          │
│  ACTOR DISTANCE: 1-1.5m (max 3m)          │
│  CENTER ZONE: 2m x 2m                      │
│                                             │
└─────────────────────────────────────────────┘
```
