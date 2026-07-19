# FreeMoCap Enhanced — Decisions Log

| Date | Time | Problem | Considered Options | Chosen Option | Reason | Status | Changelog Ref |
|------|------|---------|-------------------|---------------|--------|--------|---------------|
| 18.07.2026 | 00:00 | C: drive full, need venv | C: drive cleanup / D: drive venv / Docker | D: drive venv | 396GB free on D:, no cleanup needed | Done | [18.07.2026 00:00] |
| 18.07.2026 | 00:30 | UnicodeEncodeError on Windows | Replace Δt / Set PYTHONUTF8=1 / chcp 65001 | Replace `Δt` → `dt` in logging config | Minimal change, no side effects | Done | [18.07.2026 00:30] |
| 18.07.2026 | 02:00 | Pipeline architecture for 6 problems | Monolithic script / Modular pipeline / Class-based | Modular pipeline (7 modules + pipeline.py) | Testability, separation of concerns | Done | [18.07.2026 02:00] |
| 18.07.2026 | 14:00 | GUI i18n scope | Full FreeMoCap i18n / New screens only / None | Option B: Localize ONLY new dual-actor screens | FreeMoCap has 50+ files with hardcoded strings, too risky | Done | [18.07.2026 14:00] |
| 18.07.2026 | 14:30 | GUI integration approach | Standalone app / Patch MainWindow / Fork FreeMoCap | Patch MainWindow via monkey-patch | Preserves upstream, minimal intrusion | Done | [18.07.2026 14:30] |
| 18.07.2026 | 15:00 | Import collision: our `gui` vs FreeMoCap `gui` | Rename our package / sys.path reorder / sys.modules injection | sys.modules injection | Cleanest, no renaming, isolates at import time | Done | [18.07.2026 15:00] |
| 18.07.2026 | 16:00 | Single vs Dual actor settings | Same settings / Different settings | Different settings (Screen 2 minimal wrapper, Screen 3 full params) | User confirmed: "single and dual should be DIFFERENT" | Done | [18.07.2026 16:00] |
| 18.07.2026 | 17:00 | Live indicator on Screen 4 | Keep pulsing dot / Remove completely | Remove completely | User: "delete it entirely" | Done | [18.07.2026 17:00] |
| 19.07.2026 | 00:00 | Real data integration: when to load | On recording stop / On processing_finished / Manual button | On `processing_finished_signal` | Guarantees 2D/3D npy files exist | Done | [19.07.2026 00:30] |
| 19.07.2026 | 00:15 | Review mode bbox mapping | Per-camera actor ID / Global actor ID / Simple modulo | Modulo: camera index % 2 → actor | Simplest, works for 2-cam stereo; extensible later | Done | [19.07.2026 00:30] |
| 19.07.2026 | 00:20 | Export format | CSV / TRC / C3D / BVH | CSV | Universal, human-readable, easy to debug | Done | [19.07.2026 00:30] |
| 19.07.2026 | 00:25 | sys.modules injection cleanup | Restore all / Restore only FreeMoCap / Leave injected | Restore only FreeMoCap modules, keep ours out | Prevents leak into FreeMoCap's internal imports | Done | [19.07.2026 00:30] |
| 19.07.2026 | 04:00 | Cross-view association algorithm | Pure epipolar / Epipolar + appearance / Deep ReID | Epipolar + keypoint appearance (Hungarian on cost = epi_dist × (2-app_sim)) | Fast, no training data needed, works on Campus dataset | Done | [19.07.2026 04:00] |
| 19.07.2026 | 04:15 | Association level: per-keypoint vs per-pose | Match individual keypoints / Match full poses | Match full poses (14 keypoints × scores) | More robust, handles occlusion naturally | Done | [19.07.2026 04:15] |
| 19.07.2026 | 04:30 | Matching algorithm | Greedy nearest / Hungarian / Spectral clustering | Hungarian (linear_sum_assignment) on cost matrix | Globally optimal assignment, handles unbalanced counts | Done | [19.07.2026 04:30] |
| 19.07.2026 | 04:45 | Confidence weighting | Equal weights / Epipolar dominant / Detection dominant | 0.3 det + 0.4 epi + 0.3 app | Balanced: epipolar most important for cross-view, appearance as tiebreaker | Done | [19.07.2026 04:45] |
| 19.07.2026 | 05:00 | Test dataset for Stage 2 | Synthetic / Campus_Seq1 / Shelf / Own recording | Campus_Seq1 (3 cams, 3 actors, crossing) | Public, calibrated, multi-person, ground truth available | Done | [19.07.2026 05:00] |