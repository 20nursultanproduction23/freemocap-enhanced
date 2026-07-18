"""Append Section 13 to technical_report[ENG].docx — Research: External Repo Adaptability"""
import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

path = r"C:\freemocap-docs\technical_report[ENG].docx"
doc = docx.Document(path)

# Add page break before new section
doc.add_page_break()

# Section header
h = doc.add_heading("Section 13: Research — External Repository Adaptability Assessment", level=1)

doc.add_heading("13.1 Objective", level=2)
doc.add_paragraph(
    "Before implementing multi-person tracking, evaluate three external multi-view "
    "pose estimation repositories to determine which (if any) provides reusable "
    "cross-view person association algorithms. The key question: is association "
    "a learned neural network requiring person re-identification weights, or a "
    "purely geometric algorithm that works with any detector?"
)

doc.add_heading("13.2 Repositories Evaluated", level=2)

# --- mvpose ---
doc.add_heading("13.2.1 mvpose (zju3dv)", level=3)
doc.add_paragraph("GitHub: zju3dv/mvpose")
doc.add_paragraph("Paper: Cross-view Parsing (ECCV 2018)")

p = doc.add_paragraph()
p.add_run("Association algorithm: ").bold = True
p.add_run(
    "Two-stage system: (1) Affinity matrix construction combining geometric "
    "epipolar distance and ReID appearance features, (2) SVT (Singular Value "
    "Thresholding) matching via ADMM semidefinite relaxation."
)

p = doc.add_paragraph()
p.add_run("Input format: ").bold = True
p.add_run(
    "COCO-17 keypoints (flat list of 51 values per person: 17 x [x,y,confidence]). "
    "Requires camera calibration pickle with P (3x4 projection), K (3x3 intrinsics), "
    "RT (3x4 extrinsics). dimGroup array encodes per-camera person counts."
)

p = doc.add_paragraph()
p.add_run("Key functions: ").bold = True
p.add_run(
    "geometry_affinity() in src/m_utils/geometry.py:64-96 (epipolar distance → affinity), "
    "matchSVT() in src/models/matchSVT.py:16-94 (ADMM + SVD thresholding + doubly-stochastic "
    "projection + transitive closure)."
)

p = doc.add_paragraph()
p.add_run("Pretrained weights dependency: ").bold = True
p.add_run(
    "HYBRID. The ReID component (ResNet-50 on Market-1501, 1024-dim embeddings) requires "
    "pretrained checkpoint.pth.tar. HOWEVER, the system has an explicit 'Geometry only' mode "
    "(model_config.py metric='Geometry only') that uses ONLY epipolar affinity — zero learned "
    "parameters needed. The SVT matching algorithm is purely mathematical."
)

p = doc.add_paragraph()
p.add_run("Adaptability to RTMPose 133-point: ").bold = True
p.add_run(
    "HIGH. The SVT matcher (matchSVT.py) is fully generic — takes any NxN affinity matrix. "
    "Only the geometry_affinity() function hardcodes joint_num=17 in 3-4 places. Mapping "
    "RTMPose 133→COCO-17 subset is trivial indexing. No changes needed to the core matching."
)

# --- crossview ---
doc.add_heading("13.2.2 crossview_3d_pose_tracking (longcw)", level=3)
doc.add_paragraph("GitHub: longcw/crossview_3d_pose_tracking")
doc.add_paragraph("Paper: Cross-View Tracking for Multi-Human 3D Pose Estimation (CVPR 2020)")

p = doc.add_paragraph()
p.add_run("Association algorithm: ").bold = True
p.add_run(
    "NOT PRESENT in the repository. The README explicitly states: 'The source code is not "
    "included as this is a commercial project.' The repo contains only data loaders, camera "
    "calibration utilities, evaluation code (PCP metric), and visualization."
)

p = doc.add_paragraph()
p.add_run("Pretrained weights dependency: ").bold = True
p.add_run(
    "N/A — zero neural network code, zero model weights, zero ML framework imports "
    "in the entire repository. requirements.txt lists only prettytable, tqdm, opencv-python, vispy."
)

p = doc.add_paragraph()
p.add_run("Adaptability: ").bold = True
p.add_run(
    "ZERO for association. The calibration/triangulation utilities (calib/calibration.py) "
    "are reusable but trivial — standard DLT. Not worth cloning a 500MB repo for 50 lines of DLT math."
)

# --- multiview_pose ---
doc.add_heading("13.2.3 multiview_pose (wusize)", level=3)
doc.add_paragraph("GitHub: wusize/multiview_pose")
doc.add_paragraph("Paper: Graph-based 3D Multi-Person Pose Estimation Using Multi-View Images")

p = doc.add_paragraph()
p.add_run("Association algorithm: ").bold = True
p.add_run(
    "Two-stage: (1) Epipolar geometry scoring via StereoGeometry class (pure math — "
    "ray intersection + point-to-epipolar-line distance), (2) GCN (EdgeConvLayers) "
    "refinement using sampled CNN features."
)

p = doc.add_paragraph()
p.add_run("Input format: ").bold = True
p.add_run(
    "NOT keypoints — expects CNN feature maps (NxVxCxHxW) from ResNet-50 backbone. "
    "Person centers detected via dedicated heatmap channel. Custom 15-point Panoptic "
    "skeleton, not COCO-17."
)

p = doc.add_paragraph()
p.add_run("Pretrained weights dependency: ").bold = True
p.add_run(
    "The GCN match scoring (EdgeConvLayers) requires trained weights. HOWEVER, the "
    "geometric prior — exp(-coef × epipolar_distance) — is purely mathematical and "
    "provides a strong initial signal. The GCN only REFINES this signal."
)

p = doc.add_paragraph()
p.add_run("Adaptability to RTMPose 133-point: ").bold = True
p.add_run(
    "MODERATE. The StereoGeometry/MonocularGeometry classes (utils.py) are clean, "
    "keypoint-agnostic geometry utilities that work with any camera calibration. "
    "But the input pipeline expects CNN feature maps, not keypoints. Would need to "
    "bypass the feature-sampling step entirely."
)

doc.add_heading("13.3 Comparative Summary", level=2)

table = doc.add_table(rows=1, cols=5)
table.style = 'Table Grid'
hdr = table.rows[0].cells
hdr[0].text = "Criterion"
hdr[1].text = "mvpose"
hdr[2].text = "crossview_tracking"
hdr[3].text = "multiview_pose"
hdr[4].text = "Winner"

rows_data = [
    ("Association code present?", "Yes", "No (proprietary)", "Yes", "mvpose / multiview"),
    ("Geometric association", "Epipolar + SVT", "N/A", "Epipolar + GCN", "mvpose (SVT)"),
    ("Works without pretrained weights", "Yes (Geometry only mode)", "N/A", "Partially (geo prior)", "mvpose"),
    ("Input format match with RTMPose", "COCO-17 (easy map)", "N/A", "CNN features (hard)", "mvpose"),
    ("Algorithm sophistication", "SVT semidefinite relaxation", "N/A", "GCN graph matching", "mvpose (mature)"),
    ("Extraction effort", "Low (3-4 hardcoded values)", "N/A", "Medium (bypass CNN pipeline)", "mvpose"),
    ("Calibration format", "P/K/RT pickle", "JSON", "K/R/Torch", "All compatible"),
]

for row_data in rows_data:
    row = table.add_row().cells
    for i, val in enumerate(row_data):
        row[i].text = val

doc.add_heading("13.4 Conclusion and Recommendation", level=2)

p = doc.add_paragraph()
p.add_run("Selected approach: Custom geometric association combining best elements from mvpose and multiview_pose.\n\n").bold = True

doc.add_paragraph(
    "The crossview_3d_pose_tracking repository is rejected — its association code is "
    "proprietary and not distributed."
)

doc.add_paragraph(
    "The multiview_pose repository provides clean epipolar geometry utilities "
    "(StereoGeometry, MonocularGeometry in utils.py) that are worth extracting as "
    "reference implementations. However, its matching pipeline requires CNN feature maps "
    "as input, which is architecturally incompatible with RTMPose keypoint output."
)

doc.add_paragraph(
    "The mvpose repository is the primary reference. Its Geometry-only mode provides "
    "exactly what we need: (1) epipolar affinity computation from keypoint pairs across "
    "camera views, (2) SVT semidefinite matching that handles the full multi-camera, "
    "multi-person assignment problem optimally. The only adaptation needed is mapping "
    "RTMPose 133-point indices to COCO-17 subset for the epipolar distance calculation."
)

doc.add_paragraph(
    "However, rather than directly importing mvpose code (which has Cython dependencies, "
    "CUDA requirements, and tightly coupled configuration), we will REIMPLEMENT the core "
    "algorithms from scratch, informed by the mvpose approach:"
)

items = [
    "Epipolar distance: cv2.computeCorrespondEpilines() + mean point-to-line distance "
    "(from mvpose geometry.py:29-61)",
    "Affinity matrix: z-score normalization + sigmoid conversion "
    "(from mvpose geometry.py:93-96)",
    "Hungarian matching via scipy.optimize.linear_sum_assignment "
    "(replacing SVT for simplicity — SVT is overkill for ≤6 cameras with ≤5 persons)",
    "Transitive closure for multi-camera consensus "
    "(from mvpose pictorial.pyx:201-234)",
]

for item in items:
    doc.add_paragraph(item, style='List Bullet')

doc.add_paragraph(
    "This approach avoids all dependency on pretrained weights, ONNX models, or CUDA, "
    "while preserving the mathematical rigor of the proven mvpose association algorithm."
)

# Save
doc.save(path)
print(f"Section 13 appended to {path}")
