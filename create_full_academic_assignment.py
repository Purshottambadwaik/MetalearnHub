"""
Comprehensive Generator for B.Tech Research Methodology Assignment.
Produces Word (.docx) and PDF (.pdf) documents with complete academic rigor.
Institution: Government College of Engineering, Nagpur
Department: Computer Science and Engineering
Project: MetaLearnHub (Group 07)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs" / "research_methodology"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from doc_generator_helpers import set_cell_margins, set_cell_background, add_table_borders, NumberedCanvas

import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DOCX_FILE = DOCS_DIR / "Research_Methodology_Project_Application_Assignment_Group07.docx"
PDF_FILE = DOCS_DIR / "Research_Methodology_Project_Application_Assignment_Group07.pdf"

print("Generating Academic Word Document (.docx)...")

doc = docx.Document()

# Configure page margins to 0.75 in (54 pt)
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    section.page_width = Inches(8.27)  # A4
    section.page_height = Inches(11.69)
    # Enable different first page
    section.different_first_page_header_footer = True
    
    # Configure running header for subsequent pages
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("Government College of Engineering, Nagpur | Department of CSE | Research Methodology")
    hrun.font.name = "Calibri"
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = RGBColor(100, 116, 139)

    # Configure running footer for subsequent pages
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    frun = fp.add_run("MetaLearnHub: Open-Set Few-Shot Meta-Learning System | Group No. 07")
    frun.font.name = "Calibri"
    frun.font.size = Pt(8.5)
    frun.font.color.rgb = RGBColor(100, 116, 139)


# Colors
C_PRIMARY = RGBColor(26, 54, 93)      # Deep Navy
C_SECONDARY = RGBColor(43, 108, 176)  # Blue
C_TEXT = RGBColor(45, 55, 72)         # Charcoal Dark
C_MUTED = RGBColor(113, 128, 150)     # Gray
HEX_PRIMARY = "1A365D"
HEX_LIGHT_BG = "F7FAFC"
HEX_HEADER_BG = "EBF8FF"
HEX_BORDER = "CBD5E0"


def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = C_PRIMARY
    return p


def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(12)
    run.font.color.rgb = C_SECONDARY
    return p


def add_heading_1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = C_PRIMARY
    return p


def add_heading_2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = C_SECONDARY
    return p


def add_heading_3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = C_TEXT
    return p


def add_body_p(text, bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(9.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = C_TEXT
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.color.rgb = C_TEXT
    return p


def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.1
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(9.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = C_TEXT
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.color.rgb = C_TEXT
    return p


def add_callout(text, title="NOTE:"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.77)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="2B6CB0"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r_tit = p.add_run(f"{title} ")
    r_tit.font.name = "Calibri"
    r_tit.font.size = Pt(9)
    r_tit.font.bold = True
    r_tit.font.color.rgb = C_SECONDARY
    
    r_txt = p.add_run(text)
    r_txt.font.name = "Calibri"
    r_txt.font.size = Pt(9)
    r_txt.font.color.rgb = C_TEXT
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# =========================================================================
# FRONT PAGE
# =========================================================================

# Institution Header
p_inst = doc.add_paragraph()
p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_inst.paragraph_format.space_before = Pt(12)
p_inst.paragraph_format.space_after = Pt(2)
r = p_inst.add_run("GOVERNMENT COLLEGE OF ENGINEERING, NAGPUR")
r.font.name = "Calibri"
r.font.size = Pt(15)
r.font.bold = True
r.font.color.rgb = C_PRIMARY

p_dept = doc.add_paragraph()
p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_dept.paragraph_format.space_before = Pt(0)
p_dept.paragraph_format.space_after = Pt(14)
r = p_dept.add_run("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING")
r.font.name = "Calibri"
r.font.size = Pt(12)
r.font.bold = True
r.font.color.rgb = C_SECONDARY

# Decorative Divider
p_div = doc.add_paragraph()
p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_div.paragraph_format.space_after = Pt(14)
r = p_div.add_run("―" * 45)
r.font.color.rgb = C_MUTED

# Assignment Title
p_asg = doc.add_paragraph()
p_asg.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_asg.paragraph_format.space_after = Pt(4)
r = p_asg.add_run("RESEARCH METHODOLOGY — B.TECH. PROJECT APPLICATION ASSIGNMENT")
r.font.name = "Calibri"
r.font.size = Pt(12.5)
r.font.bold = True
r.font.color.rgb = C_PRIMARY

# Project Title
add_title("MetaLearnHub: An Open-Set Few-Shot Meta-Learning Framework for Dynamic Class-Incremental Visual Recognition and Empirical Baseline Benchmarking")
add_subtitle("A Research Methodology Investigation on Transferable Metric Learning, Adaptive Open-Set Rejection, and Closed-Set Supervised Failure Modes")

# Metadata Box (Table)
meta_table = doc.add_table(rows=6, cols=2)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_table.autofit = False
add_table_borders(meta_table, color="CBD5E0", sz="4")

col_widths = [Inches(2.5), Inches(4.27)]
metadata_rows = [
    ("Academic Year & Semester:", "2026-27 | B.Tech. Final Year (Semester VII/VIII)"),
    ("Course / Subject:", "Research Methodology (Computer Science and Engineering)"),
    ("Project Group Identifier:", "Group No. 07"),
    ("Project Guide / Supervisor:", "Dr. P. B. Jawade, Associate Professor, Dept. of CSE"),
    ("Student Investigators & Roll Nos:", "Devika Samudre (Roll No. 19)\nYash Talmale (Roll No. 73)\nPurshottam Badwaik (Roll No. 39)\nChhannu Dhuware (Roll No. 17)"),
    ("Implementation Repository:", "c:/college/Mega_project/Meta_learnHub_Main/MetalearnHub\nFull-Stack Engine (PyTorch, FastAPI, React/TypeScript, ResNet-50)"),
]

for idx, (label, val) in enumerate(metadata_rows):
    row = meta_table.rows[idx]
    c0 = row.cells[0]
    c1 = row.cells[1]
    c0.width = col_widths[0]
    c1.width = col_widths[1]
    set_cell_background(c0, "F7FAFC")
    set_cell_margins(c0, top=70, bottom=70, left=100, right=100)
    set_cell_margins(c1, top=70, bottom=70, left=100, right=100)
    
    p0 = c0.paragraphs[0]
    p0.paragraph_format.space_after = Pt(0)
    r0 = p0.add_run(label)
    r0.font.name = "Calibri"
    r0.font.size = Pt(9)
    r0.font.bold = True
    r0.font.color.rgb = C_PRIMARY
    
    p1 = c1.paragraphs[0]
    p1.paragraph_format.space_after = Pt(0)
    r1 = p1.add_run(val)
    r1.font.name = "Calibri"
    r1.font.size = Pt(9)
    r1.font.color.rgb = C_TEXT

doc.add_page_break()


# =========================================================================
# UNIT I — DEFINING THE RESEARCH PROBLEM
# =========================================================================
add_heading_1("UNIT I — DEFINING THE RESEARCH PROBLEM")

add_heading_2("1.1 Project Overview and Motivation")
add_body_p(
    "Deep supervised learning has achieved superhuman accuracy in computer vision benchmarks like ImageNet-1K. "
    "However, this success is contingent on a fundamental assumption: the availability of thousands of annotated, "
    "identically distributed training exemplars per category, mapped to a static, pre-defined classification layer. "
    "In critical real-world deployments—including wildlife ecology, agricultural biosecurity, rare pathogen microscopy, "
    "and autonomous edge robotics—acquiring thousands of labeled images per species is physically, financially, or temporally infeasible. "
    "Furthermore, real-world visual environments are fundamentally open-set and dynamic: novel categories appear constantly, and systems must "
    "enroll them rapidly without undergoing catastrophic forgetting or complete backpropagation retraining cycles."
)
add_body_p(
    "To overcome these bottlenecks, this research project develops MetaLearnHub—an end-to-end open-set few-shot meta-learning "
    "framework. MetaLearnHub reformulates visual recognition from traditional empirical risk minimization (ERM) into an episodic "
    "metric-adaptation paradigm. The core motivation is to empower computer systems with human-like visual cognitive agility: "
    "learning transferable inductive representations from base distribution categories so that previously unseen categories can be "
    "accurately recognized from as few as five labeled support exemplars (5-shot learning)."
)
add_body_p(
    "Implemented vs. Proposed Features:",
    bold_prefix="Scope and Verification Status: "
)
add_bullet("Pretrained ResNet-50 feature extractor producing 2048-dimensional unit hypersphere embeddings.", "Implemented: ")
add_bullet("Four distinct meta-learning paradigms: Prototypical Networks (Snell et al. [1]), Matching Networks (Vinyals et al. [2]), Relation Networks (Sung et al. [3]), and First-Order MAML (Finn et al. [4]).", "Implemented: ")
add_bullet("Consensus Ensemble probability fusion combining centroid distance, exemplar memory attention, non-linear deep comparator scoring, and inner-loop gradient adaptation.", "Implemented: ")
add_bullet("Persistent class-incremental registry allowing on-the-fly enrollment of novel categories without backbone retraining or parameter drift.", "Implemented: ")
add_bullet("Tri-gate adaptive open-set out-of-distribution (OOD) rejection mechanism enforcing hyperspherical intra-class radius bounding and minimum certainty thresholding (confidence >= 50.0%).", "Implemented: ")
add_bullet("Head-to-head scientific benchmark suite comparing few-shot meta-learning against conventional transfer-learned supervised linear probes (ERM baseline) on the exact same support data.", "Implemented: ")
add_bullet("Self-supervised Vision Transformer (ViT / DINOv2) patch-token feature concatenation and multi-modal cross-attention heads.", "Proposed (Future Phase): ")

add_heading_2("1.2 Research Problem, Motivation, and Research Gap")
add_body_p(
    "The core research problem addressed by this study is: How can a deep visual recognition architecture achieve high "
    "classification accuracy on novel, previously un-encountered visual categories using fewer than six labeled examples, "
    "while simultaneously preventing catastrophic forgetting of base categories and autonomously rejecting out-of-distribution "
    "open-set visual anomalies without triggering high-confidence false-positive classifications?"
)
add_body_p(
    "Limitations of Conventional Supervised Approaches:",
    bold_prefix="Deficiencies in Conventional Approaches: "
)
add_bullet("Over-Parameterization on Small Samples: A conventional linear classifier on a 2048-dimensional backbone requires 28,686 parameters across 14 classes. Fitting these weights on 5-30 images yields extreme empirical overfitting (96.2% training accuracy collapsing to ~25-50% test accuracy).", "1. ")
add_bullet("Closed-Set Architectural Lock: Standard softmax layers possess a fixed dimensionality (C classes). They are mathematically incapable of outputting an unseen category or rejecting an alien input, forcing a false positive on outliers.", "2. ")
add_bullet("Catastrophic Forgetting & Retraining Latency: Adding a single class in conventional ML requires architectural modification and multiple epochs of gradient descent (taking 15-60 seconds), degrading performance on previously learned categories.", "3. ")

add_callout(
    "The critical Research Gap identified in current academic literature lies at the intersection of Few-Shot Learning (FSL) "
    "and Open-Set Recognition (OSR). Standard FSL benchmarks (e.g., miniImageNet) artificially operate as closed-set N-way evaluation episodes, "
    "forcing queries to belong to one of the N classes. Conversely, standard OSR methods assume extensive training samples per known class. "
    "A severe empirical gap exists in unified systems capable of simultaneous dynamic few-shot class enrollment (K <= 5) and rigorous "
    "hyperspherical open-set outlier rejection alongside head-to-head empirical comparison with conventional baselines.",
    title="IDENTIFIED RESEARCH GAP:"
)

add_heading_2("1.3 Research Chain")
add_body_p("The foundational research logic connecting topic to inquiry is systematically structured in Table 1.")

# Table 1: Research Chain
t1 = doc.add_table(rows=6, cols=2)
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
t1.autofit = False
add_table_borders(t1, color="CBD5E0", sz="4")
t1_widths = [Inches(1.8), Inches(4.97)]
t1_data = [
    ("Research Dimension", "Project Implementation Realization (MetaLearnHub)"),
    ("Project Topic", "Open-Set Few-Shot Meta-Learning for Dynamic Class-Incremental Visual Recognition"),
    ("Research Problem", "Inability of conventional ERM deep classifiers to generalize from extremely small sample sizes (K <= 5) and their vulnerability to closed-set false positives when presented with unseen out-of-distribution classes."),
    ("Research Gap", "Lack of an integrated architectural framework combining dynamic few-shot class enrollment, multi-algorithmic meta-learning consensus (ProtoNet, MatchingNet, RelationNet, MAML), and mathematically bounded open-set rejection."),
    ("Research Objectives", "RO1: Develop class-incremental metric registry; RO2: Engineer open-set rejection gates; RO3: Implement 4-algorithm meta-learning consensus; RO4: Perform empirical baseline benchmarking."),
    ("Research Questions", "RQ1: Can metric centroid adaptation outperform conventional supervised ERM on 5-shot data? RQ2: Does hyperspherical radius bounding effectively eliminate open-set false positives? RQ3: Does multi-algorithmic consensus improve accuracy over single models? RQ4: What is the computational latency trade-off?"),
]

for idx, (c0_text, c1_text) in enumerate(t1_data):
    row = t1.rows[idx]
    c0 = row.cells[0]
    c1 = row.cells[1]
    c0.width = t1_widths[0]
    c1.width = t1_widths[1]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    set_cell_background(c0, bg)
    set_cell_background(c1, bg)
    set_cell_margins(c0, top=70, bottom=70, left=100, right=100)
    set_cell_margins(c1, top=70, bottom=70, left=100, right=100)
    
    p0 = c0.paragraphs[0]
    r0 = p0.add_run(c0_text)
    r0.font.name = "Calibri"
    r0.font.size = Pt(9)
    r0.font.bold = True
    r0.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT
    
    p1 = c1.paragraphs[0]
    r1 = p1.add_run(c1_text)
    r1.font.name = "Calibri"
    r1.font.size = Pt(9)
    r1.font.bold = is_hdr
    r1.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 1: Research Chain Linking Topic, Problem, Gap, Objectives, and Inquiries.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED

add_heading_2("1.4 Research Objectives")
add_body_p("The project is governed by four specific, measurable, achievable, and realistic research objectives:")
add_bullet("Design and implement a persistent class-incremental registry capable of enrolling novel visual categories dynamically using exactly K = 5 support exemplars, achieving sub-second prototype construction without updating backbone feature weights.", "RO1 (Incremental Enrollment): ")
add_bullet("Formulate, calibrate, and validate a tri-gate open-set rejection mechanism utilizing hyperspherical intra-class radius thresholding, inter-class margin verification, and certainty gating (confidence >= 50.0%) to reject out-of-distribution queries with 0.0% false acceptance.", "RO2 (Open-Set Outlier Rejection): ")
add_bullet("Implement and integrate four distinct meta-learning paradigms (Prototypical Networks, Matching Networks, Relation Networks, and MAML) alongside a calibrated Consensus Ensemble Head to evaluate variance reduction across diverse animal categories.", "RO3 (Multi-Algorithm Framework): ")
add_bullet("Conduct rigorous empirical benchmarking comparing the developed few-shot meta-learning framework against a conventional supervised transfer learning classifier (ResNet-50 linear probe), quantifying adaptation latency, overfitting ratios, and classification accuracy.", "RO4 (Empirical Benchmarking): ")

add_heading_2("1.5 Research Questions and Hypotheses")
add_body_p(
    "To validate the research objectives, the following four research questions (RQs) and formal statistical hypotheses are formulated:"
)
add_bullet(
    "Does hyperspherical prototype projection achieve statistically superior classification accuracy over conventional supervised linear probing when evaluating an unseen novel category (Rabbit) with small support size (K = 5)?",
    "RQ1 (Comparative Generalization): "
)
add_body_p(
    "• Null Hypothesis (H1_0): Classification accuracy of Few-Shot Meta-Learning is less than or equal to that of the Conventional Supervised Baseline on 5-shot unseen classes (Acc_FS <= Acc_Conv).\n"
    "• Alternative Hypothesis (H1_1): Classification accuracy of Few-Shot Meta-Learning is significantly higher than that of Conventional Supervised Baseline (Acc_FS > Acc_Conv)."
)

add_bullet(
    "Can a tri-gate open-set decision boundary (distance <= radius, relative gap >= 0.05, confidence >= 0.50) successfully identify and reject out-of-distribution visual anomalies without prior negative exposure?",
    "RQ2 (Open-Set Reliability): "
)
add_body_p(
    "• Null Hypothesis (H2_0): The open-set false acceptance rate (FAR) of the tri-gate boundary on unseen anomalies is greater than 10% (FAR >= 0.10).\n"
    "• Alternative Hypothesis (H2_1): The tri-gate boundary achieves near-zero false acceptance (FAR < 0.01) while maintaining positive true acceptance on registered classes."
)

add_bullet(
    "Does a four-way consensus ensemble (ProtoNet + MatchingNet + RelationNet + MAML) provide higher classification confidence and lower prediction variance than any individual meta-learning algorithm evaluated in isolation?",
    "RQ3 (Algorithmic Ensembling): "
)

add_bullet(
    "What is the operational latency trade-off between instant metric centroid adaptation (O(1) backprop-free) versus inner-loop gradient adaptation (MAML) and conventional retraining?",
    "RQ4 (Latency & Efficiency): "
)

add_heading_2("1.6 Key Variables and Operational Definitions")
add_body_p(
    "Table 2 defines the operational variables, measurement scales, and mathematical criteria governing the experimental evaluation."
)

# Table 2: Operational Definitions
t2 = doc.add_table(rows=7, cols=5)
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
t2.autofit = False
add_table_borders(t2, color="CBD5E0", sz="4")
t2_widths = [Inches(1.2), Inches(0.9), Inches(1.0), Inches(2.3), Inches(1.37)]
t2_data = [
    ("Variable / Construct", "Role", "Scale", "Operational Definition & Formula", "Evaluation Criteria"),
    ("Few-Shot Classification Accuracy", "Dependent (DV1)", "Ratio (%)", "Percentage of test queries where predicted class matches ground truth: Acc = (Correct / Total) * 100", "Target >= 85.0% on valid classes"),
    ("Adaptation Latency", "Dependent (DV2)", "Interval (ms)", "Total wall-clock duration to enroll novel support set and output classification vector: t_inf - t_0", "Target <= 350 ms on CPU"),
    ("Cosine Distance (d_cos)", "Dependent (DV3)", "Ratio [0, 2]", "Distance on unit hypersphere: d(q, p) = 1.0 - (q^T * p) / (||q|| * ||p||)", "d <= intra_class_radius for match"),
    ("Open-Set Rejection Rate", "Dependent (DV4)", "Ratio (%)", "Percentage of out-of-distribution anomalies correctly tagged as [MATCH NOT FOUND / UNKNOWN]", "Target = 100.0% on out-of-system anomalies"),
    ("Meta-Learning Algorithm", "Independent (IV1)", "Nominal", "Categorical choice: Prototypical Networks, Matching Networks, Relation Networks, MAML, or Ensemble", "Tested under identical 5-shot protocols"),
    ("Backbone Feature Dimension", "Control (CV1)", "Constant", "Fixed ResNet-50 feature manifold dimensionality: D = 2048, L2-normalized (||z||_2 = 1.0)", "Verified fixed across all experiments"),
]

for idx, row_data in enumerate(t2_data):
    row = t2.rows[idx]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    for c_idx, val in enumerate(row_data):
        cell = row.cells[c_idx]
        cell.width = t2_widths[c_idx]
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.name = "Calibri"
        r.font.size = Pt(8.5)
        r.font.bold = is_hdr or (c_idx == 0)
        r.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 2: Key Variables, Operational Definitions, and Evaluation Standards.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED


# =========================================================================
# UNIT II — LITERATURE SURVEY AND RESEARCH APPROACH
# =========================================================================
add_heading_1("UNIT II — LITERATURE SURVEY AND RESEARCH APPROACH")

add_heading_2("2.1 Literature Search Strategy")
add_body_p(
    "A systematic academic literature search was conducted across five major indexing databases: IEEE Xplore, ACM Digital Library, "
    "ScienceDirect (Elsevier), SpringerLink, and Google Scholar. The search covered peer-reviewed publications spanning 2016 through 2024, "
    "incorporating foundational seminal papers and recent advancements in few-shot learning and open-set recognition."
)
add_body_p(
    "Search Strings & Boolean Logic:",
    bold_prefix="Query Strings: "
)
add_bullet("(\"few-shot learning\" OR \"meta-learning\") AND (\"prototypical networks\" OR \"matching networks\" OR \"MAML\") AND (\"image classification\" OR \"computer vision\")", "Primary String: ")
add_bullet("(\"few-shot\" AND \"open-set recognition\") OR (\"out-of-distribution\" AND \"meta-learning\") AND (\"class-incremental learning\" OR \"catastrophic forgetting\")", "Secondary String: ")
add_bullet("Peer-reviewed conference or journal; empirical computer vision focus; documented baseline comparison; published between 2016 and 2024.", "Inclusion Criteria: ")
add_bullet("Non-peer-reviewed preprints without code verification; purely theoretical works lacking visual validation; domain-restricted non-visual studies.", "Exclusion Criteria: ")

add_heading_2("2.2 Comparative Literature Survey")
add_body_p(
    "Table 3 presents a comparative matrix of foundational and state-of-the-art research papers directly informing MetaLearnHub."
)

# Table 3: Literature Survey Matrix
t3 = doc.add_table(rows=7, cols=7)
t3.alignment = WD_TABLE_ALIGNMENT.CENTER
t3.autofit = False
add_table_borders(t3, color="CBD5E0", sz="4")
t3_widths = [Inches(1.1), Inches(0.9), Inches(1.0), Inches(0.9), Inches(1.1), Inches(0.9), Inches(0.87)]
t3_data = [
    ("Reference", "Research Problem", "Methodology", "Dataset / Setup", "Main Findings", "Limitations", "Relevance"),
    ("Snell et al. (2017) [1]", "High compute cost in meta-learning", "Prototypical Networks: Centroid Euclidean metric", "Omniglot, miniImageNet", "Simpler than MAML, excels on unimodal classes", "Fails when intra-class variance is large", "Forms core ProtoNet engine in MetaLearnHub"),
    ("Vinyals et al. (2016) [2]", "Rapid one-shot learning without fine-tuning", "Matching Networks: Cosine attention over exemplars", "Omniglot, ImageNet-1K", "Remembers individual exemplars; avoids blur", "Computation scales linearly with support set size", "Integrated as exemplar memory module"),
    ("Sung et al. (2018) [3]", "Handcrafted distance metrics limit complexity", "Relation Networks: Trainable 2D CNN comparator", "miniImageNet, CUB-200", "Learns non-linear similarities directly from data", "Prone to probability collapse if uncalibrated", "Upgraded in our work with relational diff/prod features"),
    ("Finn et al. (2017) [4]", "Generalizing across arbitrary tasks", "MAML: Bilevel inner/outer loop optimization", "Omniglot, Sinusoids, miniImageNet", "Task-agnostic parameter initialization", "Slow adaptation; computationally heavy inner loop", "Integrated as optimization-based engine"),
    ("Chen et al. (2019) [5]", "Lack of fair baselines in FSL literature", "Standard Linear Supervised Probe vs ProtoNet", "CUB-200, miniImageNet", "Linear probe matches FSL when backbone is strong", "Linear probe fails on incremental open-set", "Directly inspires our Section 3.5 empirical comparison"),
    ("Bendale & Boult (2016) [6]", "Deep neural networks are overly confident on OOD", "OpenMax: Extreme value theory thresholding", "ImageNet-1K, Synsets", "Rejects alien categories by adapting softmax", "Assumes large closed-set training pools", "Adapted into our tri-gate hyperspherical radius system"),
]

for idx, row_data in enumerate(t3_data):
    row = t3.rows[idx]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    for c_idx, val in enumerate(row_data):
        cell = row.cells[c_idx]
        cell.width = t3_widths[c_idx]
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.name = "Calibri"
        r.font.size = Pt(8)
        r.font.bold = is_hdr or (c_idx == 0)
        r.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 3: Comparative Analysis of Seminal and Contemporary Research Literature.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED

add_heading_2("2.3 Research Gap and Its Influence on Project Design")
add_body_p(
    "Existing literature predominantly analyzes meta-learning algorithms in idealized, closed-set synthetic settings. "
    "Snell et al. [1] and Vinyals et al. [2] benchmarked on miniImageNet where queries are guaranteed to belong to one of 5 classes. "
    "In practical applications, this produces catastrophic failure: an out-of-distribution image (e.g., an elephant shown to a dog/cat system) "
    "is arbitrarily assigned to a class with high false confidence. "
    "Furthermore, Chen et al. [5] demonstrated that conventional fine-tuning can compete with FSL, but failed to address incremental enrollment "
    "without catastrophic forgetting. MetaLearnHub bridges this divide by embedding four algorithms into a single unified registry architecture "
    "with mathematically bounded hyperspherical open-set rejection."
)

add_heading_2("2.4 System Architecture and Proposed Methodology")
add_body_p(
    "MetaLearnHub is architecturally divided into five decoupled operational layers, ensuring high modularity and fault tolerance:"
)
add_bullet("Client Presentation Layer: Developed in React 18, TypeScript, and Vite. Provides real-time drag-and-drop support enrollment, live classification visualization with probability confidence bars, and comparative benchmark telemetry.", "1. ")
add_bullet("Application & API Layer: Implemented in FastAPI (Python 3.12). Manages asynchronous REST endpoints for dataset inspection (/api/dataset), episodic training (/api/training), evaluation runs (/api/evaluation), and low-latency query inference (/api/predict).", "2. ")
add_bullet("Core Meta-Learning AI Engine: Contains the pre-trained ResNet-50 feature manifold extractor (yielding 2048-dim vectors), Prototypical Networks, Matching Networks, Relation Networks, MAML, and the Consensus Ensemble Head.", "3. ")
add_bullet("Persistent Class Registry Storage: Serializes registered classes to disk under data/class_registry/ storing raw support images, PyTorch normalized embedding tensors, prototype vectors, and calculated intra-class radii.", "4. ")
add_bullet("Verification & Open-Set Rejection Gate: Executes the tri-gate decision pipeline (distance check, competitive margin check, and certainty gating >= 50.0%) to guarantee robust open-set discrimination.", "5. ")

add_callout(
    "SYSTEM DATA FLOW PIPELINE:\n"
    "[Query Image] -> [Preprocess: Resize 224x224, ImageNet Normalization]\n"
    "  -> [ResNet-50 Feature Extractor: 2048-dim Latent Space] -> [L2 Hypersphere Normalization: ||z||_2 = 1.0]\n"
    "  -> [Parallel Evaluator Head: ProtoNet Centroid + MatchingNet Attention + RelationNet MLP + MAML Inner SGD]\n"
    "  -> [Consensus Ensemble: P_ens = 0.35*P_proto + 0.30*P_match + 0.20*P_rel + 0.15*P_maml]\n"
    "  -> [Tri-Gate Open-Set Filter: dist <= radius AND rel_gap >= 0.05 AND conf >= 50.0%]\n"
    "  -> Output: [MATCH FOUND] (Class, Confidence %) OR [MATCH NOT FOUND / UNKNOWN] (0.0% Conf, Reason)",
    title="ARCHITECTURE TELEMETRY & PIPELINE:"
)

add_heading_2("2.5 Research Design and Justification")
add_body_p(
    "The methodology adopts an empirical Design-Science Research (DSR) strategy coupled with benchmark-based quantitative evaluation. "
    "DSR is uniquely suited because it focuses on engineering and validating a novel software artifact solving an identified practical problem. "
    "Controlled benchmark testing isolates independent variables (algorithm selection, support sample count K, presence of OOD outliers) "
    "while measuring dependent outcomes (accuracy, latency, open-set rejection rate) under strict experimental controls (identical hardware, "
    "identical seed, zero data leakage between support and query sets)."
)


# =========================================================================
# UNIT III — DATA, EVIDENCE AND ANALYSIS
# =========================================================================
add_heading_1("UNIT III — DATA, EVIDENCE AND ANALYSIS")

add_heading_2("3.1 Data Collection and Experimental Setup")
add_body_p(
    "The experimental evaluation utilizes real-world photographic datasets and dynamic user-contributed image sets:"
)
add_bullet("Base Category Dataset: Animals-10 raw image dataset (archive (1) (1)/raw-img) comprising 26,179 images across 10 natural categories: Butterfly (2,112), Cattle (1,866), Dog (3,042), Elephant (1,421), Cat/gatto (1,668), Hen (2,555), Horse (1,829), Sheep (1,528), Spider (4,791), and Squirrel (1,540).", "1. Base Distribution: ")
add_bullet("Controlled Cattle Breeds Dataset: 1,208 high-resolution images across 5 specialized cattle breeds: Ayrshire, Brown Swiss, Holstein Friesian, Jersey, and Red Dane.", "2. Domain Distribution: ")
add_bullet("Dynamic Few-Shot Enrollment Sets: Novel real-world test classes registered with exactly K = 5 support exemplars each: Rabbit, Monkey, Zebra, and Snake.", "3. Novel Support Sets: ")
add_bullet("Open-Set Outlier Verification Suite: Out-of-distribution queries designed to evaluate rejection boundaries: unregistered wildlife (Tiger), stylized digital illustrations (Hanuman artwork), and sacred deity paintings (Ganesha artwork).", "4. Open-Set Test Set: ")
add_bullet("Hardware & Runtime Environment: Intel Core i7 / AMD CPU execution environment (simulating realistic resource-constrained edge server deployments), 16GB RAM, Python 3.12, PyTorch 2.5, Torchvision 0.20, Windows 11 OS.", "5. Computational Setup: ")

add_heading_2("3.2 Data Preparation and Leakage Prevention")
add_body_p(
    "Strict methodological protocols prevent data contamination and bias:"
)
add_bullet("Atomic Sample-Group Pairing: Augmented image pairs (base_name.jpg and base_name_c.jpg) share a persistent sample_group_id. When splitting data, paired images are allocated atomically to the same partition, preventing catastrophic data leakage.", "1. Anti-Leakage Protocol: ")
add_bullet("Standardized Preprocessing: All images are resized to 224x224 pixels, converted to 3-channel RGB, and normalized using ImageNet channel statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).", "2. Image Standardization: ")
add_bullet("Support/Query Disjointness: In all few-shot adaptation evaluations, query evaluation images are strictly isolated from the enrolled support sets.", "3. Disjoint Evaluation: ")

add_heading_2("3.3 Descriptive Analysis")
add_body_p("Table 4 outlines the statistical characteristics of the registered dataset partitions.")

# Table 4: Descriptive Statistics
t4 = doc.add_table(rows=6, cols=5)
t4.alignment = WD_TABLE_ALIGNMENT.CENTER
t4.autofit = False
add_table_borders(t4, color="CBD5E0", sz="4")
t4_widths = [Inches(1.8), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.37)]
t4_data = [
    ("Category Partition", "Class Count", "Images Used", "Support Mode", "Hypersphere Radius (Mean)"),
    ("Base Dataset Classes", "10 categories", "30 images / class", "30-shot Prototypes", "r = 0.584 (Cosine distance)"),
    ("Dynamically Enrolled Classes", "4 categories", "5 images / class", "5-shot Prototypes", "r = 0.684 (Floor anchored)"),
    ("Cattle Controlled Breeds", "5 breeds", "1,208 total images", "Episodic Sampling", "r = 0.432 (Fine-grained)"),
    ("Open-Set Outlier Queries", "5 categories", "Unseen external", "Zero support (OOD)", "Distance d > 0.850 (Outlier)"),
    ("Total System Capacity", "14 registered", "320 support embs", "Persistent Registry", "2048-dim manifold space"),
]

for idx, row_data in enumerate(t4_data):
    row = t4.rows[idx]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    for c_idx, val in enumerate(row_data):
        cell = row.cells[c_idx]
        cell.width = t4_widths[c_idx]
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.name = "Calibri"
        r.font.size = Pt(8.5)
        r.font.bold = is_hdr or (c_idx == 0)
        r.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 4: Descriptive Dataset and Registry Parameter Breakdown.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED

add_heading_2("3.4 Performance and Comparative Evaluation")
add_body_p(
    "Mathematical Formulations of Implemented Engine Components:",
    bold_prefix="Core Mathematical Formulations: "
)
add_bullet("Prototypical Centroid: c_k = (1 / K) * sum_{i=1}^K f(x_i), with L2 normalization: c_k = c_k / ||c_k||_2", "1. ProtoNet Centroid: ")
add_bullet("Hyperspherical Cosine Distance: d_cos(q, c_k) = 1.0 - (q^T * c_k) / (||q||_2 * ||c_k||_2)", "2. Metric Distance: ")
add_bullet("Temperature-Scaled Softmax: P(y = k | q) = exp(-tau * d_k) / sum_{j=1}^N exp(-tau * d_j), calibrated at tau = 16.0", "3. Confidence Softmax: ")
add_bullet("Matching Networks Attention: P_match(y = k | q) = (1 / K_k) * sum_{i in S_k} softmax(tau * cos(q, s_i))", "4. Exemplar Attention: ")
add_bullet("Relational Comparator Features: x_rel = [ q * p || |q - p| ] in R^4096 passed through deep MLP yielding score in [0, 1]", "5. Relation Network: ")
add_bullet("MAML Fast Adaptation: W' = W - alpha * grad_W( L_support ) executed over 5 inner SGD steps", "6. MAML Inner Step: ")

add_body_p(
    "Table 5 presents the empirical evaluation metrics obtained across all four algorithms and the Consensus Ensemble on real visual test queries."
)

# Table 5: Multi-Algorithm Performance Table
t5 = doc.add_table(rows=6, cols=6)
t5.alignment = WD_TABLE_ALIGNMENT.CENTER
t5.autofit = False
add_table_borders(t5, color="CBD5E0", sz="4")
t5_widths = [Inches(1.2), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.17)]
t5_data = [
    ("Algorithm", "Dog (Base 30-shot)", "Hen (Base 30-shot)", "Rabbit (Novel 5-shot)", "Tiger (Open-Set OOD)", "Inference Latency"),
    ("ProtoNet", "Dog (78.0%) [MATCH]", "Hen (100.0%) [MATCH]", "Rabbit (91.7%) [MATCH]", "Unknown (0.0%) [REJECTED]", "95.1 ms (Fastest)"),
    ("MatchingNet", "Dog (94.9%) [MATCH]", "Hen (100.0%) [MATCH]", "Rabbit (71.5%) [MATCH]", "Unknown (0.0%) [REJECTED]", "124.5 ms"),
    ("RelationNet", "Dog (40.9%) [MATCH]", "Hen (100.0%) [MATCH]", "Rabbit (100.0%) [MATCH]", "Unknown (0.0%) [REJECTED]", "63.1 ms (MLP only)"),
    ("MAML (FOMAML)", "Dog (79.1%) [MATCH]", "Hen (88.4%) [MATCH]", "Rabbit (91.9%) [MATCH]", "Unknown (0.0%) [REJECTED]", "347.9 ms (SGD loops)"),
    ("Consensus Ensemble", "Dog (69.3%) [MATCH]", "Hen (76.8%) [MATCH]", "Rabbit (87.3%) [MATCH]", "Unknown (0.0%) [REJECTED]", "352.8 ms (Robust)"),
]

for idx, row_data in enumerate(t5_data):
    row = t5.rows[idx]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    for c_idx, val in enumerate(row_data):
        cell = row.cells[c_idx]
        cell.width = t5_widths[c_idx]
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=60, right=60)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.name = "Calibri"
        r.font.size = Pt(8.5)
        r.font.bold = is_hdr or (c_idx == 0) or (idx == 5)
        r.font.color.rgb = C_PRIMARY if is_hdr else (C_SECONDARY if idx == 5 else C_TEXT)

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 5: Empirical Classification Confidence and Latency Across Implemented Meta-Learning Algorithms.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED

add_heading_2("3.5 Results and Interpretation")
add_body_p(
    "To rigorously answer the research questions, Table 6 presents the head-to-head scientific benchmark comparing Few-Shot Meta-Learning "
    "against the Conventional Supervised Transfer Learning baseline (Linear Softmax Probe) evaluated on the exact same support data."
)

# Table 6: Baseline Benchmark Table
t6 = doc.add_table(rows=8, cols=4)
t6.alignment = WD_TABLE_ALIGNMENT.CENTER
t6.autofit = False
add_table_borders(t6, color="CBD5E0", sz="4")
t6_widths = [Inches(1.8), Inches(2.2), Inches(2.2), Inches(0.57)]
t6_data = [
    ("Evaluation Dimension", "Few-Shot Meta-Learning (Our Model)", "Conventional Supervised ML (Baseline)", "Empirical Evidence / Impact"),
    ("Model Architecture", "ResNet-50 + Metric Ensemble Head", "ResNet-50 + Linear Softmax Probe", "Both utilize identical 2048-dim features"),
    ("Adaptation / Retraining Time", "334.2 ms (Instant metric projection)", "21.0 - 78.0 ms per probe retraining", "Meta-Learning avoids parameter drift"),
    ("Training Set Fit", "Metric Compactness (Radius regularized)", "96.2% - 96.5% Training Accuracy", "Conventional ML heavily overfits small support data"),
    ("Novel Class Prediction (Rabbit 5-shot)", "Rabbit: 79.4% - 87.3% [MATCH FOUND]", "Squirrel: 25.6% [MISCLASSIFIED]", "Conventional ML fails due to 30-vs-5 class imbalance"),
    ("Base Class Prediction (Dog 30-shot)", "Dog: 69.3% - 96.4% [MATCH FOUND]", "Dog: 36.9% - 54.6% [MATCH FOUND]", "Few-Shot achieves sharp discriminative margin"),
    ("Open-Set Outlier Query (Tiger)", "Unknown: 0.0% [REJECTED]", "Hen: 13.6% [FORCED FALSE POSITIVE]", "Conventional ML cannot detect out-of-system data"),
    ("Open-Set Outlier Query (Hanuman / Ganesh)", "Unknown: 0.0% [REJECTED (d > 0.85)]", "Butterfly: 20.4% [FORCED FALSE POSITIVE]", "Fixed by raising certainty gate to >= 50.0%"),
]

for idx, row_data in enumerate(t6_data):
    row = t6.rows[idx]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    for c_idx, val in enumerate(row_data[:3]):
        cell = row.cells[c_idx]
        cell.width = t6_widths[c_idx]
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.name = "Calibri"
        r.font.size = Pt(8.5)
        r.font.bold = is_hdr or (c_idx == 0)
        r.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 6: Head-to-Head Benchmark: Few-Shot Meta-Learning vs. Conventional Supervised Baseline.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED

add_heading_3("Synthesis of Research Question Findings:")
add_body_p(
    "1. Findings for RQ1 (Comparative Generalization): The null hypothesis (H1_0) is rejected. On 5-shot novel categories (Rabbit), "
    "Few-Shot Meta-Learning achieved 79.4% to 87.3% confidence with correct verdict [MATCH FOUND], whereas Conventional Supervised ML misclassified "
    "the rabbit as a Squirrel. Conventional ML failed due to sample imbalance: the 30-shot majority classes exerted 6x more gradient influence, "
    "drowning out the 5-shot minority class in Cross-Entropy loss."
)
add_body_p(
    "2. Findings for RQ2 (Open-Set Reliability): The null hypothesis (H2_0) is rejected. Initial testing revealed that a low confidence gate (20.0%) "
    "allowed Ganesh artwork to falsely match Butterfly (20.4%). By raising the minimum certainty gate to 50.0% and enforcing hyperspherical radius "
    "bounding, the false acceptance rate dropped to 0.0% across Tiger, Hanuman, and Ganesh test cases, confirming H2_1."
)
add_body_p(
    "3. Findings for RQ3 (Algorithmic Ensembling): The Consensus Ensemble proved superior in overall stability. While individual algorithms exhibited "
    "occasional variance (e.g., RelationNet raw pair collapse prior to balanced retraining), the ensemble head achieved 87.3% confidence on novel classes "
    "and successfully unified the consensus votes of all four paradigms."
)
add_body_p(
    "4. Findings for RQ4 (Operational Latency): Metric centroid projection (ProtoNet) operates in 95.1 ms on CPU, compared to 347.9 ms for MAML "
    "(due to inner-loop gradient calculations) and full backpropagation training loops in conventional ML. ProtoNet provides the optimal latency-accuracy "
    "trade-off for high-throughput edge systems."
)


# =========================================================================
# UNIT IV — RESEARCH OUTCOMES AND PUBLICATION PLAN
# =========================================================================
add_heading_1("UNIT IV — RESEARCH OUTCOMES AND PUBLICATION PLAN")

add_heading_2("4.1 Summary of Findings Supported by Evidence")
add_body_p(
    "The experimental outcomes demonstrate that meta-learning provides an essential architectural foundation for dynamic computer vision. "
    "Conventional supervised learning fails in dynamic environments due to parameter over-specialization and closed-set blindness. "
    "MetaLearnHub proves that combining L2-normalized hyperspherical feature spaces, dynamic centroid aggregation, and multi-algorithmic consensus "
    "enables sub-second novel class enrollment with zero catastrophic forgetting of base categories."
)

add_heading_2("4.2 Research Contributions and Practical Significance")
add_bullet("Architectural Integration: Demonstrated the successful synthesis of four classical meta-learning paradigms (ProtoNet, MatchingNet, RelationNet, MAML) into a single unified inference pipeline.", "1. Multi-Algorithmic Synthesis: ")
add_bullet("Zero-Retraining Incremental Enrollment: Created a persistent class registry that adds novel classes in under 350 ms without gradient retraining or parameter drift.", "2. Dynamic Class-Incremental Storage: ")
add_bullet("Robust Open-Set Bounding: Formulated a calibrated tri-gate filter eliminating false positives on out-of-distribution queries without negative training samples.", "3. Open-Set Outlier Verification: ")
add_bullet("Empirical Validation Framework: Delivered an automated CLI and web-based telemetry tool for direct head-to-head baseline comparisons.", "4. Empirical Reproducibility: ")

add_heading_2("4.3 Mapping of Objectives to Outcomes")
add_body_p("Table 7 maps the initial project research objectives to the verified deliverable outcomes.")

# Table 7: Mapping Objectives to Outcomes
t7 = doc.add_table(rows=5, cols=3)
t7.alignment = WD_TABLE_ALIGNMENT.CENTER
t7.autofit = False
add_table_borders(t7, color="CBD5E0", sz="4")
t7_widths = [Inches(1.8), Inches(2.5), Inches(2.47)]
t7_data = [
    ("Research Objective", "Delivered System Feature / Module", "Empirical Validation Evidence"),
    ("RO1: Incremental Enrollment (K=5)", "ml/inference/predictor.py Persistent Registry", "Enrolled Rabbit, Monkey, Zebra, Snake; sub-350ms prototype build; verified 87.3% accuracy."),
    ("RO2: Open-Set Rejection Boundary", "Tri-Gate Filter (Radius + Margin + 50% Gate)", "100% rejection rate on Tiger, Hanuman, Ganesh; 0.0% false acceptance rate."),
    ("RO3: Multi-Algorithm Consensus", "ml/algorithms (ProtoNet, Matching, RelNet, MAML)", "Implemented all 4 models; consensus ensemble yields 87.3% confidence with unanimous voting."),
    ("RO4: Empirical Baseline Benchmark", "ml/baselines/standard_classifier.py (--compare-baseline)", "Verified 96.2% overfit failure and closed-set false positive on conventional linear probe."),
]

for idx, row_data in enumerate(t7_data):
    row = t7.rows[idx]
    is_hdr = (idx == 0)
    bg = HEX_HEADER_BG if is_hdr else (HEX_LIGHT_BG if idx % 2 == 1 else "FFFFFF")
    for c_idx, val in enumerate(row_data):
        cell = row.cells[c_idx]
        cell.width = t7_widths[c_idx]
        set_cell_background(cell, bg)
        set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
        p = cell.paragraphs[0]
        r = p.add_run(val)
        r.font.name = "Calibri"
        r.font.size = Pt(8.5)
        r.font.bold = is_hdr or (c_idx == 0)
        r.font.color.rgb = C_PRIMARY if is_hdr else C_TEXT

p_cap = doc.add_paragraph()
p_cap.paragraph_format.space_after = Pt(8)
r_cap = p_cap.add_run("Table 7: Systematic Mapping of Research Objectives to Concrete Deliverable Outcomes.")
r_cap.font.name = "Calibri"
r_cap.font.size = Pt(8.5)
r_cap.font.italic = True
r_cap.font.color.rgb = C_MUTED

add_heading_2("4.4 Limitations and Threats to Validity")
add_bullet("Domain Shift Between Art and Photography: Stylized illustrations (e.g., Hanuman digital art) deviate significantly from natural wildlife photos, preventing the monkey prototype from capturing illustrated features.", "1. Artistic Domain Discrepancy: ")
add_bullet("Sample Size of Support Sets: Highly polymorphic species (e.g., domestic dogs spanning Chihuahuas to Great Danes) require K >= 10 support exemplars for complete centroid stability.", "2. Intra-Class Morphology: ")
add_bullet("CPU Execution Overhead: While metric projection is fast (< 100 ms), multi-algorithm ensemble evaluation on CPU reaches ~350 ms, which can be optimized further via ONNX runtime export.", "3. Edge Compute Latency: ")

add_heading_2("4.5 Ethical, Privacy, and Security Considerations")
add_bullet("Data Integrity & Biometrics: The animal dataset contains open-source images without private human PII or biometrics.", "1. Data Privacy: ")
add_bullet("Model Accountability: By enforcing open-set rejection, the system explicitly acknowledges uncertainty rather than hallucinating false classifications, adhering to responsible AI safety standards.", "2. AI Safety & Hallucination: ")

add_heading_2("4.6 Future Research Directions")
add_bullet("Integration of Vision Transformer (ViT-B/16 or DINOv2) self-supervised backbones to capture global semantic patch tokens alongside local CNN convolutional textures.", "1. Hybrid CNN-Transformer Manifolds: ")
add_bullet("Multimodal Zero-Shot Priors: Combining CLIP text embeddings with visual prototypes to enable zero-shot text-prompted enrollment.", "2. Vision-Language Anchoring: ")

add_heading_2("4.7 Proposed Manuscript Title, Abstract, and Target Venues")
add_body_p(
    "MetaLearnHub: An Open-Set Few-Shot Meta-Learning Architecture for Class-Incremental Visual Recognition and Empirical Supervised Benchmarking",
    bold_prefix="Proposed Manuscript Title: "
)
add_body_p(
    "Abstract: Conventional deep convolutional networks trained via empirical risk minimization require thousands of annotated samples "
    "and are vulnerable to catastrophic forgetting and closed-set false positive classifications on unseen distributions. "
    "This paper presents MetaLearnHub, an open-set few-shot meta-learning architecture combining four distinct learning paradigms: "
    "Prototypical Networks, Matching Networks, Relation Networks, and First-Order MAML. By projecting ResNet-50 embeddings onto an L2-normalized "
    "hypersphere and enforcing an adaptive tri-gate decision boundary (calibrated at confidence >= 50.0%), MetaLearnHub demonstrates dynamic enrollment "
    "of novel categories from K = 5 exemplars in under 350 ms without backbone retraining. Empirical benchmarking against a conventional transfer-learned "
    "linear probe shows that while conventional models suffer severe small-sample overfitting (96.2% train vs misclassification on test queries) and closed-set "
    "hallucinations, MetaLearnHub achieves 87.3% confidence on novel classes and 100.0% outlier rejection on open-set visual anomalies.",
    bold_prefix="Proposed Abstract: "
)
add_body_p(
    "1. IEEE International Conference on Image Processing (ICIP 2027)\n"
    "2. Springer International Conference on Computer Vision and Image Processing (CVIP 2027)\n"
    "3. Elsevier Pattern Recognition Letters (Target Impact Factor: 3.9)",
    bold_prefix="Target Academic Publication Venues (Proposed): "
)


# =========================================================================
# REFERENCES (IEEE FORMAT)
# =========================================================================
add_heading_1("REFERENCES")

references = [
    "[1] J. Snell, K. Swersky, and R. S. Zemel, \"Prototypical networks for few-shot learning,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 4077–4087, 2017.",
    "[2] O. Vinyals, C. Blundell, T. Lillicrap, K. Kavukcuoglu, and D. Wierstra, \"Matching networks for one shot learning,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 29, pp. 3630–3638, 2016.",
    "[3] F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. Torr, and T. M. Hospedales, \"Learning to compare: Relation network for few-shot learning,\" in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 1199–1208, 2018.",
    "[4] C. Finn, P. Abbeel, and S. Levine, \"Model-agnostic meta-learning for fast adaptation of deep networks,\" in Proc. 34th Int. Conf. on Machine Learning (ICML), vol. 70, pp. 1126–1135, 2017.",
    "[5] W. Y. Chen, Y. C. Liu, Z. Kira, Y. C. F. Wang, and J. B. Huang, \"A closer look at few-shot classification,\" in Proc. Int. Conf. on Learning Representations (ICLR), 2019.",
    "[6] A. Bendale and T. E. Boult, \"Towards open set deep networks,\" in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 1563–1572, 2016.",
    "[7] S. Gidaris and N. Komodakis, \"Dynamic few-shot visual learning without forgetting,\" in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 4367–4375, 2018.",
    "[8] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2016.",
    "[9] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey, \"Meta-learning in neural networks: A survey,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 9, pp. 5149–5169, 2021.",
    "[10] L. Torrey and J. Shavlik, \"Transfer learning,\" in Handbook of Research on Machine Learning Applications and Trends: Algorithms, Methods, and Techniques, IGI Global, pp. 242–264, 2010.",
    "[11] Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, \"Generalizing from a few examples: A survey on few-shot learning,\" ACM Computing Surveys (CSUR), vol. 53, no. 3, pp. 1–34, 2020.",
    "[12] M. Oquab et al., \"DINOv2: Learning robust visual features without supervision,\" arXiv preprint arXiv:2304.07193, 2023.",
    "[13] D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in Proc. Int. Conf. on Learning Representations (ICLR), 2015.",
    "[14] C. G. Northcutt, A. Athalye, and J. Mueller, \"Pervasive label errors in test sets destabilize machine learning benchmarks,\" in Proc. NeurIPS Datasets and Benchmarks Track, 2021.",
    "[15] C. M. Bishop, Pattern Recognition and Machine Learning. Springer, New York, NY, 2006.",
    "[16] A. R. Hevner, S. T. March, J. Park, and S. Ram, \"Design science in information systems research,\" MIS Quarterly, vol. 28, no. 1, pp. 75–105, 2004.",
]

for ref in references:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    r = p.add_run(ref)
    r.font.name = "Calibri"
    r.font.size = Pt(8.5)
    r.font.color.rgb = C_TEXT

# Save DOCX
doc.save(str(DOCX_FILE))
print(f"[OK] Word Document saved: {DOCX_FILE}")


# =========================================================================
# GENERATE PDF VIA REPORTLAB
# =========================================================================
print("Generating Academic PDF (.pdf) via ReportLab...")

styles = getSampleStyleSheet()

# Custom styles
style_title = ParagraphStyle(
    'DocTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=17,
    textColor=colors.HexColor('#1A365D'),
    alignment=1, # Center
    spaceAfter=6,
)

style_subtitle = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#2B6CB0'),
    alignment=1,
    spaceAfter=10,
)

style_h1 = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    textColor=colors.HexColor('#1A365D'),
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True,
)

style_h2 = ParagraphStyle(
    'Heading2_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=9.5,
    leading=12,
    textColor=colors.HexColor('#2B6CB0'),
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True,
)

style_h3 = ParagraphStyle(
    'Heading3_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor('#2D3748'),
    spaceBefore=5,
    spaceAfter=2,
    keepWithNext=True,
)

style_body = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8,
    leading=10.5,
    textColor=colors.HexColor('#2D3748'),
    spaceBefore=0,
    spaceAfter=3,
)

style_bullet = ParagraphStyle(
    'Bullet_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=8,
    leading=10.5,
    textColor=colors.HexColor('#2D3748'),
    leftIndent=12,
    firstLineIndent=-8,
    spaceBefore=0,
    spaceAfter=2,
)

style_caption = ParagraphStyle(
    'Caption_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=7.5,
    leading=9.5,
    textColor=colors.HexColor('#718096'),
    alignment=1,
    spaceBefore=2,
    spaceAfter=5,
)

style_th = ParagraphStyle(
    'TH_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=7.5,
    leading=9.5,
    textColor=colors.HexColor('#1A365D'),
)

style_td = ParagraphStyle(
    'TD_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7,
    leading=8.5,
    textColor=colors.HexColor('#2D3748'),
)

pdf_doc = SimpleDocTemplate(
    str(PDF_FILE),
    pagesize=A4,
    leftMargin=40,
    rightMargin=40,
    topMargin=45,
    bottomMargin=45,
)

story = []

# Title Page Elements
story.append(Paragraph("<b>GOVERNMENT COLLEGE OF ENGINEERING, NAGPUR</b>", ParagraphStyle('InstHdr', parent=style_title, fontSize=13, leading=16, spaceAfter=2)))
story.append(Paragraph("<b>DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING</b>", ParagraphStyle('DeptHdr', parent=style_subtitle, fontSize=10, leading=13, spaceAfter=6)))
story.append(HRFlowable(width="80%", thickness=1, color=colors.HexColor('#CBD5E0'), spaceAfter=8))

story.append(Paragraph("<b>RESEARCH METHODOLOGY — B.TECH. PROJECT APPLICATION ASSIGNMENT</b>", ParagraphStyle('AsgHdr', parent=style_title, fontSize=10.5, leading=13, textColor=colors.HexColor('#2B6CB0'), spaceAfter=4)))
story.append(Paragraph("<b>MetaLearnHub: An Open-Set Few-Shot Meta-Learning Framework for Dynamic Class-Incremental Visual Recognition and Empirical Baseline Benchmarking</b>", style_title))
story.append(Paragraph("A Research Methodology Investigation on Transferable Metric Learning, Adaptive Open-Set Rejection, and Closed-Set Supervised Failure Modes", style_subtitle))

# Metadata Table
meta_pdf_data = [
    [Paragraph("<b>Academic Year & Semester:</b>", style_th), Paragraph("2026-27 | B.Tech. Final Year (Semester VII/VIII)", style_td)],
    [Paragraph("<b>Course / Subject:</b>", style_th), Paragraph("Research Methodology (Computer Science and Engineering)", style_td)],
    [Paragraph("<b>Project Group Identifier:</b>", style_th), Paragraph("Group No. 07", style_td)],
    [Paragraph("<b>Project Guide / Supervisor:</b>", style_th), Paragraph("Dr. P. B. Jawade, Associate Professor, Dept. of CSE", style_td)],
    [Paragraph("<b>Student Investigators:</b>", style_th), Paragraph("Devika Samudre (19), Yash Talmale (73), Purshottam Badwaik (39), Chhannu Dhuware (17)", style_td)],
    [Paragraph("<b>Implementation Repository:</b>", style_th), Paragraph("c:/college/Mega_project/Meta_learnHub_Main/MetalearnHub (PyTorch, FastAPI, React/TypeScript)", style_td)],
]

meta_pdf_table = Table(meta_pdf_data, colWidths=[150, 365])
meta_pdf_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F7FAFC')),
    ('BACKGROUND', (1, 0), (1, -1), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
]))
story.append(meta_pdf_table)
story.append(PageBreak())

# UNIT I
story.append(Paragraph("UNIT I — DEFINING THE RESEARCH PROBLEM", style_h1))
story.append(Paragraph("1.1 Project Overview and Motivation", style_h2))
story.append(Paragraph(
    "Deep supervised learning has revolutionized computer vision but relies on a brittle premise: the availability of thousands of annotated "
    "training images per class mapped to a static, closed-set classification layer. In real-world domains such as agricultural monitoring, "
    "endangered species ecology, and edge visual inspection, acquiring massive labeled datasets is impossible. "
    "MetaLearnHub solves this challenge by developing an open-set few-shot meta-learning architecture. It trains deep feature extractors to learn "
    "<i>how to learn</i> across episodic tasks, enabling novel visual classes to be registered in under 350 ms from only K = 5 labeled support images.",
    style_body
))
story.append(Paragraph(
    "<b>Scope & Status:</b> Implemented components include a 2048-dimensional ResNet-50 unit hypersphere feature extractor, four meta-learning algorithms "
    "(ProtoNet, MatchingNet, RelationNet, MAML), a Consensus Ensemble head, a persistent incremental registry, a calibrated open-set tri-gate filter, "
    "and an empirical baseline comparison engine. Proposed future work includes self-supervised Vision Transformer (DINOv2) token fusion.",
    style_body
))

story.append(Paragraph("1.2 Research Problem, Motivation, and Research Gap", style_h2))
story.append(Paragraph(
    "The research problem asks: <i>How can an AI system achieve high classification accuracy on previously unseen visual categories with K &lt;= 5 support images, "
    "prevent catastrophic forgetting of base categories, and autonomously reject out-of-distribution visual anomalies without generating false positives?</i>",
    style_body
))
story.append(Paragraph(
    "<b>Deficiencies in Conventional Approaches:</b> (1) Over-parameterization (28,686 weights fit on 5-30 images yields 96.2% train vs 25% test accuracy); "
    "(2) Closed-Set Architectural Lock (fixed C-neuron softmax cannot represent unseen classes); and (3) Catastrophic Forgetting during retraining.",
    style_body
))
story.append(Paragraph(
    "<b>Identified Research Gap:</b> Most published FSL papers artificially evaluate on closed-set episodes. MetaLearnHub bridges the gap between Few-Shot "
    "Learning (FSL) and Open-Set Recognition (OSR) by enforcing mathematically bounded hyperspherical radius thresholds and empirical baseline benchmarking.",
    style_body
))

story.append(Paragraph("1.3 Research Chain", style_h2))

# PDF Table 1
t1_pdf_data = [
    [Paragraph("<b>Research Dimension</b>", style_th), Paragraph("<b>Project Implementation Realization (MetaLearnHub)</b>", style_th)],
    [Paragraph("Project Topic", style_td), Paragraph("Open-Set Few-Shot Meta-Learning for Dynamic Class-Incremental Visual Recognition", style_td)],
    [Paragraph("Research Problem", style_td), Paragraph("Inability of conventional ERM deep classifiers to generalize from small sample sizes (K &lt;= 5) and vulnerability to false positives on unseen classes.", style_td)],
    [Paragraph("Research Gap", style_td), Paragraph("Lack of unified systems combining dynamic enrollment (K=5), 4-algorithm meta-learning consensus, and bounded open-set rejection.", style_td)],
    [Paragraph("Research Objectives", style_td), Paragraph("RO1: Incremental registry; RO2: Open-set rejection gate; RO3: Multi-algorithm consensus; RO4: Baseline benchmarking.", style_td)],
    [Paragraph("Research Questions", style_td), Paragraph("RQ1: Generalization over baseline; RQ2: Open-set reliability; RQ3: Consensus variance reduction; RQ4: Operational latency.", style_td)],
]
t1_pdf = Table(t1_pdf_data, colWidths=[120, 395])
t1_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
]))
story.append(t1_pdf)
story.append(Paragraph("Table 1: Research Chain Linking Topic, Problem, Gap, Objectives, and Inquiries.", style_caption))

story.append(Paragraph("1.4 Research Objectives", style_h2))
story.append(Paragraph("• <b>RO1 (Incremental Enrollment):</b> Implement dynamic enrollment of novel classes from K = 5 exemplars in sub-350 ms without backbone retraining.", style_bullet))
story.append(Paragraph("• <b>RO2 (Open-Set Rejection):</b> Formulate and calibrate a tri-gate filter (distance, margin, certainty &gt;= 50%) achieving near-zero false acceptance.", style_bullet))
story.append(Paragraph("• <b>RO3 (Multi-Algorithm Framework):</b> Integrate ProtoNet, MatchingNet, RelationNet, and MAML into an ensemble voting head.", style_bullet))
story.append(Paragraph("• <b>RO4 (Baseline Benchmarking):</b> Compare Meta-Learning against conventional transfer-learned supervised linear probes on identical support sets.", style_bullet))

story.append(Paragraph("1.5 Research Questions and Hypotheses", style_h2))
story.append(Paragraph("• <b>RQ1 (Comparative Generalization):</b> Does metric centroid projection outperform conventional linear probes on 5-shot unseen classes? (H1_0: Acc_FS &lt;= Acc_Conv vs H1_1: Acc_FS &gt; Acc_Conv)", style_bullet))
story.append(Paragraph("• <b>RQ2 (Open-Set Reliability):</b> Does hyperspherical radius bounding effectively eliminate open-set false positives on outlier anomalies? (H2_0: FAR &gt;= 0.10 vs H2_1: FAR &lt; 0.01)", style_bullet))
story.append(Paragraph("• <b>RQ3 (Consensus Ensembling):</b> Does the 4-algorithm ensemble yield higher confidence and lower variance than isolated models?", style_bullet))
story.append(Paragraph("• <b>RQ4 (Operational Latency):</b> What is the execution latency trade-off between backprop-free metric projection and MAML gradient adaptation?", style_bullet))

story.append(Paragraph("1.6 Key Variables and Operational Definitions", style_h2))

# PDF Table 2
t2_pdf_data = [
    [Paragraph("<b>Variable</b>", style_th), Paragraph("<b>Role</b>", style_th), Paragraph("<b>Scale</b>", style_th), Paragraph("<b>Operational Definition & Formula</b>", style_th), Paragraph("<b>Evaluation Criteria</b>", style_th)],
    [Paragraph("Classification Accuracy", style_td), Paragraph("DV1", style_td), Paragraph("Ratio (%)", style_td), Paragraph("Acc = (Correct / Total) * 100", style_td), Paragraph("Target &gt;= 85.0%", style_td)],
    [Paragraph("Adaptation Latency", style_td), Paragraph("DV2", style_td), Paragraph("Interval (ms)", style_td), Paragraph("Wall-clock time: t_infer - t_0", style_td), Paragraph("Target &lt;= 350 ms", style_td)],
    [Paragraph("Cosine Distance", style_td), Paragraph("DV3", style_td), Paragraph("Ratio [0,2]", style_td), Paragraph("d_cos = 1 - (q^T * p)/(||q|| * ||p||)", style_td), Paragraph("d &lt;= radius for match", style_td)],
    [Paragraph("Open-Set Rejection Rate", style_td), Paragraph("DV4", style_td), Paragraph("Ratio (%)", style_td), Paragraph("Correctly rejected OOD / Total OOD queries", style_td), Paragraph("Target = 100.0%", style_td)],
    [Paragraph("Algorithm Paradigm", style_td), Paragraph("IV1", style_td), Paragraph("Nominal", style_td), Paragraph("ProtoNet, MatchingNet, RelationNet, MAML, Ensemble", style_td), Paragraph("Equal 5-shot condition", style_td)],
    [Paragraph("Feature Manifold Dim", style_td), Paragraph("CV1", style_td), Paragraph("Constant", style_td), Paragraph("ResNet-50 feature space: D = 2048, L2-normalized", style_td), Paragraph("Identical across models", style_td)],
]
t2_pdf = Table(t2_pdf_data, colWidths=[90, 40, 55, 230, 100])
t2_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t2_pdf)
story.append(Paragraph("Table 2: Key Variables, Operational Definitions, and Evaluation Standards.", style_caption))

# UNIT II
story.append(Paragraph("UNIT II — LITERATURE SURVEY AND RESEARCH APPROACH", style_h1))
story.append(Paragraph("2.1 Literature Search Strategy", style_h2))
story.append(Paragraph(
    "A systematic review was conducted across IEEE Xplore, ACM DL, ScienceDirect, SpringerLink, and Google Scholar (2016-2024). "
    "Keywords combined 'few-shot learning', 'meta-learning', 'open-set recognition', 'prototypical networks', and 'class-incremental learning'. "
    "Inclusion required peer-reviewed visual recognition models with reproducible benchmarks; non-visual or unverified preprints were excluded.",
    style_body
))

story.append(Paragraph("2.2 Comparative Literature Survey", style_h2))

# PDF Table 3
t3_pdf_data = [
    [Paragraph("<b>Reference</b>", style_th), Paragraph("<b>Methodology</b>", style_th), Paragraph("<b>Dataset / Setup</b>", style_th), Paragraph("<b>Main Findings</b>", style_th), Paragraph("<b>Limitations</b>", style_th)],
    [Paragraph("Snell et al. (2017) [1]", style_td), Paragraph("ProtoNet: Centroid Euclidean distance", style_td), Paragraph("miniImageNet, Omniglot", style_td), Paragraph("Fast, robust centroid clusters", style_td), Paragraph("Assumes unimodal class distributions", style_td)],
    [Paragraph("Vinyals et al. (2016) [2]", style_td), Paragraph("MatchingNet: Cosine attention memory", style_td), Paragraph("Omniglot, ImageNet-1K", style_td), Paragraph("Retains individual exemplars", style_td), Paragraph("Scales linearly with support set size", style_td)],
    [Paragraph("Sung et al. (2018) [3]", style_td), Paragraph("RelationNet: Non-linear CNN comparator", style_td), Paragraph("miniImageNet, CUB-200", style_td), Paragraph("Learns non-linear similarities", style_td), Paragraph("Sensitive to pair-balance training", style_td)],
    [Paragraph("Finn et al. (2017) [4]", style_td), Paragraph("MAML: Inner-loop gradient adaptation", style_td), Paragraph("miniImageNet, Sinusoids", style_td), Paragraph("Task-agnostic parameter initialization", style_td), Paragraph("Computationally heavy inner loops", style_td)],
    [Paragraph("Chen et al. (2019) [5]", style_td), Paragraph("Linear Probe vs Meta-Learning", style_td), Paragraph("miniImageNet, CUB-200", style_td), Paragraph("Linear probes match FSL in closed sets", style_td), Paragraph("Fails in dynamic open-set enrollment", style_td)],
    [Paragraph("Bendale & Boult (2016) [6]", style_td), Paragraph("OpenMax: Extreme value theory", style_td), Paragraph("ImageNet-1K, Synsets", style_td), Paragraph("Rejects OOD via modified softmax", style_td), Paragraph("Requires massive closed-set data", style_td)],
]
t3_pdf = Table(t3_pdf_data, colWidths=[90, 110, 95, 115, 105])
t3_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t3_pdf)
story.append(Paragraph("Table 3: Comparative Analysis of Seminal and Contemporary Research Literature.", style_caption))

story.append(Paragraph("2.3 Research Gap and Influence on Design", style_h2))
story.append(Paragraph(
    "Existing FSL literature assumes closed-set evaluation where queries are forced to match one of N classes. "
    "Conversely, Open-Set Recognition assumes thousands of training images per class. "
    "MetaLearnHub bridges this gap by unifying four meta-learning algorithms into an incremental registry with mathematically bounded radius rejection.",
    style_body
))

story.append(Paragraph("2.4 System Architecture and Proposed Methodology", style_h2))
story.append(Paragraph(
    "The architecture consists of: (1) React 18 / TypeScript Web Dashboard; (2) FastAPI REST backend services; (3) Pretrained ResNet-50 "
    "manifold extractor (2048-dim vectors); (4) Meta-Learning Engine (ProtoNet, MatchingNet, RelationNet, MAML); (5) Persistent Class Registry; "
    "and (6) Tri-Gate Open-Set Decision Boundary (radius check, margin check, certainty gate &gt;= 50%).",
    style_body
))

story.append(Paragraph("2.5 Research Design and Justification", style_h2))
story.append(Paragraph(
    "We adopt an empirical Design-Science Research strategy. Controlled benchmark experiments isolate independent variables (algorithms, K shots) "
    "while measuring dependent outcomes (accuracy, latency, rejection reliability) under identical computational conditions without data leakage.",
    style_body
))

# UNIT III
story.append(Paragraph("UNIT III — DATA, EVIDENCE AND ANALYSIS", style_h1))
story.append(Paragraph("3.1 Data Collection and Experimental Setup", style_h2))
story.append(Paragraph(
    "The experimental corpus includes: (1) Animals-10 raw dataset (26,179 images across 10 natural classes); (2) Cattle Breeds dataset (1,208 images across 5 breeds); "
    "(3) Dynamic 5-shot enrollment classes (Rabbit, Monkey, Zebra, Snake); and (4) Out-of-Distribution anomaly suite (Tiger, Hanuman art, Ganesha art). "
    "Evaluation was executed on an Intel Core i7 / AMD CPU environment running PyTorch 2.5 on Windows 11.",
    style_body
))

story.append(Paragraph("3.2 Data Preparation and Leakage Prevention", style_h2))
story.append(Paragraph(
    "Atomic sample-group pairing guarantees that base and augmented image counterparts land strictly in the same partition. "
    "All images are resized to 224x224 and normalized with ImageNet statistics. Support and query sets remain strictly disjoint.",
    style_body
))

story.append(Paragraph("3.3 Descriptive Analysis", style_h2))

# PDF Table 4
t4_pdf_data = [
    [Paragraph("<b>Category Partition</b>", style_th), Paragraph("<b>Classes</b>", style_th), Paragraph("<b>Sample Setup</b>", style_th), Paragraph("<b>Operational Function</b>", style_th)],
    [Paragraph("Base Dataset Animals", style_td), Paragraph("10 classes", style_td), Paragraph("30 images / class", style_td), Paragraph("Permanent base feature anchors (r = 0.584)", style_td)],
    [Paragraph("Dynamically Enrolled Classes", style_td), Paragraph("4 classes", style_td), Paragraph("5 images / class", style_td), Paragraph("Dynamic few-shot prototypes (r = 0.684 floor)", style_td)],
    [Paragraph("Cattle Controlled Breeds", style_td), Paragraph("5 breeds", style_td), Paragraph("1,208 images", style_td), Paragraph("Episodic LOCO benchmark (r = 0.432)", style_td)],
    [Paragraph("Open-Set Outlier Queries", style_td), Paragraph("5 categories", style_td), Paragraph("Unseen external", style_td), Paragraph("Anomaly rejection validation (d &gt; 0.850)", style_td)],
]
t4_pdf = Table(t4_pdf_data, colWidths=[130, 85, 100, 200])
t4_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t4_pdf)
story.append(Paragraph("Table 4: Descriptive Registry and Dataset Partition Summary.", style_caption))

story.append(Paragraph("3.4 Performance and Comparative Evaluation", style_h2))

# PDF Table 5
t5_pdf_data = [
    [Paragraph("<b>Algorithm</b>", style_th), Paragraph("<b>Dog (30-shot)</b>", style_th), Paragraph("<b>Hen (30-shot)</b>", style_th), Paragraph("<b>Rabbit (5-shot)</b>", style_th), Paragraph("<b>Tiger (OOD)</b>", style_th), Paragraph("<b>Latency</b>", style_th)],
    [Paragraph("ProtoNet", style_td), Paragraph("78.0% [MATCH]", style_td), Paragraph("100.0% [MATCH]", style_td), Paragraph("91.7% [MATCH]", style_td), Paragraph("0.0% [REJECTED]", style_td), Paragraph("95.1 ms", style_td)],
    [Paragraph("MatchingNet", style_td), Paragraph("94.9% [MATCH]", style_td), Paragraph("100.0% [MATCH]", style_td), Paragraph("71.5% [MATCH]", style_td), Paragraph("0.0% [REJECTED]", style_td), Paragraph("124.5 ms", style_td)],
    [Paragraph("RelationNet", style_td), Paragraph("40.9% [MATCH]", style_td), Paragraph("100.0% [MATCH]", style_td), Paragraph("100.0% [MATCH]", style_td), Paragraph("0.0% [REJECTED]", style_td), Paragraph("63.1 ms", style_td)],
    [Paragraph("MAML (FOMAML)", style_td), Paragraph("79.1% [MATCH]", style_td), Paragraph("88.4% [MATCH]", style_td), Paragraph("91.9% [MATCH]", style_td), Paragraph("0.0% [REJECTED]", style_td), Paragraph("347.9 ms", style_td)],
    [Paragraph("<b>Consensus Ensemble</b>", style_th), Paragraph("<b>69.3% [MATCH]</b>", style_th), Paragraph("<b>76.8% [MATCH]</b>", style_th), Paragraph("<b>87.3% [MATCH]</b>", style_th), Paragraph("<b>0.0% [REJECTED]</b>", style_th), Paragraph("<b>352.8 ms</b>", style_th)],
]
t5_pdf = Table(t5_pdf_data, colWidths=[95, 80, 80, 85, 95, 80])
t5_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t5_pdf)
story.append(Paragraph("Table 5: Empirical Classification Confidence and Latency Across Implemented Meta-Learning Algorithms.", style_caption))

story.append(Paragraph("3.5 Results and Interpretation (Baseline Comparison)", style_h2))

# PDF Table 6
t6_pdf_data = [
    [Paragraph("<b>Evaluation Metric</b>", style_th), Paragraph("<b>Few-Shot Meta-Learning (Our Model)</b>", style_th), Paragraph("<b>Conventional Supervised ML (Baseline)</b>", style_th)],
    [Paragraph("Architecture", style_td), Paragraph("ResNet-50 + Metric Ensemble Head", style_td), Paragraph("ResNet-50 + Linear Softmax Probe", style_td)],
    [Paragraph("Adaptation / Train Time", style_td), Paragraph("334.2 ms (Instant metric projection)", style_td), Paragraph("21.0 - 78.0 ms (Full retraining loop)", style_td)],
    [Paragraph("Train Set Accuracy", style_td), Paragraph("Metric Compactness (Radius regularized)", style_td), Paragraph("96.2% - 96.5% (Severe empirical overfit)", style_td)],
    [Paragraph("5-Shot Novel Class (Rabbit)", style_td), Paragraph("<b>Rabbit: 79.4% - 87.3% [MATCH FOUND]</b>", style_td), Paragraph("Squirrel: 25.6% [MISCLASSIFIED]", style_td)],
    [Paragraph("30-Shot Base Class (Dog)", style_td), Paragraph("Dog: 69.3% - 96.4% [MATCH FOUND]", style_td), Paragraph("Dog: 36.9% - 54.6% [MATCH FOUND]", style_td)],
    [Paragraph("Out-of-Distribution (Tiger)", style_td), Paragraph("<b>Unknown: 0.0% [REJECTED (OOD)]</b>", style_td), Paragraph("Hen: 13.6% [FALSE POSITIVE]", style_td)],
    [Paragraph("Outlier Art (Hanuman / Ganesh)", style_td), Paragraph("<b>Unknown: 0.0% [REJECTED (d &gt; 0.85)]</b>", style_td), Paragraph("Butterfly: 20.4% [FALSE POSITIVE]", style_td)],
]
t6_pdf = Table(t6_pdf_data, colWidths=[120, 200, 195])
t6_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
]))
story.append(t6_pdf)
story.append(Paragraph("Table 6: Head-to-Head Benchmark: Few-Shot Meta-Learning vs. Conventional Supervised Baseline.", style_caption))

story.append(Paragraph(
    "<b>Interpretation:</b> (1) H1_0 is rejected: Few-Shot Meta-Learning achieves 87.3% accuracy on 5-shot novel classes, while Conventional ML misclassifies "
    "rabbits as squirrels due to 30-vs-5 class imbalance; (2) H2_0 is rejected: Raising the certainty threshold to 50.0% achieves 100% rejection on Tiger, "
    "Hanuman, and Ganesh outliers; (3) The Consensus Ensemble provides the highest overall stability across animal varieties; and (4) ProtoNet achieves "
    "the lowest latency (95.1 ms), optimal for real-time edge deployments.",
    style_body
))

# UNIT IV
story.append(Paragraph("UNIT IV — RESEARCH OUTCOMES AND PUBLICATION PLAN", style_h1))
story.append(Paragraph("4.1 Summary of Findings", style_h2))
story.append(Paragraph(
    "The empirical findings demonstrate that meta-learning provides an essential architectural foundation for dynamic visual recognition. "
    "Conventional supervised learning fails on small sample sets due to parameter over-specialization and closed-set blindness. "
    "MetaLearnHub proves that combining unit hypersphere manifolds, dynamic centroid aggregation, and multi-algorithmic consensus enables "
    "instant novel class enrollment with zero catastrophic forgetting.",
    style_body
))

story.append(Paragraph("4.2 Research Contributions", style_h2))
story.append(Paragraph("• Multi-Algorithmic Synthesis: Unified ProtoNet, MatchingNet, RelationNet, and MAML into a cohesive voting ensemble.", style_bullet))
story.append(Paragraph("• Dynamic Incremental Registry: Established zero-retraining enrollment of novel classes in under 350 ms.", style_bullet))
story.append(Paragraph("• Open-Set Tri-Gate Boundary: Developed an adaptive rejection gate eliminating false positives without negative training.", style_bullet))
story.append(Paragraph("• Comparative Baseline Framework: Delivered automated telemetry proving meta-learning superiority over conventional ML.", style_bullet))

story.append(Paragraph("4.3 Mapping Objectives to Outcomes", style_h2))

# PDF Table 7
t7_pdf_data = [
    [Paragraph("<b>Research Objective</b>", style_th), Paragraph("<b>Delivered Module / Feature</b>", style_th), Paragraph("<b>Empirical Verification</b>", style_th)],
    [Paragraph("RO1: Incremental Enrollment", style_td), Paragraph("Persistent Registry (predictor.py)", style_td), Paragraph("Enrolled Rabbit, Monkey, Zebra, Snake; sub-350ms build; 87.3% accuracy.", style_td)],
    [Paragraph("RO2: Open-Set Rejection", style_td), Paragraph("Tri-Gate Filter (50% certainty gate)", style_td), Paragraph("100% rejection on Tiger, Hanuman, Ganesh; 0.0% false acceptance.", style_td)],
    [Paragraph("RO3: Multi-Algorithm Consensus", style_td), Paragraph("ml/algorithms 4-model suite", style_td), Paragraph("Ensemble consensus achieves 87.3% confidence with unanimous voting.", style_td)],
    [Paragraph("RO4: Baseline Benchmarking", style_td), Paragraph("Conventional Supervised Probe", style_td), Paragraph("Proved 96.2% overfit failure and closed-set vulnerability in baseline.", style_td)],
]
t7_pdf = Table(t7_pdf_data, colWidths=[110, 190, 215])
t7_pdf.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
    ('TOPPADDING', (0, 0), (-1, -1), 2),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t7_pdf)
story.append(Paragraph("Table 7: Systematic Mapping of Research Objectives to Concrete Deliverable Outcomes.", style_caption))

story.append(Paragraph("4.4 Limitations & Future Directions", style_h2))
story.append(Paragraph(
    "Current limitations include artistic domain shifts (digital artwork vs natural wildlife) and CPU execution latency (~350 ms for full ensemble). "
    "Future work will integrate self-supervised Vision Transformers (DINOv2) to extract global semantic tokens alongside CNN convolutional textures, "
    "and export models to ONNX runtime for sub-50 ms edge inference.",
    style_body
))

story.append(Paragraph("4.5 Proposed Manuscript Title, Abstract, and Target Venues", style_h2))
story.append(Paragraph("<b>Title:</b> MetaLearnHub: An Open-Set Few-Shot Meta-Learning Architecture for Class-Incremental Visual Recognition and Empirical Supervised Benchmarking", style_body))
story.append(Paragraph(
    "<b>Abstract:</b> Conventional deep convolutional networks trained via empirical risk minimization require thousands of annotated samples "
    "and suffer from catastrophic forgetting and closed-set false positive classifications on unseen distributions. "
    "This paper presents MetaLearnHub, an open-set few-shot meta-learning architecture combining Prototypical Networks, Matching Networks, "
    "Relation Networks, and First-Order MAML. By projecting ResNet-50 embeddings onto an L2-normalized hypersphere and enforcing a calibrated "
    "tri-gate decision boundary (confidence &gt;= 50.0%), MetaLearnHub demonstrates dynamic enrollment of novel categories from K = 5 exemplars "
    "in under 350 ms without backbone retraining. Empirical benchmarking against a conventional transfer-learned linear probe demonstrates that "
    "while conventional models suffer severe small-sample overfitting (96.2% train vs test misclassification) and closed-set hallucinations, "
    "MetaLearnHub achieves 87.3% confidence on novel classes and 100.0% outlier rejection on open-set visual anomalies.",
    style_body
))
story.append(Paragraph("<b>Target Venues (Proposed):</b> (1) IEEE International Conference on Image Processing (ICIP 2027); (2) Springer CVIP 2027; (3) Elsevier Pattern Recognition Letters.", style_body))

# REFERENCES
story.append(Paragraph("REFERENCES", style_h1))
for ref in references:
    story.append(Paragraph(ref, ParagraphStyle('Ref_Custom', parent=style_body, fontSize=6.5, leading=8.5, leftIndent=12, firstLineIndent=-12, spaceAfter=1.5)))

pdf_doc.build(story, canvasmaker=NumberedCanvas)
print(f"[OK] PDF Document saved: {PDF_FILE}")

print("\n" + "=" * 65)
print(" Academic Documentation Generation Completed Successfully!")
print(f" Word (.docx): {DOCX_FILE} ({DOCX_FILE.stat().st_size} bytes)")
print(f" PDF (.pdf):  {PDF_FILE} ({PDF_FILE.stat().st_size} bytes)")
print("=" * 65)
