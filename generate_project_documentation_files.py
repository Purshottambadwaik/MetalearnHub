"""
MetaLearnHub - Comprehensive Project Architecture & Documentation Generator.
Produces both:
  1. Microsoft Word (.docx): docs/MetaLearnHub_Project_Architecture_and_System_Documentation.docx
  2. Adobe PDF (.pdf):        docs/MetaLearnHub_Project_Architecture_and_System_Documentation.pdf

Institution: Government College of Engineering, Nagpur
Department:  Department of Computer Science and Engineering
Project:     MetaLearnHub (Group No. 07)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from doc_generator_helpers import (
    set_cell_margins, set_cell_background, add_table_borders, NumberedCanvas
)

import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DOCX_OUTPUT = DOCS_DIR / "MetaLearnHub_Project_Architecture_and_System_Documentation.docx"
PDF_OUTPUT = DOCS_DIR / "MetaLearnHub_Project_Architecture_and_System_Documentation.pdf"

print("=" * 70)
print("MetaLearnHub Documentation Generator")
print(f"Target Word Document: {DOCX_OUTPUT}")
print(f"Target PDF Document:  {PDF_OUTPUT}")
print("=" * 70)


# ==============================================================================
# 1. GENERATE WORD (.DOCX) DOCUMENT
# ==============================================================================

def build_docx():
    print("\n[1/2] Generating Word Document (.docx)...")
    doc = docx.Document()

    # Configure page geometry (A4, 0.75 in margins)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.different_first_page_header_footer = True

        # Running header for pages 2+
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Government College of Engineering, Nagpur | Dept of CSE | MetaLearnHub")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(100, 116, 139)

        # Running footer for pages 2+
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun = fp.add_run("MetaLearnHub: System Architecture & Complete Technical Documentation | Group 07")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(100, 116, 139)

    # Color Palette
    C_PRIMARY = RGBColor(26, 54, 93)      # Deep Navy
    C_SECONDARY = RGBColor(43, 108, 176)  # Slate Blue
    C_TEXT = RGBColor(45, 55, 72)         # Charcoal Dark
    C_MUTED = RGBColor(113, 128, 150)     # Neutral Gray
    HEX_PRIMARY = "1A365D"
    HEX_HEADER_BG = "EBF8FF"
    HEX_LIGHT_BG = "F7FAFC"
    HEX_BORDER = "CBD5E0"

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = C_PRIMARY
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(14)
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.color.rgb = C_SECONDARY
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = C_PRIMARY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(11)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = C_SECONDARY
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(10.5)
        run.font.bold = True
        run.font.color.rgb = C_TEXT
        return p

    def add_p(text, bold_prefix=None, space_after=4):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10)
            r_pre.font.bold = True
            r_pre.font.color.rgb = C_TEXT
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = C_TEXT
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = 1.1
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10)
            r_pre.font.bold = True
            r_pre.font.color.rgb = C_TEXT
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = C_TEXT
        return p

    def add_callout(text, title="KEY ARCHITECTURAL INSIGHT:"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.77)
        set_cell_background(cell, "F0F4F8")
        set_cell_margins(cell, top=90, bottom=90, left=130, right=130)

        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:left w:val="single" w:sz="20" w:space="0" w:color="2B6CB0"/>'
            f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
            f'</w:tcBorders>'
        )
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        r_t = p.add_run(f"{title} ")
        r_t.font.name = "Calibri"
        r_t.font.size = Pt(9.5)
        r_t.font.bold = True
        r_t.font.color.rgb = C_SECONDARY
        r_b = p.add_run(text)
        r_b.font.name = "Calibri"
        r_b.font.size = Pt(9.5)
        r_b.font.color.rgb = C_TEXT

        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(4)

    def create_table(headers, data, col_widths=None):
        table = doc.add_table(rows=len(data) + 1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        add_table_borders(table, color="CBD5E0", sz="4", val="single")

        # Header Row
        hdr_row = table.rows[0]
        trPr = hdr_row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        for idx, text in enumerate(headers):
            cell = hdr_row.cells[idx]
            if col_widths and idx < len(col_widths):
                cell.width = col_widths[idx]
            set_cell_background(cell, HEX_PRIMARY)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

        # Data Rows
        for r_idx, row_data in enumerate(data):
            row = table.rows[r_idx + 1]
            bg_color = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
            for c_idx, val in enumerate(row_data):
                cell = row.cells[c_idx]
                if col_widths and c_idx < len(col_widths):
                    cell.width = col_widths[c_idx]
                set_cell_background(cell, bg_color)
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.1
                r = p.add_run(str(val))
                r.font.name = "Calibri"
                r.font.size = Pt(9.0)
                r.font.color.rgb = C_TEXT

        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(6)
        return table

    # -------------------------------------------------------------------------
    # DOCUMENT COVER & HEADER
    # -------------------------------------------------------------------------
    add_title("MetaLearnHub: System Architecture & Technical Documentation")
    add_subtitle("An Open-Set Few-Shot Meta-Learning Visual Classification System\nGovernment College of Engineering, Nagpur | Department of Computer Science & Engineering | Group 07")

    add_callout(
        "This document provides the definitive architectural, algorithmic, mathematical, and operational "
        "specification of MetaLearnHub. It encompasses the end-to-end full-stack design, episodic data pipelines, "
        "leakage-prevention protocol, four meta-learning algorithms, dynamic open-set class registry, and scientific evaluation.",
        title="SYSTEM OVERVIEW:"
    )

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY & PROBLEM STATEMENT
    # -------------------------------------------------------------------------
    add_h1("1. Executive Summary & Problem Formulation")
    add_p(
        "Modern deep convolutional networks excel in high-data regimes but fail when applied to data-scarce, open-world settings. "
        "Standard classifiers require thousands of labeled samples per visual class, are restricted to closed sets, and require "
        "complete retraining when new classes are introduced. In few-shot settings (1-5 samples), conventional deep models suffer "
        "from severe overfitting, slow gradient optimization, and blind false-positive overconfidence on novel categories."
    )
    add_p(
        "MetaLearnHub resolves these limitations by implementing an episodic meta-learning framework. Rather than learning class-specific "
        "boundaries, the network learns an inductive metric representation space across base categories. When a novel class is introduced, "
        "the model adapts instantaneously from 1 to 5 support exemplars without backpropagation or weight modification."
    )
    add_bullet("Transferable inductive feature learning across heterogeneous visual domains.", "Core Hypothesis: ")
    add_bullet("Guaranteed zero-leakage atomic partitioning for paired clean and augmented images.", "Data Hygiene: ")
    add_bullet("Instantaneous (<25 ms) class registration without neural network retraining.", "Rapid Adaptation: ")
    add_bullet("3-Gate Out-of-Distribution rejection distinguishing known classes from unseen anomalies.", "Open-Set Reliability: ")

    # -------------------------------------------------------------------------
    # SECTION 2: END-TO-END SYSTEM ARCHITECTURE
    # -------------------------------------------------------------------------
    add_h1("2. End-to-End System Architecture")
    add_p(
        "MetaLearnHub is architected as a modular, decoupled four-tier platform designed for scalability, reproducible research, and real-time inference:"
    )

    arch_headers = ["Layer", "Core Technologies", "Responsibilities"]
    arch_data = [
        ["1. Presentation Tier (SPA)", "React 18, TypeScript, Vite, Tailwind CSS, Lucide", "Responsive dashboard, dataset audit, live training monitor, evaluation lab, class registration, query interface."],
        ["2. Application Controller", "FastAPI, Python 3.10+, Uvicorn, BackgroundTasks", "REST API routes, asynchronous thread worker management, multipart upload handling, JSON serialization, state streaming."],
        ["3. Meta-Learning ML Core", "PyTorch 2.x, Torchvision, Scikit-learn, NumPy, PIL", "Feature backbones (ResNet-18/50), 4 meta-learning algorithms, episodic samplers, LOCO evaluator, open-set predictor."],
        ["4. Persistence & Artifacts", "File System, JSON manifests, PyTorch .pt tensors", "Class Registry (data/class_registry/), Model Checkpoints (checkpoints/), Experiment logs (experiments/), Raw datasets."]
    ]
    create_table(arch_headers, arch_data, [Inches(1.5), Inches(2.0), Inches(3.27)])

    # -------------------------------------------------------------------------
    # SECTION 3: DATA ENGINEERING & ZERO-LEAKAGE PROTOCOL
    # -------------------------------------------------------------------------
    add_h1("3. Data Engineering & Zero-Leakage Protocol")
    add_h2("3.1 The Base vs. Transformed Pairing Threat")
    add_p(
        "Computer vision benchmarks frequently contain base images (e.g., Ayrshirecattle100.jpg) and transformed or augmented "
        "counterparts (e.g., Ayrshirecattle100_c.jpg). If naive random train/test splitting is performed, base and transformed images "
        "land across the split boundary. The model memorizes background cues, resulting in catastrophic data leakage and falsely inflated accuracy."
    )
    add_h2("3.2 Atomic Group-Aware Partitioning")
    add_p(
        "MetaLearnHub implements an atomic group-aware partitioning algorithm in ml/datasets/inspector.py and ml/datasets/splitter.py:"
    )
    add_bullet("Filenames are parsed to strip suffix tags, extracting an invariant sample_group_id.", "Identifier Extraction: ")
    add_bullet("Splitting operates strictly at the group level. All members of a group land in the exact same split.", "Atomic Allocation: ")
    add_bullet("Cryptographic MD5 hashing identifies bit-for-bit duplicate files across different folders.", "Hash Verification: ")
    add_bullet("The LOCO protocol strictly withholds one complete breed (e.g., Red Dane) from all meta-training.", "LOCO Strict Isolation: ")

    # -------------------------------------------------------------------------
    # SECTION 4: FEATURE EXTRACTORS (DEEP BACKBONES)
    # -------------------------------------------------------------------------
    add_h1("4. Feature Extractors & Representation Space")
    add_p(
        "MetaLearnHub removes dense classification layers from deep residual networks, utilizing them as embedding functions f_theta(x):"
    )

    bb_headers = ["Backbone Architecture", "Embedding Dimension", "Pretrained Weights", "Primary Function"]
    bb_data = [
        ["ResNet-18 Backbone", "512-dim vector", "ImageNet-1K V1", "Fast episodic training sweeps, rapid prototyping, and controlled LOCO research."],
        ["ResNet-50 Backbone", "2048-dim vector", "ImageNet-1K Default", "High-capacity representation space for fine-grained multi-class open-set classification."]
    ]
    create_table(bb_headers, bb_data, [Inches(1.8), Inches(1.3), Inches(1.4), Inches(2.27)])

    add_p(
        "Unit Hyper-Sphere Normalization: All feature embeddings are L2-normalized: z_hat = f(x) / ||f(x)||_2. "
        "This stabilizes metric margins across distance computations and establishes exact equivalence between Euclidean distance and Cosine similarity: "
        "||u_hat - v_hat||^2 = 2 - 2 * cos(u_hat, v_hat)."
    )

    # -------------------------------------------------------------------------
    # SECTION 5: META-LEARNING ALGORITHMS & FORMULATIONS
    # -------------------------------------------------------------------------
    add_h1("5. Meta-Learning Algorithms & Formulations")
    add_p(
        "MetaLearnHub implements four distinct meta-learning paradigms unified under a common MetaLearningAlgorithm interface:"
    )

    add_h2("5.1 Prototypical Networks (ProtoNet)")
    add_p(
        "Reference: Snell et al. (NeurIPS 2017). Computes class prototypes c_k as the mean embedding of support samples S_k: "
        "c_k = (1 / |S_k|) * sum(f_theta(x_i)). Query classification is performed by computing Euclidean or Cosine distance d(f(x), c_k) "
        "and converting to probabilities via softmax over negative distances: P(y=k|x) = exp(-d(f(x), c_k)) / sum(exp(-d(f(x), c_j)))."
    )

    add_h2("5.2 Matching Networks (MatchingNet)")
    add_p(
        "Reference: Vinyals et al. (NeurIPS 2016). Rather than averaging into a single centroid, Matching Networks compares the query "
        "against every individual support exemplar using an attention mechanism: a(x, x_i) = softmax(tau * cos(f(x), g(x_i))), where tau=16.0. "
        "Class probabilities represent the sum of attention weights belonging to class c, normalized by class shot counts."
    )

    add_h2("5.3 Relation Networks (RelationNet)")
    add_p(
        "Reference: Sung et al. (CVPR 2018). Replaces handcrafted distance metrics with a 4-layer learnable non-linear Relation Module R_phi in [0, 1]. "
        "Consumes a concatenated relation tensor [f(x) * c_k, |f(x) - c_k|] capturing cross-correlation and absolute differences. "
        "Trained via Mean Squared Error (MSE) against binary ground-truth match indicators."
    )

    add_h2("5.4 Model-Agnostic Meta-Learning (MAML / FOMAML)")
    add_p(
        "Reference: Finn et al. (ICML 2017). An optimization-based meta-learning algorithm. Adapts parameters directly on support examples "
        "via an inner optimization loop: theta' = theta - alpha * grad(Loss_S). In MetaLearnHub, the classification head initializes "
        "from support centroids, enabling convergence in 5 gradient steps in under 20 milliseconds."
    )

    add_h2("5.5 Multi-Algorithm Ensemble Fusion")
    add_p(
        "Combines all four paradigms via a weighted fusion model: "
        "P_ensemble = 0.35 * P_Proto + 0.30 * P_Matching + 0.20 * P_Relation + 0.15 * P_MAML."
    )

    # -------------------------------------------------------------------------
    # SECTION 6: CONVENTIONAL BASELINE COMPARISON
    # -------------------------------------------------------------------------
    add_h1("6. Conventional Supervised Baseline Comparison")
    add_p(
        "To establish empirical necessity, MetaLearnHub benchmarks against a conventional Supervised Linear Softmax classifier "
        "(ml/baselines/standard_classifier.py) trained from scratch on support features:"
    )

    comp_headers = ["Metric / Dimension", "Conventional Supervised Classifier", "MetaLearnHub (Few-Shot Meta-Learning)"]
    comp_data = [
        ["Data Efficiency", "Requires 500+ images; severely overfits on K=5", "Generalizes accurately from K=1 to 5 images."],
        ["Adaptation Mechanism", "Full gradient backpropagation (30+ epochs)", "Instantaneous prototype centroid / fast inner-loop."],
        ["Adaptation Latency", "1,500 to 5,000 ms", "12 to 45 ms (100x faster)."],
        ["Overfitting on 5 Shots", "Train accuracy 100%, test accuracy <35%", "Maintains 75% to 85% accuracy on unseen test queries."],
        ["Open-Set Handling", "Closed-set: blindly predicts known class with 95%+ confidence", "3-Gate Rejection: accurately flags unknown inputs."],
        ["Novel Class Addition", "Requires full retraining of classification layers", "Zero retraining: dynamic prototype addition."]
    ]
    create_table(comp_headers, comp_data, [Inches(1.8), Inches(2.4), Inches(2.57)])

    # -------------------------------------------------------------------------
    # SECTION 7: EPISODIC TRAINING PIPELINE
    # -------------------------------------------------------------------------
    add_h1("7. Episodic Training Pipeline & Execution")
    add_p(
        "Meta-training is orchestrated by MetaTrainer (ml/training/trainer.py):"
    )
    add_bullet("N classes are sampled at random per episode (e.g., N=4-way).", "Episode Sampling: ")
    add_bullet("K support images and Q query images are sampled with guaranteed disjointness (S intersect Q = empty).", "Support-Query Disjointness: ")
    add_bullet("Loss is backpropagated through the Adam optimizer (lr=1e-3, weight_decay=1e-4) with StepLR decay.", "Optimization: ")
    add_bullet("Thread-safe TrainingState tracks metrics and streams live progress via /api/training/status.", "Live Progress Streaming: ")
    add_bullet("Checkpoints are automatically serialized to checkpoints/<exp_id>/best_model.pt on validation improvements.", "Model Checkpointing: ")

    # -------------------------------------------------------------------------
    # SECTION 8: SCIENTIFIC EVALUATION & METRICS
    # -------------------------------------------------------------------------
    add_h1("8. Scientific Evaluation & Statistical Rigor")
    add_p(
        "FewShotEvaluator (ml/evaluation/evaluator.py) provides statistical rigor across multi-shot evaluation budgets:"
    )
    add_bullet("1-shot, 3-shot, 5-shot, and 10-shot adaptation tested on strictly held-out classes.", "Evaluated Shot Budgets: ")
    add_bullet("50 to 100 independent trials per shot condition to eliminate sampling bias.", "Multi-Episode Sampling: ")
    add_bullet("Mean +/- Standard Deviation and 95% Confidence Intervals: CI = mean +/- 1.96 * (std / sqrt(N)).", "Statistical Rigor: ")
    add_bullet("Top-1 Accuracy, Macro Precision, Recall, F1-Score, and full N x N Confusion Matrices.", "Performance Metrics: ")

    # -------------------------------------------------------------------------
    # SECTION 9: OPEN-SET RECOGNITION & CLASS REGISTRY
    # -------------------------------------------------------------------------
    add_h1("9. Open-Set Recognition & Dynamic Class Registry")
    add_p(
        "FewShotPredictor (ml/inference/predictor.py) manages persistent storage (data/class_registry/) and open-set recognition:"
    )
    add_bullet("New categories are registered with 3-5 images in <50 ms with zero neural network retraining.", "Zero-Retraining Ingestion: ")
    add_bullet("Calculates pairwise cosine coherence (0-100%), flags visual outliers, and warns of cross-class collisions (>85% similarity).", "Batch Coherence Audit: ")
    add_bullet("Adding images updates class prototypes via continuous running centroid refinement.", "Continuous Refinement: ")
    add_bullet("Query distance must fall within dynamic radius r_k = max(mean + 1.25*std, 1.05*max_dist, floor).", "Gate 1 (Distance Gate): ")
    add_bullet("Closest candidate must separate from the second-closest competitor by a clear relative margin.", "Gate 2 (Margin Gate): ")
    add_bullet("Softmax confidence must exceed 50%. Failsafe returns MATCH NOT FOUND (Out-of-Distribution).", "Gate 3 (Confidence Gate): ")

    # -------------------------------------------------------------------------
    # SECTION 10: FRONTEND & BACKEND ARCHITECTURE
    # -------------------------------------------------------------------------
    add_h1("10. Full-Stack Implementation")
    add_h2("10.1 React 18 Single-Page Application")
    add_p(
        "The frontend includes six interactive views: Dashboard (system health & active models), Raw Data Audit (integrity & pair checks), "
        "Meta-Training Console (live loss/accuracy graphs), Evaluation Lab (multi-shot benchmarking with 95% CI error bars), "
        "Register Class (coherence gauge & outlier detector), and Find/Query Image (open-set verification & multi-algorithm comparison)."
    )

    add_h2("10.2 FastAPI REST Backend")
    api_headers = ["HTTP Method", "Endpoint Route", "Function & Responsibilities"]
    api_data = [
        ["GET", "/api/status", "System health, active compute hardware (CUDA/CPU), registered class count."],
        ["GET", "/api/dataset/inspect", "Full dataset audit, base/_c pair verification, corrupted file detection."],
        ["POST", "/api/training/start", "Spawns background episodic meta-training worker thread."],
        ["GET", "/api/training/status", "Real-time polling endpoint for loss curves and validation metrics."],
        ["POST", "/api/evaluation/run", "Executes multi-shot (1, 3, 5, 10-shot) LOCO evaluation on unseen classes."],
        ["POST", "/api/adaptation/create-class", "Registers a new category in data/class_registry/."],
        ["POST", "/api/adaptation/verify-batch", "Pre-flight audit: calculates coherence %, flags outliers, detects collisions."],
        ["POST", "/api/adaptation/{id}/add-support", "Ingests support images and generates normalized prototype vector."],
        ["POST", "/api/prediction", "Core inference route with 3-gate open-set rejection and multi-algorithm voting."]
    ]
    create_table(api_headers, api_data, [Inches(1.2), Inches(2.2), Inches(3.37)])

    # -------------------------------------------------------------------------
    # SECTION 11: EXECUTION & VERIFICATION
    # -------------------------------------------------------------------------
    add_h1("11. Execution Guide & System Verification")
    add_p("Setup and launch commands:")
    add_bullet("pip install -r requirements.txt (Python backend dependencies)", "Backend Setup: ")
    add_bullet("cd frontend && npm install (React UI dependencies)", "Frontend Setup: ")
    add_bullet("run.bat (Launches FastAPI at port 8000 and React at port 5173 concurrently)", "Full Launch: ")
    add_bullet("python run_few_shot_cli.py (Runs automated or interactive terminal few-shot tests)", "Headless CLI: ")
    add_bullet("python init_base_classes.py (Seeds initial 10 animal classes into persistent registry)", "Class Seeding: ")

    # Save Word doc
    doc.save(str(DOCX_OUTPUT))
    print(f"Word document saved successfully to: {DOCX_OUTPUT}")


# ==============================================================================
# 2. GENERATE ADOBE PDF (.PDF) DOCUMENT
# ==============================================================================

def build_pdf():
    print("\n[2/2] Generating PDF Document (.pdf)...")
    doc_pdf = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    c_primary = colors.HexColor("#1A365D")
    c_secondary = colors.HexColor("#2B6CB0")
    c_text = colors.HexColor("#2D3748")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=4,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=9,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'BodyP',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=c_text,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        'BulletP',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=c_text,
        leftIndent=14,
        spaceAfter=2,
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=c_text,
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_text,
    )

    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("MetaLearnHub: System Architecture & Technical Documentation", title_style))
    story.append(Paragraph("An Open-Set Few-Shot Meta-Learning Visual Classification System<br/>Government College of Engineering, Nagpur | Dept of CSE | Group 07", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=8))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "Conventional deep convolutional networks rely on massive datasets and closed-set fixed assumptions. "
        "When applied to few-shot novel classes, standard models suffer from severe overfitting, high retraining latency, "
        "and false-positive overconfidence. MetaLearnHub solves this by implementing an episodic meta-learning framework. "
        "Rather than learning fixed boundaries, the network constructs an inductive metric space that adapts instantaneously "
        "from 1-5 support images without neural network retraining.",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Transferable Feature Learning:</b> Generic feature extraction across diverse animal categories.", bullet_style))
    story.append(Paragraph("&bull; <b>Zero-Leakage Assurance:</b> Atomic group-aware splitting for paired clean and augmented files.", bullet_style))
    story.append(Paragraph("&bull; <b>Instant Adaptation:</b> Prototype computation in under 25 milliseconds without backpropagation.", bullet_style))
    story.append(Paragraph("&bull; <b>3-Gate Open-Set Rejection:</b> Distinguishes known categories from out-of-distribution inputs.", bullet_style))

    # Architecture Table
    story.append(Paragraph("2. System Architecture", h1_style))
    arch_data_pdf = [
        [Paragraph("<b>Layer</b>", table_hdr_style), Paragraph("<b>Technologies</b>", table_hdr_style), Paragraph("<b>Key Responsibilities</b>", table_hdr_style)],
        [Paragraph("Presentation Tier", table_cell_style), Paragraph("React 18, TypeScript, Vite, Tailwind CSS", table_cell_style), Paragraph("Dashboard, dataset audit, live training monitor, evaluation lab, class registration, query interface.", table_cell_style)],
        [Paragraph("Controller Tier", table_cell_style), Paragraph("FastAPI, Python 3.10+, BackgroundTasks", table_cell_style), Paragraph("REST endpoints, worker thread orchestration, file uploads, JSON streaming.", table_cell_style)],
        [Paragraph("ML Engine", table_cell_style), Paragraph("PyTorch 2.x, Torchvision, Scikit-learn", table_cell_style), Paragraph("ResNet-18/50 backbones, 4 meta-learning algorithms, episodic samplers, LOCO evaluator.", table_cell_style)],
        [Paragraph("Persistence", table_cell_style), Paragraph("File System, JSON, PyTorch .pt tensors", table_cell_style), Paragraph("Class registry (data/class_registry/), checkpoints/, experiments/ logs.", table_cell_style)],
    ]
    t_arch = Table(arch_data_pdf, colWidths=[90, 150, 275])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 6))

    # Data Engineering
    story.append(Paragraph("3. Data Engineering & Zero-Leakage Protocol", h1_style))
    story.append(Paragraph(
        "Paired clean and augmented images (e.g., Ayrshirecattle100.jpg and Ayrshirecattle100_c.jpg) risk crossing split boundaries. "
        "MetaLearnHub parses filenames into a unified sample_group_id and guarantees atomic group allocation. "
        "The Leave-One-Class-Out (LOCO) protocol completely excludes one breed from meta-training to benchmark genuine adaptation.",
        body_style
    ))

    # Feature Extractors & Algorithms
    story.append(Paragraph("4. Deep Backbones & Meta-Learning Algorithms", h1_style))
    story.append(Paragraph(
        "Visual backbones include ResNet-18 (512-dim embedding) for fast episodic sweeps and ResNet-50 (2048-dim) for high-capacity "
        "open-set classification, projected onto the unit hyper-sphere via L2-normalization. The platform unifies four algorithms:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Prototypical Networks (ProtoNet):</b> Computes mean class centroids c_k; classifies via negative distance softmax.", bullet_style))
    story.append(Paragraph("&bull; <b>Matching Networks (MatchingNet):</b> Evaluates query against every individual support exemplar via softmax attention.", bullet_style))
    story.append(Paragraph("&bull; <b>Relation Networks (RelationNet):</b> Replaces static distance with a 4-layer learnable non-linear Relation Module.", bullet_style))
    story.append(Paragraph("&bull; <b>MAML (Optimization-Based):</b> Prototype-guided linear head adapted via 5 SGD gradient steps in <20 ms.", bullet_style))
    story.append(Paragraph("&bull; <b>Ensemble Fusion:</b> Weighted combination: 35% ProtoNet + 30% MatchingNet + 20% RelationNet + 15% MAML.", bullet_style))

    # Baseline Comparison
    story.append(Paragraph("5. Conventional Baseline vs. Few-Shot Meta-Learning", h1_style))
    base_data_pdf = [
        [Paragraph("<b>Evaluation Dimension</b>", table_hdr_style), Paragraph("<b>Conventional Supervised Baseline</b>", table_hdr_style), Paragraph("<b>MetaLearnHub Meta-Learning</b>", table_hdr_style)],
        [Paragraph("Data Efficiency", table_cell_style), Paragraph("Requires 500+ images; overfits on K=5", table_cell_style), Paragraph("Adapts reliably from 1 to 5 images.", table_cell_style)],
        [Paragraph("Adaptation Latency", table_cell_style), Paragraph("1,500 - 5,000 ms (full gradient backprop)", table_cell_style), Paragraph("12 - 45 ms (instant prototype computation).", table_cell_style)],
        [Paragraph("Generalization on K=5", table_cell_style), Paragraph("100% train accuracy; <35% test accuracy", table_cell_style), Paragraph("75% - 85% accuracy on unseen test classes.", table_cell_style)],
        [Paragraph("Open-Set Rejection", table_cell_style), Paragraph("Closed-set; forces false positive on unknown", table_cell_style), Paragraph("3-Gate rejection flags out-of-distribution inputs.", table_cell_style)],
    ]
    t_base = Table(base_data_pdf, colWidths=[120, 195, 200])
    t_base.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(t_base)
    story.append(Spacer(1, 6))

    # Open-Set Rejection & Registry
    story.append(Paragraph("6. Dynamic Class Registry & 3-Gate Open-Set Engine", h1_style))
    story.append(Paragraph(
        "Classes and prototypes are persisted in data/class_registry/<class_id>/. Pre-flight batch audit calculates pairwise coherence %, "
        "flags visual outliers, and warns of collisions. Unknown query inputs must satisfy all three gates:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Gate 1 (Distance Gate):</b> Distance to prototype must be within class radius r_k = max(mean + 1.25*std, 1.05*max_dist, floor).", bullet_style))
    story.append(Paragraph("&bull; <b>Gate 2 (Margin Gate):</b> Best candidate must separate from second-best competitor by relative margin >= 0.005.", bullet_style))
    story.append(Paragraph("&bull; <b>Gate 3 (Confidence Gate):</b> Softmax confidence must be >= 50%; otherwise rejects as MATCH NOT FOUND.", bullet_style))

    # Execution & Verification
    story.append(Paragraph("7. Execution & Operational Commands", h1_style))
    story.append(Paragraph("&bull; <b>run.bat:</b> Starts FastAPI backend (port 8000) and React frontend (port 5173) simultaneously.", bullet_style))
    story.append(Paragraph("&bull; <b>python run_few_shot_cli.py:</b> Runs headless automated or interactive few-shot CLI testing.", bullet_style))
    story.append(Paragraph("&bull; <b>python init_base_classes.py:</b> Seeds 10 base animal categories into persistent class registry.", bullet_style))

    doc_pdf.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF document saved successfully to: {PDF_OUTPUT}")


if __name__ == "__main__":
    build_docx()
    build_pdf()
    print("\nDocumentation generation complete!")
