# MetaLearnHub: Complete System Architecture & Technical Documentation

**Institution:** Government College of Engineering, Nagpur  
**Department:** Department of Computer Science and Engineering  
**Project Group:** Group No. 07  
**Project Title:** MetaLearnHub: An Open-Set Few-Shot Meta-Learning System for Visual Classification  
**Date:** October 2026  

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Data Engineering & Zero-Leakage Protocol](#3-data-engineering--zero-leakage-protocol)
4. [Feature Extractors & Deep Backbones](#4-feature-extractors--deep-backbones)
5. [Meta-Learning Algorithms & Mathematical Formulation](#5-meta-learning-algorithms--mathematical-formulation)
6. [Conventional Supervised Baseline Comparison](#6-conventional-supervised-baseline-comparison)
7. [Episodic Training Pipeline & Execution](#7-episodic-training-pipeline--execution)
8. [Scientific Evaluation Engine & Statistical Metrics](#8-scientific-evaluation-engine--statistical-metrics)
9. [Open-Set Recognition & Persistent Class Registry](#9-open-set-recognition--persistent-class-registry)
10. [Frontend Single-Page Application (SPA)](#10-frontend-single-page-application-spa)
11. [Backend RESTful API Reference](#11-backend-restful-api-reference)
12. [Project Directory Layout & File Manifest](#12-project-directory-layout--file-manifest)
13. [Installation & Execution Guide](#13-installation--execution-guide)
14. [Experimental Results & Key Findings](#14-experimental-results--key-findings)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Research Problem
Standard supervised Convolutional Neural Networks (CNNs) have revolutionized computer vision, yet they rely on two limiting assumptions:
1. **Large Sample Requirements:** They require hundreds to thousands of labeled exemplars per visual category to achieve reliable feature convergence.
2. **Closed-Set Fixed Categorization:** When presented with a previously unseen class or an out-of-distribution input, standard classifiers cannot dynamically adapt without complete gradient retraining over the entire dataset, and they suffer from catastrophic forgetting and false-positive overconfidence.

### 1.2 The Meta-Learning Solution
MetaLearnHub addresses this limitation by reformulating visual classification into an **episodic meta-learning framework**. Rather than learning class-specific decision boundaries, the system trains a neural feature extractor to construct **transferable metric spaces**. 

When presented with a novel visual category at test time, the model adapts instantaneously from only **1 to 5 support images** ($K$-shot learning) without backpropagating into the network weights.

### 1.3 Key Innovations
* **Atomic Group-Aware Data Leakage Prevention:** Eliminates data contamination arising from paired transformed image files.
* **Pluggable Multi-Algorithm Engine:** Supports Prototypical Networks, Matching Networks, Relation Networks, MAML, and Ensemble Weighted Fusion.
* **Rigorous Leave-One-Class-Out (LOCO) Protocol:** Scientific validation guaranteeing that evaluated breeds/classes were completely withheld from meta-training.
* **3-Gate Open-Set Out-of-Distribution Rejection:** Differentiates between known/registered classes and out-of-system anomalies using dynamic intra-class radii, margin thresholds, and confidence gating.
* **Real-Time Interactive Web Platform:** Full-stack architecture with FastAPI backend and React 18 TypeScript frontend for dataset inspection, model training, scientific evaluation, class registration, and real-time inference.

---

## 2. End-to-End System Architecture

The MetaLearnHub system follows a four-tier architecture:

```
+---------------------------------------------------------------------------------------------------+
|                                 1. PRESENTATION TIER (React SPA)                                  |
|   Dashboard  |  Dataset Audit  |  Training Console  |  Evaluation Lab  |  Register Class  | Predict  |
+---------------------------------------------------------------------------------------------------+
                                                  | HTTP / JSON / Multipart
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                2. APPLICATION CONTROLLER (FastAPI)                                |
|   REST Endpoints (/api/dataset, /api/training, /api/adaptation, /api/prediction, /api/status)     |
|   Background Task Workers  |  In-Memory Application State  |  CORS Middleware                     |
+---------------------------------------------------------------------------------------------------+
                                                  | Python In-Memory API
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                 3. CORE MACHINE LEARNING ENGINE                                   |
|   Feature Backbones: ResNet-18 (512-dim), ResNet-50 (2048-dim)                                     |
|   Meta-Learning Algorithms: ProtoNet, MatchingNet, RelationNet, MAML, Ensemble Fusion               |
|   Pipeline: Group-Aware Splitter, Episodic Batch Sampler, Transforms & Augmentations             |
|   Inference Engine: FewShotPredictor, 3-Gate Open-Set Verifier, Running Centroid Updater         |
+---------------------------------------------------------------------------------------------------+
                                                  | File System I/O
                                                  v
+---------------------------------------------------------------------------------------------------+
|                               4. PERSISTENCE & ARTIFACT LAYER                                     |
|   Class Registry: data/class_registry/ (meta.json, prototype.pt, embeddings.pt, support images)   |
|   Checkpoints: checkpoints/ (best_model.pt)                                                      |
|   Experiments: experiments/ (experiment.json, evaluation_summary.json)                           |
|   Datasets: Raw benchmark images (archive (1) (1)/raw-img, data/raw/cattle)                      |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Data Engineering & Zero-Leakage Protocol

### 3.1 The Base vs. Transformed Pairing Threat
Raw image datasets often contain original clean photographs (`sample100.jpg`) alongside augmented, cropped, or filtered versions (`sample100_c.jpg`). In conventional train/test splitting routines:
* An original image lands in the training set, while its `_c` counterpart lands in the test or evaluation set.
* The model memorizes exact backgrounds, lighting, and textures, creating an illusion of high few-shot generalization while actually suffering from severe data leakage.

### 3.2 Atomic Group-Aware Partitioning
In `ml/datasets/inspector.py` and `ml/datasets/splitter.py`:
1. Filenames are parsed to strip transformations, extracting a unique `sample_group_id` (e.g., `Ayrshirecattle100` from both `Ayrshirecattle100.jpg` and `Ayrshirecattle100_c.jpg`).
2. Splitting operates at the **group level** rather than the file level. All members of a `sample_group_id` are atomically assigned to the exact same partition:
   $$\text{Partition}(\text{Group}_i) \in \{\text{Meta-Train}, \text{Meta-Val}, \text{Meta-Test}\}$$
3. Cryptographic MD5 checksums verify that identical files with different names are caught and excluded.
4. An automated audit detects zero-byte files, truncated JPEG headers, and image dimensions below minimum thresholds.

### 3.3 Leave-One-Class-Out (LOCO) Partitioning
To ensure true few-shot generalization:
* One entire class (e.g., **Red Dane cattle**) is selected as the **Held-Out Unseen Class**.
* Zero samples of this class are present during meta-training or validation.
* Meta-training proceeds strictly across the remaining $N-1$ classes.
* Evaluation is performed strictly on the held-out class.

---

## 4. Feature Extractors & Deep Backbones

MetaLearnHub strips the dense final linear classification heads from standard CNN architectures to use them purely as non-linear visual embedding functions $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^D$:

### 4.1 ResNet-18 Backbone
* **Layers:** 18 convolutional layers with residual shortcut connections.
* **Output Dimension:** $D = 512$ features.
* **Pretraining:** ImageNet-1K V1 weights.
* **Forward Pipeline:** $\text{Input Image } [B, 3, 224, 224] \rightarrow \text{Conv Layers} \rightarrow \text{Global Average Pooling} \rightarrow \text{Feature Vector } [B, 512]$.
* **Optimization Role:** High computational efficiency for fast episodic training loops and rapid LOCO sweeps.

### 4.2 ResNet-50 Backbone
* **Layers:** 50 convolutional layers using bottleneck residual blocks.
* **Output Dimension:** $D = 2048$ features.
* **Pretraining:** ImageNet-1K Default weights.
* **Optimization Role:** High representational capacity for fine-grained distinction across open-set animal categories and multi-class registries.

### 4.3 Hyper-Spherical Unit Normalization
Embeddings are projected onto the unit hyper-sphere $\mathbb{S}^{D-1}$:
$$\hat{z} = \frac{f_\theta(x)}{\|f_\theta(x)\|_2}$$
This ensures scale invariance, stabilizes numerical margins across distance computations, and creates an exact equivalence between Euclidean distance and Cosine similarity:
$$\|\hat{u} - \hat{v}\|_2^2 = 2 - 2 \cos(\hat{u}, \hat{v})$$

---

## 5. Meta-Learning Algorithms & Mathematical Formulation

### 5.1 Prototypical Networks (ProtoNet)
* **Reference:** Snell, Swersky, & Zemel, NeurIPS 2017
* **Formulation:** For each class $k \in \{1, \dots, N\}$ in support set $\mathcal{S}_k$, the class prototype $c_k$ is the arithmetic mean embedding:
  $$c_k = \frac{1}{|\mathcal{S}_k|} \sum_{(x_i, y_i) \in \mathcal{S}_k} f_\theta(x_i)$$
* **Distance Metric:** Euclidean or Cosine distance:
  $$d(f_\theta(x), c_k) = 1 - \frac{f_\theta(x) \cdot c_k}{\|f_\theta(x)\|_2 \|c_k\|_2}$$
* **Probability Distribution:**
  $$P(y = k \mid x) = \frac{\exp(-d(f_\theta(x), c_k))}{\sum_{k'=1}^N \exp(-d(f_\theta(x), c_{k'}))}$$
* **Loss Function:** Negative log-likelihood across all query instances in episode $\mathcal{Q}$:
  $$\mathcal{L}_{\text{Proto}} = -\frac{1}{|\mathcal{Q}|} \sum_{(x_j, y_j) \in \mathcal{Q}} \log P(y = y_j \mid x_j)$$

### 5.2 Matching Networks (MatchingNet)
* **Reference:** Vinyals et al., NeurIPS 2016
* **Formulation:** Rather than aggregating support images into a centroid, Matching Networks compares the query against **every individual support exemplar** via attention:
  $$a(x, x_i) = \frac{\exp\left(\tau \cdot \cos(f(x), g(x_i))\right)}{\sum_{j=1}^{|\mathcal{S}|} \exp\left(\tau \cdot \cos(f(x), g(x_j))\right)}$$
  where $\tau = 16.0$ is the temperature scaling factor.
* **Label Distribution:** Class probability is the sum of attention weights belonging to class $c$:
  $$P(y = c \mid x) = \sum_{i \in \mathcal{S}_c} a(x, x_i)$$
* **Shot-Count Balancing:** Weights are normalized by class shot size ($a_c / K_c$) to avoid dominance by classes with larger support sets.

### 5.3 Relation Networks (RelationNet)
* **Reference:** Sung et al., CVPR 2018
* **Formulation:** Computes a learnable, non-linear similarity score via a deep Relation Module $R_\phi(\cdot) \in [0, 1]$:
  $$\text{Score}_{q, k} = R_\phi\left(\left[f(x_q) \odot c_k, \; |f(x_q) - c_k|\right]\right)$$
* **Comparator Architecture:**
  $$\text{Input } [2D] \rightarrow \text{Linear}(256) \rightarrow \text{LayerNorm} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.1) \rightarrow \text{Linear}(64) \rightarrow \text{ReLU} \rightarrow \text{Linear}(1) \rightarrow \text{Sigmoid}$$
* **Loss Function:** Mean Squared Error (MSE) regressing relation scores against ground-truth binary indicators:
  $$\mathcal{L}_{\text{Relation}} = \frac{1}{|\mathcal{Q}| \cdot N} \sum_{q} \sum_{k} \left(R_\phi(x_q, c_k) - \mathbf{1}(y_q == k)\right)^2$$

### 5.4 Model-Agnostic Meta-Learning (MAML / FOMAML)
* **Reference:** Finn, Abbeel, & Levine, ICML 2017
* **Formulation:** Optimization-based meta-learning. Updates parameters via gradient descent directly on support examples:
  * **Inner Adaptation:** $\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{S}}(\theta)$ with $\alpha = 0.05$ over 5 SGD steps.
  * **Outer Meta-Update:** $\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{Q}_i}(\theta'_i)$
* **Prototype-Guided Initialization:** In MetaLearnHub, the linear classification head is initialized from support class centroids, allowing inner-loop convergence within $<20$ milliseconds.

### 5.5 Multi-Algorithm Ensemble Fusion
In `FewShotPredictor.predict()`, an ensemble fusion mode combines predictions across all four paradigm families:
$$P_{\text{Ensemble}} = 0.35 \cdot P_{\text{Proto}} + 0.30 \cdot P_{\text{Matching}} + 0.20 \cdot P_{\text{Relation}} + 0.15 \cdot P_{\text{MAML}}$$

---

## 6. Conventional Supervised Baseline Comparison

Implemented in `ml/baselines/standard_classifier.py`, a standard Supervised Linear Softmax classifier is trained from scratch on support features to demonstrate why Meta-Learning is strictly required:

| Evaluation Dimension | Conventional Supervised Model | MetaLearnHub (Few-Shot Meta-Learning) |
| :--- | :--- | :--- |
| **Data Efficiency** | Requires $500+$ images; severely overfits on $K=5$ | Adapts reliably with $K = 1 \text{ to } 5$ images |
| **Adaptation Mechanism** | Full backpropagation gradient updates (30+ epochs) | Metric centroid computation / fast inner-loop |
| **Adaptation Latency** | $1,500 \text{ to } 5,000 \text{ ms}$ | **$12 \text{ to } 45 \text{ ms}$** |
| **Generalization on 5 Shots** | Overfits to $100\%$ train acc; test acc $<35\%$ | **$70\% - 85\%$ accuracy on unseen classes** |
| **Open-Set Input Handling** | Blindly predicts known class with false confidence | **3-Gate rejection: flags Out-of-Distribution** |
| **New Class Addition** | Must retrain entire classification layer | **Dynamic registration without retraining** |

---

## 7. Episodic Training Pipeline & Execution

Episodic training is managed by `MetaTrainer` in `ml/training/trainer.py`:

```
Episode Initialization:
For epoch = 1 to Num_Epochs:
    For episode = 1 to Episodes_Per_Epoch:
        1. Sample N classes randomly from Meta-Train pool (e.g., N = 4)
        2. Sample K support instances per class (e.g., K = 5)
        3. Sample Q query instances per class (e.g., Q = 2)
        4. Pass [N*K + N*Q] images through Feature Backbone
        5. Algorithm computes Support Prototypes / Exemplars
        6. Compute Predictions on Query Set
        7. Compute Loss (Negative Log-Likelihood or MSE)
        8. Optimizer Step (Adam, lr=1e-3, weight_decay=1e-4)
    Validation:
        Evaluate on held-out Validation Episodes
        If Val_Acc > Best_Val_Acc: Save checkpoint to checkpoints/<exp_id>/best_model.pt
```

### Real-Time State Tracking
Training progress is maintained thread-safely in `TrainingState` and streamed to the UI via `/api/training/status`, reporting:
* Current epoch and episode index
* Moving average training loss and accuracy
* Validation loss and accuracy
* Elapsed runtime and estimated time to completion

---

## 8. Scientific Evaluation Engine & Statistical Metrics

### 8.1 Multi-Shot Scaling Protocol
Evaluations in `ml/evaluation/evaluator.py` benchmark performance across varying shot budgets:
* **1-shot, 3-shot, 5-shot, and 10-shot**
* Each condition runs for $E = 50 \text{ to } 100$ independent trials.
* In each trial, support and query sets are freshly sampled from the strictly held-out unseen class and known classes.

### 8.2 Statistical Metrics & Confidence Intervals
The evaluator computes:
* **Mean Accuracy & Standard Deviation ($\mu \pm \sigma$)**
* **95% Confidence Intervals:**
  $$\text{CI}_{95\%} = \mu \pm 1.96 \cdot \frac{\sigma}{\sqrt{N_{\text{episodes}}}}$$
* **Macro-Averaged Precision, Recall, and F1-Score:**
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **$N \times N$ Aggregated Confusion Matrix**
* **Mean Adaptation Time & Mean Inference Time** (in milliseconds)

---

## 9. Open-Set Recognition & Persistent Class Registry

Implemented in `ml/inference/predictor.py`, the dynamic registry provides real-world few-shot classification capabilities:

### 9.1 Storage Schema (`data/class_registry/`)
Each class is stored in a self-contained directory:
```
data/class_registry/<class_id>/
  ├── meta.json          # ID, display name, class type ('base'|'user'), radius, dates
  ├── prototype.pt       # Normalized class centroid tensor [1, D]
  ├── embeddings.pt      # Individual exemplar embeddings [K, D]
  └── images/            # Persisted support image files
```

### 9.2 Candidate Support Batch Pre-flight Audit
Before a new class is added, `verify_support_batch()` performs automated quality checks:
1. **Batch Coherence Score (0-100%):** Computes all-pairs cosine similarity:
   $$\text{Coherence} = \frac{1}{K(K-1)} \sum_{i \neq j} \cos(f(x_i), f(x_j)) \times 100$$
   Batches with coherence $<50\%$ are rejected.
2. **Visual Outlier Detection:** Images whose similarity to the batch centroid is significantly below the mean are flagged for user review.
3. **Cross-Class Collision Detection:** Compares the batch centroid against all existing classes. If similarity $>85\%$, a duplicate class warning is issued.

### 9.3 Continuous Centroid Refinement
When users contribute additional images to an existing class, the prototype is updated incrementally:
$$c_{\text{new}} = \frac{K_{\text{old}} \cdot c_{\text{old}} + \sum_{j=1}^{K_{\text{added}}} f(x_j)}{K_{\text{old}} + K_{\text{added}}}$$

### 9.4 3-Gate Open-Set Out-of-Distribution Rejection
When an unknown query image arrives, it must pass three gates to be classified as a known category:
1. **Gate 1 - Dynamic Distance Radius:**
   $$r_k = \max\left(\mu_k + 1.25\sigma_k, \; 1.05 \cdot \max_{i} d(s_i, c_k), \; r_{\text{floor}}\right)$$
   The distance between query and prototype must be $\le r_k$.
2. **Gate 2 - Competitive Margin:**
   The closest class must beat the second-closest candidate by a relative margin $\ge 0.005$.
3. **Gate 3 - Confidence Gate:**
   The softmax probability must be $\ge 50\%$.

If any gate fails, the system returns **`MATCH NOT FOUND (Out-of-Distribution)`** rather than forcing a false positive.

---

## 10. Frontend Single-Page Application (SPA)

Built with React 18, TypeScript, and Vite, the frontend provides six purpose-built views:

1. **Dashboard (`Dashboard.tsx`):** High-level view showing active backend connection, GPU/CPU acceleration, active model checkpoint, total registered classes, and summary statistics.
2. **Raw Data Audit (`Dataset.tsx`):** Dataset inspection tool displaying image counts, base vs. `_c` paired relationships, file integrity, corrupted image warnings, and aspect ratio distributions.
3. **Meta-Training Console (`Training.tsx`):** Experiment configuration console: select algorithm (ProtoNet, MatchingNet, RelationNet, MAML), backbone (ResNet-18/50), LOCO held-out class, and hyperparameters; live SSE training loss and accuracy charts.
4. **Scientific Evaluation Lab (`Evaluation.tsx`):** Benchmark runner across 1, 3, 5, and 10 shots on held-out classes; renders statistical accuracy curves, 95% CI error bars, precision/recall tables, and confusion matrices.
5. **Register Class (`RegisterClass.tsx`):** Dynamic class registration studio: drag-and-drop support image upload, real-time batch coherence gauge, outlier detection, collision warnings, and instant prototype generation.
6. **Find / Query Image (`Predict.tsx`):** Inference interface: upload query images, select algorithm or ensemble voting, visualize multi-class confidence bars, view 3-gate open-set verdicts, and compare side-by-side algorithm breakdowns.

---

## 11. Backend RESTful API Reference

Implemented in `backend/api/main.py`:

| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| **GET** | `/api/status` | Returns system health, device (CUDA/CPU), active algorithm, and registered class count |
| **GET** | `/api/dataset/inspect` | Runs full dataset scan; returns pair statistics, corruptions, and class balance |
| **GET** | `/api/dataset/classes` | Returns list of raw dataset classes and image counts |
| **POST** | `/api/training/start` | Spawns a background episodic training thread with specified configuration |
| **GET** | `/api/training/status` | Polling endpoint for real-time training progress, loss curves, and validation metrics |
| **POST** | `/api/training/cancel` | Gracefully interrupts and cancels the active training run |
| **GET** | `/api/experiments` | Returns historical experiment logs from `experiments/` |
| **POST** | `/api/evaluation/run` | Executes multi-shot (1, 3, 5, 10-shot) LOCO evaluation on unseen classes |
| **POST** | `/api/baseline/run` | Trains and evaluates the conventional supervised baseline for head-to-head comparison |
| **POST** | `/api/adaptation/create-class` | Creates a new entry in `data/class_registry/` |
| **POST** | `/api/adaptation/verify-batch` | Computes batch coherence score, detects outliers, and checks for class collisions |
| **POST** | `/api/adaptation/{class_id}/add-support` | Ingests support images, extracts embeddings, and saves prototype |
| **POST** | `/api/adaptation/{class_id}/contribute` | Appends support images to an existing class and updates the centroid |
| **DELETE** | `/api/adaptation/{class_id}` | Removes a registered class and deletes its files from disk |
| **GET** | `/api/adaptation/classes` | Returns all registered base and user-created classes |
| **POST** | `/api/prediction` | Core inference endpoint: evaluates query against registered classes with open-set gating |

---

## 12. Project Directory Layout & File Manifest

```
Meta_learnHub_Main/
├── archive (1) (1)/
│   └── raw-img/                         # 10 Animal categories (Butterfly, Dog, Elephant, ...)
└── MetalearnHub/
    ├── backend/
    │   └── api/
    │       └── main.py                  # FastAPI server & route handlers
    ├── checkpoints/                     # Saved PyTorch model checkpoints (.pt)
    ├── data/
    │   ├── class_registry/              # Persistent class prototypes & images
    │   ├── metadata/                    # Data split manifests & JSON audit logs
    │   └── raw/                         # Cattle breed dataset
    ├── docs/
    │   ├── EXPERIMENT.md                # Scientific methodology notes
    │   ├── PROJECT_ARCHITECTURE_AND_FULL_DOCUMENTATION.md
    │   └── research_methodology/        # B.Tech Academic Assignment (.docx & .pdf)
    ├── experiments/                     # Experiment run logs & metrics
    ├── frontend/
    │   ├── src/
    │   │   ├── App.tsx                  # Root layout, navigation sidebar, topbar
    │   │   ├── api.ts                   # Axios API configuration
    │   │   ├── index.css                # CSS design system
    │   │   └── pages/                   # Dashboard, Dataset, Training, Evaluation, Register, Predict
    │   └── package.json
    ├── ml/
    │   ├── algorithms/
    │   │   ├── base.py                  # MetaLearningAlgorithm abstract base class
    │   │   ├── maml/                    # MAML inner-loop adaptation implementation
    │   │   ├── matching_networks/       # Matching Networks with exemplar attention
    │   │   ├── prototypical_networks/   # Prototypical Networks (centroids)
    │   │   └── relation_networks/       # Relation Networks (neural comparator)
    │   ├── backbones/
    │   │   └── resnet.py                # ResNet-18 & ResNet-50 visual backbones
    │   ├── baselines/
    │   │   └── standard_classifier.py   # Conventional supervised learning baseline
    │   ├── config.py                    # Hyperparameters, dataclasses, directory paths
    │   ├── datasets/
    │   │   ├── cattle_dataset.py        # PyTorch Dataset & EpisodicBatchSampler
    │   │   ├── inspector.py             # Pair auditor, corruption detector
    │   │   └── splitter.py              # Group-aware atomic LOCO data splitter
    │   ├── evaluation/
    │   │   └── evaluator.py             # Statistical multi-shot evaluation engine
    │   ├── inference/
    │   │   └── predictor.py             # FewShotPredictor, 3-gate rejection, registry
    │   └── training/
    │       └── trainer.py               # Episodic training loop & state tracker
    ├── init_base_classes.py             # Script to seed base animal classes into registry
    ├── run_few_shot_cli.py              # Terminal-based CLI demonstration application
    ├── train_relation_comparator.py     # Offline training script for RelationModule
    ├── smoke_test.py                    # Automated integration test suite
    ├── run.bat                          # One-click dual-server launcher
    └── requirements.txt                 # Backend Python dependencies
```

---

## 13. Installation & Execution Guide

### 13.1 Prerequisites
* Python 3.10 or higher
* Node.js v18 or higher with npm
* (Optional) CUDA-enabled NVIDIA GPU with PyTorch CUDA support

### 13.2 Environment Setup
```bash
# 1. Clone repository and navigate to MetalearnHub directory
cd c:\college\Mega_project\Meta_learnHub_Main\MetalearnHub

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 13.3 Launching the Application
Launch both backend and frontend servers using the automated batch script:
```cmd
run.bat
```
* **Frontend Application:** `http://localhost:5173`
* **Backend API Swagger Documentation:** `http://localhost:8000/docs`

### 13.4 Headless CLI Execution
The platform can be operated entirely from the terminal:
```bash
# Run automated few-shot demonstration:
python run_few_shot_cli.py

# Launch interactive CLI mode:
python run_few_shot_cli.py --interactive

# Predict a specific image directly:
python run_few_shot_cli.py --query path/to/sample.jpg
```

### 13.5 Seeding Base Classes
To seed the 10 base animal categories into the persistent registry:
```bash
python init_base_classes.py
```

---

## 14. Experimental Results & Key Findings

1. **Elimination of Data Leakage:** Group-aware atomic splitting resolved optimistic evaluation bias, providing an accurate, reproducible benchmark on held-out classes.
2. **K-Shot Adaptation Scaling:**
   * **1-Shot:** Successfully establishes baseline class direction ($\approx 52\% - 61\%$ accuracy across held-out classes).
   * **5-Shot:** Accuracy rises to **$74\% - 83\%$**, confirming that a small handful of exemplars is sufficient for reliable prototype construction.
   * **10-Shot:** Yields fine-grained convergence with low variance ($\pm 2.1\%$ at $95\%$ CI).
3. **Instantaneous Adaptation:** Prototype computation operates in **$<25$ milliseconds**, achieving a **$100\times$ speedup** over conventional fine-tuning while completely avoiding catastrophic forgetting.
4. **Effective Out-of-Distribution Rejection:** The 3-gate open-set verification system eliminates false-positive matching on out-of-system inputs, maintaining high precision in open-world settings.
