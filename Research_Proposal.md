# Research Proposal

## Addressing the Absence Detection Paradox: A Decoupled Sentry-Judge Architecture Combining YOLOv11m and SAM 3 for Real-Time PPE Compliance Monitoring

**Student:** [Your Name]
**Date:** March 2026

---

## 1. Introduction

### 1.1 Background

Construction site safety remains a critical global challenge, with over 1,000 annual fatalities directly linked to Personal Protective Equipment (PPE) non-compliance (OSHA, 2024). Despite significant advances in computer vision and deep learning, existing AI-powered safety monitoring systems face a fundamental limitation we term the **Absence Detection Paradox**: object detection models excel at identifying *present* PPE (helmets, vests) but fundamentally struggle at detecting *absent* equipment — workers *without* required protective gear.

Current approaches rely heavily on single-stage detectors such as YOLO variants for PPE monitoring. While these systems achieve impressive speeds, they face an inherent asymmetry: detecting a helmet (a clear visual target) is fundamentally different from detecting the *absence* of a helmet (reasoning about what *should* be present). This gap leads to unacceptable false positive and false negative rates in real-world deployment.

### 1.2 Problem Statement

The core technical challenge lies in three areas:

1. **Presence vs. Absence Asymmetry:** Object detectors are trained to find objects, not to reason about missing objects. Detecting `no_helmet` requires contextual understanding beyond standard bounding box regression.

2. **Real-Time Constraint:** Construction monitoring demands 25+ FPS processing, yet semantic verification models (e.g., SAM 3) operate at ~2-3 FPS. Using both simultaneously in a coupled pipeline creates an unacceptable bottleneck.

3. **Alert Fatigue:** Continuous monitoring generates thousands of detections per hour. Without intelligent deduplication, the same worker triggers hundreds of repeated alerts, overwhelming safety managers and undermining system trust.

### 1.3 Research Questions

1. How significant is the performance gap between presence and absence detection in PPE compliance, and can it be quantified?
2. Can a decoupled asynchronous architecture separate fast detection from semantic verification while maintaining real-time throughput?
3. What is the optimal cooldown and tracking strategy to eliminate alert fatigue while preserving violation coverage?
4. Can vision-language foundation models (SAM 3) serve as an effective "Judge" to verify or override the initial detector's uncertain cases?

---

## 2. Literature Review

### 2.1 PPE Detection with Deep Learning

Recent works employ YOLO-family detectors for PPE monitoring. Wang et al. (2023) demonstrated YOLOv8-based helmet detection achieving 92% mAP, while Zhang et al. (2024) extended detection to multiple PPE classes. However, these works focus on presence detection and report aggregated metrics that mask the absence detection gap. No existing work specifically addresses or quantifies the asymmetry between detecting present versus absent safety equipment.

### 2.2 The Absence Detection Challenge

Absence detection has been explored in anomaly detection (Cao et al., 2023), where models learn expected configurations and flag deviations. However, these methods require scene-specific training and lack generalizability. The specific challenge of detecting *missing* safety equipment in *uncontrolled* construction environments — with varying lighting, occlusion, and camera angles — remains underexplored in the literature.

### 2.3 Vision-Language Foundation Models

SAM 3 (Meta AI, 2025) introduces **Promptable Concept Segmentation (PCS)**, enabling text-prompted semantic segmentation. Unlike traditional detectors that require task-specific training, SAM 3 can reason about visual concepts through natural language prompts (e.g., "helmet", "safety vest"). This capability offers a potential solution for semantic verification without retraining, but its integration into real-time pipelines has not been studied.

### 2.4 Multi-Object Tracking in Safety Systems

ByteTrack (Zhang et al., 2022) and BoT-SORT (Aharon et al., 2022) enable persistent identity assignment across video frames. While widely used in surveillance, their application to PPE monitoring — specifically for violation deduplication and session-based tracking — has not been systematically explored.

### 2.5 Hybrid and Cascade Architectures

Cascade architectures (Cai and Vasconcelos, 2019) balance speed and accuracy through multi-stage refinement. Recent works combine specialized models for different subtasks (Li et al., 2024). However, existing cascades are *synchronous* — the second stage blocks the first. This research proposes an *asynchronous* decoupled architecture where both stages operate independently.

---

## 3. Proposed Methodology

### 3.1 Decoupled Sentry-Judge Framework

We propose a **decoupled asynchronous Sentry-Judge architecture** consisting of two independent components connected by a message queue:

**The Sentry (YOLOv11m + ByteTrack):**
- Processes every video frame at real-time speed (target: 25-30 FPS)
- Detects persons and PPE items (Helmet, Vest, no_helmet)
- Assigns persistent Person IDs via ByteTrack multi-object tracking
- Performs 5-path triage to classify each detection
- Applies per-person cooldown to prevent duplicate alerts
- Crops Regions of Interest (ROIs) and pushes violation candidates to the queue
- **Never blocks on the Judge** — maintains constant throughput

**The Judge (SAM 3):**
- Runs as a separate asynchronous process/thread
- Consumes violation candidates from the queue
- Performs semantic verification on cropped ROI images using text prompts
- **Confirms or rejects** the Sentry's initial assessment
- Writes only confirmed violations to the database

### 3.2 Five-Path Decision Logic

The Sentry implements an intelligent triage framework:

| Path | YOLO Output | Action |
|------|-------------|--------|
| **Path 0: Fast Safe** | Helmet + Vest detected | No queue, no Judge |
| **Path 1: Fast Violation** | `no_helmet` class detected | Queue HEAD ROI for Judge verification |
| **Path 2: Rescue Head** | Vest detected, no helmet info | Queue HEAD ROI |
| **Path 3: Rescue Body** | Helmet detected, no vest info | Queue TORSO ROI |
| **Path 4: Critical** | No PPE detected at all | Queue both ROIs |

Path 0 bypasses the Judge entirely, achieving target bypass rates of 70-80%. This is the key efficiency gain — the expensive Judge only processes uncertain cases.

### 3.3 Cooldown-Based Deduplication

Instead of queuing every frame where a worker violates:
- The Sentry maintains an in-memory dictionary: `{Person_ID: {violation_type: last_timestamp}}`
- A configurable cooldown period (default: 5 minutes) prevents re-queuing the same worker for the same violation
- **Result:** Instead of 300 SAM calls for one worker over 10 minutes, only 2 calls are made (one at detection, one after cooldown expires)

### 3.4 Geometric Prompt Engineering

Rather than processing full images, the Judge receives **cropped ROI regions** based on geometric constraints:
- **Head ROI:** Top 40% of person bounding box (for helmet verification)
- **Torso ROI:** 20-100% of person bounding box (for vest verification)

This approach reduces SAM 3's computational cost by ~60-70% while focusing its semantic reasoning on the most relevant image regions.

### 3.5 Agentic Compliance System

An autonomous end-of-day agent will:
1. **Query** the database of Judge-verified violations
2. **Aggregate** violations by Person ID to identify serial offenders
3. **Generate** a professional safety summary using an LLM API with an OSHA Safety Officer prompt
4. **Produce** a PDF report containing the LLM narrative, violation table, and embedded ROI evidence images
5. **Distribute** reports to site managers via email

### 3.6 Training Strategy

The Sentry (YOLOv11m) will be trained on the **Construction-PPE dataset** with:
- **4-class detection:** Helmet, Vest, Person, no_helmet
- **Resolution:** 1280×1280 for small PPE detection at distance
- **Optimizer ablation:** SGD vs. AdamW to evaluate impact on minority class (absence) detection

---

## 4. System Architecture

The complete system will be implemented as a full-stack application:

- **Detection Pipeline:** Python-based, using `ultralytics` for YOLO and SAM 3
- **Message Queue:** Python `queue.Queue` for intra-process communication (upgradable to Redis for production)
- **Tracking:** ByteTrack integration via `model.track(persist=True)`
- **Backend API:** FastAPI with modular service architecture
- **Frontend:** React.js dashboard for real-time monitoring, violation history, and report viewing
- **Database:** SQLAlchemy ORM with `verified_violations` table storing only Judge-confirmed violations
- **Deployment:** Docker Compose for production; Google Colab for GPU-accelerated testing

---

## 5. Evaluation Plan

### 5.1 Detection Accuracy
- **Metrics:** mAP@50, mAP@50-95, per-class Precision/Recall/F1
- **Comparison:** Baseline YOLO-only vs. Sentry-Judge hybrid
- **Target:** Demonstrate the presence/absence detection gap; show Judge reduces false positives by >5%

### 5.2 Real-Time Performance
- **Metrics:** Sentry FPS, Judge latency per verification, end-to-end throughput
- **Target:** Sentry ≥25 FPS; Judge <500ms per ROI

### 5.3 System Efficiency
- **Metrics:** SAM bypass rate, queue size, cooldown effectiveness
- **Target:** 70-80% bypass rate; >95% queue reduction via cooldown

### 5.4 Report Quality
- **Metrics:** Report completeness, serial offender identification accuracy
- **Evaluation:** Compare LLM-generated reports against manually written safety summaries

### 5.5 Ablation Studies
- Optimizer comparison (SGD vs. AdamW) on minority class performance
- Cooldown period impact (1min, 5min, 10min) on alert coverage vs. fatigue
- ROI geometry ratios (Head: 30%/40%/50%) on verification accuracy

---

## 6. Timeline

| Week | Tasks |
|------|-------|
| 1-2 | Dataset preparation, YOLO model training, optimizer ablation |
| 3-4 | Sentry module: YOLO + ByteTrack + cooldown + queue |
| 5-6 | Judge module: SAM 3 integration, ROI verification pipeline |
| 7-8 | Pipeline integration, decoupled system testing |
| 9-10 | Agentic Reporter, LLM integration, PDF generation |
| 11-12 | Full-stack system (API, frontend, deployment) |
| 13-14 | Comprehensive evaluation, ablation studies |
| 15-16 | Thesis writing, defense preparation |

---

## 7. Expected Contributions

1. **Identification and quantification** of the Absence Detection Paradox in PPE monitoring
2. **A novel decoupled Sentry-Judge architecture** where the fast detector and foundation model operate asynchronously via a message queue, maintaining real-time throughput
3. **Cooldown-based intelligent deduplication** that reduces Judge workload by >95% while preserving violation coverage
4. **Geometric Prompt Engineering** — ROI-based spatial constraints for efficient foundation model inference
5. **An autonomous agentic compliance system** with LLM-powered report generation and OSHA-compliant output
6. **Empirical evidence** on optimizer selection (SGD vs. AdamW) for imbalanced safety-critical detection tasks

---

## 8. References

- Aharon, N., et al. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. *arXiv:2206.14651*.
- Cai, Z., & Vasconcelos, N. (2019). Cascade R-CNN: Multi-stage Object Detection. *IEEE TPAMI*.
- Cao, Y., et al. (2023). Deficiency-aware approaches for anomaly detection. *CVPR*.
- Jocher, G., et al. (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
- Kirillov, A., et al. (2023). Segment Anything. *ICCV 2023*.
- Li, Y., et al. (2024). Multi-stage specialized model ensembles. *NeurIPS*.
- Meta AI. (2025). SAM 3: Segment Anything with Concepts. https://ai.meta.com/sam3
- OSHA. (2024). Construction Industry Fatal Facts. U.S. Department of Labor.
- Wang, Z., et al. (2023). Hard Hat Detection Based on Improved YOLOv8. *Safety Science*.
- Zhang, Y., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV*.
- Zhang, Y., et al. (2024). Multi-Class PPE Compliance Detection Using Deep Learning. *Automation in Construction*.