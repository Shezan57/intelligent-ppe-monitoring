# Chapter 5: Experiments and Results

## 5.1 Introduction

This chapter presents the experimental evaluation of the Intelligent PPE Compliance Monitoring System. The evaluation is structured around four areas: model performance comparison between the baseline and combined dataset models, per-class accuracy analysis, confidence threshold sensitivity, and system-level latency profiling. All experiments were conducted on the held-out test split (10% of the 29,053-image dataset) unless otherwise stated.


## 5.2 Evaluation Metrics

The following standard object detection metrics are used throughout this chapter:

- **mAP@50 (mean Average Precision at IoU=0.50):** The primary benchmark metric. Measures the area under the Precision-Recall curve at a 50% IoU overlap threshold.
- **mAP@50-95:** Average mAP across IoU thresholds from 0.50 to 0.95 in steps of 0.05. A stricter measure of localization accuracy.
- **Precision (P):** The proportion of positive detections that are correct.
- **Recall (R):** The proportion of true positive objects that are successfully detected.
- **F1 Score:** The harmonic mean of Precision and Recall.
- **Frames Per Second (FPS):** Inference throughput, measured as the number of frames processed per second.


## 5.3 Experiment 1: Baseline vs. Combined Dataset Model

### 5.3.1 Experimental Setup

Two model variants were trained and evaluated:

- **Baseline Model:** Trained on 29,053 images from the six primary source datasets [29][30][31][32][33][34].
- **Combined Model:** Trained on 29,053 + 2,550 = 31,603 images, incorporating the additional person-detection dataset [35].

Both models used identical training hyperparameters as specified in Table 4.2. The Combined Model is the primary model evaluated throughout the remainder of this chapter, as it demonstrated superior real-world generalization despite a marginally lower validation mAP.

### 5.3.2 Overall Performance Comparison

Table 5.1 presents the overall detection performance of both models on the test set.

**Table 5.1: Overall Model Performance Comparison**

| Metric | Baseline Model | Combined Model | Difference |
|--------|:--------------:|:--------------:|:----------:|
| mAP@50 | 89.7% | 89.3% | −0.4% |
| mAP@50-95 | 64.2% | 65.9% | **+1.7%** |
| Precision | 88.5% | 87.8% | −0.7% |
| Recall | 80.1% | 82.7% | **+2.6%** |
| F1 Score | 84.1% | 85.1% | **+1.0%** |

*[Insert Figure 5.1: D11_mAP_comparison.png — Model Performance Comparison Bar Chart]*

**Key finding:** The Combined Model achieves higher recall (+2.6%) and mAP@50-95 (+1.7%) despite a slight reduction in mAP@50. This indicates improved localization precision and reduced missed detections — both critical properties for a safety monitoring system where false negatives (workers without PPE going undetected) carry direct safety consequences.

Importantly, real-world deployment trials confirmed that the Combined Model correctly identified ambiguous edge cases (workers partially occluded by scaffolding, workers at low viewing angles) at a confidence threshold of 0.25, whereas the Baseline Model failed to produce detections at the same threshold. This demonstrates the generalization benefit of incorporating varied person-specific training data, consistent with findings in related literature [17].


## 5.4 Experiment 2: Per-Class AP Analysis

Table 5.2 presents the Average Precision (AP@50) for each class across both models. This breakdown reveals class-level performance differences that aggregate metrics obscure.

**Table 5.2: Per-Class Average Precision (AP@50)**

| Class | Baseline Model | Combined Model | Change |
|-------|:--------------:|:--------------:|:------:|
| `helmet` | 93.1% | 92.8% | −0.3% |
| `vest` | 91.4% | 91.7% | +0.3% |
| `person` | 85.3% | 83.1% | −2.2% |
| `no-helmet` | 91.8% | 92.3% | **+0.5%** |
| `no-vest` | 87.2% | 87.6% | +0.4% |

**Key findings:**

1. **`no-helmet` AP improved** from 91.8% to 92.3% in the Combined Model. This is the most safety-critical class, and the improvement represents a direct reduction in missed helmet violations.

2. **`person` AP dropped** from 85.3% to 83.1% in the Combined Model. This is attributed to domain mismatch: the 2,550 additional person images included non-construction stock photography, introducing a distribution shift that reduced precision for the base `person` class. This finding underscores the risk of naive data augmentation without careful domain alignment.

3. **`helmet` class reached 93.1%** in the Baseline Model — the highest AP of any class — confirming that helmet presence detection is well-solved by the trained architecture.

*[Insert Figure 5.2: D12_PR_curve.png — Precision-Recall Curves (All Classes)]*
*[Insert Figure 5.3: D12b_F1_curve.png — F1-Confidence Curves (All Classes)]*


## 5.5 Experiment 3: Confusion Matrix Analysis

*[Insert Figure 5.4: D9_confusion_matrix.png — Raw Confusion Matrix]*
*[Insert Figure 5.5: D10_confusion_matrix_normalized.png — Normalized Confusion Matrix]*

The normalized confusion matrix reveals the following patterns:

- The `helmet` and `no-helmet` classes show minimal cross-confusion, indicating that the model has learned to distinguish between workers wearing helmets and those without helmets reliably.
- The `person` class exhibits the highest background confusion rate (~12%), consistent with the domain mismatch identified in Experiment 2.
- The `no-vest` class errors are predominantly misclassified as `person` rather than `vest`, suggesting that the absence detector correctly avoids false positives (classifying unprotected workers as safe) — a conservative error mode that is preferable from a safety standpoint.


## 5.6 Experiment 4: Confidence Threshold Sensitivity

To determine the optimal operating confidence threshold, inference was run on the test set at six threshold values. Table 5.3 summarizes the Precision, Recall, and F1 score at each threshold.

**Table 5.3: Confidence Threshold Sensitivity Analysis**

| Threshold | Precision | Recall | F1 Score | Notes |
|-----------|:---------:|:------:|:--------:|-------|
| 0.10 | 74.3% | 93.8% | 82.9% | High false positives |
| 0.15 | 79.1% | 91.2% | 84.7% | More FP than FN |
| 0.20 | 83.4% | 88.6% | 85.9% | Good recall, some FP |
| **0.25** | **86.2%** | **86.1%** | **86.1%** | **Balanced operating point** |
| 0.30 | 87.8% | 82.7% | 85.1% | **System default** |
| 0.50 | 91.3% | 71.4% | 80.1% | High precision, many misses |

**Key finding:** The system default threshold of 0.30 was selected as a conservative operating point that prioritizes precision (fewer false alarms). However, for deployment scenarios where missed violations are the primary concern, a threshold of 0.25 provides a better F1 score and comparable precision to the 0.30 baseline — with 3.7% higher recall.

*[Insert Figure 5.6: D12c_Precision_curve.png — Precision-Confidence Curve]*
*[Insert Figure 5.7: D12d_Recall_curve.png — Recall-Confidence Curve]*


## 5.7 Experiment 5: 5-Path Distribution Analysis

To characterize the real-world load distribution across the five triage paths, inference was run on a 10-minute test video (18,000 frames at 30 FPS). Table 5.4 summarizes the distribution of detections across each path.

**Table 5.4: 5-Path Triage Distribution (10-Minute Test Video)**

| Path | Description | Detection Count | % of Total | SAM Invoked? |
|------|-------------|:-----------:|:----------:|:------------:|
| Path 1 | Fast Safe (helmet + vest) | 10,842 | 60.2% | No |
| Path 2 | Fast Violation (no-helmet) | 4,210 | 23.4% | No |
| Path 3 | Rescue Head | 1,726 | 9.6% | Yes |
| Path 4 | Rescue Body | 849 | 4.7% | Yes |
| Path 5 | Critical (both missing) | 373 | 2.1% | Yes |
| | **Total** | **18,000** | **100%** | |

**Key finding:** Paths 1 and 2 together accounted for 83.6% of all detections, confirming the system's ability to resolve the majority of cases without SAM invocation. Only 16.4% of detections required SAM verification, which — combined with the 5-minute cooldown — keeps the verification queue well within the Judge's processing capacity.


## 5.8 Experiment 6: System Latency Profiling

Table 5.5 presents the measured processing latency for each stage of the pipeline on a system equipped with an NVIDIA T4 GPU (16 GB VRAM).

**Table 5.5: Pipeline Stage Latency Breakdown**

| Stage | Mean Latency | Per-Frame Budget (30 FPS = 33ms) |
|-------|:-----------:|:---------------------------------:|
| YOLO inference (per frame) | 18.4 ms | ✅ Within budget |
| IoU tracking (per frame) | 1.2 ms | ✅ Within budget |
| 5-path triage (per frame) | 0.8 ms | ✅ Within budget |
| ROI crop + enqueue | 0.6 ms | ✅ Within budget |
| **Total Sentry (per frame)** | **21.0 ms** | **✅ ~47 FPS capacity** |
| SAM verification (per ROI) | ~800 ms | Async — does not block Sentry |
| Database write (per violation) | 2.3 ms | Async |

**Key finding:** The Sentry pipeline completes well within the 33 ms per-frame budget required for 30 FPS operation, with ~47 FPS theoretical capacity. The SAM verification latency of ~800 ms per ROI is entirely absorbed by the asynchronous consumer thread and does not affect the real-time video output.


## 5.9 Sample Detection Outputs

Figures 5.8 and 5.9 present sample detection outputs from the validation set, showing both ground truth annotations and model predictions.

*[Insert Figure 5.8: D15c_ground_truth.jpg — Validation Set Ground Truth Labels]*
*[Insert Figure 5.9: D15a_sample_predictions.jpg — Validation Set Model Predictions]*
*[Insert Figure 5.10: D15b_sample_predictions.jpg — Additional Sample Predictions]*
*[Insert Figure 5.11: D15d_training_samples.jpg — Training Batch Samples]*

Qualitative inspection of the sample outputs confirms that the model correctly localizes helmet and vest regions on workers at multiple distances and angles. The model demonstrates robustness to partial occlusion by scaffolding structures, a common challenge in construction site imagery.


## 5.10 Chapter Summary

This chapter presented six experiments evaluating the Intelligent PPE Compliance Monitoring System. The Combined Model achieved 89.3% mAP@50, 65.9% mAP@50-95, and 92.3% AP for the safety-critical `no-helmet` class. Confidence threshold analysis identified 0.25 as the balanced operating point and 0.30 as the precision-optimized default. The five-path triage analysis confirmed that 83.6% of detections are resolved without SAM invocation. System latency profiling demonstrated that the Sentry pipeline achieves ~47 FPS on a T4 GPU, comfortably meeting the 30 FPS real-time requirement. Together, these results validate the system's design objectives.
