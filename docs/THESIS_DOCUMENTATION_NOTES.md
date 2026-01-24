# Thesis Documentation Notes

This document collects all information needed for writing the thesis document.

---

## 📚 Chapter 1: Introduction

### 1.1 Background
- Construction industry safety is critical
- PPE compliance monitoring is essential
- Manual monitoring is inefficient and error-prone
- Current automated systems have limitations

### 1.2 Problem Statement
**The Absence Detection Paradox:**
- Standard detectors excel at presence detection (94-96% accuracy)
- Struggle with absence detection (41% accuracy baseline)
- This is a fundamental challenge in computer vision

**Why Absence Detection is Hard:**
- No positive visual features ("no helmet" = hair/sky)
- Class imbalance: 4.4:1 ratio (helmet:no_helmet)
- Visual ambiguity (hair vs. no-helmet)
- VLM hallucination of absent objects

### 1.3 Research Contribution
**Hybrid Architecture with Intelligent Bypass:**
- Only 20.2% of detections require expensive semantic verification
- Achieves real-time performance (28.5 FPS)
- Improves precision by 14.3%

**Automated Reporting Agent:**
- Zero human intervention required
- Automatic violation collection and evidence storage
- Daily PDF reports with email delivery

### 1.4 Thesis Objectives
1. Develop a hybrid YOLO+SAM detection system
2. Implement intelligent bypass mechanism (5-path logic)
3. Create automated violation reporting system
4. Achieve >60% precision while maintaining real-time performance
5. Build complete full-stack application

---

## 📚 Chapter 2: Literature Review

### 2.1 Object Detection Evolution
- Traditional methods (HOG, SVM)
- Deep learning revolution (R-CNN, Fast R-CNN, Faster R-CNN)
- Single-shot detectors (YOLO series, SSD)
- Current state: YOLOv11 variants

### 2.2 PPE Detection Research
- **SC-YOLO (Saeheaw, 2025):** 96.3% mAP on helmet detection
- **Ordrick et al. (2025):** YOLOv11 for PPE
- **YOLO-World (2024):** Open-vocabulary detection

### 2.3 Semantic Segmentation
- **SAM (Kirillov et al., 2023):** Segment Anything Model
- SAM2 improvements
- SAM3 with text-guided prompts

### 2.4 Hybrid Approaches
- **Cabral et al. (2025):** YOLO+SAM hybrid paradigm
- Multi-model ensembles

### 2.5 Absence Detection Challenge
- **Kim et al. (2025):** VLM absence detection challenge
- Negative class detection strategies
- Visual Language Model limitations

### Key References
1. Kim et al. (2025) - VLM absence detection challenge
2. Saeheaw (2025) - SC-YOLO (96.3% mAP baseline)
3. Cabral et al. (2025) - YOLO+SAM hybrid paradigm
4. Ordrick et al. (2025) - YOLOv11 for PPE
5. Kirillov et al. (2023) - Segment Anything

---

## 📚 Chapter 3: Methodology

### 3.1 System Architecture
```
Input Image → YOLO Sentry (fast) → 5-Path Decision Logic
                                    ↓
                        ┌───────────┴────────────┐
                        │                        │
                    ✅ Certain              ❓ Uncertain
                    (79.8%)                 (20.2%)
                        │                        │
                        │                    SAM Judge
                        │                 (semantic verify)
                        │                        │
                        └────────┬───────────────┘
                                 ↓
                          Final Decision
```

### 3.2 5-Path Decision Logic

| Path | Condition | Action | Frequency |
|------|-----------|--------|-----------|
| Path 0 | Helmet + Vest detected | SAFE, no SAM | ~45% |
| Path 1 | "no_helmet" class detected | VIOLATION, no SAM | ~35% |
| Path 2 | Vest found, helmet missing | SAM checks HEAD ROI | ~10% |
| Path 3 | Helmet found, vest missing | SAM checks TORSO ROI | ~5% |
| Path 4 | Both missing | SAM checks both ROIs | ~5% |

### 3.3 ROI Extraction
- **Head ROI:** Top 40% of person bounding box
- **Torso ROI:** 20% to 100% of person bounding box

### 3.4 SAM Verification
- SAM receives CROPPED ROI, not full image (critical optimization)
- Mask coverage threshold: 5%
- Prompts: ["helmet", "hard hat"] or ["vest", "safety vest"]

### 3.5 Training Configuration
- Model: YOLOv11m
- Optimizer: SGD (not AdamW) - +6.3% precision improvement
- Learning rate: 0.01
- Epochs: 200 (early stopping at ~65)
- Image size: 640
- Batch size: 16

---

## 📚 Chapter 4: Implementation

### 4.1 Technology Stack

**Backend:**
- Python 3.9+
- FastAPI (async REST API)
- YOLOv11m + SAM3 (detection)
- PostgreSQL (database)
- SQLAlchemy (ORM)
- APScheduler (task scheduling)

**Frontend:**
- React 18 + Vite 5
- Axios (API client)
- React Dropzone (file upload)
- Canvas API (bbox visualization)

**DevOps:**
- Docker + Docker Compose
- GitHub Actions (CI/CD)

### 4.2 Database Schema
- `violations` table: stores all detection results with evidence
- `daily_reports` table: tracks generated reports

### 4.3 API Endpoints
- `POST /api/detect`: Run detection on uploaded image
- `POST /api/upload`: Upload image for processing
- `GET /api/history`: Retrieve violation history

---

## 📚 Chapter 5: Experimental Results

### 5.1 Dataset
- **Construction-PPE Dataset:** 1,416 images
- Train: 1,132 (80%), Val: 143 (10%), Test: 141 (10%)
- Classes: helmet, vest, person, no_helmet

### 5.2 Performance Metrics

| Metric | YOLO-only | Hybrid (Ours) | Improvement |
|--------|-----------|---------------|-------------|
| Precision | 58.8% | 62.5% | +6.3% |
| Recall | 54.2% | 55.1% | +0.9% |
| F1 Score | 56.4% | 58.5% | +2.1% |
| FPS | 35.5 | 28.5 | -7.0 |

### 5.3 Class-wise mAP@50
- Helmet: 84.2%
- Vest: 84.2%
- Person: 92.2%
- No_helmet: 41.1% → 62.5% (with hybrid)

### 5.4 SAM Activation Analysis
- Total bypass rate: 79.8%
- SAM activation: 20.2%
- False positive reduction: 14.3%

---

## 📚 Chapter 6: Discussion

### 6.1 Key Findings
1. Intelligent bypass mechanism maintains real-time performance
2. SGD optimizer outperforms AdamW (+6.3% precision)
3. ROI-based SAM verification is more efficient than full-image
4. Automated reporting reduces manual effort to zero

### 6.2 Limitations
- Limited to helmet and vest detection (2 PPE types)
- Requires GPU for optimal performance
- Dataset size relatively small (1,416 images)

### 6.3 Future Work
- Expand to more PPE types (gloves, boots, goggles)
- Worker re-identification across cameras
- Edge deployment optimization
- Real-time camera streaming at scale

---

## 📚 Chapter 7: Conclusion

### Summary
- Developed hybrid YOLO+SAM detection system
- Achieved 62.5% precision (+6.3% over baseline)
- Maintained real-time performance (28.5 FPS)
- Implemented fully automated reporting

### Contributions
1. Novel 5-path intelligent bypass mechanism
2. Complete working full-stack system
3. Automated reporting pipeline
4. Comprehensive evaluation framework

---

## 📊 Figures to Create

- [ ] System architecture diagram
- [ ] 5-path decision flowchart
- [ ] ROI extraction visualization
- [ ] Training loss curves
- [ ] Precision-recall curves
- [ ] Confusion matrix
- [ ] SAM activation distribution
- [ ] Processing time breakdown
- [ ] Screenshot: Upload interface
- [ ] Screenshot: Detection results
- [ ] Screenshot: Violation history
- [ ] Sample PDF report

---

## 📈 Tables to Create

- [ ] Dataset statistics
- [ ] Class distribution
- [ ] Training hyperparameters
- [ ] Model comparison (YOLO-only vs Hybrid)
- [ ] Performance metrics summary
- [ ] Processing time comparison
- [ ] Ablation study results
