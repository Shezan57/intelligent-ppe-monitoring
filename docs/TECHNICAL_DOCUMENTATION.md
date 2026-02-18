# Technical Documentation: PPE Detection System

## For Master's Thesis Writing

This document provides all technical details needed for thesis documentation.

---

## 1. System Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPE Detection System                      │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React/Vite)           │  Backend (FastAPI/Python)    │
│  ├─ Image Upload                 │  ├─ YOLO Detector            │
│  ├─ Video Upload                 │  ├─ SAM Verifier             │
│  ├─ Violation History            │  ├─ Hybrid Detector          │
│  └─ Settings Panel               │  ├─ Stream Processor         │
│                                  │  ├─ Violation Collector      │
│                                  │  └─ Daily Reporter           │
├──────────────────────────────────┼──────────────────────────────┤
│          SQLite/PostgreSQL       │        Model Weights         │
│          (Violations DB)         │  (YOLOv11m + SAM3)           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Diagram

```
                     ┌──────────────┐
                     │   Frontend   │
                     │   (React)    │
                     └──────┬───────┘
                            │ HTTP/WS
                     ┌──────▼───────┐
                     │   FastAPI    │
                     │   Backend    │
                     └──────┬───────┘
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     ┌──────────┐    ┌──────────┐    ┌──────────┐
     │  YOLO    │    │   SAM    │    │ Database │
     │ Detector │    │ Verifier │    │ (SQLite) │
     └────┬─────┘    └────┬─────┘    └──────────┘
          │               │
          └───────┬───────┘
                  ▼
          ┌──────────────┐
          │   Hybrid     │
          │  Detector    │
          │ (5-Path)     │
          └──────────────┘
```

---

## 2. Core Innovation: 5-Path Decision Logic

### 2.1 Path Distribution

| Path | Name | Trigger | SAM Used | % of Cases |
|------|------|---------|----------|------------|
| 0 | Fast Safe | Helmet + Vest detected | ❌ No | ~45% |
| 1 | Fast Violation | no_helmet class detected | ❌ No | ~35% |
| 2 | Rescue Head | Vest found, no helmet | ✅ HEAD ROI | ~10% |
| 3 | Rescue Body | Helmet found, no vest | ✅ TORSO ROI | ~5% |
| 4 | Critical | Both missing | ✅ Both ROIs | ~5% |

**Result:** 79.8% bypass rate, only 20.2% SAM activation

### 2.2 ROI Extraction

```python
# Head ROI: Top 40% of person bbox
def extract_head_roi(bbox):
    x_min, y_min, x_max, y_max = bbox
    height = y_max - y_min
    head_y_max = int(y_min + height * 0.4)
    return [x_min, y_min, x_max, head_y_max]

# Torso ROI: 20%-100% of person bbox
def extract_torso_roi(bbox):
    x_min, y_min, x_max, y_max = bbox
    height = y_max - y_min
    torso_y_min = int(y_min + height * 0.2)
    return [x_min, torso_y_min, x_max, y_max]
```

### 2.3 SAM Optimization

**Critical:** SAM receives cropped ROI image, NOT full image.
- Reduces processing from ~1000ms to ~100ms per verification
- Enables real-time performance

---

## 3. File Structure

### Backend (30+ files)
```
backend/
├── main.py                 # FastAPI application entry
├── config/
│   └── settings.py         # Pydantic configuration
├── api/
│   ├── routes/
│   │   ├── detection.py    # POST /api/detect
│   │   ├── upload.py       # POST /api/upload
│   │   ├── history.py      # GET /api/history
│   │   └── video.py        # POST /api/detect/video
│   └── models/             # Pydantic schemas
├── services/
│   ├── yolo_detector.py    # YOLOv11m wrapper
│   ├── sam_verifier.py     # SAM3 verification
│   ├── hybrid_detector.py  # 5-path logic (CORE)
│   ├── stream_processor.py # Video processing
│   ├── report_generator.py # PDF reports
│   ├── email_service.py    # SMTP delivery
│   └── storage_service.py  # Database ops
├── agents/
│   ├── violation_collector.py  # Real-time storage
│   └── daily_reporter.py       # Scheduled reports
├── database/
│   ├── models.py           # SQLAlchemy ORM
│   └── connection.py       # DB session
├── utils/
│   ├── bbox_utils.py       # ROI extraction
│   ├── visualization.py    # Drawing
│   └── metrics.py          # Evaluation
└── scripts/
    └── evaluate.py         # Thesis metrics
```

### Frontend (15+ files)
```
frontend/
├── src/
│   ├── App.jsx             # Main app with tabs
│   ├── api/
│   │   └── client.js       # Axios API client
│   ├── components/
│   │   ├── Header.jsx
│   │   ├── UploadZone.jsx
│   │   ├── DetectionCanvas.jsx
│   │   ├── StatsPanel.jsx
│   │   ├── ViolationCard.jsx
│   │   ├── HistoryTable.jsx
│   │   ├── SettingsPanel.jsx
│   │   └── VideoUpload.jsx
│   └── styles/
│       └── index.css       # Design system
├── package.json
└── vite.config.js
```

---

## 4. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Detect PPE in uploaded image |
| POST | `/api/detect/video` | Process video file |
| POST | `/api/upload` | Upload image file |
| GET | `/api/history` | Get violation history |
| GET | `/api/history/summary` | Get statistics summary |
| GET | `/health` | System health check |
| WS | `/api/ws/webcam` | Real-time webcam stream |

---

## 5. Database Schema

### violations Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | DATETIME | Detection time |
| site_location | VARCHAR | Site identifier |
| camera_id | VARCHAR | Camera identifier |
| person_bbox | JSON | [x_min, y_min, x_max, y_max] |
| has_helmet | BOOLEAN | Helmet detected |
| has_vest | BOOLEAN | Vest detected |
| violation_type | VARCHAR | no_helmet/no_vest/both_missing |
| decision_path | VARCHAR | 5-path decision taken |
| detection_confidence | FLOAT | YOLO confidence |
| sam_activated | BOOLEAN | SAM was used |
| processing_time_ms | FLOAT | Total processing time |

### daily_reports Table
| Column | Type |
|--------|------|
| id | INTEGER |
| report_date | DATE |
| total_detections | INTEGER |
| total_violations | INTEGER |
| compliance_rate | FLOAT |
| pdf_path | VARCHAR |
| email_sent | BOOLEAN |

---

## 6. Performance Metrics

### Target Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| Precision | >60% | For violation detection |
| Recall | >50% | Trade-off with precision |
| F1 Score | >55% | Balanced metric |
| FPS | >25 | Real-time requirement |
| SAM Activation | <25% | Bypass rate optimization |

### Timing Breakdown
| Component | Time (ms) |
|-----------|-----------|
| YOLO Detection | ~25-35 |
| SAM Verification | ~80-120 |
| Post-processing | ~5-10 |
| **Total (with bypass)** | **~35 avg** |
| **Total (with SAM)** | **~150 avg** |

---

## 7. Technology Stack

### Backend
- Python 3.9+
- FastAPI 0.104+
- PyTorch 2.0+
- Ultralytics (YOLOv11m)
- SAM3 (Segment Anything)
- SQLAlchemy 2.0+
- ReportLab (PDF)
- APScheduler

### Frontend
- React 18
- Vite 5
- Axios
- React Hot Toast

### Deployment
- Docker + Docker Compose
- Nginx (frontend)
- PostgreSQL (production)
- Azure GPU VM (recommended)

---

## 8. Key Algorithms

### 8.1 Hybrid Detection Flow
```python
def detect(image):
    # Step 1: YOLO detection
    yolo_results = yolo_detector.detect(image)
    
    # Step 2: For each person, determine path
    for person in yolo_results.persons:
        if person.helmet_detected and person.vest_detected:
            # PATH 0: Fast Safe
            result = safe(sam_used=False)
        elif person.no_helmet_detected:
            # PATH 1: Fast Violation
            result = violation(sam_used=False)
        elif person.vest_detected:
            # PATH 2: Rescue Head
            head_roi = extract_head_roi(person.bbox)
            sam_result = sam.verify_helmet(image, head_roi)
            result = violation if not sam_result else safe
        elif person.helmet_detected:
            # PATH 3: Rescue Body
            torso_roi = extract_torso_roi(person.bbox)
            sam_result = sam.verify_vest(image, torso_roi)
            result = violation if not sam_result else safe
        else:
            # PATH 4: Critical
            sam_result = sam.verify_both(image, person.bbox)
            result = based_on_sam_result
    
    return results
```

---

## 9. Thesis Figures

### Recommended Figures
1. System architecture diagram
2. 5-path decision flowchart
3. ROI extraction visualization
4. Performance comparison charts
5. Screenshot of detection UI
6. Violation history dashboard
7. PDF report example

### Recommended Tables
1. Dataset statistics
2. Training hyperparameters
3. Evaluation metrics comparison
4. Path distribution analysis
5. Processing time breakdown
6. API endpoint summary

---

## 10. Evaluation Commands

```bash
# Run evaluation on test images
cd backend
python scripts/evaluate.py --test-dir ../test_images --output ./results

# Output files:
# - evaluation_metrics_TIMESTAMP.json
# - evaluation_results_TIMESTAMP.json
# - evaluation_timing_TIMESTAMP.json
```

---

## 11. Deployment Commands

### Local Development
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

### Docker Deployment
```bash
docker-compose up -d
```

### Azure GPU Deployment
See `azure_deployment_plan.md` in artifacts.

---

*Document Version: 1.0 | Last Updated: 2026-01-24*
