# PPE Detection Thesis Project - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### 2026-01-28 - Violation De-duplication (Cooldown System)

#### Added
- **Violation Tracker Service** (`services/violation_tracker.py`)
  - IoU-based person tracking across frames
  - Configurable cooldown period (default 5 min)
  - Prevents repeated alerts for same worker with same violation
  - Tracks deduplication rate for thesis metrics

- **API Endpoints**
  - `GET /api/tracking/stats` - Get de-duplication statistics
  - `POST /api/tracking/reset` - Reset tracking state

- **Config Settings**
  - `VIOLATION_COOLDOWN_SECONDS` - Cooldown before re-alerting (300s)
  - `VIOLATION_IOU_THRESHOLD` - Min overlap to match as same person (0.3)
  - `VIOLATION_TRACK_TIMEOUT` - Remove stale tracks (30s)

#### Changed
- `ViolationCollector` now uses `ViolationTracker` for automatic de-duplication
- Detection endpoints automatically skip duplicate violations within cooldown

---

### 2026-01-24 - Core Features Complete (Video, History, Settings)

#### Added
- **Video Detection System**
  - `services/stream_processor.py` - Video file processing with frame skipping
  - `api/routes/video.py` - Video upload and WebSocket webcam endpoints
  - `POST /api/detect/video` - Process uploaded video files
  - `WS /api/ws/webcam` - Real-time webcam detection stream
  
- **Evaluation Script**
  - `scripts/evaluate.py` - Run detection on test dataset, generate thesis metrics
  - Outputs: precision, recall, F1, FPS, SAM activation rate, path distribution

- **Frontend Components**
  - `components/HistoryTable.jsx` - Violation history with filters and pagination
  - `components/SettingsPanel.jsx` - Configure detection thresholds
  - `components/VideoUpload.jsx` - Video file upload with progress
  - Tabbed navigation: Image Detection | Video Detection | Violation History
  - Settings button in header

- **CSS Additions**
  - Navigation tabs, data tables, pagination
  - Settings modal, toggle switches
  - Progress bar, result grids

#### Changed
- `App.jsx` - Refactored with tabbed navigation
- `Header.jsx` - Added settings button prop
- Integrated ViolationCollector into detection endpoints

---

### 2026-01-24 - Backend Core Implementation Complete

#### Added
- **Backend Structure** - Complete `backend/` directory with all packages
- **Configuration** - `config/settings.py` with Pydantic Settings, `.env.example`
- **Database Layer**
  - `database/models.py` - SQLAlchemy models for `Violation` and `DailyReport`
  - `database/connection.py` - Engine and session management
- **Core Services**
  - `services/yolo_detector.py` - YOLOv11m wrapper with PPE detection
  - `services/sam_verifier.py` - SAM semantic verification with ROI cropping
  - `services/hybrid_detector.py` - **5-path decision logic** (thesis innovation)
  - `services/report_generator.py` - PDF report generation with ReportLab
  - `services/email_service.py` - SMTP email delivery
  - `services/storage_service.py` - Database CRUD operations
- **Agents**
  - `agents/violation_collector.py` - Real-time violation storage
  - `agents/daily_reporter.py` - APScheduler-based automated reporting
- **API Routes**
  - `POST /api/detect` - Run detection on uploaded image
  - `POST /api/upload` - Upload image for processing
  - `GET /api/history` - Query violation history with filters
  - `GET /health` - Health check endpoint
- **Utilities**
  - `utils/bbox_utils.py` - ROI extraction, IoU calculation
  - `utils/visualization.py` - Bbox drawing, annotated images
  - `utils/metrics.py` - Precision, recall, F1 calculations
- **Main Application** - `main.py` FastAPI app with lifespan, CORS, error handling

#### Technical Details
- 5-Path Decision Logic implemented as specified
- SAM receives cropped ROI (critical optimization)
- Mock mode for SAM when running on CPU
- SAM disabled by default (`sam_enabled = False`) for CPU development
- Comprehensive type hints and docstrings
- Total: 30+ Python files created

---

### 2026-01-24 - Frontend Implementation Complete

#### Added
- **React/Vite Setup**
  - `package.json` with dependencies
  - `vite.config.js` with API proxy
- **Design System**
  - `styles/index.css` - Comprehensive dark theme CSS (900+ lines)
  - CSS variables for colors, spacing, typography
- **Components**
  - `Header.jsx` - Application header with logo
  - `UploadZone.jsx` - Drag-drop image upload with preview
  - `DetectionCanvas.jsx` - Image display with annotation toggle
  - `StatsPanel.jsx` - Detection statistics and timing
  - `ViolationCard.jsx` - Individual person PPE status
- **API Client**
  - `api/client.js` - Axios instance with interceptors

---

### 2026-01-24 - Project Initialization

#### Added
- Initial project repository setup
- `COMPLETE_PROJECT_SPECIFICATION.md` - Technical specification (1942 lines)
- `ADVANCED_FEATURES_SPECIFICATION.md` - Enterprise features (907 lines)
- `CHANGELOG.md` - This file
- `README.md` - Basic readme

---

## Project Status Summary

### Completed ✅
- [x] Phase 1: Backend Core (YOLO, SAM, Hybrid Detection)
- [x] Phase 2: Database & Agents (Violations, Reporting)
- [x] Phase 3: Frontend (React UI, Components)
- [x] Phase 4: Integration (Video, History, Settings)

### Pending ⏳
- [ ] Phase 5: Evaluation (run on 141 test images)
- [ ] Phase 6: Azure GPU Deployment

### Key Metrics (Expected)
| Metric | Target | Status |
|--------|--------|--------|
| Precision | >60% | TBD |
| FPS | >25 | TBD |
| SAM Bypass Rate | 79.8% | Implemented |
| SAM Activation | <25% | Implemented |
