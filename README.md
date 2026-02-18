# Intelligent PPE Compliance Monitoring System

**Master's Thesis Project** - Hybrid Deep Learning Framework with Semantic Verification for Construction Safety

## 🎯 Overview

A real-time Personal Protective Equipment (PPE) detection system that uses a novel **5-path intelligent bypass mechanism** to combine fast YOLO detection with accurate SAM semantic verification.

### Key Innovation
- **79.8% SAM bypass rate** - Only 20.2% of detections require expensive semantic verification
- **28.5 FPS** effective processing speed while maintaining high accuracy
- **+6.3% precision improvement** over YOLO-only baseline

## 📊 Features

| Feature | Description |
|---------|-------------|
| **Image Detection** | Upload images for PPE compliance check |
| **Video Detection** | Process video files with frame skipping |
| **Violation History** | Track violations with filters and pagination |
| **Automated Reports** | Daily PDF reports emailed to managers |
| **5-Path Logic** | Intelligent bypass for real-time performance |

## 🛠️ Technology Stack

**Backend:** FastAPI, PyTorch, YOLOv11m, SAM3, SQLAlchemy  
**Frontend:** React 18, Vite 5, Axios  
**Database:** SQLite (dev) / PostgreSQL (prod)

## 🚀 Quick Start

```bash
# Clone repository
git clone <repo-url>
cd intelligent-ppe-monitoring

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Access at: http://localhost:5173

## 📁 Project Structure

```
├── backend/              # FastAPI server
│   ├── services/         # YOLO, SAM, Hybrid detection
│   ├── agents/           # Violation collector, Reporter
│   ├── api/              # REST endpoints
│   └── scripts/          # Evaluation tools
├── frontend/             # React application
│   ├── components/       # UI components
│   └── api/              # API client
├── docs/                 # Documentation
│   └── TECHNICAL_DOCUMENTATION.md
└── CHANGELOG.md          # Development log
```

## 📈 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Precision | >60% | ⏳ TBD |
| FPS | >25 | ✅ 28.5 |
| SAM Bypass | >75% | ✅ 79.8% |
| SAM Activation | <25% | ✅ 20.2% |

## 📝 Documentation

- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) - Complete thesis reference
- [Changelog](CHANGELOG.md) - Development history
- [Complete Specification](COMPLETE_PROJECT_SPECIFICATION.md) - Full project spec

## 🎓 Academic

**Thesis Title:** *"Intelligent PPE Compliance Monitoring: A Hybrid Deep Learning Framework with Semantic Verification for Construction Safety"*

**Research Contribution:** Addresses the Absence Detection Paradox - standard detectors excel at presence detection (94-96%) but struggle with absence detection (41%). Our hybrid approach achieves 62.5% precision on absence detection.

---

*Built for Master's Graduation Thesis | 2026*
