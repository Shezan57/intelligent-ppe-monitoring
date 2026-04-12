# Architecture

## Pattern & Layers
This project follows a Client-Server and decoupled AI Pipeline pattern:
1. **Frontend (Presentation Layer)**: A React-based Single Page Application (SPA).
2. **Backend API (Service Layer)**: Built with FastAPI, handling REST API requests, database queries, and async process scheduling.
3. **AI Pipeline (AI/ML Layer)**: A complex hybrid detection system broken into 'Sentry' and 'Judge':
   - **Sentry**: Uses YOLOv11 to detect objects in streams.
   - **Judge**: Asynchronous verification step often using `async_sam_verifier.py`/`sam_verifier.py` to confirm YOLO's detections via SAM (Segment Anything Model).
   - **Tracking & Deduplication**: The `ViolationTracker` manages cross-frame tracking to prevent duplicate violation alerting using intersection-over-union (IoU).

## Data Flow
`Camera/Upload -> API Endpoint -> Stream Processor / Storage -> Sentry (Detector) -> Tracker & Cropper -> Queue -> Judge (Verifier) -> Database -> UI/Alerts`

## Abstractions & Entry Points
- **Backend Entry Point**: `backend/main.py`.
- **Frontend Entry Point**: `frontend/src/main.jsx`.
- **Key Abstractions**: The separation of `Sentry` and `Judge` handles distinct ML tasks, allowing YOLO flexibility and SAM precision without blocking video frames.
