# External Integrations

## Databases
- **PostgreSQL**: Used as the primary relational database for tracking violations, sessions, etc. (Managed via SQLAlchemy). SQLite fallback might be present based on `ppe_detection.db`.

## Machine Learning Models
- **Ultralytics YOLOv11**: Object detection model file located at `backend/models/best.pt`.
- **SAM (Segment Anything Model)**: Secondary model configuration for segmentation tasks (`sam3.pt`).

## Services
- **SMTP Email Integration**: Scheduled reports and notifications via standard SMTP (e.g., smtp.gmail.com).

## Internal APIs
- The React Frontend integrates with the FastAPI Backend locally and via production configurations handling CORS.
