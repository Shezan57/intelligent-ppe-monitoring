# Tech Stack

## Backend
- **Language**: Python 3
- **Web Framework**: FastAPI (running on Uvicorn ASGI server)
- **Database ORM**: SQLAlchemy 2.0 with Alembic for migrations
- **Database**: PostgreSQL (psycopg2-binary)
- **Machine Learning**: 
  - Ultralytics YOLOv11 (Object Detection)
  - PyTorch & Torchvision
  - Facebook Segment Anything Model (SAM)
- **Image Processing**: OpenCV, Pillow, Numpy
- **Configuration Validation**: Pydantic & Pydantic-Settings
- **Background Tasks**: APScheduler
- **Reporting**: ReportLab, Matplotlib, Seaborn
- **Testing**: PyTest, Pytest-Asyncio, HTTPX

## Frontend
- **Framework**: React 18
- **Build Tool**: Vite 5
- **HTTP Client**: Axios
- **Components**: React Dropzone, React Hot Toast
- **Serving**: Uses Node.js for development, potential Nginx for production (`nginx.conf` present)

## Environment & Deployment
- Docker and Docker Compose configured.
