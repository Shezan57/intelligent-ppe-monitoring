"""
PPE Detection Backend - FastAPI Application

Main entry point for the PPE compliance monitoring system.
Implements hybrid YOLO+SAM detection with automated reporting.

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from api.routes import detection_router, upload_router, history_router
from database.connection import engine
from database.models import create_tables


# === Lifespan Management ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Startup:
    - Create database tables
    - Load ML models
    - Start background schedulers
    
    Shutdown:
    - Stop schedulers
    - Cleanup resources
    """
    # === STARTUP ===
    print("🚀 Starting PPE Detection Backend...")
    
    # Create database tables
    print("📦 Creating database tables...")
    create_tables(engine)
    
    # Pre-load models (optional - can be lazy loaded)
    if not settings.debug:
        print("🤖 Pre-loading ML models...")
        try:
            from services.yolo_detector import get_yolo_detector
            from services.sam_verifier import get_sam_verifier
            
            yolo = get_yolo_detector()
            yolo.load_model()
            
            sam = get_sam_verifier()
            sam.load_model()
            
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"⚠️ Model pre-loading failed: {e}")
            print("   Models will be loaded on first request")
    
    # Start daily reporter (if email is configured)
    if settings.sender_email and settings.sender_password:
        print("📧 Starting automated reporter...")
        try:
            from agents.daily_reporter import get_daily_reporter
            reporter = get_daily_reporter()
            reporter.start()
            print(f"✅ Daily reporter scheduled for {settings.report_time}")
        except Exception as e:
            print(f"⚠️ Reporter start failed: {e}")
    
    print("✅ Backend ready!")
    print(f"📍 API docs: http://localhost:8000/docs")
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    print("👋 Shutting down PPE Detection Backend...")
    
    # Stop scheduler
    try:
        from agents.daily_reporter import get_daily_reporter
        reporter = get_daily_reporter()
        reporter.stop()
    except:
        pass
    
    print("✅ Shutdown complete")


# === Create FastAPI App ===

app = FastAPI(
    title="PPE Detection API",
    description="""
    ## Intelligent PPE Compliance Monitoring System
    
    ### Key Features
    - **Hybrid Detection:** YOLO + SAM with 5-path intelligent bypass
    - **79.8% Bypass Rate:** Only 20.2% of detections need SAM verification
    - **Automated Reporting:** Daily PDF reports sent to managers
    - **Real-time API:** Sub-second detection response
    
    ### Detection Classes
    - Helmet (presence)
    - Vest (presence)  
    - Person (base)
    - No_helmet (absence)
    
    ### 5-Path Decision Logic
    1. **Fast Safe:** Helmet + Vest detected → SAFE
    2. **Fast Violation:** no_helmet class → VIOLATION
    3. **Rescue Head:** SAM verifies helmet in HEAD ROI
    4. **Rescue Body:** SAM verifies vest in TORSO ROI
    5. **Critical:** SAM verifies both ROIs
    
    ---
    Built for Master's Thesis - Intelligent PPE Compliance Monitoring
    """,
    version="1.0.0",
    lifespan=lifespan
)


# === CORS Middleware ===

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Error Handlers ===

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


# === Register Routers ===

app.include_router(detection_router)
app.include_router(upload_router)
app.include_router(history_router)


# === Static Files ===

# Serve uploaded images
uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Serve reports
reports_dir = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")


# === Health Check ===

@app.get("/", tags=["health"])
async def root():
    """Root endpoint - API info."""
    return {
        "name": "PPE Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all system components.
    """
    from api.models.response_models import HealthResponse
    
    # Check YOLO
    yolo_loaded = False
    try:
        from services.yolo_detector import get_yolo_detector
        yolo = get_yolo_detector()
        yolo_loaded = yolo.is_loaded()
    except:
        pass
    
    # Check SAM
    sam_loaded = False
    try:
        from services.sam_verifier import get_sam_verifier
        sam = get_sam_verifier()
        sam_loaded = sam.is_loaded()
    except:
        pass
    
    # Check database
    db_connected = False
    try:
        from database.connection import engine
        with engine.connect() as conn:
            db_connected = True
    except:
        pass
    
    # Determine overall status
    if yolo_loaded and db_connected:
        status = "healthy"
    elif db_connected:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        yolo_loaded=yolo_loaded,
        sam_loaded=sam_loaded,
        database_connected=db_connected
    )


# === Development Server ===

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
