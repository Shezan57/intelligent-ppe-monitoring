# API Routes package
from .detection import router as detection_router
from .upload import router as upload_router
from .history import router as history_router

__all__ = ["detection_router", "upload_router", "history_router"]
