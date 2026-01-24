"""
Upload API Routes

Endpoints for file upload handling.
"""

import os
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import cv2
import numpy as np

from api.models.response_models import UploadResponse


router = APIRouter(prefix="/api", tags=["upload"])


# Allowed file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    run_detection: bool = Form(default=False)
):
    """
    Upload an image file for later processing.
    
    Args:
        file: Image file to upload
        run_detection: Whether to run detection immediately
        
    Returns:
        UploadResponse with file path and optional detection result
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read and validate content
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Validate it's a valid image
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    new_filename = f"upload_{timestamp}_{unique_id}{file_ext}"
    
    # Save file
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    file_path = os.path.join(uploads_dir, new_filename)
    cv2.imwrite(file_path, image)
    
    # Run detection if requested
    detection_result = None
    if run_detection:
        from services.hybrid_detector import get_hybrid_detector
        from api.models.response_models import DetectionResponse, PersonDetection, TimingInfo, DetectionStats
        
        detector = get_hybrid_detector()
        annotated_path = file_path.replace(file_ext, f"_annotated{file_ext}")
        
        result = detector.detect(
            image,
            save_annotated=True,
            output_path=annotated_path
        )
        
        persons = [
            PersonDetection(
                person_id=p["person_id"],
                bbox=p["bbox"],
                confidence=p["confidence"],
                has_helmet=p["has_helmet"],
                has_vest=p["has_vest"],
                is_violation=p["is_violation"],
                violation_type=p["violation_type"],
                decision_path=p["decision_path"],
                sam_activated=p["sam_activated"]
            )
            for p in result["persons"]
        ]
        
        detection_result = DetectionResponse(
            success=True,
            message="Detection completed",
            image_path=file_path,
            annotated_image_path=annotated_path,
            persons=persons,
            timing=TimingInfo(**result["timing"]),
            stats=DetectionStats(
                total_persons=result["stats"]["total_persons"],
                total_violations=result["stats"]["total_violations"],
                compliance_rate=result["stats"]["compliance_rate"],
                sam_activations=result["stats"]["sam_activations"],
                bypass_rate=result["stats"]["bypass_rate"]
            )
        )
    
    return UploadResponse(
        success=True,
        message="File uploaded successfully",
        file_path=file_path,
        detection_result=detection_result
    )


@router.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """
    Serve an uploaded file.
    
    Args:
        filename: Name of the file to retrieve
        
    Returns:
        The file as a response
    """
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    file_path = os.path.join(uploads_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security: prevent directory traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(uploads_dir)):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(file_path)
