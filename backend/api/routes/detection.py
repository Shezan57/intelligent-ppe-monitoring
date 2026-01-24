"""
Detection API Routes

Endpoints for running PPE detection on images.
"""

import os
import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session
import cv2
import numpy as np

from api.models.response_models import DetectionResponse, PersonDetection, TimingInfo, DetectionStats
from services.hybrid_detector import get_hybrid_detector
from database.connection import get_db
from config.settings import settings
from agents.violation_collector import ViolationCollector


router = APIRouter(prefix="/api", tags=["detection"])


@router.post("/detect", response_model=DetectionResponse)
async def detect_violations(
    file: UploadFile = File(...),
    site_location: Optional[str] = Form(default=None),
    camera_id: Optional[str] = Form(default=None),
    save_annotated: bool = Form(default=True),
    db: Session = Depends(get_db)
):
    """
    Run PPE detection on uploaded image.
    
    This endpoint:
    1. Receives an uploaded image
    2. Runs hybrid YOLO+SAM detection
    3. Returns detected persons with PPE status
    4. Optionally saves annotated image
    
    Args:
        file: Uploaded image file (JPG, PNG)
        site_location: Site location for tracking
        camera_id: Camera ID for tracking
        save_annotated: Whether to save annotated image
        
    Returns:
        DetectionResponse with all detection results
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG)"
        )
    
    # Read image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading image: {str(e)}"
        )
    
    # Generate file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    original_filename = f"detection_{timestamp}_{unique_id}.jpg"
    annotated_filename = f"detection_{timestamp}_{unique_id}_annotated.jpg"
    
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    original_path = os.path.join(uploads_dir, original_filename)
    annotated_path = os.path.join(uploads_dir, annotated_filename) if save_annotated else None
    
    # Save original image
    cv2.imwrite(original_path, image)
    
    # Run detection
    try:
        detector = get_hybrid_detector()
        result = detector.detect(
            image,
            save_annotated=save_annotated,
            output_path=annotated_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
    
    # Convert to response model
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
    
    timing = TimingInfo(
        total_ms=result["timing"]["total_ms"],
        yolo_ms=result["timing"]["yolo_ms"],
        sam_ms=result["timing"]["sam_ms"],
        postprocess_ms=result["timing"]["postprocess_ms"]
    )
    
    stats = DetectionStats(
        total_persons=result["stats"]["total_persons"],
        total_violations=result["stats"]["total_violations"],
        compliance_rate=result["stats"]["compliance_rate"],
        sam_activations=result["stats"]["sam_activations"],
        bypass_rate=result["stats"]["bypass_rate"]
    )
    
    # === Store violations in database ===
    try:
        collector = ViolationCollector(
            db=db,
            site_location=site_location or settings.default_site_location,
            camera_id=camera_id or settings.default_camera_id
        )
        collector.store_detection_results(
            detection_result=result,
            image_path=original_path,
            annotated_path=annotated_path,
            site_location=site_location,
            camera_id=camera_id
        )
    except Exception as e:
        # Log but don't fail detection if storage fails
        print(f"⚠️ Warning: Failed to store violations: {e}")
    
    return DetectionResponse(
        success=True,
        message="Detection completed successfully",
        image_path=original_path,
        annotated_image_path=annotated_path,
        persons=persons,
        timing=timing,
        stats=stats
    )


@router.post("/detect/base64", response_model=DetectionResponse)
async def detect_from_base64(
    image_base64: str,
    site_location: Optional[str] = None,
    camera_id: Optional[str] = None,
    save_annotated: bool = True,
    db: Session = Depends(get_db)
):
    """
    Run detection on base64-encoded image.
    
    Alternative to file upload for API integrations.
    """
    import base64
    
    try:
        # Decode base64
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode base64 image"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image: {str(e)}"
        )
    
    # Generate file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    original_path = os.path.join(uploads_dir, f"detection_{timestamp}_{unique_id}.jpg")
    annotated_path = os.path.join(uploads_dir, f"detection_{timestamp}_{unique_id}_annotated.jpg") if save_annotated else None
    
    cv2.imwrite(original_path, image)
    
    # Run detection
    detector = get_hybrid_detector()
    result = detector.detect(
        image,
        save_annotated=save_annotated,
        output_path=annotated_path
    )
    
    # Build response (same as above)
    persons = [
        PersonDetection(**{
            "person_id": p["person_id"],
            "bbox": p["bbox"],
            "confidence": p["confidence"],
            "has_helmet": p["has_helmet"],
            "has_vest": p["has_vest"],
            "is_violation": p["is_violation"],
            "violation_type": p["violation_type"],
            "decision_path": p["decision_path"],
            "sam_activated": p["sam_activated"]
        })
        for p in result["persons"]
    ]
    
    # Store violations in database
    try:
        collector = ViolationCollector(
            db=db,
            site_location=site_location or settings.default_site_location,
            camera_id=camera_id or settings.default_camera_id
        )
        collector.store_detection_results(
            detection_result=result,
            image_path=original_path,
            annotated_path=annotated_path,
            site_location=site_location,
            camera_id=camera_id
        )
    except Exception as e:
        print(f"⚠️ Warning: Failed to store violations: {e}")
    
    return DetectionResponse(
        success=True,
        message="Detection completed successfully",
        image_path=original_path,
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
