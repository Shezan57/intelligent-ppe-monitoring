"""
Detection API Routes

Endpoints for running PPE detection on images.

Async SAM Pipeline:
    YOLO detects instantly → response returned → SAM verifies in background
    SAM result updates DB during the violation cooldown window.
"""

import os
import uuid
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session
import cv2
import numpy as np

from api.models.response_models import DetectionResponse, PersonDetection, TimingInfo, DetectionStats
from services.hybrid_detector import get_hybrid_detector
from database.connection import get_db, SessionLocal
from config.settings import settings
from agents.violation_collector import ViolationCollector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detection"])


def _make_sam_db_callback(
    violation_db_id: Optional[int],
    site_location: str,
    camera_id: str,
    original_path: str,
    annotated_path: Optional[str]
):
    """
    Create a callback that updates the DB violation record when SAM finishes.

    This is the bridge between async SAM and the database.
    Called in a background thread when SAM completes.

    Args:
        violation_db_id: DB row ID to update (None = create new record)
        site_location: Site for new record creation
        camera_id: Camera for new record creation
        original_path: Image path for new record
        annotated_path: Annotated image path

    Returns:
        Callback function(SAMVerificationResult)
    """
    def on_sam_complete(sam_result):
        """
        Called when SAM finishes verifying.
        Updates the DB record with refined PPE status.
        """
        try:
            # Use a fresh DB session (background thread)
            db = SessionLocal()
            try:
                if violation_db_id is not None:
                    # Update existing violation record with SAM result
                    from database.models import Violation
                    violation = db.query(Violation).filter(
                        Violation.id == violation_db_id
                    ).first()

                    if violation:
                        # Update with SAM's refined result
                        violation.has_helmet = sam_result.has_helmet
                        violation.has_vest = sam_result.has_vest
                        violation.violation_type = sam_result.violation_type or violation.violation_type
                        violation.sam_activated = True
                        violation.processing_time_ms = (
                            (violation.processing_time_ms or 0) + sam_result.sam_latency_ms
                        )
                        db.commit()

                        if not sam_result.yolo_was_correct:
                            logger.info(
                                f"🔄 SAM corrected violation #{violation_db_id}: "
                                f"YOLO={'violation' if sam_result.yolo_initial_violation else 'safe'} "
                                f"→ SAM={'violation' if sam_result.is_violation else 'SAFE'}"
                            )
                        else:
                            logger.debug(
                                f"✅ SAM confirmed violation #{violation_db_id} "
                                f"in {sam_result.sam_latency_ms:.1f}ms"
                            )
                else:
                    logger.debug("SAM job completed but no DB record to update")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"SAM DB callback failed: {e}")

    return on_sam_complete


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
    
    # Run detection (YOLO instant, SAM async in background)
    try:
        detector = get_hybrid_detector()
        result = detector.detect_async(
            image,
            save_annotated=save_annotated,
            output_path=annotated_path
            # on_sam_complete wired after DB record created below
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
    
    # === Store YOLO violations in database + wire SAM async callback ===
    violation_db_ids = []  # DB IDs for SAM to update later
    try:
        collector = ViolationCollector(
            db=db,
            site_location=site_location or settings.default_site_location,
            camera_id=camera_id or settings.default_camera_id
        )
        stored = collector.store_detection_results(
            detection_result=result,
            image_path=original_path,
            annotated_path=annotated_path,
            site_location=site_location,
            camera_id=camera_id
        ) or []
        # Extract integer IDs from Violation ORM objects
        violation_db_ids = [v.id for v in stored if hasattr(v, 'id')]
    except Exception as e:
        logger.warning(f"Failed to store violations: {e}")

    # === Wire SAM callbacks to update DB when SAM finishes ===
    # Each SAM job gets a callback pointing to its DB violation record
    if result.get("sam_jobs"):
        from services.async_sam_verifier import get_async_sam_verifier
        async_sam = get_async_sam_verifier()

        for i, job_id in enumerate(result["sam_jobs"]):
            db_id = violation_db_ids[i] if i < len(violation_db_ids) else None
            callback = _make_sam_db_callback(
                violation_db_id=db_id,
                site_location=site_location or settings.default_site_location,
                camera_id=camera_id or settings.default_camera_id,
                original_path=original_path,
                annotated_path=annotated_path
            )
            # Attach callback to already-submitted job
            job = async_sam._jobs.get(job_id)
            if job:
                job.on_complete = callback

    sam_pending = result.get("sam_jobs_pending", 0)
    msg = "Detection completed"
    if sam_pending > 0:
        msg += f" (SAM verifying {sam_pending} person(s) in background)"

    return DetectionResponse(
        success=True,
        message=msg,
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
        logger.warning(f"Failed to store violations (base64): {e}")

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


@router.get("/tracking/stats")
async def get_tracking_stats():
    """
    Get violation tracking + async SAM statistics.

    Returns de-duplication metrics and SAM accuracy for thesis:
    - Total violations detected / stored / deduplicated
    - SAM false positives/negatives caught
    - Average SAM latency
    - YOLO accuracy rate
    """
    from services.violation_tracker import get_violation_tracker
    from services.async_sam_verifier import get_async_sam_verifier

    tracker = get_violation_tracker()
    async_sam = get_async_sam_verifier()

    return {
        "success": True,
        "tracking_stats": tracker.get_stats(),
        "async_sam_stats": async_sam.get_stats(),
        "config": {
            "cooldown_seconds": settings.violation_cooldown_seconds,
            "iou_threshold": settings.violation_iou_threshold,
            "track_timeout_seconds": settings.violation_track_timeout
        }
    }


@router.get("/sam/stats")
async def get_sam_stats():
    """
    Get async SAM verification statistics.

    Key thesis metrics:
    - jobs_submitted: How many SAM verifications were triggered
    - false_positives_caught: YOLO said violation, SAM corrected to SAFE
    - false_negatives_caught: YOLO missed, SAM caught
    - avg_sam_latency_ms: Average SAM processing time
    - yolo_accuracy_rate: % of YOLO results SAM agreed with
    - pending_jobs: SAM jobs currently running in background
    """
    from services.async_sam_verifier import get_async_sam_verifier
    async_sam = get_async_sam_verifier()

    stats = async_sam.get_stats()
    return {
        "success": True,
        "stats": stats,
        "interpretation": {
            "false_positives_caught": "YOLO over-detected, SAM corrected (reduces false alarms)",
            "false_negatives_caught": "YOLO missed, SAM caught (increases recall)",
            "yolo_accuracy_rate": "% of YOLO decisions SAM agreed with",
            "avg_sam_latency_ms": "SAM processing time (runs during cooldown window)"
        }
    }


@router.post("/tracking/reset")
async def reset_tracking():
    """
    Reset all violation tracking state.

    Use this when starting a new monitoring session or for testing.
    """
    from services.violation_tracker import get_violation_tracker

    tracker = get_violation_tracker()
    tracker.reset()

    return {
        "success": True,
        "message": "Tracking state reset successfully"
    }

