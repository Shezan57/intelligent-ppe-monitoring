"""
Video Detection API Routes

Endpoints for video file upload and webcam detection.
"""

import os
import uuid
import tempfile
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np

from services.stream_processor import get_stream_processor, frame_to_base64
from services.hybrid_detector import get_hybrid_detector
from utils.visualization import draw_detections

router = APIRouter(prefix="/api", tags=["video"])


ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB


@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    frame_skip: int = 5
):
    """
    Process an uploaded video file for PPE detection.
    
    Args:
        file: Video file (.mp4, .avi, .mov)
        frame_skip: Process every Nth frame (default: 5)
        
    Returns:
        Aggregated detection results for entire video
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Save uploaded file temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    input_path = os.path.join(uploads_dir, f"video_{timestamp}_{unique_id}{ext}")
    output_path = os.path.join(uploads_dir, f"video_{timestamp}_{unique_id}_annotated.mp4")
    
    try:
        # Save uploaded file
        contents = await file.read()
        if len(contents) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=400, detail="Video file too large (max 100MB)")
        
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        # Process video
        processor = get_stream_processor()
        processor.frame_skip = frame_skip
        
        result = processor.process_video_file(
            video_path=input_path,
            output_path=output_path
        )
        
        # Add paths to result - use actual output path from processing
        result["input_video_path"] = input_path
        actual_output = result.get("output_video_path", output_path)
        result["output_video_url"] = f"/uploads/{os.path.basename(actual_output)}"
        
        return result
        
    except Exception as e:
        # Cleanup on error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@router.post("/detect/video/pipeline")
async def detect_video_pipeline(
    file: UploadFile = File(...),
    cooldown_seconds: float = 300.0,
    camera_zone: str = "CAM-001",
):
    """
    Process video through the decoupled Sentry-Judge pipeline.

    Flow: Sentry (YOLO+ByteTrack) → Queue → Judge (SAM 3) → DB

    Returns combined Sentry + Judge results with verified violations.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]

    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    roi_dir = os.path.join(uploads_dir, f"rois_{timestamp}_{unique_id}")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)

    input_path = os.path.join(uploads_dir, f"video_{timestamp}_{unique_id}{ext}")
    output_path = os.path.join(uploads_dir, f"pipeline_{timestamp}_{unique_id}_annotated.mp4")

    try:
        contents = await file.read()
        if len(contents) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=400, detail="Video file too large (max 100MB)")

        with open(input_path, 'wb') as f:
            f.write(contents)

        # Run the Sentry-Judge pipeline
        from run_pipeline import run_pipeline
        from database.connection import SessionLocal
        from database.models import VerifiedViolation

        results = run_pipeline(
            video_path=input_path,
            output_path=output_path,
            cooldown_seconds=cooldown_seconds,
            roi_save_dir=roi_dir,
            camera_zone=camera_zone,
        )

        # Fetch verified violations from DB
        session = SessionLocal()
        verified = session.query(VerifiedViolation).order_by(
            VerifiedViolation.timestamp.desc()
        ).limit(50).all()

        verified_list = [
            {
                "id": v.id,
                "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                "person_id": v.person_id,
                "violation_type": v.violation_type,
                "image_path": v.image_path,
                "camera_zone": v.camera_zone,
                "judge_confidence": v.judge_confidence,
                "decision_path": v.decision_path,
            }
            for v in verified
        ]
        session.close()

        # Build response
        sentry = results.get("sentry", {})
        judge = results.get("judge", {})

        return {
            "pipeline": True,
            "output_video_url": f"/uploads/{os.path.basename(output_path)}" if os.path.exists(output_path) else None,
            "sentry": {
                "frames_processed": sentry.get("frames_processed", 0),
                "effective_fps": round(sentry.get("effective_fps", 0), 1),
                "avg_latency_ms": round(sentry.get("avg_latency_ms", 0), 1),
                "unique_persons": sentry.get("unique_persons", 0),
                "violations_queued": sentry.get("violations_queued", 0),
                "cooldown_skipped": sentry.get("violations_cooldown_skipped", 0),
                "safe_count": sentry.get("safe_count", 0),
                "filtered_false_persons": sentry.get("filtered_false_persons", 0),
                "path_distribution": sentry.get("path_distribution", {}),
            },
            "judge": {
                "total_processed": judge.get("total_processed", 0),
                "confirmed": judge.get("confirmed", 0),
                "rejected": judge.get("rejected", 0),
                "not_person_rejected": judge.get("not_person_rejected", 0),
                "avg_time_ms": round(judge.get("avg_time_ms", 0), 1),
                "confirmation_rate": round(judge.get("confirmation_rate", 0), 1),
                "sam_mock_mode": judge.get("sam_mock_mode", True),
            },
            "bypass_rate": round(
                (sentry.get("safe_count", 0) / max(sentry.get("total_detections", 1), 1)) * 100, 1
            ),
            "total_time_seconds": round(results.get("total_time_seconds", 0), 1),
            "verified_violations": verified_list,
        }

    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@router.post("/detect/webcam")
async def detect_webcam_frame(camera_index: int = 0):
    """
    Capture and process a single frame from webcam.
    
    Args:
        camera_index: Webcam device index (default: 0)
        
    Returns:
        Detection result with base64 annotated image
    """
    try:
        processor = get_stream_processor()
        result = processor.capture_webcam_frame(camera_index)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webcam capture failed: {str(e)}")


@router.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket, camera_index: int = 0):
    """
    WebSocket endpoint for real-time webcam detection.
    
    Streams detection results as JSON messages.
    Client can send 'stop' to end the stream.
    """
    await websocket.accept()
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        await websocket.send_json({"error": "Could not open webcam"})
        await websocket.close()
        return
    
    detector = get_hybrid_detector()
    frame_count = 0
    frame_skip = 3  # Process every 3rd frame for real-time performance
    
    try:
        while True:
            # Check for client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.01
                )
                if data == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            # Run detection
            result = detector.detect(frame, save_annotated=False)
            annotated = draw_detections(frame, result)
            
            # Send result
            await websocket.send_json({
                "frame_number": frame_count,
                "annotated_base64": frame_to_base64(annotated, quality=70),
                "persons": result["persons"],
                "stats": result["stats"],
                "timing": result["timing"]
            })
            
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        try:
            await websocket.close()
        except:
            pass


# Import asyncio for WebSocket
import asyncio
