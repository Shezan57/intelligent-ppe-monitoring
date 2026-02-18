"""
API Response Models

Pydantic models for API response serialization.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PersonDetection(BaseModel):
    """
    Detection result for a single person.
    """
    person_id: int = Field(description="Unique ID for this person in the image")
    bbox: List[float] = Field(description="Bounding box [x_min, y_min, x_max, y_max]")
    confidence: float = Field(description="Detection confidence")
    has_helmet: bool = Field(description="Whether helmet was detected")
    has_vest: bool = Field(description="Whether vest was detected")
    is_violation: bool = Field(description="Whether this is a PPE violation")
    violation_type: Optional[str] = Field(
        default=None,
        description="Type: no_helmet, no_vest, both_missing, or null if safe"
    )
    decision_path: str = Field(
        description="Decision path used: Fast Safe, Fast Violation, Rescue Head, Rescue Body, Critical"
    )
    sam_activated: bool = Field(description="Whether SAM verification was used")


class TimingInfo(BaseModel):
    """
    Processing time breakdown.
    """
    total_ms: float = Field(description="Total processing time in milliseconds")
    yolo_ms: float = Field(description="YOLO detection time")
    sam_ms: float = Field(description="SAM verification time (0 if not used)")
    postprocess_ms: float = Field(description="Post-processing time")


class DetectionStats(BaseModel):
    """
    Summary statistics for detection.
    """
    total_persons: int = Field(description="Total persons detected")
    total_violations: int = Field(description="Number of violations")
    compliance_rate: float = Field(description="Compliance percentage (0-100)")
    sam_activations: int = Field(description="Number of SAM activations")
    bypass_rate: float = Field(description="Percentage that bypassed SAM")


class DetectionResponse(BaseModel):
    """
    Full detection response.
    """
    success: bool = Field(description="Whether detection succeeded")
    message: str = Field(description="Status message")
    image_path: Optional[str] = Field(default=None, description="Path to original image")
    annotated_image_path: Optional[str] = Field(
        default=None,
        description="Path to annotated image with bboxes"
    )
    persons: List[PersonDetection] = Field(
        default_factory=list,
        description="List of detected persons with PPE status"
    )
    timing: Optional[TimingInfo] = Field(
        default=None,
        description="Processing time breakdown"
    )
    stats: Optional[DetectionStats] = Field(
        default=None,
        description="Detection statistics"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Detection completed",
                "image_path": "/uploads/image_001.jpg",
                "annotated_image_path": "/uploads/image_001_annotated.jpg",
                "persons": [
                    {
                        "person_id": 1,
                        "bbox": [100, 50, 200, 350],
                        "confidence": 0.92,
                        "has_helmet": True,
                        "has_vest": True,
                        "is_violation": False,
                        "violation_type": None,
                        "decision_path": "Fast Safe",
                        "sam_activated": False
                    },
                    {
                        "person_id": 2,
                        "bbox": [300, 60, 400, 360],
                        "confidence": 0.88,
                        "has_helmet": False,
                        "has_vest": True,
                        "is_violation": True,
                        "violation_type": "no_helmet",
                        "decision_path": "Fast Violation",
                        "sam_activated": False
                    }
                ],
                "timing": {
                    "total_ms": 45.2,
                    "yolo_ms": 35.1,
                    "sam_ms": 0.0,
                    "postprocess_ms": 10.1
                },
                "stats": {
                    "total_persons": 2,
                    "total_violations": 1,
                    "compliance_rate": 50.0,
                    "sam_activations": 0,
                    "bypass_rate": 100.0
                }
            }
        }


class ViolationResponse(BaseModel):
    """
    Violation session record from database.

    Each row = 1 violation SESSION (not 1 detection).
    A session starts when a worker first violates and is updated
    every cooldown cycle until they become compliant or leave.
    """
    id: int
    timestamp: datetime
    site_location: str
    camera_id: str
    violation_type: str
    has_helmet: bool
    has_vest: bool
    decision_path: str
    detection_confidence: Optional[float]
    sam_activated: bool
    processing_time_ms: Optional[float]
    original_image_path: Optional[str]
    annotated_image_path: Optional[str]
    report_sent: bool
    # Session tracking fields
    session_start: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    occurrence_count: int = 1
    total_duration_minutes: float = 0.0
    is_active_session: bool = True


class HistoryResponse(BaseModel):
    """
    Response for history query.
    """
    success: bool
    total_count: int = Field(description="Total number of violations matching query")
    returned_count: int = Field(description="Number of violations in this response")
    violations: List[ViolationResponse]
    summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary statistics if requested"
    )


class UploadResponse(BaseModel):
    """
    Response for file upload.
    """
    success: bool
    message: str
    file_path: str = Field(description="Path to uploaded file")
    detection_result: Optional[DetectionResponse] = Field(
        default=None,
        description="Detection result if run_detection was True"
    )


class ReportResponse(BaseModel):
    """
    Response for report generation.
    """
    success: bool
    message: str
    report_date: str
    pdf_path: Optional[str] = Field(default=None, description="Path to generated PDF")
    email_sent: bool = Field(default=False)
    total_violations: int = Field(default=0)
    compliance_rate: float = Field(default=100.0)


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(description="Service status: healthy, degraded, unhealthy")
    version: str = Field(description="API version")
    yolo_loaded: bool = Field(description="Whether YOLO model is loaded")
    sam_loaded: bool = Field(description="Whether SAM model is loaded")
    database_connected: bool = Field(description="Whether database is connected")
