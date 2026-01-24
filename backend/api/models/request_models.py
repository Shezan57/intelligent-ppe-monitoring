"""
API Request Models

Pydantic models for API request validation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """
    Request model for detection endpoint.
    
    Used when submitting image data directly (base64) or 
    referencing an already uploaded file.
    """
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image data"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Path to uploaded image file"
    )
    site_location: Optional[str] = Field(
        default=None,
        description="Site location for violation tracking"
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="Camera ID for violation tracking"
    )
    save_annotated: bool = Field(
        default=True,
        description="Whether to save annotated image"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_path": "/uploads/image_001.jpg",
                "site_location": "Construction Site A",
                "camera_id": "CAM-001",
                "save_annotated": True
            }
        }


class UploadRequest(BaseModel):
    """
    Metadata for file upload.
    
    Sent along with multipart form file upload.
    """
    site_location: Optional[str] = Field(
        default=None,
        description="Site location identifier"
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="Camera ID"
    )
    run_detection: bool = Field(
        default=True,
        description="Whether to run detection automatically after upload"
    )


class HistoryQuery(BaseModel):
    """
    Query parameters for history endpoint.
    """
    start_date: Optional[str] = Field(
        default=None,
        description="Start date (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date (YYYY-MM-DD)"
    )
    violation_type: Optional[str] = Field(
        default=None,
        description="Filter by violation type: no_helmet, no_vest, both_missing"
    )
    camera_id: Optional[str] = Field(
        default=None,
        description="Filter by camera ID"
    )
    site_location: Optional[str] = Field(
        default=None,
        description="Filter by site location"
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of results"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Pagination offset"
    )


class ReportRequest(BaseModel):
    """
    Request model for manual report generation.
    """
    report_date: Optional[str] = Field(
        default=None,
        description="Date for report (YYYY-MM-DD), defaults to today"
    )
    send_email: bool = Field(
        default=True,
        description="Whether to send email after generation"
    )
    recipients: Optional[List[str]] = Field(
        default=None,
        description="Override default recipients"
    )
