"""
History API Routes

Endpoints for retrieving violation history.
"""

from typing import Optional, List
from datetime import datetime, date, timedelta

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from api.models.response_models import HistoryResponse, ViolationResponse
from database.connection import get_db
from database.models import Violation


router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history", response_model=HistoryResponse)
async def get_violation_history(
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    violation_type: Optional[str] = Query(default=None, description="Filter: no_helmet, no_vest, both_missing"),
    camera_id: Optional[str] = Query(default=None, description="Filter by camera ID"),
    site_location: Optional[str] = Query(default=None, description="Filter by site"),
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db)
):
    """
    Get violation history with optional filters.
    
    Returns paginated list of violations from the database.
    """
    # Build query
    query = db.query(Violation)
    
    # Apply filters
    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(Violation.timestamp >= start)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            # Add one day to include the end date fully
            end = end + timedelta(days=1)
            query = query.filter(Violation.timestamp < end)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format")
    
    if violation_type:
        if violation_type not in ["no_helmet", "no_vest", "both_missing"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid violation_type. Use: no_helmet, no_vest, or both_missing"
            )
        query = query.filter(Violation.violation_type == violation_type)
    
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    
    if site_location:
        query = query.filter(Violation.site_location == site_location)
    
    # Get total count before pagination
    total_count = query.count()
    
    # Apply pagination
    violations = query.order_by(desc(Violation.timestamp)).offset(offset).limit(limit).all()
    
    # Convert to response model
    violation_responses = [
        ViolationResponse(
            id=v.id,
            timestamp=v.timestamp,
            site_location=v.site_location,
            camera_id=v.camera_id,
            violation_type=v.violation_type,
            has_helmet=v.has_helmet,
            has_vest=v.has_vest,
            decision_path=v.decision_path,
            detection_confidence=v.detection_confidence,
            sam_activated=v.sam_activated,
            processing_time_ms=v.processing_time_ms,
            original_image_path=v.original_image_path,
            annotated_image_path=v.annotated_image_path,
            report_sent=v.report_sent,
            # Session fields
            session_start=v.session_start,
            last_seen=v.last_seen,
            occurrence_count=v.occurrence_count or 1,
            total_duration_minutes=v.total_duration_minutes or 0.0,
            is_active_session=v.is_active_session if v.is_active_session is not None else True
        )
        for v in violations
    ]
    
    return HistoryResponse(
        success=True,
        total_count=total_count,
        returned_count=len(violation_responses),
        violations=violation_responses
    )


@router.get("/history/summary")
async def get_history_summary(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to summarize"),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for recent violations.
    
    Returns:
        Summary with counts, compliance rates, trends
    """
    from sqlalchemy import func
    
    start_date = datetime.now() - timedelta(days=days)
    
    # Total violations
    total_violations = db.query(Violation).filter(
        Violation.timestamp >= start_date
    ).count()
    
    # By type
    by_type = db.query(
        Violation.violation_type,
        func.count(Violation.id).label("count")
    ).filter(
        Violation.timestamp >= start_date
    ).group_by(Violation.violation_type).all()
    
    type_counts = {t: c for t, c in by_type}
    
    # By camera
    by_camera = db.query(
        Violation.camera_id,
        func.count(Violation.id).label("count")
    ).filter(
        Violation.timestamp >= start_date
    ).group_by(Violation.camera_id).all()
    
    camera_counts = {cam: cnt for cam, cnt in by_camera}
    
    # Daily trend
    daily = db.query(
        func.date(Violation.timestamp).label("date"),
        func.count(Violation.id).label("count")
    ).filter(
        Violation.timestamp >= start_date
    ).group_by(func.date(Violation.timestamp)).all()
    
    daily_trend = [{"date": str(d), "count": c} for d, c in daily]
    
    return {
        "success": True,
        "period_days": days,
        "total_violations": total_violations,
        "by_type": type_counts,
        "by_camera": camera_counts,
        "daily_trend": daily_trend
    }


@router.get("/history/{violation_id}", response_model=ViolationResponse)
async def get_violation_detail(
    violation_id: int,
    db: Session = Depends(get_db)
):
    """
    Get details for a specific violation.
    """
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return ViolationResponse(
        id=violation.id,
        timestamp=violation.timestamp,
        site_location=violation.site_location,
        camera_id=violation.camera_id,
        violation_type=violation.violation_type,
        has_helmet=violation.has_helmet,
        has_vest=violation.has_vest,
        decision_path=violation.decision_path,
        detection_confidence=violation.detection_confidence,
        sam_activated=violation.sam_activated,
        processing_time_ms=violation.processing_time_ms,
        original_image_path=violation.original_image_path,
        annotated_image_path=violation.annotated_image_path,
        report_sent=violation.report_sent
    )
