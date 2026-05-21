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
            cropped_roi_path=getattr(v, 'cropped_roi_path', None),
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

    Aggregates from BOTH tables:
    - violations      → image/basic-video detection results
    - verified_violations → Sentry-Judge pipeline results

    Returns:
        Summary with counts, compliance rates, trends
    """
    from sqlalchemy import func
    from database.models import VerifiedViolation, DailyReport

    start_date = datetime.now() - timedelta(days=days)

    # ── violations table (image / basic-video detection) ──────────────
    v_total = db.query(Violation).filter(Violation.timestamp >= start_date).count()

    v_by_type = db.query(
        Violation.violation_type,
        func.count(Violation.id).label("count")
    ).filter(Violation.timestamp >= start_date).group_by(Violation.violation_type).all()
    v_type = {t: c for t, c in v_by_type}

    v_sam = db.query(func.count(Violation.id)).filter(
        Violation.timestamp >= start_date,
        Violation.sam_activated == True
    ).scalar() or 0

    v_camera = db.query(
        Violation.camera_id,
        func.count(Violation.id).label("count")
    ).filter(Violation.timestamp >= start_date).group_by(Violation.camera_id).all()
    camera_counts = {cam: cnt for cam, cnt in v_camera}

    # ── verified_violations table (Sentry-Judge pipeline) ─────────────
    vv_total = db.query(VerifiedViolation).filter(
        VerifiedViolation.timestamp >= start_date
    ).count()

    vv_by_type = db.query(
        VerifiedViolation.violation_type,
        func.count(VerifiedViolation.id).label("count")
    ).filter(VerifiedViolation.timestamp >= start_date).group_by(VerifiedViolation.violation_type).all()
    vv_type = {t: c for t, c in vv_by_type}

    # ── Combine both tables ────────────────────────────────────────────
    total_violations = v_total + vv_total
    no_helmet_count  = v_type.get("no_helmet", 0)  + vv_type.get("no_helmet", 0)
    no_vest_count    = v_type.get("no_vest", 0)    + vv_type.get("no_vest", 0)
    both_missing_count = v_type.get("both_missing", 0) + vv_type.get("both_missing", 0)
    # Every row in verified_violations passed SAM; v_sam tracks SAM in basic detection
    sam_activations  = v_sam + vv_total

    # ── Compliance rate ────────────────────────────────────────────────
    # Prefer the most recent daily report value; fall back to null
    compliance_rate = None
    try:
        latest = db.query(DailyReport).order_by(DailyReport.report_date.desc()).first()
        if latest:
            compliance_rate = round(float(latest.compliance_rate), 1)
    except Exception:
        pass

    # ── Daily trend (combined) ─────────────────────────────────────────
    v_daily = db.query(
        func.date(Violation.timestamp).label("date"),
        func.count(Violation.id).label("count")
    ).filter(Violation.timestamp >= start_date).group_by(func.date(Violation.timestamp)).all()

    vv_daily = db.query(
        func.date(VerifiedViolation.timestamp).label("date"),
        func.count(VerifiedViolation.id).label("count")
    ).filter(VerifiedViolation.timestamp >= start_date).group_by(func.date(VerifiedViolation.timestamp)).all()

    trend_map: dict = {}
    for d, c in v_daily:
        trend_map[str(d)] = trend_map.get(str(d), 0) + c
    for d, c in vv_daily:
        trend_map[str(d)] = trend_map.get(str(d), 0) + c
    daily_trend = [{"date": k, "count": v} for k, v in sorted(trend_map.items())]

    return {
        "success": True,
        "period_days": days,
        # Flat fields the frontend StatCards read directly
        "total_violations": total_violations,
        "no_helmet_count": no_helmet_count,
        "no_vest_count": no_vest_count,
        "both_missing_count": both_missing_count,
        "sam_activations": sam_activations,
        "compliance_rate": compliance_rate,
        # Legacy nested structure kept for backward compat
        "by_type": {
            "no_helmet": no_helmet_count,
            "no_vest": no_vest_count,
            "both_missing": both_missing_count,
        },
        "by_camera": camera_counts,
        "daily_trend": daily_trend,
    }


@router.get("/history/verified")
async def get_verified_violations(
    violation_type: Optional[str] = Query(default=None),
    camera_zone: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Get Judge-confirmed verified violations.

    These are only violations that passed SAM 3 verification.
    """
    from database.models import VerifiedViolation

    query = db.query(VerifiedViolation)

    if violation_type:
        query = query.filter(VerifiedViolation.violation_type == violation_type)
    if camera_zone:
        query = query.filter(VerifiedViolation.camera_zone == camera_zone)

    total = query.count()
    violations = query.order_by(desc(VerifiedViolation.timestamp)).offset(offset).limit(limit).all()

    return {
        "success": True,
        "total_count": total,
        "returned_count": len(violations),
        "violations": [
            {
                "id": v.id,
                "timestamp": v.timestamp.isoformat() if v.timestamp else None,
                "person_id": v.person_id,
                "violation_type": v.violation_type,
                "image_path": v.image_path,
                "camera_zone": v.camera_zone,
                "judge_confirmed": v.judge_confirmed,
                "judge_confidence": v.judge_confidence,
                "judge_processing_time_ms": v.judge_processing_time_ms,
                "sentry_confidence": v.sentry_confidence,
                "decision_path": v.decision_path,
                "person_bbox": v.person_bbox,
            }
            for v in violations
        ],
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
        cropped_roi_path=getattr(violation, 'cropped_roi_path', None),
        report_sent=violation.report_sent
    )
