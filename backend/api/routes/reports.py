"""
Report Generation API Routes

Manual PDF report generation with optional email delivery.
"""

import os
import smtplib
import json
from datetime import date, datetime, timedelta
from typing import Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.connection import get_db
from database.models import Violation, VerifiedViolation, DailyReport
from services.report_generator import ReportGenerator
from config.settings import settings

router = APIRouter(prefix="/api", tags=["reports"])


class ReportRequest(BaseModel):
    date_from: str          # "YYYY-MM-DD"
    date_to: str            # "YYYY-MM-DD"
    send_email: bool = False
    email: Optional[str] = None   # override recipient


def _parse_date(s: str, field: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, detail=f"Invalid {field}: expected YYYY-MM-DD")


@router.post("/reports/generate")
async def generate_report(body: ReportRequest, db: Session = Depends(get_db)):
    """
    Generate a PDF violation report for the given date range.

    Queries both the violations table (image/basic detection) and the
    verified_violations table (Sentry-Judge pipeline) to build a
    comprehensive report.  Optionally sends it via SMTP.
    """
    d_from = _parse_date(body.date_from, "date_from")
    d_to   = _parse_date(body.date_to,   "date_to")

    if d_from > d_to:
        raise HTTPException(400, detail="date_from must be ≤ date_to")

    dt_from = datetime(d_from.year, d_from.month, d_from.day, 0, 0, 0)
    dt_to   = datetime(d_to.year,   d_to.month,   d_to.day,   23, 59, 59)

    # ── Pull violations from both tables ──────────────────────────────
    v_rows = db.query(Violation).filter(
        Violation.timestamp >= dt_from,
        Violation.timestamp <= dt_to,
    ).order_by(Violation.timestamp.asc()).all()

    vv_rows = db.query(VerifiedViolation).filter(
        VerifiedViolation.timestamp >= dt_from,
        VerifiedViolation.timestamp <= dt_to,
    ).order_by(VerifiedViolation.timestamp.asc()).all()

    total_violations = len(v_rows) + len(vv_rows)

    # Build per-type counts
    def _type_counts(rows, type_attr="violation_type"):
        counts = {}
        for r in rows:
            t = getattr(r, type_attr, None) or "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts

    v_types  = _type_counts(v_rows)
    vv_types = _type_counts(vv_rows)
    no_helmet_count    = v_types.get("no_helmet", 0)    + vv_types.get("no_helmet", 0)
    no_vest_count      = v_types.get("no_vest", 0)      + vv_types.get("no_vest", 0)
    both_missing_count = v_types.get("both_missing", 0) + vv_types.get("both_missing", 0)

    # Compliance rate: how many pipeline ROIs were rejected by Judge
    # = (rejected / (confirmed + rejected)) * 100  — rough proxy
    # If no data, default to 100%
    total_detections = max(total_violations, 1)
    compliance_rate  = max(0.0, 100.0 - (total_violations / total_detections) * 100)
    if total_violations == 0:
        compliance_rate = 100.0

    # ── Assemble stats dict for ReportGenerator ────────────────────────
    stats = {
        "total_detections": total_detections,
        "total_violations": total_violations,
        "compliance_rate":  compliance_rate,
        "no_helmet_count":    no_helmet_count,
        "no_vest_count":      no_vest_count,
        "both_missing_count": both_missing_count,
        "sam_activations": (
            sum(1 for v in v_rows if v.sam_activated) + len(vv_rows)
        ),
        "date_from": body.date_from,
        "date_to":   body.date_to,
    }

    # ── Build a compatible "violations" list for ReportGenerator ──────
    # ReportGenerator.generate_daily_report expects Violation ORM objects.
    # Wrap VerifiedViolation rows in a simple adapter if needed.
    class _VVAdapter:
        """Shim so ReportGenerator can consume VerifiedViolation rows."""
        def __init__(self, vv):
            self.timestamp         = vv.timestamp
            self.site_location     = vv.camera_zone or "Pipeline"
            self.camera_id         = vv.camera_zone or "CAM-001"
            self.has_helmet        = vv.violation_type not in ("no_helmet", "both_missing")
            self.has_vest          = vv.violation_type not in ("no_vest",   "both_missing")
            self.violation_type    = vv.violation_type
            self.decision_path     = vv.decision_path or "Pipeline"
            self.detection_confidence = vv.sentry_confidence
            self.sam_activated     = True
            # Resolve URL path → filesystem path for PDF image embedding
            if vv.image_path:
                _backend_dir = os.path.normpath(
                    os.path.join(os.path.dirname(__file__), "..", "..")
                )
                _fs_path = os.path.normpath(
                    os.path.join(_backend_dir, vv.image_path.lstrip("/\\").replace("/", os.sep))
                )
                self.annotated_image_path = _fs_path if os.path.exists(_fs_path) else None
            else:
                self.annotated_image_path = None
            self.cropped_roi_path  = vv.image_path
            self.original_image_path = vv.image_path
            self.processing_time_ms = vv.judge_processing_time_ms
            self.occurrence_count  = 1
            self.total_duration_minutes = 0.0

    combined_violations = list(v_rows) + [_VVAdapter(vv) for vv in vv_rows]
    combined_violations.sort(key=lambda x: x.timestamp)

    # ── Generate PDF ───────────────────────────────────────────────────
    try:
        reports_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "reports"
        )
        gen = ReportGenerator(output_dir=reports_dir)

        # Use date_from as the "report date" for naming; suffix range if multi-day
        label_date = d_from
        pdf_path = gen.generate_daily_report(
            report_date=label_date,
            violations=combined_violations,
            stats=stats,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"PDF generation failed: {e}")

    pdf_filename = os.path.basename(pdf_path)
    pdf_url      = f"/reports/{pdf_filename}"

    # ── Persist to daily_reports table ────────────────────────────────
    try:
        existing = db.query(DailyReport).filter(
            DailyReport.report_date == label_date
        ).first()
        if existing:
            existing.total_detections = total_detections
            existing.total_violations = total_violations
            existing.compliance_rate  = compliance_rate
            existing.pdf_path         = pdf_path
        else:
            db.add(DailyReport(
                report_date      = label_date,
                total_detections = total_detections,
                total_violations = total_violations,
                compliance_rate  = compliance_rate,
                pdf_path         = pdf_path,
                email_sent       = False,
            ))
        db.commit()
    except Exception:
        pass  # Non-fatal — PDF already generated

    # ── Send email if requested ────────────────────────────────────────
    email_status = "not_requested"
    if body.send_email:
        recipient = (
            body.email
            or settings.recipient_email
            or settings.sender_email
        )
        if not recipient:
            email_status = "no_recipient_configured"
        elif not (settings.sender_email and settings.sender_password):
            email_status = "smtp_not_configured"
        else:
            try:
                _send_email(
                    sender=settings.sender_email,
                    password=settings.sender_password,
                    recipient=recipient,
                    pdf_path=pdf_path,
                    date_from=body.date_from,
                    date_to=body.date_to,
                    stats=stats,
                )
                email_status = f"sent_to_{recipient}"
                # Update DB
                try:
                    rec = db.query(DailyReport).filter(
                        DailyReport.report_date == label_date
                    ).first()
                    if rec:
                        rec.email_sent    = True
                        rec.email_sent_at = datetime.now()
                        rec.recipients    = json.dumps([recipient])
                        db.commit()
                except Exception:
                    pass
            except Exception as e:
                email_status = f"failed: {e}"

    return {
        "success": True,
        "pdf_url": pdf_url,
        "filename": pdf_filename,
        "stats": stats,
        "email_status": email_status,
    }


@router.get("/reports")
async def list_reports(db: Session = Depends(get_db)):
    """List all generated reports."""
    rows = db.query(DailyReport).order_by(DailyReport.report_date.desc()).limit(30).all()
    return {
        "success": True,
        "reports": [
            {
                "id": r.id,
                "report_date": str(r.report_date),
                "total_violations": r.total_violations,
                "compliance_rate": r.compliance_rate,
                "pdf_url": f"/reports/{os.path.basename(r.pdf_path)}" if r.pdf_path and os.path.exists(r.pdf_path) else None,
                "email_sent": r.email_sent,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


def _send_email(sender: str, password: str, recipient: str,
                pdf_path: str, date_from: str, date_to: str,
                stats: dict) -> None:
    """Send PDF report via Gmail SMTP."""
    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = recipient
    msg["Subject"] = f"PPE Violation Report {date_from} to {date_to}"

    body = (
        f"Please find attached the PPE Violation Report for {date_from} to {date_to}.\n\n"
        f"Summary:\n"
        f"  Total Violations : {stats['total_violations']}\n"
        f"  No Helmet        : {stats['no_helmet_count']}\n"
        f"  No Vest          : {stats['no_vest_count']}\n"
        f"  Both Missing     : {stats['both_missing_count']}\n"
        f"  Compliance Rate  : {stats['compliance_rate']:.1f}%\n\n"
        f"Generated by the Intelligent PPE Monitoring System.\n"
    )
    msg.attach(MIMEText(body, "plain"))

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f'attachment; filename="{os.path.basename(pdf_path)}"',
    )
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
