"""
Daily Reporter Agent

Automated daily report generation and email delivery.
Runs as a scheduled background task at end of day.
"""

import json
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.models import Violation, DailyReport
from config.settings import settings


class DailyReporter:
    """
    Scheduled agent for daily report generation and delivery.
    
    Uses APScheduler to run at a configured time (default 23:59)
    to generate PDF reports and email them to managers.
    
    Workflow:
    1. Fetch unreported violations for the day
    2. Generate PDF report with statistics and evidence
    3. Send email with PDF attachment
    4. Mark violations as reported
    5. Create DailyReport record
    
    Attributes:
        scheduler: APScheduler background scheduler
        report_time: Time to run daily (HH:MM)
    """
    
    def __init__(self):
        """Initialize the daily reporter."""
        self.scheduler = BackgroundScheduler()
        self.report_time = settings.report_time
        self._is_running = False
    
    def start(self) -> None:
        """
        Start the scheduled reporter.
        
        Adds a cron job to run at the configured time daily.
        """
        if self._is_running:
            print("⚠️ Daily reporter already running")
            return
        
        # Parse time
        try:
            hour, minute = map(int, self.report_time.split(":"))
        except ValueError:
            hour, minute = 23, 59
        
        # Add daily job
        self.scheduler.add_job(
            self.generate_and_send_report,
            trigger='cron',
            hour=hour,
            minute=minute,
            id='daily_ppe_report',
            replace_existing=True
        )
        
        self.scheduler.start()
        self._is_running = True
        print(f"✅ Daily reporter started (runs at {hour:02d}:{minute:02d})")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            print("✅ Daily reporter stopped")
    
    def generate_and_send_report(
        self,
        target_date: Optional[date] = None,
        send_email: bool = True
    ) -> Dict[str, Any]:
        """
        Generate and send daily report.
        
        This is the main entry point, called by scheduler or manually.
        
        Args:
            target_date: Date to report on (defaults to today)
            send_email: Whether to send email (can be disabled for testing)
            
        Returns:
            Report summary dict
        """
        if target_date is None:
            target_date = date.today()
        
        print(f"📊 Generating report for {target_date}")
        
        # Get database session
        db = SessionLocal()
        
        try:
            # Fetch violations
            violations = db.query(Violation).filter(
                Violation.report_date == target_date,
                Violation.report_sent == False
            ).all()
            
            if not violations:
                print("ℹ️ No violations to report")
                return {
                    "success": True,
                    "message": "No violations to report",
                    "report_date": str(target_date),
                    "total_violations": 0
                }
            
            # Calculate statistics
            total_detections = len(violations)
            stats = self._calculate_stats(violations)
            
            # Generate PDF
            pdf_path = self._generate_pdf(target_date, violations, stats)
            
            # Send email if configured
            email_sent = False
            if send_email and settings.sender_email:
                email_sent = self._send_email(target_date, pdf_path, stats)
            
            # Mark violations as reported
            for violation in violations:
                violation.report_sent = True
            
            # Create report record
            report = DailyReport(
                report_date=target_date,
                total_detections=total_detections,
                total_violations=stats["total_violations"],
                compliance_rate=stats["compliance_rate"],
                pdf_path=pdf_path,
                email_sent=email_sent,
                email_sent_at=datetime.now() if email_sent else None,
                recipients=json.dumps(settings.get_manager_emails_list())
            )
            db.add(report)
            db.commit()
            
            print(f"✅ Report completed for {target_date}")
            
            return {
                "success": True,
                "message": "Report generated successfully",
                "report_date": str(target_date),
                "pdf_path": pdf_path,
                "email_sent": email_sent,
                "total_violations": stats["total_violations"],
                "compliance_rate": stats["compliance_rate"]
            }
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            db.rollback()
            return {
                "success": False,
                "message": str(e),
                "report_date": str(target_date)
            }
        
        finally:
            db.close()
    
    def _calculate_stats(self, violations: List[Violation]) -> Dict[str, Any]:
        """Calculate session-based report statistics."""
        total_sessions = len(violations)  # 1 row = 1 session (not 1 detection!)

        no_helmet = sum(1 for v in violations if v.violation_type == "no_helmet")
        no_vest = sum(1 for v in violations if v.violation_type == "no_vest")
        both_missing = sum(1 for v in violations if v.violation_type == "both_missing")

        # Count compliant sessions
        compliant = sum(1 for v in violations if v.has_helmet and v.has_vest)
        total_violations = total_sessions - compliant

        compliance_rate = (compliant / total_sessions * 100) if total_sessions > 0 else 100.0

        # SAM activation stats
        sam_activations = sum(1 for v in violations if v.sam_activated)

        # === Session-based metrics (thesis contribution) ===
        # Total re-detections across all sessions
        total_occurrences = sum(
            (v.occurrence_count or 1) for v in violations
        )
        # Total violation time across all sessions
        total_duration_minutes = sum(
            (v.total_duration_minutes or 0.0) for v in violations
        )
        # Average session duration
        avg_duration_minutes = (
            total_duration_minutes / total_violations
            if total_violations > 0 else 0.0
        )
        # Longest single violation session
        longest_session_minutes = max(
            (v.total_duration_minutes or 0.0) for v in violations
        ) if violations else 0.0

        return {
            # Basic counts
            "total_sessions": total_sessions,
            "total_detections": total_sessions,       # kept for compatibility
            "total_violations": total_violations,
            "compliance_rate": compliance_rate,
            "no_helmet_count": no_helmet,
            "no_vest_count": no_vest,
            "both_missing_count": both_missing,
            "sam_activations": sam_activations,
            # Session-based metrics
            "total_occurrences": total_occurrences,
            "total_duration_minutes": round(total_duration_minutes, 1),
            "total_duration_hours": round(total_duration_minutes / 60, 2),
            "avg_session_duration_minutes": round(avg_duration_minutes, 1),
            "longest_session_minutes": round(longest_session_minutes, 1),
        }
    
    def _generate_pdf(
        self,
        report_date: date,
        violations: List[Violation],
        stats: Dict[str, Any]
    ) -> str:
        """
        Generate PDF report.
        
        Uses ReportLab for PDF generation.
        """
        import os
        
        try:
            from services.report_generator import ReportGenerator
            
            generator = ReportGenerator(output_dir=settings.report_output_dir)
            pdf_path = generator.generate_daily_report(report_date, violations, stats)
            return pdf_path
            
        except ImportError:
            # Fallback: create simple text file
            print("⚠️ ReportLab not available, creating text report")
            
            reports_dir = settings.report_output_dir
            os.makedirs(reports_dir, exist_ok=True)
            
            filename = f"PPE_Report_{report_date.strftime('%Y-%m-%d')}.txt"
            filepath = os.path.join(reports_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(f"PPE Compliance Report - {report_date}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Detections: {stats['total_detections']}\n")
                f.write(f"Total Violations: {stats['total_violations']}\n")
                f.write(f"Compliance Rate: {stats['compliance_rate']:.1f}%\n\n")
                f.write(f"No Helmet: {stats['no_helmet_count']}\n")
                f.write(f"No Vest: {stats['no_vest_count']}\n")
                f.write(f"Both Missing: {stats['both_missing_count']}\n")
            
            return filepath
    
    def _send_email(
        self,
        report_date: date,
        pdf_path: str,
        stats: Dict[str, Any]
    ) -> bool:
        """
        Send email with PDF attachment.
        """
        try:
            from services.email_service import EmailService
            
            email_service = EmailService()
            
            success = email_service.send_daily_report(
                recipients=settings.get_manager_emails_list(),
                report_date=report_date,
                pdf_path=pdf_path,
                summary_stats=stats
            )
            
            return success
            
        except Exception as e:
            print(f"⚠️ Email sending failed: {e}")
            return False


# Global instance
_reporter_instance: Optional[DailyReporter] = None


def get_daily_reporter() -> DailyReporter:
    """Get or create the global daily reporter instance."""
    global _reporter_instance
    
    if _reporter_instance is None:
        _reporter_instance = DailyReporter()
    
    return _reporter_instance
