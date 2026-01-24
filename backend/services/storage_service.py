"""
Storage Service

Database operations for violations and reports.
Provides CRUD operations and query helpers.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from database.models import Violation, DailyReport


class StorageService:
    """
    Database storage service for violations and reports.
    
    Provides convenient methods for common database operations.
    """
    
    def __init__(self, db: Session):
        """
        Initialize storage service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
    
    # === Violation Operations ===
    
    def get_violation_by_id(self, violation_id: int) -> Optional[Violation]:
        """Get a single violation by ID."""
        return self.db.query(Violation).filter(
            Violation.id == violation_id
        ).first()
    
    def get_violations(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        violation_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        site_location: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Violation]:
        """
        Get violations with optional filters.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            violation_type: Filter by type
            camera_id: Filter by camera
            site_location: Filter by site
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of Violation records
        """
        query = self.db.query(Violation)
        
        if start_date:
            query = query.filter(Violation.timestamp >= datetime.combine(start_date, datetime.min.time()))
        
        if end_date:
            query = query.filter(Violation.timestamp < datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
        
        if violation_type:
            query = query.filter(Violation.violation_type == violation_type)
        
        if camera_id:
            query = query.filter(Violation.camera_id == camera_id)
        
        if site_location:
            query = query.filter(Violation.site_location == site_location)
        
        return query.order_by(desc(Violation.timestamp)).offset(offset).limit(limit).all()
    
    def get_violations_count(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> int:
        """Get total count of violations."""
        query = self.db.query(func.count(Violation.id))
        
        if start_date:
            query = query.filter(Violation.timestamp >= datetime.combine(start_date, datetime.min.time()))
        
        if end_date:
            query = query.filter(Violation.timestamp < datetime.combine(end_date + timedelta(days=1), datetime.min.time()))
        
        return query.scalar()
    
    def get_today_violations(self) -> List[Violation]:
        """Get all violations from today."""
        today = date.today()
        return self.get_violations(start_date=today, end_date=today)
    
    def get_unreported_violations(self, target_date: Optional[date] = None) -> List[Violation]:
        """Get violations not yet included in a report."""
        query = self.db.query(Violation).filter(Violation.report_sent == False)
        
        if target_date:
            query = query.filter(Violation.report_date == target_date)
        
        return query.order_by(Violation.timestamp).all()
    
    # === Report Operations ===
    
    def get_report_by_date(self, report_date: date) -> Optional[DailyReport]:
        """Get daily report for a specific date."""
        return self.db.query(DailyReport).filter(
            DailyReport.report_date == report_date
        ).first()
    
    def get_recent_reports(self, days: int = 30) -> List[DailyReport]:
        """Get reports from the last N days."""
        cutoff = date.today() - timedelta(days=days)
        return self.db.query(DailyReport).filter(
            DailyReport.report_date >= cutoff
        ).order_by(desc(DailyReport.report_date)).all()
    
    # === Statistics ===
    
    def get_daily_stats(self, target_date: date) -> Dict[str, Any]:
        """Get statistics for a specific date."""
        violations = self.get_violations(start_date=target_date, end_date=target_date)
        
        total = len(violations)
        
        if total == 0:
            return {
                "date": str(target_date),
                "total_detections": 0,
                "total_violations": 0,
                "compliance_rate": 100.0,
                "no_helmet": 0,
                "no_vest": 0,
                "both_missing": 0
            }
        
        no_helmet = sum(1 for v in violations if v.violation_type == "no_helmet")
        no_vest = sum(1 for v in violations if v.violation_type == "no_vest")
        both_missing = sum(1 for v in violations if v.violation_type == "both_missing")
        
        total_violations = no_helmet + no_vest + both_missing
        compliance_rate = ((total - total_violations) / total * 100) if total > 0 else 100.0
        
        return {
            "date": str(target_date),
            "total_detections": total,
            "total_violations": total_violations,
            "compliance_rate": compliance_rate,
            "no_helmet": no_helmet,
            "no_vest": no_vest,
            "both_missing": both_missing
        }
    
    def get_weekly_trend(self) -> List[Dict[str, Any]]:
        """Get daily stats for the last 7 days."""
        today = date.today()
        trend = []
        
        for i in range(7):
            day = today - timedelta(days=i)
            stats = self.get_daily_stats(day)
            trend.append(stats)
        
        return trend
    
    def get_camera_stats(self, days: int = 7) -> Dict[str, int]:
        """Get violation counts by camera."""
        cutoff = datetime.now() - timedelta(days=days)
        
        results = self.db.query(
            Violation.camera_id,
            func.count(Violation.id).label("count")
        ).filter(
            Violation.timestamp >= cutoff
        ).group_by(Violation.camera_id).all()
        
        return {camera: count for camera, count in results}
    
    def get_hourly_distribution(self, target_date: Optional[date] = None) -> Dict[int, int]:
        """Get violation count by hour of day."""
        query = self.db.query(
            func.extract('hour', Violation.timestamp).label("hour"),
            func.count(Violation.id).label("count")
        )
        
        if target_date:
            query = query.filter(Violation.report_date == target_date)
        
        results = query.group_by("hour").all()
        
        # Fill in missing hours with 0
        distribution = {h: 0 for h in range(24)}
        for hour, count in results:
            distribution[int(hour)] = count
        
        return distribution
