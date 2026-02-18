"""
Database ORM Models

SQLAlchemy models for PPE Detection system.
Based on thesis specification database schema.
"""

from datetime import datetime, date
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, 
    DateTime, Date, Text, JSON, Index
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Violation(Base):
    """
    Violation detection record.
    
    Stores all PPE violations detected by the hybrid system,
    including evidence images and decision path used.
    """
    __tablename__ = "violations"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Location info
    site_location = Column(String(255), nullable=False)
    camera_id = Column(String(50), nullable=False)
    
    # Detection details
    person_bbox = Column(JSON, nullable=False)  # [x_min, y_min, x_max, y_max]
    has_helmet = Column(Boolean, nullable=False)
    has_vest = Column(Boolean, nullable=False)
    violation_type = Column(String(50), nullable=False)  # 'no_helmet', 'no_vest', 'both_missing'
    
    # Evidence paths
    original_image_path = Column(String(500), nullable=True)
    annotated_image_path = Column(String(500), nullable=True)
    
    # System details
    decision_path = Column(String(50), nullable=False)  # 'Fast Safe', 'Fast Violation', etc.
    detection_confidence = Column(Float, nullable=True)
    sam_activated = Column(Boolean, default=False)
    processing_time_ms = Column(Float, nullable=True)
    
    # Reporting
    report_sent = Column(Boolean, default=False)
    report_date = Column(Date, nullable=True)

    # === Session Tracking ===
    # Instead of creating a new row every 5 minutes for the same worker,
    # we UPDATE this single row to track the full violation session.
    #
    # Example: Worker without helmet for 2 hours:
    #   - occurrence_count = 24  (re-detected every 5 min)
    #   - total_duration_minutes = 120
    #   - session_start = 08:00
    #   - last_seen = 10:00
    #   → Report shows 1 row, not 24!
    session_start = Column(DateTime, nullable=True)          # When violation first started
    last_seen = Column(DateTime, nullable=True)              # Last time this worker was seen violating
    occurrence_count = Column(Integer, default=1)            # How many times re-detected in this session
    total_duration_minutes = Column(Float, default=0.0)      # Total violation duration in minutes
    is_active_session = Column(Boolean, default=True)        # Is worker still in frame?
    
    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_violations_timestamp', 'timestamp'),
        Index('idx_violations_report_date', 'report_date'),
        Index('idx_violations_violation_type', 'violation_type'),
        Index('idx_violations_report_sent', 'report_sent'),
    )
    
    def __repr__(self) -> str:
        return f"<Violation(id={self.id}, type={self.violation_type}, time={self.timestamp})>"
    
    @property
    def is_violation(self) -> bool:
        """Check if this record represents a violation."""
        return not (self.has_helmet and self.has_vest)
    
    @property
    def missing_items(self) -> list:
        """Get list of missing PPE items."""
        items = []
        if not self.has_helmet:
            items.append("helmet")
        if not self.has_vest:
            items.append("vest")
        return items


class DailyReport(Base):
    """
    Daily report record.
    
    Tracks generated PDF reports and their delivery status.
    """
    __tablename__ = "daily_reports"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Report date (unique)
    report_date = Column(Date, unique=True, nullable=False)
    
    # Statistics
    total_detections = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    compliance_rate = Column(Float, default=100.0)
    
    # PDF file
    pdf_path = Column(String(500), nullable=True)
    
    # Email status
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime, nullable=True)
    recipients = Column(Text, nullable=True)  # JSON array of emails
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_daily_reports_report_date', 'report_date'),
    )
    
    def __repr__(self) -> str:
        return f"<DailyReport(date={self.report_date}, violations={self.total_violations})>"


def create_tables(engine):
    """
    Create all database tables.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """
    Drop all database tables.
    
    WARNING: This will delete all data!
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(bind=engine)
