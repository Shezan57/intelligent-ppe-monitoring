"""
Violation Collector Agent

Real-time storage of violations in database.
Called automatically after each detection to persist results.
"""

import json
from datetime import datetime, date
from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session

from database.models import Violation
from config.settings import settings


class ViolationCollector:
    """
    Stores violations in database immediately after detection.
    
    This agent runs as part of the detection pipeline, capturing
    all violations with their evidence for later reporting.
    
    Attributes:
        db: SQLAlchemy database session
        site_location: Default site location
        camera_id: Default camera ID
    """
    
    def __init__(
        self,
        db: Session,
        site_location: Optional[str] = None,
        camera_id: Optional[str] = None
    ):
        """
        Initialize violation collector.
        
        Args:
            db: Database session
            site_location: Default site location (uses settings if not provided)
            camera_id: Default camera ID (uses settings if not provided)
        """
        self.db = db
        self.site_location = site_location or settings.default_site_location
        self.camera_id = camera_id or settings.default_camera_id
    
    def store_detection_results(
        self,
        detection_result: Dict[str, Any],
        image_path: str,
        annotated_path: Optional[str] = None,
        site_location: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> List[Violation]:
        """
        Store all violations from a detection result.
        
        CRITICAL: This should be called after every detection to ensure
        no violations are missed for daily reporting.
        
        Args:
            detection_result: Output from HybridDetector.detect()
            image_path: Path to original image
            annotated_path: Path to annotated image with bboxes
            site_location: Override site location
            camera_id: Override camera ID
            
        Returns:
            List of created Violation records
        """
        stored_violations = []
        
        site = site_location or self.site_location
        camera = camera_id or self.camera_id
        
        persons = detection_result.get("persons", [])
        timing = detection_result.get("timing", {})
        
        for person in persons:
            # Only store violations (not safe detections)
            if not person.get("is_violation", False):
                continue
            
            violation = self._create_violation(
                person=person,
                image_path=image_path,
                annotated_path=annotated_path,
                site_location=site,
                camera_id=camera,
                processing_time_ms=timing.get("total_ms", 0)
            )
            
            self.db.add(violation)
            stored_violations.append(violation)
        
        # Commit all violations
        if stored_violations:
            self.db.commit()
            print(f"📝 Stored {len(stored_violations)} violation(s)")
        
        return stored_violations
    
    def _create_violation(
        self,
        person: Dict[str, Any],
        image_path: str,
        annotated_path: Optional[str],
        site_location: str,
        camera_id: str,
        processing_time_ms: float
    ) -> Violation:
        """
        Create a Violation record from person detection.
        
        Args:
            person: Person detection dict with PPE status
            image_path: Path to original image
            annotated_path: Path to annotated image
            site_location: Site location
            camera_id: Camera ID
            processing_time_ms: Processing time
            
        Returns:
            Violation ORM model (not yet committed)
        """
        return Violation(
            # Timestamp
            timestamp=datetime.now(),
            
            # Location
            site_location=site_location,
            camera_id=camera_id,
            
            # Detection details
            person_bbox=json.dumps(person["bbox"]),
            has_helmet=person.get("has_helmet", False),
            has_vest=person.get("has_vest", False),
            violation_type=person.get("violation_type", "unknown"),
            
            # Evidence
            original_image_path=image_path,
            annotated_image_path=annotated_path,
            
            # System details
            decision_path=person.get("decision_path", "Unknown"),
            detection_confidence=person.get("confidence", 0.0),
            sam_activated=person.get("sam_activated", False),
            processing_time_ms=processing_time_ms,
            
            # Reporting
            report_sent=False,
            report_date=date.today()
        )
    
    def store_single_violation(
        self,
        bbox: List[float],
        has_helmet: bool,
        has_vest: bool,
        decision_path: str,
        confidence: float,
        sam_activated: bool,
        image_path: str,
        annotated_path: Optional[str] = None,
        site_location: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> Violation:
        """
        Store a single violation manually.
        
        Useful for testing or manual entry.
        """
        # Determine violation type
        if not has_helmet and not has_vest:
            violation_type = "both_missing"
        elif not has_helmet:
            violation_type = "no_helmet"
        elif not has_vest:
            violation_type = "no_vest"
        else:
            violation_type = "none"  # Not actually a violation
        
        violation = Violation(
            timestamp=datetime.now(),
            site_location=site_location or self.site_location,
            camera_id=camera_id or self.camera_id,
            person_bbox=json.dumps(bbox),
            has_helmet=has_helmet,
            has_vest=has_vest,
            violation_type=violation_type,
            original_image_path=image_path,
            annotated_image_path=annotated_path,
            decision_path=decision_path,
            detection_confidence=confidence,
            sam_activated=sam_activated,
            processing_time_ms=0.0,
            report_sent=False,
            report_date=date.today()
        )
        
        self.db.add(violation)
        self.db.commit()
        
        return violation
    
    def get_unreported_violations(
        self,
        target_date: Optional[date] = None
    ) -> List[Violation]:
        """
        Get all violations that haven't been included in a report.
        
        Args:
            target_date: Date to filter by (defaults to today)
            
        Returns:
            List of unreported Violation records
        """
        query = self.db.query(Violation).filter(
            Violation.report_sent == False
        )
        
        if target_date:
            query = query.filter(Violation.report_date == target_date)
        
        return query.all()
    
    def mark_as_reported(
        self,
        violations: List[Violation]
    ) -> None:
        """
        Mark violations as included in a report.
        
        Args:
            violations: List of violations to mark
        """
        for violation in violations:
            violation.report_sent = True
        
        self.db.commit()


def get_violation_collector(db: Session) -> ViolationCollector:
    """
    Factory function to create a ViolationCollector.
    
    Args:
        db: Database session
        
    Returns:
        ViolationCollector instance
    """
    return ViolationCollector(db)
