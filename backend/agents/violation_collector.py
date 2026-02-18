"""
Violation Collector Agent

Real-time storage of violations in database using SESSION-BASED tracking.

SESSION-BASED APPROACH (Thesis Innovation):
============================================
PROBLEM: A worker violating for 2 hours would create 24 DB rows (one per 5-min cooldown).
         The daily report would show that worker 24 times — noisy and misleading.

SOLUTION: One DB row per violation SESSION per worker.
          When the same worker is re-detected violating:
          - We UPDATE the existing row (not create a new one)
          - Increment occurrence_count
          - Update total_duration_minutes
          - Update last_seen timestamp

RESULT:
  Worker A (no helmet, 08:00-10:00) → 1 DB row:
    session_start=08:00, last_seen=10:00,
    occurrence_count=24, total_duration_minutes=120

  Report shows: "Worker A | No Helmet | 08:00 | 2 hours | 24 detections"
  NOT: 24 separate rows!

HOW SESSION ENDS:
  - Worker leaves frame (track timeout in ViolationTracker)
  - Worker becomes compliant (puts on helmet/vest)
  - New day starts (report_date changes)
  - Cooldown expires AND worker is gone (new session starts)
"""

import json
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session

from database.models import Violation
from config.settings import settings
from services.violation_tracker import get_violation_tracker

logger = logging.getLogger(__name__)


class ViolationCollector:
    """
    Stores violations in database using session-based tracking.

    Key behaviour:
    - First detection of a violation → CREATE new Violation row (session starts)
    - Same worker re-detected within cooldown → SKIP (cooldown active)
    - Same worker re-detected after cooldown → UPDATE existing row (session continues)
    - Worker leaves frame → session marked inactive (is_active_session=False)

    This ensures 1 row per worker per violation session, not 1 row per cooldown cycle.
    """

    def __init__(
        self,
        db: Session,
        site_location: Optional[str] = None,
        camera_id: Optional[str] = None,
        enable_deduplication: bool = True
    ):
        self.db = db
        self.site_location = site_location or settings.default_site_location
        self.camera_id = camera_id or settings.default_camera_id
        self.enable_deduplication = enable_deduplication
        self.tracker = get_violation_tracker() if enable_deduplication else None

    def store_detection_results(
        self,
        detection_result: Dict[str, Any],
        image_path: str,
        annotated_path: Optional[str] = None,
        site_location: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> List[Violation]:
        """
        Store violations using session-based logic.

        For each violating person:
        - NEW violation → create 1 DB row, mark session_start
        - COOLDOWN ACTIVE → skip entirely (same session, no update needed)
        - COOLDOWN EXPIRED → find existing session row and UPDATE it
                             (increment count, update duration, update last_seen)

        Args:
            detection_result: Output from HybridDetector.detect_async()
            image_path: Path to original image
            annotated_path: Path to annotated image
            site_location: Override site location
            camera_id: Override camera ID

        Returns:
            List of Violation records (new OR updated this cycle)
        """
        touched_violations = []  # new + updated this cycle
        skipped_count = 0

        site = site_location or self.site_location
        camera = camera_id or self.camera_id

        persons = detection_result.get("persons", [])
        timing = detection_result.get("timing", {})
        now = datetime.now()

        for person in persons:
            if not person.get("is_violation", False):
                continue

            violation_type = person.get("violation_type", "unknown")
            bbox = person.get("bbox", [])

            if self.enable_deduplication and self.tracker:
                should_store, reason = self.tracker.should_store_violation(
                    person, violation_type
                )

                if not should_store:
                    # Still within cooldown — same session, no DB action needed
                    skipped_count += 1
                    continue

                if reason == "cooldown_expired":
                    # ── SESSION CONTINUES ──────────────────────────────────
                    # Cooldown expired means we've seen this worker again
                    # after 5 minutes. Instead of creating a new row,
                    # find the existing session row and UPDATE it.
                    existing = self._find_active_session(
                        site=site,
                        camera=camera,
                        violation_type=violation_type,
                        bbox=bbox,
                        today=date.today()
                    )

                    if existing:
                        self._update_session(existing, now)
                        touched_violations.append(existing)
                        logger.debug(
                            f"📝 Session updated: violation #{existing.id} | "
                            f"count={existing.occurrence_count} | "
                            f"duration={existing.total_duration_minutes:.1f}min"
                        )
                        continue  # Don't fall through to create new row

                # reason == "new_violation" → fall through to create new row

            # ── NEW SESSION ────────────────────────────────────────────────
            # First time we see this worker violating → create 1 DB row
            violation = self._create_session(
                person=person,
                image_path=image_path,
                annotated_path=annotated_path,
                site_location=site,
                camera_id=camera,
                processing_time_ms=timing.get("total_ms", 0),
                now=now
            )
            self.db.add(violation)
            touched_violations.append(violation)

        # Commit all changes (new rows + updated rows)
        if touched_violations:
            self.db.commit()
            new_count = sum(1 for v in touched_violations if v.occurrence_count == 1)
            updated_count = len(touched_violations) - new_count
            logger.info(
                f"📝 {new_count} new session(s), {updated_count} session(s) updated, "
                f"{skipped_count} skipped (within cooldown)"
            )
        elif skipped_count > 0:
            logger.debug(f"⏭️ {skipped_count} violation(s) within cooldown — no DB action")

        return touched_violations

    def _find_active_session(
        self,
        site: str,
        camera: str,
        violation_type: str,
        bbox: List[float],
        today: date
    ) -> Optional[Violation]:
        """
        Find an active violation session for this worker.

        Looks for a row from today with the same site/camera/violation_type
        that is still marked as an active session.

        We match by violation_type (not exact bbox) because the worker
        may have moved slightly between cooldown cycles.

        Args:
            site: Site location
            camera: Camera ID
            violation_type: Type of violation
            bbox: Current bounding box (for future IoU matching if needed)
            today: Today's date

        Returns:
            Existing Violation row or None
        """
        # Find the most recent active session for this violation type today
        existing = (
            self.db.query(Violation)
            .filter(
                Violation.site_location == site,
                Violation.camera_id == camera,
                Violation.violation_type == violation_type,
                Violation.report_date == today,
                Violation.is_active_session == True
            )
            .order_by(Violation.last_seen.desc())
            .first()
        )
        return existing

    def _update_session(self, violation: Violation, now: datetime) -> None:
        """
        Update an existing violation session with new detection.

        Called when the same worker is re-detected after cooldown expires.
        Updates:
        - last_seen → current time
        - occurrence_count → +1
        - total_duration_minutes → time from session_start to now
        - is_active_session → True (still active)

        Args:
            violation: Existing Violation ORM object
            now: Current datetime
        """
        violation.last_seen = now
        violation.occurrence_count = (violation.occurrence_count or 1) + 1
        violation.is_active_session = True

        # Calculate total duration from session start to now
        if violation.session_start:
            delta = now - violation.session_start
            violation.total_duration_minutes = delta.total_seconds() / 60.0

    def _create_session(
        self,
        person: Dict[str, Any],
        image_path: str,
        annotated_path: Optional[str],
        site_location: str,
        camera_id: str,
        processing_time_ms: float,
        now: datetime
    ) -> Violation:
        """
        Create a new violation session record.

        This is the first time we see this worker violating.
        Sets session_start = now, occurrence_count = 1.

        Args:
            person: Person detection dict
            image_path: Original image path
            annotated_path: Annotated image path
            site_location: Site location
            camera_id: Camera ID
            processing_time_ms: Detection time
            now: Current datetime

        Returns:
            New Violation ORM object (not yet committed)
        """
        return Violation(
            # Timestamp (when session started)
            timestamp=now,

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
            report_date=date.today(),

            # Session tracking (NEW)
            session_start=now,
            last_seen=now,
            occurrence_count=1,
            total_duration_minutes=0.0,
            is_active_session=True
        )

    def close_inactive_sessions(self, site: str = None, camera: str = None) -> int:
        """
        Mark sessions as inactive when workers leave the frame.

        Call this when a track times out (worker not seen for track_timeout_seconds).
        This cleanly closes the session so the next detection starts a fresh one.

        Args:
            site: Site location filter (optional)
            camera: Camera ID filter (optional)

        Returns:
            Number of sessions closed
        """
        query = self.db.query(Violation).filter(
            Violation.is_active_session == True
        )
        if site:
            query = query.filter(Violation.site_location == site)
        if camera:
            query = query.filter(Violation.camera_id == camera)

        sessions = query.all()
        now = datetime.now()

        for session in sessions:
            session.is_active_session = False
            if session.session_start:
                delta = now - session.session_start
                session.total_duration_minutes = delta.total_seconds() / 60.0

        if sessions:
            self.db.commit()
            logger.info(f"🔒 Closed {len(sessions)} inactive violation session(s)")

        return len(sessions)

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
        """Store a single violation manually (for testing or manual entry)."""
        if not has_helmet and not has_vest:
            violation_type = "both_missing"
        elif not has_helmet:
            violation_type = "no_helmet"
        elif not has_vest:
            violation_type = "no_vest"
        else:
            violation_type = "none"

        now = datetime.now()
        violation = Violation(
            timestamp=now,
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
            report_date=date.today(),
            session_start=now,
            last_seen=now,
            occurrence_count=1,
            total_duration_minutes=0.0,
            is_active_session=True
        )
        self.db.add(violation)
        self.db.commit()
        return violation

    def get_unreported_violations(
        self,
        target_date: Optional[date] = None
    ) -> List[Violation]:
        """Get all violations that haven't been included in a report."""
        query = self.db.query(Violation).filter(
            Violation.report_sent == False
        )
        if target_date:
            query = query.filter(Violation.report_date == target_date)
        return query.all()

    def mark_as_reported(self, violations: List[Violation]) -> None:
        """Mark violations as included in a report."""
        for violation in violations:
            violation.report_sent = True
        self.db.commit()


def get_violation_collector(db: Session) -> ViolationCollector:
    """Factory function to create a ViolationCollector."""
    return ViolationCollector(db)
