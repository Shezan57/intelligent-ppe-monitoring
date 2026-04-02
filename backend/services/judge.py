"""
The Judge — Asynchronous SAM 3 Verification Consumer

The Judge runs in a separate thread/process, listening to the queue
populated by the Sentry. For each violation candidate:
1. Load the ROI image from disk
2. Run SAM 3 verification
3. If confirmed → write to `verified_violations` DB table
4. If rejected  → delete temp ROI image, do nothing

The Judge never touches the video stream. It only processes queued ROIs.
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Judge:
    """
    Asynchronous SAM 3 verification consumer.

    Reads violation candidates from the queue (pushed by Sentry),
    verifies each using SAM 3, and writes confirmed violations to DB.

    Can run as a background thread or a separate process.
    """

    def __init__(
        self,
        queue,
        db_session_factory=None,
        roi_cleanup: bool = True,
    ):
        """
        Initialize the Judge.

        Args:
            queue: multiprocessing.Queue or queue.Queue — reads violation payloads
            db_session_factory: SQLAlchemy sessionmaker for DB writes
            roi_cleanup: Delete temp ROI images after processing
        """
        self.queue = queue
        self.db_session_factory = db_session_factory
        self.roi_cleanup = roi_cleanup
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.stats = {
            "total_processed": 0,
            "confirmed": 0,
            "rejected": 0,
            "not_person_rejected": 0,
            "errors": 0,
            "total_time_ms": 0.0,
        }

        # Load SAM 3
        from services.sam_verifier import get_sam_verifier
        self.sam = get_sam_verifier()
        self.sam.load_model()

    def start_background(self):
        """Start the Judge as a background daemon thread."""
        if self._running:
            logger.warning("Judge already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        print("Judge started (background thread)")

    def stop(self):
        """Stop the Judge gracefully."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        print("Judge stopped")

    def _consume_loop(self):
        """
        Main consumption loop. Runs until STOP signal (None) received.
        """
        logger.info("Judge: listening for violations...")
        print("Judge: listening for violations...")

        while self._running:
            try:
                # Block with timeout so we can check _running flag
                try:
                    payload = self.queue.get(timeout=1.0)
                except Exception:
                    continue

                # STOP signal
                if payload is None:
                    print("Judge: received STOP signal")
                    break

                # Process the violation candidate
                self._process_payload(payload)

            except Exception as e:
                logger.error(f"Judge error: {e}")
                self.stats["errors"] += 1

        self._running = False
        print(f"Judge finished. Stats: {self.get_stats()}")

    def _process_payload(self, payload: Dict[str, Any]):
        """
        Process a single violation candidate from the Sentry.

        Args:
            payload: Dict with timestamp, person_id, roi_image_path,
                    suspected_violation, decision_path, etc.
        """
        t0 = time.time()
        person_id = payload["person_id"]
        violation_type = payload["suspected_violation"]
        roi_path = payload["roi_image_path"]
        path = payload.get("decision_path", "Unknown")

        logger.info(f"Judge processing: Person {person_id}, {violation_type}")

        # Load ROI image
        if not os.path.exists(roi_path):
            logger.warning(f"ROI image not found: {roi_path}")
            self.stats["errors"] += 1
            return

        roi_image = cv2.imread(roi_path)
        if roi_image is None:
            logger.warning(f"Failed to read ROI image: {roi_path}")
            self.stats["errors"] += 1
            return

        # === Run SAM 3 verification ===
        confirmed = False
        judge_confidence = 0.0

        if self.sam.is_mock():
            # Mock mode: use dummy verification
            result = self.sam._mock_verification(
                "helmet" if "helmet" in violation_type else "vest"
            )
            # In mock mode, confirm all violations (for testing)
            confirmed = True
            judge_confidence = 0.5
        else:
            # Real SAM 3 verification
            confirmed, judge_confidence = self._verify_with_sam(
                roi_image, violation_type, payload.get("person_bbox", [])
            )

        processing_time = (time.time() - t0) * 1000
        self.stats["total_processed"] += 1
        self.stats["total_time_ms"] += processing_time

        if confirmed:
            # === CONFIRMED: Write to database ===
            self.stats["confirmed"] += 1
            self._store_violation(payload, judge_confidence, processing_time)
            logger.info(
                f"Judge CONFIRMED: Person {person_id} - {violation_type} "
                f"(conf={judge_confidence:.2f}, {processing_time:.0f}ms)"
            )
            print(f"  Judge CONFIRMED: Person {person_id} - {violation_type} ({processing_time:.0f}ms)")
        else:
            # === REJECTED: Clean up ROI image ===
            self.stats["rejected"] += 1
            if self.roi_cleanup and os.path.exists(roi_path):
                os.remove(roi_path)
            logger.info(
                f"Judge REJECTED: Person {person_id} - {violation_type} "
                f"(SAM found PPE, conf={judge_confidence:.2f})"
            )
            print(f"  Judge REJECTED: Person {person_id} - {violation_type} (SAM found PPE)")

    def _is_person(self, roi_image: np.ndarray) -> bool:
        """
        Pre-check: Is this ROI actually a person?

        Two-way verification:
          1. POSITIVE: SAM searches for 'person/human/worker'
          2. NEGATIVE: SAM searches for 'machine/building/vehicle'
        
        Reject if:
          - No person found at all, OR
          - Machine/building confidence > person confidence

        Returns:
            True if SAM confirms this is a person (not a building/machine)
        """
        # Positive check: is there a person?
        person_result = self.sam._run_sam3_verification(
            roi_image,
            ["person", "human", "worker", "man", "woman"],
            "person"
        )
        person_found = person_result.get("person_found", False)
        person_conf = person_result.get("confidence", 0.0)

        # Negative check: is this a machine/building/object?
        object_result = self.sam._run_sam3_verification(
            roi_image,
            ["machine", "building", "vehicle", "crane", "excavator",
             "construction equipment", "truck", "wall", "structure"],
            "object"
        )
        object_found = object_result.get("object_found", False)
        object_conf = object_result.get("confidence", 0.0)

        print(f"    Person check: person={person_found} (conf={person_conf:.3f}), "
              f"object={object_found} (conf={object_conf:.3f})")

        # Reject if no person found
        if not person_found:
            return False

        # Reject if object/machine has higher confidence than person
        if object_found and object_conf > person_conf:
            print(f"    -> Object confidence ({object_conf:.3f}) > Person ({person_conf:.3f}): NOT a person")
            return False

        return True

    def _verify_with_sam(
        self,
        roi_image: np.ndarray,
        violation_type: str,
        person_bbox: list,
    ) -> tuple:
        """
        Run SAM 3 verification on the ROI.

        For 'no_helmet': check if SAM finds a helmet → if YES, reject violation
        For 'no_vest': check if SAM finds a vest → if YES, reject violation
        For 'both_missing': check both

        Returns:
            (confirmed: bool, confidence: float)
            confirmed=True means violation IS real (SAM didn't find the PPE)
        """
        # === STEP 0: Verify this is actually a person ===
        if not self._is_person(roi_image):
            self.stats["not_person_rejected"] += 1
            print(f"    -> Not a person! Skipping PPE check.")
            return (False, 0.0)  # Reject — not a real person

        # === STEP 1: Check PPE based on violation type ===
        if violation_type == "no_helmet":
            # SAM checks for helmet in the head ROI
            result = self.sam._run_sam3_verification(
                roi_image,
                ["helmet", "hard hat", "safety helmet"],
                "helmet"
            )
            helmet_found = result.get("helmet_found", False)
            confidence = result.get("confidence", 0.0)
            # If SAM found a helmet → REJECT (YOLO was wrong)
            return (not helmet_found, confidence)

        elif violation_type == "no_vest":
            result = self.sam._run_sam3_verification(
                roi_image,
                ["safety vest", "high visibility vest", "reflective vest"],
                "vest"
            )
            vest_found = result.get("vest_found", False)
            confidence = result.get("confidence", 0.0)
            return (not vest_found, confidence)

        elif violation_type == "both_missing":
            # Check both
            helmet_result = self.sam._run_sam3_verification(
                roi_image,
                ["helmet", "hard hat", "safety helmet"],
                "helmet"
            )
            vest_result = self.sam._run_sam3_verification(
                roi_image,
                ["safety vest", "high visibility vest", "reflective vest"],
                "vest"
            )
            helmet_found = helmet_result.get("helmet_found", False)
            vest_found = vest_result.get("vest_found", False)
            avg_conf = (
                helmet_result.get("confidence", 0) +
                vest_result.get("confidence", 0)
            ) / 2

            # Confirmed if at least one is still missing
            confirmed = not (helmet_found and vest_found)
            return (confirmed, avg_conf)

        # Unknown type → confirm by default
        return (True, 0.0)

    def _store_violation(
        self,
        payload: Dict[str, Any],
        judge_confidence: float,
        processing_time: float,
    ):
        """
        Store a confirmed violation in the database.
        """
        if self.db_session_factory is None:
            logger.warning("No DB session factory — skipping DB write")
            return

        try:
            from database.models import VerifiedViolation

            session = self.db_session_factory()
            violation = VerifiedViolation(
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                person_id=payload["person_id"],
                violation_type=payload["suspected_violation"],
                image_path=payload.get("roi_image_path"),
                camera_zone=payload.get("camera_zone", "zone_1"),
                judge_confirmed=True,
                judge_confidence=judge_confidence,
                judge_processing_time_ms=processing_time,
                sentry_confidence=payload.get("sentry_confidence"),
                decision_path=payload.get("decision_path"),
                person_bbox=payload.get("person_bbox"),
            )
            session.add(violation)
            session.commit()
            session.close()

        except Exception as e:
            logger.error(f"DB write error: {e}")
            self.stats["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get Judge processing statistics."""
        stats = self.stats.copy()
        if stats["total_processed"] > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_processed"]
            stats["confirmation_rate"] = stats["confirmed"] / stats["total_processed"] * 100
        else:
            stats["avg_time_ms"] = 0.0
            stats["confirmation_rate"] = 0.0
        stats["sam_mock_mode"] = self.sam.is_mock()
        return stats
