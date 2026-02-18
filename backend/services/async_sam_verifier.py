"""
Async SAM Verification Service

This is the core implementation of the thesis's real-time innovation:
YOLO detects instantly → response returned to user → SAM verifies in background.

Pipeline:
    t=0ms   → YOLO detects person, flags as potential violation
    t=35ms  → Response returned to user (real-time!)
    t=35ms  → SAM job submitted to background thread pool
    t=1500ms → SAM finishes → DB record updated with refined result
    t=0-300s → Cooldown active → no duplicate alerts during SAM processing

This decouples detection latency from verification accuracy.
The cooldown window (5 min default) always covers SAM processing time (~1-2s).

Academic Framing:
    "Two-stage asynchronous pipeline: synchronous YOLO for real-time response,
    asynchronous SAM for semantic verification within the cooldown window."
"""

import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SAMJob:
    """A pending SAM verification job."""
    job_id: str
    person_id: int
    bbox: List[float]
    image: np.ndarray          # Frame snapshot for SAM to process
    violation_type: str        # What YOLO suspected
    yolo_result: Dict[str, Any]  # Original YOLO result
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    sam_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Callback to call when SAM finishes (e.g., update DB)
    on_complete: Optional[Callable] = None


@dataclass
class SAMVerificationResult:
    """Result from async SAM verification."""
    job_id: str
    person_id: int
    # Refined PPE status from SAM
    has_helmet: bool
    has_vest: bool
    is_violation: bool
    violation_type: Optional[str]
    # Timing
    sam_latency_ms: float
    # Changed from YOLO's initial guess?
    yolo_was_correct: bool
    yolo_initial_violation: bool


class AsyncSAMVerifier:
    """
    Runs SAM verification asynchronously in a background thread pool.

    YOLO provides instant detection results. For uncertain cases (Paths 2,3,4),
    SAM verification is submitted as a background job. The result is used to:
    1. Update the DB violation record with the refined result
    2. Potentially cancel a false-positive violation

    This achieves real-time performance (YOLO speed) with SAM accuracy,
    using the violation cooldown window as the processing budget.

    Usage:
        verifier = get_async_sam_verifier()

        # Submit SAM job (non-blocking)
        job_id = verifier.submit(
            person_id=0,
            bbox=[x1,y1,x2,y2],
            image=frame,
            violation_type="no_helmet",
            yolo_result=person_dict,
            on_complete=update_db_callback
        )

        # Check result later (optional)
        result = verifier.get_result(job_id)
    """

    def __init__(self, max_workers: int = 2):
        """
        Initialize async SAM verifier.

        Args:
            max_workers: Number of parallel SAM threads.
                         Keep low (1-2) since SAM is GPU/CPU intensive.
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sam_worker"
        )
        self._jobs: Dict[str, SAMJob] = {}
        self._results: Dict[str, SAMVerificationResult] = {}
        self._lock = threading.Lock()
        self._job_counter = 0

        # Stats for thesis
        self.stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "false_positives_caught": 0,   # YOLO said violation, SAM said safe
            "false_negatives_caught": 0,   # YOLO said safe, SAM said violation
            "total_sam_latency_ms": 0.0,
            "avg_sam_latency_ms": 0.0,
        }

        logger.info(f"AsyncSAMVerifier initialized with {max_workers} workers")

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        with self._lock:
            self._job_counter += 1
            return f"sam_job_{self._job_counter}_{int(time.time()*1000)}"

    def submit(
        self,
        person_id: int,
        bbox: List[float],
        image: np.ndarray,
        violation_type: str,
        yolo_result: Dict[str, Any],
        on_complete: Optional[Callable[[SAMVerificationResult], None]] = None
    ) -> str:
        """
        Submit a SAM verification job (non-blocking).

        YOLO has already returned its result. This submits SAM to verify
        in the background. The caller gets a job_id to check later.

        Args:
            person_id: Person index from YOLO
            bbox: Person bounding box [x1, y1, x2, y2]
            image: Frame snapshot (copied to avoid mutation)
            violation_type: What YOLO detected ('no_helmet', 'no_vest', 'both_missing')
            yolo_result: Full YOLO person dict
            on_complete: Callback(SAMVerificationResult) called when SAM finishes

        Returns:
            job_id: Use to check status/result later
        """
        job_id = self._generate_job_id()

        # Copy image to avoid mutation during async processing
        image_copy = image.copy()

        job = SAMJob(
            job_id=job_id,
            person_id=person_id,
            bbox=bbox,
            image=image_copy,
            violation_type=violation_type,
            yolo_result=yolo_result,
            on_complete=on_complete
        )

        with self._lock:
            self._jobs[job_id] = job
            self.stats["jobs_submitted"] += 1

        # Submit to thread pool (non-blocking)
        future = self._executor.submit(self._run_sam_job, job)
        future.add_done_callback(lambda f: self._on_job_done(job_id, f))

        logger.debug(f"SAM job {job_id} submitted for person {person_id} ({violation_type})")
        return job_id

    def _run_sam_job(self, job: SAMJob) -> SAMVerificationResult:
        """
        Execute SAM verification (runs in background thread).

        This is the actual SAM call. It runs in a worker thread so it
        doesn't block the main request/response cycle.
        """
        start_time = time.time()

        try:
            from services.sam_verifier import get_sam_verifier
            sam = get_sam_verifier()

            # Determine what SAM needs to verify based on violation type
            has_helmet = job.yolo_result.get("has_helmet", False)
            has_vest = job.yolo_result.get("has_vest", False)
            yolo_was_violation = job.yolo_result.get("is_violation", True)

            if job.violation_type == "no_helmet":
                # Path 2: Rescue Head - SAM checks HEAD ROI
                result = sam.verify_helmet(job.image, job.bbox)
                has_helmet = result.get("helmet_found", False)
                # has_vest stays from YOLO

            elif job.violation_type == "no_vest":
                # Path 3: Rescue Body - SAM checks TORSO ROI
                result = sam.verify_vest(job.image, job.bbox)
                has_vest = result.get("vest_found", False)
                # has_helmet stays from YOLO

            elif job.violation_type == "both_missing":
                # Path 4: Critical - SAM checks both ROIs
                result = sam.verify_both(job.image, job.bbox)
                has_helmet = result.get("helmet_found", False)
                has_vest = result.get("vest_found", False)

            # Determine refined violation status
            is_violation = not (has_helmet and has_vest)
            violation_type = None
            if is_violation:
                if not has_helmet and not has_vest:
                    violation_type = "both_missing"
                elif not has_helmet:
                    violation_type = "no_helmet"
                else:
                    violation_type = "no_vest"

            sam_latency = (time.time() - start_time) * 1000

            # Was YOLO correct?
            yolo_was_correct = (yolo_was_violation == is_violation)

            return SAMVerificationResult(
                job_id=job.job_id,
                person_id=job.person_id,
                has_helmet=has_helmet,
                has_vest=has_vest,
                is_violation=is_violation,
                violation_type=violation_type,
                sam_latency_ms=sam_latency,
                yolo_was_correct=yolo_was_correct,
                yolo_initial_violation=yolo_was_violation
            )

        except Exception as e:
            logger.error(f"SAM job {job.job_id} failed: {e}")
            sam_latency = (time.time() - start_time) * 1000

            # On error, fall back to YOLO result
            return SAMVerificationResult(
                job_id=job.job_id,
                person_id=job.person_id,
                has_helmet=job.yolo_result.get("has_helmet", False),
                has_vest=job.yolo_result.get("has_vest", False),
                is_violation=job.yolo_result.get("is_violation", True),
                violation_type=job.violation_type,
                sam_latency_ms=sam_latency,
                yolo_was_correct=True,  # Assume YOLO was right on error
                yolo_initial_violation=job.yolo_result.get("is_violation", True)
            )

    def _on_job_done(self, job_id: str, future: Future):
        """Called when a SAM job completes (in background thread)."""
        try:
            result = future.result()
            job = self._jobs.get(job_id)

            with self._lock:
                self._results[job_id] = result
                self.stats["jobs_completed"] += 1
                self.stats["total_sam_latency_ms"] += result.sam_latency_ms
                self.stats["avg_sam_latency_ms"] = (
                    self.stats["total_sam_latency_ms"] /
                    max(self.stats["jobs_completed"], 1)
                )

                if job:
                    job.completed_at = time.time()
                    job.sam_result = result

                    # Track accuracy stats
                    if not result.yolo_was_correct:
                        if result.yolo_initial_violation and not result.is_violation:
                            self.stats["false_positives_caught"] += 1
                            logger.info(
                                f"✅ SAM caught false positive for job {job_id}: "
                                f"YOLO said violation, SAM says SAFE"
                            )
                        elif not result.yolo_initial_violation and result.is_violation:
                            self.stats["false_negatives_caught"] += 1
                            logger.info(
                                f"⚠️ SAM caught false negative for job {job_id}: "
                                f"YOLO said safe, SAM says VIOLATION"
                            )

            # Fire callback (e.g., update DB)
            if job and job.on_complete:
                try:
                    job.on_complete(result)
                    logger.debug(f"SAM job {job_id} callback completed")
                except Exception as e:
                    logger.error(f"SAM job {job_id} callback failed: {e}")

            logger.debug(
                f"SAM job {job_id} done in {result.sam_latency_ms:.1f}ms | "
                f"violation={result.is_violation} | "
                f"yolo_correct={result.yolo_was_correct}"
            )

        except Exception as e:
            with self._lock:
                self.stats["jobs_failed"] += 1
            logger.error(f"SAM job {job_id} future failed: {e}")

    def get_result(self, job_id: str) -> Optional[SAMVerificationResult]:
        """
        Get the result of a completed SAM job.

        Returns None if job is still running.
        """
        with self._lock:
            return self._results.get(job_id)

    def is_complete(self, job_id: str) -> bool:
        """Check if a SAM job has finished."""
        with self._lock:
            return job_id in self._results

    def wait_for(self, job_id: str, timeout: float = 10.0) -> Optional[SAMVerificationResult]:
        """
        Wait for a SAM job to complete (blocking).

        Only use this when you need the SAM result before responding.
        For real-time use, use submit() + on_complete callback instead.

        Args:
            job_id: Job to wait for
            timeout: Max seconds to wait

        Returns:
            SAMVerificationResult or None if timeout
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_result(job_id)
            if result is not None:
                return result
            time.sleep(0.05)  # Poll every 50ms
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get async SAM statistics for thesis metrics."""
        with self._lock:
            stats = self.stats.copy()
            stats["pending_jobs"] = len(self._jobs) - len(self._results)
            stats["completed_jobs"] = len(self._results)

            completed = max(stats["jobs_completed"], 1)
            stats["yolo_accuracy_rate"] = (
                (completed - stats["false_positives_caught"] - stats["false_negatives_caught"])
                / completed * 100
            )
            return stats

    def cleanup_old_jobs(self, max_age_seconds: float = 600.0):
        """Remove old completed jobs to free memory."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            old_jobs = [
                jid for jid, job in self._jobs.items()
                if job.completed_at and job.completed_at < cutoff
            ]
            for jid in old_jobs:
                del self._jobs[jid]
                self._results.pop(jid, None)

    def shutdown(self):
        """Shutdown the thread pool gracefully."""
        self._executor.shutdown(wait=True)
        logger.info("AsyncSAMVerifier shutdown complete")


# ─── Global Instance ───────────────────────────────────────────────────────────

_async_sam_verifier: Optional[AsyncSAMVerifier] = None


def get_async_sam_verifier() -> AsyncSAMVerifier:
    """Get or create the global async SAM verifier."""
    global _async_sam_verifier
    if _async_sam_verifier is None:
        _async_sam_verifier = AsyncSAMVerifier(max_workers=2)
    return _async_sam_verifier
