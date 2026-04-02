"""
Sentry-Judge Pipeline Runner

Orchestrates the decoupled detection pipeline:
1. Creates the message queue
2. Starts the Judge (consumer) in a background thread
3. Runs the Sentry (producer) on video input
4. Collects and prints combined results

Usage (Colab or local):
    python run_pipeline.py --video /path/to/video.mp4 --output /path/to/output.mp4

Or import and use programmatically:
    from run_pipeline import run_pipeline
    results = run_pipeline("video.mp4", "output.mp4")
"""

import os
import sys
import time
import argparse
import queue  # thread-safe queue (simpler than multiprocessing for same-process)
from datetime import datetime

# Ensure backend modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_pipeline(
    video_path: str,
    output_path: str = None,
    max_frames: int = None,
    cooldown_seconds: float = 300.0,
    roi_save_dir: str = "temp_rois",
    camera_zone: str = "zone_1",
):
    """
    Run the full Sentry-Judge pipeline on a video.

    Args:
        video_path: Path to input video file
        output_path: Path for annotated output video (optional)
        max_frames: Maximum frames to process (None = all)
        cooldown_seconds: Cooldown between alerts for same person
        roi_save_dir: Directory for temporary ROI images
        camera_zone: Camera zone identifier

    Returns:
        Dict with combined Sentry + Judge results
    """
    from services.sentry import Sentry
    from services.judge import Judge

    # Setup database
    from database.connection import engine, SessionLocal
    from database.models import Base
    Base.metadata.create_all(bind=engine)

    # === 1. Create the message queue ===
    violation_queue = queue.Queue()
    print(f"\n{'='*60}")
    print("SENTRY-JUDGE PIPELINE")
    print(f"{'='*60}")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_path or 'None'}")
    print(f"  Cooldown: {cooldown_seconds}s")
    print(f"  Camera: {camera_zone}")
    print(f"{'='*60}\n")

    # === 2. Start the Judge (background consumer) ===
    judge = Judge(
        queue=violation_queue,
        db_session_factory=SessionLocal,
        roi_cleanup=False,  # Keep ROIs for thesis evidence
    )
    judge.start_background()

    # === 3. Run the Sentry (producer) ===
    sentry = Sentry(
        queue=violation_queue,
        cooldown_seconds=cooldown_seconds,
        roi_save_dir=roi_save_dir,
        camera_zone=camera_zone,
    )

    t_start = time.time()
    sentry_results = sentry.process_video(
        video_path=video_path,
        output_path=output_path,
        max_frames=max_frames,
    )

    # === 4. Wait for Judge to finish processing queue ===
    print("\nWaiting for Judge to finish...")
    # The Sentry already sent None (STOP signal) at the end
    # Wait for Judge thread to complete
    if judge._thread:
        judge._thread.join(timeout=300)  # Max 5 min wait

    total_time = time.time() - t_start
    judge_stats = judge.get_stats()

    # === 5. Generate Report ===
    report_result = {}
    try:
        from agents.agentic_reporter import AgenticReporter
        reporter = AgenticReporter(
            output_dir=os.path.join(os.getcwd(), "reports"),
            llm_provider="google",  # or "openai" or "none"
        )
        report_result = reporter.generate_report(
            db_session_factory=SessionLocal,
        )
    except Exception as e:
        print(f"  Reporter warning: {e}")

    # === 6. Query verified violations from DB ===
    verified_count = 0
    try:
        from database.models import VerifiedViolation
        session = SessionLocal()
        verified_count = session.query(VerifiedViolation).count()
        session.close()
    except Exception as e:
        print(f"  DB query warning: {e}")

    # === 6. Print combined results ===
    print(f"\n{'='*60}")
    print("COMBINED PIPELINE RESULTS")
    print(f"{'='*60}")

    print(f"\n  SENTRY (YOLO + ByteTrack):")
    print(f"    Frames processed: {sentry_results['frames_processed']}")
    print(f"    Avg latency: {sentry_results['avg_latency_ms']:.1f}ms")
    print(f"    FPS: {sentry_results['effective_fps']:.1f}")
    print(f"    Unique persons: {sentry_results['unique_persons']}")
    print(f"    Violations queued: {sentry_results['violations_queued']}")
    print(f"    Cooldown skips: {sentry_results['violations_cooldown_skipped']}")

    print(f"\n  JUDGE (SAM 3):")
    print(f"    Total processed: {judge_stats['total_processed']}")
    print(f"    Confirmed: {judge_stats['confirmed']}")
    print(f"    Rejected: {judge_stats['rejected']}")
    print(f"    Avg time: {judge_stats['avg_time_ms']:.1f}ms")
    print(f"    Confirmation rate: {judge_stats['confirmation_rate']:.1f}%")
    print(f"    SAM mock mode: {judge_stats['sam_mock_mode']}")

    print(f"\n  DATABASE:")
    print(f"    Verified violations stored: {verified_count}")

    print(f"\n  PIPELINE:")
    print(f"    Total time: {total_time:.1f}s")

    # Path distribution
    if sentry_results["path_distribution"]:
        total_paths = sum(sentry_results["path_distribution"].values()) or 1
        fast = sentry_results["path_distribution"].get("Fast Safe", 0)
        sam_bypass = fast / total_paths * 100
        print(f"    SAM bypass rate: {sam_bypass:.1f}%")

    print(f"{'='*60}\n")

    return {
        "sentry": sentry_results,
        "judge": judge_stats,
        "verified_violations_in_db": verified_count,
        "total_time_seconds": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Sentry-Judge PPE Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path for annotated output video")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--cooldown", type=float, default=300.0, help="Cooldown seconds (default: 300)")
    parser.add_argument("--roi-dir", default="temp_rois", help="Directory for ROI images")
    parser.add_argument("--camera", default="zone_1", help="Camera zone identifier")

    args = parser.parse_args()

    results = run_pipeline(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        cooldown_seconds=args.cooldown,
        roi_save_dir=args.roi_dir,
        camera_zone=args.camera,
    )

    return results


if __name__ == "__main__":
    main()
