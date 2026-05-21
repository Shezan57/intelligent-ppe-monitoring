"""
Demo Report Generator
=====================
Generates a real PDF report WITHOUT needing YOLO or SAM.
Seeds the DB with synthetic violations, uses REAL ROI evidence images,
calls ReportGenerator, and prints the generation time.

Run from backend/ directory:
    python scripts/demo_report.py
"""

import os
import sys
import time
from datetime import datetime, date, timedelta
import random

# Set UTF-8 output encoding for Windows console
sys.stdout.reconfigure(encoding="utf-8")

# -- Add backend root to path -------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# -- Import project modules ---------------------------------------------------
from database.connection import SessionLocal, engine
from database.models import Base, Violation
from services.report_generator import ReportGenerator

REPORTS_DIR = os.path.join(BACKEND_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Actual paths provided by user
REAL_IMAGES = [
    (r"d:\SHEZAN\AI\intelligent-ppe-monitoring\Video\person1_full_roi.jpg", "no_vest"),
    (r"d:\SHEZAN\AI\intelligent-ppe-monitoring\Video\person2_full_roi.jpg", "no_vest"),
    (r"d:\SHEZAN\AI\intelligent-ppe-monitoring\Video\person3_full_roi.jpg", "no_vest"),
    (r"d:\SHEZAN\AI\intelligent-ppe-monitoring\Video\person4_full_roi.jpg", "both_missing"),
]

def seed_violations(db, report_date: date):
    """
    Insert realistic violations into the DB for report_date.
    Clears existing violations for that date first.
    Includes the 4 specific real images as the main evidence.
    """
    # Remove existing test rows for today
    db.query(Violation).filter(
        Violation.report_date == report_date
    ).delete()
    db.commit()

    cameras = ["CAM-001 (Main Gate)", "CAM-002 (Zone A)", "CAM-003 (Zone B)"]
    sites   = ["Tower A Construction", "Foundation Zone"]
    dpaths  = ["YOLO Confirmed (High Conf)", "SAM Verified", "SAM Verified", "Fallback YOLO"]

    # First, insert the 4 real violations we want to feature prominently
    inserted = 0
    times_for_real = [(8, 15), (10, 42), (13, 5), (15, 30)]
    
    for i, ((img_path, vtype), (h, m)) in enumerate(zip(REAL_IMAGES, times_for_real)):
        ts = datetime(report_date.year, report_date.month, report_date.day, h, m, random.randint(0, 59))
        
        has_helmet = False if vtype in ("no_helmet", "both_missing") else True
        has_vest = False if vtype in ("no_vest", "both_missing") else True

        v = Violation(
            timestamp=ts,
            site_location=random.choice(sites),
            camera_id=random.choice(cameras),
            person_bbox=[100, 100, 400, 500], # dummy
            has_helmet=has_helmet,
            has_vest=has_vest,
            violation_type=vtype,
            original_image_path=img_path, # store original path too
            annotated_image_path=img_path, # We use the raw ROI directly as evidence
            decision_path=random.choice(dpaths),
            detection_confidence=round(random.uniform(0.78, 0.96), 2),
            sam_activated=True,
            processing_time_ms=round(random.uniform(25.0, 45.0), 1),
            report_sent=False,
            report_date=report_date,
            session_start=ts,
            last_seen=ts + timedelta(minutes=random.randint(1, 15)),
            occurrence_count=random.randint(1, 3),
            total_duration_minutes=round(random.uniform(1.0, 15.0), 1),
            is_active_session=False,
        )
        db.add(v)
        inserted += 1

    # Insert some background violations without images to fill the graph
    bg_specs = [
        (7,  12, "no_helmet"),
        (9,  20, "no_vest"),
        (11, 33, "both_missing"),
        (14, 40, "no_helmet"),
        (16, 30, "no_vest"),
        (17,  5, "no_helmet")
    ]
    
    for (h, m, vtype) in bg_specs:
        ts = datetime(report_date.year, report_date.month, report_date.day, h, m, random.randint(0, 59))
        has_helmet = False if vtype in ("no_helmet", "both_missing") else True
        has_vest = False if vtype in ("no_vest", "both_missing") else True
        
        v = Violation(
            timestamp=ts,
            site_location=random.choice(sites),
            camera_id=random.choice(cameras),
            person_bbox=[0,0,0,0],
            has_helmet=has_helmet,
            has_vest=has_vest,
            violation_type=vtype,
            annotated_image_path=None,
            decision_path="Fast Violation",
            detection_confidence=round(random.uniform(0.65, 0.85), 2),
            sam_activated=False,
            processing_time_ms=15.0,
            report_sent=False,
            report_date=report_date,
            session_start=ts,
            last_seen=ts,
            occurrence_count=1,
            total_duration_minutes=0.5,
            is_active_session=False,
        )
        db.add(v)
        inserted += 1

    db.commit()
    print(f"  [OK] Seeded {inserted} violations for {report_date}")
    return inserted


def build_stats(violations, total_detections: int):
    """Build the stats dict expected by report_generator."""
    viol_list = [v for v in violations if not (v.has_helmet and v.has_vest)]
    total_v = len(viol_list)
    compliance = ((total_detections - total_v) / total_detections * 100) if total_detections else 100.0

    return {
        "total_detections":   total_detections,
        "total_violations":   total_v,
        "compliance_rate":    round(compliance, 1),
        "no_helmet_count":    sum(1 for v in viol_list if v.violation_type == "no_helmet"),
        "no_vest_count":      sum(1 for v in viol_list if v.violation_type == "no_vest"),
        "both_missing_count": sum(1 for v in viol_list if v.violation_type == "both_missing"),
    }


def main():
    print("\n" + "=" * 60)
    print("  PPE Report Generator - Demo Mode")
    print("=" * 60)

    Base.metadata.create_all(bind=engine)

    print("\n[1/3] Seeding violation database with user ROI images...")
    report_date = date.today()
    db = SessionLocal()
    try:
        n_inserted = seed_violations(db, report_date)

        violations = db.query(Violation).filter(
            Violation.report_date == report_date
        ).all()

        total_detections = n_inserted + 145 # Simulate many correct workers
        stats = build_stats(violations, total_detections)

        print(f"  [Stats] {stats}")

        print("\n[2/3] Generating PDF report...")
        gen = ReportGenerator(output_dir=REPORTS_DIR)

        t0 = time.perf_counter()
        pdf_path = gen.generate_daily_report(report_date, violations, stats)
        elapsed = time.perf_counter() - t0

        print("\n[3/3] Done!")
        print("=" * 60)
        print(f"  PDF saved       : {pdf_path}")
        print(f"  Generation time : {elapsed:.2f} seconds")
        print("=" * 60 + "\n")

    finally:
        db.close()


if __name__ == "__main__":
    main()
