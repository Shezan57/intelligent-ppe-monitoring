"""
The Sentry — Real-Time YOLO + ByteTrack Producer

The Sentry is the fast first-pass analyzer in the decoupled architecture.
It processes video frames at real-time speed, detects persons with PPE status,
tracks them with ByteTrack, applies cooldown to prevent spam, and pushes
uncertain/violation candidates to the Judge queue.

Flow:
    Video Frame → YOLO detect → ByteTrack assign ID → PPE association →
    5-Path Triage → Cooldown check → Crop ROI → Push to Queue

The Sentry NEVER blocks on SAM. It runs independently at full YOLO speed.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class Sentry:
    """
    Real-time YOLO + ByteTrack PPE detection producer.

    Processes video frames, detects persons, tracks them across frames,
    checks PPE compliance, applies cooldown, and queues violations for
    the Judge (SAM 3) to verify.

    Attributes:
        yolo_detector: YOLO model wrapper
        queue: multiprocessing.Queue to push violation candidates
        cooldown_seconds: Time before re-alerting same person for same violation
        roi_save_dir: Directory to save cropped ROI images
    """

    def __init__(
        self,
        queue,
        cooldown_seconds: float = 300.0,
        roi_save_dir: str = "temp_rois",
        min_person_aspect_ratio: float = 1.2,
        min_person_confidence: float = 0.40,
        camera_zone: str = "zone_1",
    ):
        """
        Initialize the Sentry.

        Args:
            queue: multiprocessing.Queue or similar — push violation payloads here
            cooldown_seconds: Minimum seconds between alerts for same person+violation
            roi_save_dir: Directory for temporary ROI crop images
            min_person_aspect_ratio: Filter false person detections (buildings)
            min_person_confidence: Minimum confidence for Person class
            camera_zone: Camera identifier for logging
        """
        self.queue = queue
        self.cooldown_seconds = cooldown_seconds
        self.roi_save_dir = roi_save_dir
        self.min_aspect_ratio = min_person_aspect_ratio
        self.min_confidence = min_person_confidence
        self.camera_zone = camera_zone

        # Cooldown tracker: {person_id: {violation_type: last_timestamp}}
        self.cooldown_dict: Dict[int, Dict[str, float]] = {}

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "filtered_false_persons": 0,
            "violations_queued": 0,
            "violations_cooldown_skipped": 0,
            "safe_count": 0,
            "path_counts": Counter(),
            "unique_persons": set(),
        }

        # Ensure ROI save directory exists
        os.makedirs(roi_save_dir, exist_ok=True)

        # Load YOLO
        from services.yolo_detector import get_yolo_detector
        self.yolo = get_yolo_detector()
        self.yolo.load_model()

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        process_every_n: int = 1,
    ) -> Dict[str, Any]:
        """
        Process an entire video file.

        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video
            max_frames: Maximum frames to process (None = all)
            process_every_n: Process every Nth frame (1 = all frames)

        Returns:
            Summary dict with stats and timing
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Sentry processing: {video_path} ({w}x{h}, {fps_in:.1f}fps, {total_frames} frames)")
        print(f"Sentry processing: {video_path}")
        print(f"  Resolution: {w}x{h}, FPS: {fps_in:.1f}, Total frames: {total_frames}")

        # Setup output video writer
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

        frame_idx = 0
        processed = 0
        all_latencies = []
        t_start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % process_every_n != 0:
                continue
            if max_frames and processed >= max_frames:
                break

            t0 = time.time()

            # Process single frame
            annotated, frame_results = self._process_frame(frame, w, h)

            latency = (time.time() - t0) * 1000
            all_latencies.append(latency)
            processed += 1
            self.stats["frames_processed"] = processed

            # Add frame overlay
            n_unique = len(self.stats["unique_persons"])
            cv2.putText(
                annotated,
                f"Frame:{processed} | {latency:.0f}ms | Unique:{n_unique} | Queued:{self.stats['violations_queued']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            if out_writer:
                out_writer.write(annotated)

            # Progress logging
            if processed % 30 == 0:
                avg = np.mean(all_latencies[-30:])
                print(f"  Frame {processed}/{total_frames}: {latency:.0f}ms (avg {avg:.0f}ms, {1000/avg:.1f} FPS)")

        cap.release()
        if out_writer:
            out_writer.release()

        total_time = time.time() - t_start

        # Send STOP signal to Judge
        self.queue.put(None)

        # Final summary
        summary = self._build_summary(all_latencies, total_time, w, h, fps_in)
        self._print_summary(summary)

        return summary

    def _process_frame(
        self, frame: np.ndarray, frame_w: int, frame_h: int
    ) -> tuple:
        """
        Process a single frame through YOLO + ByteTrack + triage.

        Returns:
            Tuple of (annotated_frame, list_of_frame_results)
        """
        # === YOLO with Track (ByteTrack integrated) ===
        track_results = self.yolo.model.track(
            frame,
            conf=settings.yolo_confidence_threshold,
            imgsz=settings.yolo_imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        annotated = frame.copy()
        frame_results = []

        if track_results[0].boxes is None or len(track_results[0].boxes) == 0:
            return annotated, frame_results

        boxes = track_results[0].boxes.xyxy.cpu().numpy()
        cls_ids = track_results[0].boxes.cls.cpu().numpy().astype(int)
        confs = track_results[0].boxes.conf.cpu().numpy()
        model_names = self.yolo.model.names

        # Get IDs if available from tracker
        track_ids = track_results[0].boxes.id
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)
        else:
            # Fallback if tracker fails to assign ID in this frame
            track_ids = np.zeros(len(boxes), dtype=int)

        frame_area = frame_w * frame_h

        for box, cls_id, conf, tid in zip(boxes, cls_ids, confs, track_ids):
            # Only process Person class
            name = model_names.get(int(cls_id), "").lower()
            if name != "person":
                continue

            # Skip unassigned IDs
            if tid == 0:
                continue

            # === Filter false persons (buildings, objects) ===
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            aspect_ratio = bh / bw if bw > 0 else 0
            bbox_area = bw * bh

            if aspect_ratio < self.min_aspect_ratio:
                self.stats["filtered_false_persons"] += 1
                continue
            if bbox_area / frame_area > 0.25:
                self.stats["filtered_false_persons"] += 1
                continue
            if bh < 80:
                self.stats["filtered_false_persons"] += 1
                continue
            if conf < self.min_confidence:
                self.stats["filtered_false_persons"] += 1
                continue

            self.stats["unique_persons"].add(tid)
            self.stats["total_detections"] += 1
            bbox = box

            # === Associate PPE items with this person ===
            helmet_detected = False
            vest_detected = False
            no_helmet_detected = False
            no_vest_detected = False

            for b2, c2, _ in zip(boxes, cls_ids, confs):
                # Check by name, not hardcoded ID
                name = model_names.get(int(c2), "").lower()
                if name == "person":
                    continue
                from utils.bbox_utils import is_inside_bbox, expand_bbox
                ppe_bbox = b2.tolist()
                # Expand person bbox by 20% to catch floating helmets
                expanded_bbox = expand_bbox(bbox, expand_ratio=0.2)
                if is_inside_bbox(ppe_bbox, expanded_bbox, threshold=0.01):
                    if name == "helmet":
                        helmet_detected = True
                    elif name == "vest":
                        vest_detected = True
                    elif name == "no-helmet":
                        no_helmet_detected = True
                    elif name == "no-vest":
                        no_vest_detected = True

            # === 5-Path Triage ===
            path, violation_type = self._triage(
                helmet_detected, vest_detected, no_helmet_detected, no_vest_detected
            )
            self.stats["path_counts"][path] += 1

            # === Handle based on path ===
            is_violation = False

            if path == "Fast Safe":
                # Path 0: Both PPE detected → SAFE, no action needed
                self.stats["safe_count"] += 1
            else:
                # Paths 1-4: Potential violation → check cooldown → queue for Judge
                is_violation = True
                queued = self._check_cooldown_and_queue(
                    tid, violation_type, bbox, conf, path, frame
                )

            # === Draw annotation ===
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tid} {path}"
            h_label = "H:Y" if helmet_detected else "H:N"
            v_label = "V:Y" if vest_detected else "V:N"
            label += f" {h_label} {v_label}"
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
            )

            frame_results.append({
                "person_id": int(tid),
                "path": path,
                "violation_type": violation_type,
                "is_violation": is_violation,
                "helmet": helmet_detected,
                "vest": vest_detected,
            })

        return annotated, frame_results

    def _triage(
        self, helmet: bool, vest: bool, no_helmet: bool, no_vest: bool
    ) -> tuple:
        """
        5-Path decision triage. Returns (path_name, violation_type).
        """
        # Path 0: Fast Safe
        if helmet and vest:
            return "Fast Safe", None

        # Path 1: Fast Violation (explicit no_helmet and/or no_vest)
        if no_helmet and no_vest:
            return "Fast Violation", "both_missing"
        if no_helmet:
            return "Fast Violation", "no_helmet" if vest else "both_missing"
        if no_vest:
            return "Fast Violation", "no_vest" if helmet else "both_missing"

        # Path 2: Rescue Head (vest found, no helmet)
        if vest and not helmet:
            return "Rescue Head", "no_helmet"

        # Path 3: Rescue Body (helmet found, no vest)
        if helmet and not vest:
            return "Rescue Body", "no_vest"

        # Path 4: Critical (nothing found)
        return "Critical", "both_missing"

    def _check_cooldown_and_queue(
        self,
        person_id: int,
        violation_type: str,
        bbox: list,
        confidence: float,
        path: str,
        frame: np.ndarray,
    ) -> bool:
        """
        Check cooldown and queue for Judge if not in cooldown.

        Returns:
            True if queued, False if skipped (cooldown)
        """
        now = time.time()

        # Check cooldown
        if person_id in self.cooldown_dict:
            person_cooldowns = self.cooldown_dict[person_id]
            if violation_type in person_cooldowns:
                last_time = person_cooldowns[violation_type]
                if (now - last_time) < self.cooldown_seconds:
                    self.stats["violations_cooldown_skipped"] += 1
                    return False

        # === Not in cooldown → crop ROI and queue ===

        # Crop ROI based on violation type
        roi_path = self._crop_and_save_roi(
            frame, bbox, person_id, violation_type
        )

        # Build payload
        payload = {
            "timestamp": datetime.now().isoformat(),
            "person_id": int(person_id),
            "roi_image_path": roi_path,
            "suspected_violation": violation_type,
            "decision_path": path,
            "sentry_confidence": float(confidence),
            "person_bbox": bbox,
            "camera_zone": self.camera_zone,
        }

        # Push to queue
        self.queue.put(payload)
        self.stats["violations_queued"] += 1

        # Update cooldown
        if person_id not in self.cooldown_dict:
            self.cooldown_dict[person_id] = {}
        self.cooldown_dict[person_id][violation_type] = now

        logger.info(
            f"Sentry queued: Person {person_id}, {violation_type}, path={path}"
        )

        return True

    def _crop_and_save_roi(
        self,
        frame: np.ndarray,
        bbox: list,
        person_id: int,
        violation_type: str,
    ) -> str:
        """
        Crop the relevant ROI and save to disk.

        - no_helmet → crop HEAD region (top 40%)
        - no_vest → crop TORSO region (20-100%)
        - both_missing → crop full person bbox
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person_h = y2 - y1

        if violation_type == "no_helmet":
            # Head ROI: top 40%
            head_y2 = y1 + int(person_h * 0.4)
            roi = frame[y1:head_y2, x1:x2].copy()
        elif violation_type == "no_vest":
            # Torso ROI: 20% to 100%
            torso_y1 = y1 + int(person_h * 0.2)
            roi = frame[torso_y1:y2, x1:x2].copy()
        else:
            # both_missing → full person crop
            roi = frame[y1:y2, x1:x2].copy()

        # Save
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"roi_p{person_id}_{violation_type}_{timestamp_str}.jpg"
        filepath = os.path.join(self.roi_save_dir, filename)
        cv2.imwrite(filepath, roi)

        return filepath

    def _build_summary(
        self, latencies: list, total_time: float, w: int, h: int, fps: float
    ) -> Dict[str, Any]:
        """Build final summary dict."""
        warm = latencies[3:] if len(latencies) > 3 else latencies
        return {
            "frames_processed": self.stats["frames_processed"],
            "total_time_seconds": total_time,
            "avg_latency_ms": float(np.mean(warm)) if warm else 0,
            "min_latency_ms": float(np.min(warm)) if warm else 0,
            "max_latency_ms": float(np.max(warm)) if warm else 0,
            "effective_fps": 1000 / np.mean(warm) if warm else 0,
            "resolution": f"{w}x{h}",
            "unique_persons": len(self.stats["unique_persons"]),
            "total_detections": self.stats["total_detections"],
            "violations_queued": self.stats["violations_queued"],
            "violations_cooldown_skipped": self.stats["violations_cooldown_skipped"],
            "safe_count": self.stats["safe_count"],
            "filtered_false_persons": self.stats["filtered_false_persons"],
            "path_distribution": dict(self.stats["path_counts"]),
        }

    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted summary."""
        print(f"\n{'='*50}")
        print("SENTRY RESULTS")
        print(f"{'='*50}")
        print(f"  Frames: {summary['frames_processed']}")
        print(f"  Time: {summary['total_time_seconds']:.1f}s")
        print(f"  Avg latency: {summary['avg_latency_ms']:.1f}ms")
        print(f"  FPS: {summary['effective_fps']:.1f}")
        print(f"  Unique persons: {summary['unique_persons']}")
        print(f"  Violations queued for Judge: {summary['violations_queued']}")
        print(f"  Violations skipped (cooldown): {summary['violations_cooldown_skipped']}")
        print(f"  Safe detections: {summary['safe_count']}")
        print(f"  False persons filtered: {summary['filtered_false_persons']}")
        print(f"\n  Path distribution:")
        total = sum(summary['path_distribution'].values()) or 1
        for path, count in sorted(summary['path_distribution'].items(), key=lambda x: -x[1]):
            print(f"    {path}: {count} ({count/total*100:.1f}%)")
