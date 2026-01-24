"""
Stream Processor Service

Handles video file upload and webcam stream processing.
Frame-by-frame detection with configurable frame skip.
"""

import cv2
import time
import base64
import tempfile
import os
from typing import Dict, Any, Optional, Generator, List
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from services.hybrid_detector import get_hybrid_detector
from utils.visualization import draw_detections
from config.settings import settings


@dataclass
class FrameResult:
    """Result for a single frame detection."""
    frame_number: int
    timestamp_ms: float
    persons: List[Dict[str, Any]]
    stats: Dict[str, Any]
    annotated_frame: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "persons": self.persons,
            "stats": self.stats
        }


class StreamProcessor:
    """
    Processes video streams (files or webcam) for PPE detection.
    
    Supports:
    - Video file upload (.mp4, .avi, .mov)
    - Webcam capture
    - Frame skipping for performance
    - Real-time results streaming
    """
    
    def __init__(
        self,
        frame_skip: int = 5,
        max_fps: float = 10.0
    ):
        """
        Initialize stream processor.
        
        Args:
            frame_skip: Process every Nth frame (default: 5 = 6 FPS for 30 FPS video)
            max_fps: Maximum processing FPS cap
        """
        self.frame_skip = frame_skip
        self.max_fps = max_fps
        self.detector = get_hybrid_detector()
        self.is_processing = False
        
    def process_video_file(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process a video file and return aggregated results.
        
        Args:
            video_path: Path to video file
            output_path: Optional path for annotated output video
            progress_callback: Optional callback(progress_pct, frame_result)
            
        Returns:
            Aggregated detection results for entire video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = fps / self.frame_skip  # Reduced FPS for output
            out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_count = 0
        total_violations = 0
        total_persons = 0
        all_persons = []
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Run detection
                result = self.detector.detect(frame, save_annotated=False)
                
                # Create annotated frame
                annotated = draw_detections(frame, result)
                
                # Store result
                timestamp_ms = (frame_count / fps) * 1000
                frame_result = FrameResult(
                    frame_number=frame_count,
                    timestamp_ms=timestamp_ms,
                    persons=result["persons"],
                    stats=result["stats"],
                    annotated_frame=annotated
                )
                frame_results.append(frame_result)
                
                # Update aggregates
                processed_count += 1
                total_persons += result["stats"]["total_persons"]
                total_violations += result["stats"]["total_violations"]
                all_persons.extend(result["persons"])
                
                # Write to output video
                if out:
                    out.write(annotated)
                
                # Progress callback
                if progress_callback:
                    progress_pct = (frame_count / total_frames) * 100
                    progress_callback(progress_pct, frame_result)
                    
        finally:
            cap.release()
            if out:
                out.release()
            self.is_processing = False
        
        # Calculate aggregated stats
        total_time = time.time() - start_time
        effective_fps = processed_count / total_time if total_time > 0 else 0
        compliance_rate = (
            (total_persons - total_violations) / total_persons * 100
            if total_persons > 0 else 100.0
        )
        
        return {
            "success": True,
            "video_info": {
                "path": video_path,
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": total_frames / fps if fps > 0 else 0,
                "resolution": f"{width}x{height}"
            },
            "processing_info": {
                "frames_processed": processed_count,
                "frame_skip": self.frame_skip,
                "processing_time_seconds": total_time,
                "effective_fps": effective_fps
            },
            "aggregated_stats": {
                "total_persons_detected": total_persons,
                "total_violations": total_violations,
                "compliance_rate": compliance_rate,
                "unique_violation_frames": sum(
                    1 for r in frame_results if r.stats["total_violations"] > 0
                )
            },
            "frame_results": [r.to_dict() for r in frame_results],
            "output_video_path": output_path
        }
    
    def process_video_file_streaming(
        self,
        video_path: str
    ) -> Generator[FrameResult, None, None]:
        """
        Process video file with streaming results.
        
        Yields FrameResult for each processed frame.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        self.is_processing = True
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % self.frame_skip != 0:
                    continue
                
                result = self.detector.detect(frame, save_annotated=False)
                annotated = draw_detections(frame, result)
                
                timestamp_ms = (frame_count / fps) * 1000
                
                yield FrameResult(
                    frame_number=frame_count,
                    timestamp_ms=timestamp_ms,
                    persons=result["persons"],
                    stats=result["stats"],
                    annotated_frame=annotated
                )
        finally:
            cap.release()
            self.is_processing = False
    
    def capture_webcam_frame(self, camera_index: int = 0) -> Dict[str, Any]:
        """
        Capture and process a single frame from webcam.
        
        Args:
            camera_index: Webcam device index (default: 0)
            
        Returns:
            Detection result with annotated frame
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam: {camera_index}")
        
        try:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read frame from webcam")
            
            # Run detection
            result = self.detector.detect(frame, save_annotated=False)
            annotated = draw_detections(frame, result)
            
            # Encode annotated frame as base64
            _, buffer = cv2.imencode('.jpg', annotated)
            annotated_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "success": True,
                "frame": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "annotated_base64": annotated_base64
                },
                "persons": result["persons"],
                "stats": result["stats"],
                "timing": result["timing"]
            }
        finally:
            cap.release()
    
    def stop_processing(self):
        """Stop any ongoing video processing."""
        self.is_processing = False


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Convert OpenCV frame to base64 JPEG string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return base64.b64encode(buffer).decode('utf-8')


# Global instance
_stream_processor: Optional[StreamProcessor] = None


def get_stream_processor() -> StreamProcessor:
    """Get or create global stream processor instance."""
    global _stream_processor
    
    if _stream_processor is None:
        _stream_processor = StreamProcessor()
    
    return _stream_processor
