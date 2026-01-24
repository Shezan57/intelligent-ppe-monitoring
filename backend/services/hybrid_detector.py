"""
Hybrid Detector Service

The core innovation of the thesis: 5-path intelligent bypass mechanism.
Combines YOLO (fast) with SAM (accurate) for optimal performance.

5-Path Decision Logic:
- Path 0 (Fast Safe):      ~45% - Helmet + vest detected → SAFE, no SAM
- Path 1 (Fast Violation): ~35% - "no_helmet" detected → VIOLATION, no SAM
- Path 2 (Rescue Head):    ~10% - Vest found, no helmet → SAM checks HEAD ROI
- Path 3 (Rescue Body):    ~5%  - Helmet found, no vest → SAM checks TORSO ROI  
- Path 4 (Critical):       ~5%  - Both missing → SAM checks both ROIs

Result: 79.8% bypass rate, 20.2% SAM activation
Performance: 28.5 FPS (vs 35 FPS YOLO-only, vs 0.79 FPS SAM-only)
Precision: 62.5% (+6.3% improvement over YOLO-only baseline)
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2

from services.yolo_detector import YOLODetector, get_yolo_detector
from services.sam_verifier import SAMVerifier, get_sam_verifier
from utils.visualization import draw_detections
from config.settings import settings


class DecisionPath(Enum):
    """5-path decision paths."""
    FAST_SAFE = "Fast Safe"           # Path 0: Both detected
    FAST_VIOLATION = "Fast Violation" # Path 1: no_helmet detected
    RESCUE_HEAD = "Rescue Head"       # Path 2: SAM verify helmet
    RESCUE_BODY = "Rescue Body"       # Path 3: SAM verify vest
    CRITICAL = "Critical"             # Path 4: SAM verify both


@dataclass
class PersonResult:
    """Detection result for a single person."""
    person_id: int
    bbox: List[float]
    confidence: float
    has_helmet: bool
    has_vest: bool
    is_violation: bool
    violation_type: Optional[str]
    decision_path: str
    sam_activated: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "has_helmet": self.has_helmet,
            "has_vest": self.has_vest,
            "is_violation": self.is_violation,
            "violation_type": self.violation_type,
            "decision_path": self.decision_path,
            "sam_activated": self.sam_activated
        }


class HybridDetector:
    """
    Hybrid YOLO + SAM detector with 5-path intelligent bypass.
    
    This is the main detection service that implements the thesis's
    key innovation: using YOLO for fast detection and SAM for
    semantic verification only when needed (20.2% of cases).
    
    Process:
    1. YOLO detects all persons and PPE items
    2. For each person, determine decision path
    3. Execute path (bypass SAM if possible)
    4. Collect results and statistics
    
    Attributes:
        yolo_detector: YOLO detector instance
        sam_verifier: SAM verifier instance
        enable_sam: Whether to enable SAM (can disable for speed testing)
    """
    
    def __init__(
        self,
        yolo_detector: Optional[YOLODetector] = None,
        sam_verifier: Optional[SAMVerifier] = None,
        enable_sam: bool = True
    ):
        """
        Initialize hybrid detector.
        
        Args:
            yolo_detector: YOLO detector (uses global instance if not provided)
            sam_verifier: SAM verifier (uses global instance if not provided)
            enable_sam: Enable SAM verification (default True)
        """
        self.yolo_detector = yolo_detector or get_yolo_detector()
        self.sam_verifier = sam_verifier or get_sam_verifier()
        self.enable_sam = enable_sam
    
    def detect(
        self,
        image: np.ndarray,
        save_annotated: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run hybrid detection on an image.
        
        This is the main entry point for detection.
        
        Args:
            image: Input image (BGR, H×W×C)
            save_annotated: Whether to save annotated image
            output_path: Path for annotated image (if save_annotated)
            
        Returns:
            Detection result dict containing:
            - success: Whether detection completed
            - persons: List of PersonResult dicts
            - timing: Processing time breakdown
            - stats: Summary statistics
            - annotated_image: Annotated image (if requested)
        """
        total_start = time.time()
        
        # === Step 1: YOLO Detection ===
        yolo_start = time.time()
        yolo_results = self.yolo_detector.detect(image)
        yolo_time = (time.time() - yolo_start) * 1000
        
        persons_raw = yolo_results["persons"]
        
        # === Step 2: Process each person through 5-path logic ===
        sam_start = time.time()
        persons_processed = []
        path_counts = {path.value: 0 for path in DecisionPath}
        sam_activations = 0
        
        for i, person in enumerate(persons_raw):
            result, path, used_sam = self._process_person(
                person, image, person_id=i
            )
            persons_processed.append(result)
            path_counts[path.value] += 1
            if used_sam:
                sam_activations += 1
        
        sam_time = (time.time() - sam_start) * 1000
        
        # === Step 3: Calculate statistics ===
        total_persons = len(persons_processed)
        total_violations = sum(1 for p in persons_processed if p.is_violation)
        compliance_rate = (
            (total_persons - total_violations) / total_persons * 100
            if total_persons > 0 else 100.0
        )
        bypass_rate = (
            (total_persons - sam_activations) / total_persons * 100
            if total_persons > 0 else 100.0
        )
        
        postprocess_start = time.time()
        
        # === Step 4: Generate annotated image ===
        annotated_image = None
        annotated_path = None
        
        if save_annotated or output_path:
            detection_result_for_viz = {
                "persons": [p.to_dict() for p in persons_processed]
            }
            annotated_image = draw_detections(image, detection_result_for_viz)
            
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                annotated_path = output_path
        
        postprocess_time = (time.time() - postprocess_start) * 1000
        total_time = (time.time() - total_start) * 1000
        
        return {
            "success": True,
            "message": "Detection completed",
            "persons": [p.to_dict() for p in persons_processed],
            "timing": {
                "total_ms": total_time,
                "yolo_ms": yolo_time,
                "sam_ms": sam_time,
                "postprocess_ms": postprocess_time
            },
            "stats": {
                "total_persons": total_persons,
                "total_violations": total_violations,
                "compliance_rate": compliance_rate,
                "sam_activations": sam_activations,
                "bypass_rate": bypass_rate,
                "path_distribution": path_counts
            },
            "annotated_image_path": annotated_path
        }
    
    def _process_person(
        self,
        person: Dict[str, Any],
        image: np.ndarray,
        person_id: int
    ) -> tuple:
        """
        Process a single person through 5-path decision logic.
        
        Args:
            person: Person detection from YOLO
            image: Full input image
            person_id: Unique ID for this person
            
        Returns:
            Tuple of (PersonResult, DecisionPath, sam_was_used)
        """
        bbox = person["bbox"]
        confidence = person["confidence"]
        
        # Get YOLO's PPE detection results
        helmet_detected = person.get("helmet_detected", False)
        vest_detected = person.get("vest_detected", False)
        no_helmet_detected = person.get("no_helmet_detected", False)
        
        # === 5-PATH DECISION LOGIC ===
        
        # PATH 0: Fast Safe
        # Both helmet and vest detected → Safe, no SAM needed
        if helmet_detected and vest_detected:
            return self._create_result(
                person_id, bbox, confidence,
                has_helmet=True,
                has_vest=True,
                path=DecisionPath.FAST_SAFE,
                sam_used=False
            )
        
        # PATH 1: Fast Violation
        # YOLO explicitly detected "no_helmet" class → Violation, no SAM needed
        if no_helmet_detected:
            return self._create_result(
                person_id, bbox, confidence,
                has_helmet=False,
                has_vest=vest_detected,
                path=DecisionPath.FAST_VIOLATION,
                sam_used=False
            )
        
        # === UNCERTAIN PATHS - Need SAM verification ===
        
        if not self.enable_sam:
            # SAM disabled - make best guess
            return self._create_result(
                person_id, bbox, confidence,
                has_helmet=helmet_detected,
                has_vest=vest_detected,
                path=DecisionPath.FAST_VIOLATION,
                sam_used=False
            )
        
        # PATH 2: Rescue Head
        # Vest found but no helmet → SAM checks HEAD ROI
        if vest_detected and not helmet_detected:
            sam_result = self.sam_verifier.verify_helmet(image, bbox)
            return self._create_result(
                person_id, bbox, confidence,
                has_helmet=sam_result.get("helmet_found", False),
                has_vest=True,
                path=DecisionPath.RESCUE_HEAD,
                sam_used=True
            )
        
        # PATH 3: Rescue Body
        # Helmet found but no vest → SAM checks TORSO ROI
        if helmet_detected and not vest_detected:
            sam_result = self.sam_verifier.verify_vest(image, bbox)
            return self._create_result(
                person_id, bbox, confidence,
                has_helmet=True,
                has_vest=sam_result.get("vest_found", False),
                path=DecisionPath.RESCUE_BODY,
                sam_used=True
            )
        
        # PATH 4: Critical
        # Nothing detected → SAM checks both ROIs
        sam_result = self.sam_verifier.verify_both(image, bbox)
        return self._create_result(
            person_id, bbox, confidence,
            has_helmet=sam_result.get("helmet_found", False),
            has_vest=sam_result.get("vest_found", False),
            path=DecisionPath.CRITICAL,
            sam_used=True
        )
    
    def _create_result(
        self,
        person_id: int,
        bbox: List[float],
        confidence: float,
        has_helmet: bool,
        has_vest: bool,
        path: DecisionPath,
        sam_used: bool
    ) -> tuple:
        """Create PersonResult from detection data."""
        is_violation = not (has_helmet and has_vest)
        
        violation_type = None
        if is_violation:
            if not has_helmet and not has_vest:
                violation_type = "both_missing"
            elif not has_helmet:
                violation_type = "no_helmet"
            else:
                violation_type = "no_vest"
        
        result = PersonResult(
            person_id=person_id,
            bbox=bbox,
            confidence=confidence,
            has_helmet=has_helmet,
            has_vest=has_vest,
            is_violation=is_violation,
            violation_type=violation_type,
            decision_path=path.value,
            sam_activated=sam_used
        )
        
        return result, path, sam_used


# Global instance
_hybrid_detector: Optional[HybridDetector] = None


def get_hybrid_detector(enable_sam: Optional[bool] = None) -> HybridDetector:
    """
    Get or create the global hybrid detector instance.
    
    Args:
        enable_sam: Whether to enable SAM verification. 
                    If None, uses settings.sam_enabled
        
    Returns:
        HybridDetector instance
    """
    global _hybrid_detector
    
    if _hybrid_detector is None:
        # Use settings.sam_enabled if not explicitly specified
        sam_enabled = enable_sam if enable_sam is not None else settings.sam_enabled
        _hybrid_detector = HybridDetector(enable_sam=sam_enabled)
    
    return _hybrid_detector
