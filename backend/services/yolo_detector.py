"""
YOLO Detector Service

YOLOv11m wrapper for PPE detection.
Handles model loading, inference, and result parsing.

Classes detected by new best.pt model:
    0: Helmet    (PPE presence — worker IS wearing helmet)
    1: Vest      (PPE presence — worker IS wearing vest)
    2: Person    (base person detection)
    3: no_helmet (PPE absence  — worker is NOT wearing helmet)
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config.settings import settings


# ── Class mapping for new best.pt model ──────────────────────────────────────
# New model has 4 classes only (simplified from old 10-class model)
CLASS_NAMES = {
    0: "Helmet",      # Worker IS wearing a helmet  (presence)
    1: "Vest",        # Worker IS wearing a vest    (presence)
    2: "Person",      # Base person detection
    3: "no_helmet",   # Worker is NOT wearing helmet (absence — direct violation signal)
}

# PPE presence classes: detected when worker HAS the equipment
PPE_PRESENCE_CLASSES = {"Helmet", "Vest"}

# PPE absence classes: detected when worker is MISSING equipment
# Note: new model only has no_helmet; vest absence is inferred from Vest not being detected
PPE_ABSENCE_CLASSES = {"no_helmet"}

# Person class ID in the new model
PERSON_CLASS_ID = 2


class YOLODetector:
    """
    YOLOv11m detector for PPE detection.
    
    This service handles:
    - Model loading and initialization
    - Running inference on images
    - Parsing detection results
    - Associating PPE items with persons
    
    Attributes:
        model: Loaded YOLO model
        confidence_threshold: Minimum confidence for detections
        device: Inference device (cuda/cpu)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLOv11m weights (uses settings if not provided)
            confidence_threshold: Min confidence (uses settings if not provided)
            device: Inference device (uses settings if not provided)
        """
        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = confidence_threshold or settings.yolo_confidence_threshold
        self.device = device or settings.sam_device  # Use same device as SAM
        
        self.model: Optional[YOLO] = None
        self._is_loaded = False
    
    def load_model(self) -> bool:
        """
        Load YOLO model from weights file.
        
        Returns:
            True if model loaded successfully
            
        Raises:
            FileNotFoundError: If model weights not found
        """
        if self._is_loaded:
            return True
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found at: {model_path}\n"
                f"Please place your trained yolov11m_best.pt in the models/ directory."
            )
        
        try:
            self.model = YOLO(str(model_path))
            self._is_loaded = True
            print(f"✅ YOLO model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def detect(
        self,
        image: np.ndarray,
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Run detection on an image.
        
        Args:
            image: Input image (BGR, H×W×C)
            return_raw: If True, also return raw YOLO results
            
        Returns:
            Detection results dict containing:
            - persons: List of person detections with bboxes
            - ppe_items: List of PPE item detections
            - raw_results: Raw YOLO output (if return_raw=True)
            - inference_time_ms: Processing time
        """
        if not self._is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # Run YOLO inference at training resolution
        results = self.model(
            image,
            conf=self.confidence_threshold,
            imgsz=settings.yolo_imgsz,  # Must match training resolution (1280)
            verbose=False,
            device=self.device
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Parse results
        persons, ppe_items = self._parse_results(results[0])
        
        # Associate PPE with persons
        persons_with_ppe = self._associate_ppe_with_persons(persons, ppe_items)
        
        result = {
            "persons": persons_with_ppe,
            "ppe_items": ppe_items,
            "inference_time_ms": inference_time,
            "total_detections": len(persons) + len(ppe_items)
        }
        
        if return_raw:
            result["raw_results"] = results
        
        return result
    
    def _parse_results(
        self,
        result
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse YOLO result into persons and PPE items.
        
        Args:
            result: Single YOLO result object
            
        Returns:
            Tuple of (persons list, ppe_items list)
        """
        persons = []
        ppe_items = []
        
        if result.boxes is None:
            return persons, ppe_items
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            bbox = box.tolist()  # [x_min, y_min, x_max, y_max]
            
            detection = {
                "id": i,
                "bbox": bbox,
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
            }
            
            if cls_id == PERSON_CLASS_ID:
                persons.append(detection)
            elif cls_id in CLASS_NAMES:
                ppe_items.append(detection)
        
        return persons, ppe_items
    
    def _associate_ppe_with_persons(
        self,
        persons: List[Dict],
        ppe_items: List[Dict]
    ) -> List[Dict]:
        """
        Associate PPE items with their corresponding persons.
        
        A PPE item is associated with a person if:
        - The PPE bbox overlaps significantly with the person bbox
        - The overlap ratio exceeds the threshold
        
        Args:
            persons: List of person detections
            ppe_items: List of PPE item detections
            
        Returns:
            Updated persons list with PPE associations
        """
        from utils.bbox_utils import is_inside_bbox
        
        for person in persons:
            person["helmet_detected"] = False
            person["vest_detected"] = False
            person["no_helmet_detected"] = False
            person["associated_ppe"] = []
        
        for ppe in ppe_items:
            ppe_bbox = ppe["bbox"]
            ppe_class = ppe["class_name"]
            best_person = None
            best_overlap = 0.0
            
            for person in persons:
                person_bbox = person["bbox"]
                
                # Check if PPE is inside person bbox
                if is_inside_bbox(ppe_bbox, person_bbox, threshold=0.5):
                    # Calculate overlap for ranking
                    overlap = self._calculate_overlap(ppe_bbox, person_bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_person = person
            
            if best_person is not None:
                best_person["associated_ppe"].append(ppe)
                
                # Map new model class names to detection flags
                if ppe_class == "Helmet":
                    best_person["helmet_detected"] = True
                elif ppe_class == "Vest":
                    best_person["vest_detected"] = True
                elif ppe_class == "no_helmet":
                    best_person["no_helmet_detected"] = True
        
        return persons
    
    def _calculate_overlap(
        self,
        inner_bbox: List[float],
        outer_bbox: List[float]
    ) -> float:
        """Calculate what fraction of inner bbox is inside outer bbox."""
        x_min = max(inner_bbox[0], outer_bbox[0])
        y_min = max(inner_bbox[1], outer_bbox[1])
        x_max = min(inner_bbox[2], outer_bbox[2])
        y_max = min(inner_bbox[3], outer_bbox[3])
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        inner_area = (inner_bbox[2] - inner_bbox[0]) * (inner_bbox[3] - inner_bbox[1])
        
        if inner_area <= 0:
            return 0.0
        
        return intersection / inner_area
    
    def get_class_names(self) -> Dict[int, str]:
        """Get the class ID to name mapping."""
        return CLASS_NAMES.copy()


# Global detector instance (singleton pattern)
_detector_instance: Optional[YOLODetector] = None


def get_yolo_detector() -> YOLODetector:
    """
    Get or create the global YOLO detector instance.
    
    Returns:
        YOLODetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = YOLODetector()
    
    return _detector_instance
