"""
SAM Verifier Service

SAM (Segment Anything Model) wrapper for semantic verification.
Used to verify helmet/vest presence when YOLO is uncertain.

CRITICAL: SAM receives CROPPED ROI, not full image!
This is the key optimization from the thesis.
"""

import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from config.settings import settings
from utils.bbox_utils import extract_head_roi, extract_torso_roi, crop_roi


# Text prompts for semantic verification
HELMET_PROMPTS = ["helmet", "hard hat", "safety helmet", "construction helmet"]
VEST_PROMPTS = ["vest", "safety vest", "high visibility vest", "reflective vest"]


class SAMVerifier:
    """
    SAM semantic verification service.
    
    Used for the "Rescue" paths in the 5-path decision logic:
    - Rescue Head: Verify helmet presence using HEAD ROI
    - Rescue Body: Verify vest presence using TORSO ROI
    - Critical: Verify both using respective ROIs
    
    CRITICAL IMPLEMENTATION DETAIL:
    SAM receives CROPPED ROI images, not full images.
    This dramatically reduces computation time.
    
    Attributes:
        model: Loaded SAM model
        device: Inference device (cuda/cpu)
        mask_threshold: Minimum mask coverage (5%)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: Optional[float] = None
    ):
        """
        Initialize SAM verifier.
        
        Args:
            model_path: Path to SAM weights (uses settings if not provided)
            device: Inference device (uses settings if not provided)
            mask_threshold: Minimum mask coverage (uses settings if not provided)
        """
        self.model_path = model_path or settings.sam_model_path
        self.device = device or settings.sam_device
        self.mask_threshold = mask_threshold or settings.sam_mask_threshold
        
        self.model = None
        self._is_loaded = False
        self._use_mock = False  # Use mock for CPU-only development
    
    def load_model(self) -> bool:
        """
        Load SAM model from weights file.
        
        Returns:
            True if model loaded successfully
            
        Note:
            If model loading fails and we're on CPU, falls back to mock mode.
        """
        if self._is_loaded:
            return True
        
        try:
            # Try to load SAM model
            # Note: This requires the segment-anything package
            from ultralytics import SAM
            
            self.model = SAM(self.model_path)
            self._is_loaded = True
            print(f"✅ SAM model loaded from {self.model_path}")
            return True
            
        except ImportError:
            print("⚠️ SAM package not installed. Install with:")
            print("   pip install git+https://github.com/facebookresearch/segment-anything.git")
            self._use_mock = True
            self._is_loaded = True
            print("📌 Using mock SAM verifier for development")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load SAM model: {e}")
            self._use_mock = True
            self._is_loaded = True
            print("📌 Using mock SAM verifier for development")
            return True
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def verify_helmet(
        self,
        full_image: np.ndarray,
        person_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Verify helmet presence in HEAD ROI.
        
        This is called for Path 2 (Rescue Head):
        - YOLO detected vest but no helmet
        - SAM checks the HEAD region for helmet
        
        Args:
            full_image: Full input image (BGR, H×W×C)
            person_bbox: Person bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            Verification result dict:
            - helmet_found: Boolean
            - confidence: Mask coverage ratio
            - roi_bbox: The HEAD ROI used
            - processing_time_ms: Time taken
        """
        head_roi = extract_head_roi(person_bbox)
        return self._verify_roi(
            full_image,
            head_roi,
            HELMET_PROMPTS,
            item_type="helmet"
        )
    
    def verify_vest(
        self,
        full_image: np.ndarray,
        person_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Verify vest presence in TORSO ROI.
        
        This is called for Path 3 (Rescue Body):
        - YOLO detected helmet but no vest
        - SAM checks the TORSO region for vest
        
        Args:
            full_image: Full input image (BGR, H×W×C)
            person_bbox: Person bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            Verification result dict:
            - vest_found: Boolean
            - confidence: Mask coverage ratio
            - roi_bbox: The TORSO ROI used
            - processing_time_ms: Time taken
        """
        torso_roi = extract_torso_roi(person_bbox)
        return self._verify_roi(
            full_image,
            torso_roi,
            VEST_PROMPTS,
            item_type="vest"
        )
    
    def verify_both(
        self,
        full_image: np.ndarray,
        person_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Verify both helmet and vest presence.
        
        This is called for Path 4 (Critical):
        - YOLO detected nothing for this person
        - SAM checks both HEAD and TORSO regions
        
        Args:
            full_image: Full input image (BGR, H×W×C)
            person_bbox: Person bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            Combined verification result
        """
        start_time = time.time()
        
        helmet_result = self.verify_helmet(full_image, person_bbox)
        vest_result = self.verify_vest(full_image, person_bbox)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "helmet_found": helmet_result["helmet_found"],
            "vest_found": vest_result["vest_found"],
            "helmet_confidence": helmet_result["confidence"],
            "vest_confidence": vest_result["confidence"],
            "head_roi": helmet_result["roi_bbox"],
            "torso_roi": vest_result["roi_bbox"],
            "processing_time_ms": total_time
        }
    
    def _verify_roi(
        self,
        full_image: np.ndarray,
        roi_bbox: List[int],
        prompts: List[str],
        item_type: str
    ) -> Dict[str, Any]:
        """
        Run SAM verification on a cropped ROI.
        
        CRITICAL: We crop the ROI first, then run SAM on the crop.
        This is MUCH faster than running SAM on the full image.
        
        Args:
            full_image: Full input image
            roi_bbox: ROI bounding box to crop
            prompts: Text prompts for SAM
            item_type: "helmet" or "vest"
            
        Returns:
            Verification result dict
        """
        if not self._is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        # CRITICAL: Crop the ROI first!
        roi_crop = crop_roi(full_image, roi_bbox)
        
        # Validate crop size
        if roi_crop.size == 0 or min(roi_crop.shape[:2]) < 20:
            return {
                f"{item_type}_found": False,
                "confidence": 0.0,
                "roi_bbox": roi_bbox,
                "processing_time_ms": 0.0,
                "error": "ROI too small"
            }
        
        # Use mock mode for development without GPU
        if self._use_mock:
            result = self._mock_verification(item_type)
        else:
            result = self._run_sam_verification(roi_crop, prompts, item_type)
        
        processing_time = (time.time() - start_time) * 1000
        result["roi_bbox"] = roi_bbox
        result["processing_time_ms"] = processing_time
        
        return result
    
    def _run_sam_verification(
        self,
        roi_crop: np.ndarray,
        prompts: List[str],
        item_type: str
    ) -> Dict[str, Any]:
        """
        Run actual SAM inference on cropped ROI.
        
        Args:
            roi_crop: Cropped ROI image
            prompts: Text prompts
            item_type: Item being verified
            
        Returns:
            Verification result
        """
        try:
            # Run SAM with text prompts
            results = self.model(
                roi_crop,
                texts=prompts,
                imgsz=640,
                verbose=False
            )
            
            # Check mask coverage
            if not results or not results[0].masks:
                return {
                    f"{item_type}_found": False,
                    "confidence": 0.0
                }
            
            # Calculate max mask coverage
            max_coverage = 0.0
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                coverage = np.sum(mask_np) / mask_np.size
                max_coverage = max(max_coverage, coverage)
            
            # Check against threshold
            found = max_coverage > self.mask_threshold
            
            return {
                f"{item_type}_found": found,
                "confidence": float(max_coverage)
            }
            
        except Exception as e:
            print(f"⚠️ SAM verification error: {e}")
            return {
                f"{item_type}_found": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _mock_verification(self, item_type: str) -> Dict[str, Any]:
        """
        Mock verification for development without SAM.
        
        Returns random-ish results for testing the pipeline.
        """
        import random
        
        # Mock: 30% chance of finding the item
        found = random.random() < 0.3
        confidence = random.uniform(0.1, 0.8) if found else random.uniform(0.0, 0.04)
        
        return {
            f"{item_type}_found": found,
            "confidence": confidence,
            "mock": True
        }


# Global verifier instance (singleton pattern)
_verifier_instance: Optional[SAMVerifier] = None


def get_sam_verifier() -> SAMVerifier:
    """
    Get or create the global SAM verifier instance.
    
    Returns:
        SAMVerifier instance
    """
    global _verifier_instance
    
    if _verifier_instance is None:
        _verifier_instance = SAMVerifier()
    
    return _verifier_instance
