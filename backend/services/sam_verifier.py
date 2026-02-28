"""
SAM 3 Verifier Service — The "Judge"

SAM 3 (Segment Anything with Concepts) wrapper for semantic PPE verification.
Uses Promptable Concept Segmentation (PCS) to verify helmet/vest presence
when YOLO is uncertain.

CRITICAL: SAM receives CROPPED ROI, not full image!
This is the key optimization from the thesis — Geometric Prompt Engineering.

API: Uses SAM3SemanticPredictor from Ultralytics (not SAM() class).
Requires: ultralytics >= 8.3.237, correct CLIP package.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import cv2

from config.settings import settings
from utils.bbox_utils import extract_head_roi, extract_torso_roi, crop_roi


logger = logging.getLogger(__name__)

# ── Text prompts for SAM 3 concept segmentation ──────────────────────────────
# SAM 3 uses short noun phrases for concept matching.
# Multiple prompts increase recall — SAM 3 picks the best match.
HELMET_PROMPTS = ["helmet", "hard hat", "safety helmet", "construction helmet"]
VEST_PROMPTS = ["safety vest", "high visibility vest", "reflective vest", "hi-vis jacket"]


class SAMVerifier:
    """
    SAM 3 semantic verification service — the "Judge" in Sentry-Judge architecture.

    Used for the "Rescue" paths in the 5-path decision logic:
    - Path 1 override: Verify YOLO's no_helmet false positives
    - Path 2 Rescue Head: Verify helmet presence using HEAD ROI
    - Path 3 Rescue Body: Verify vest presence using TORSO ROI
    - Path 4 Critical: Verify both using respective ROIs

    CRITICAL IMPLEMENTATION DETAIL:
    SAM receives CROPPED ROI images, not full images.
    This dramatically reduces computation and is the "Geometric Prompt Engineering"
    contribution described in the thesis.

    Attributes:
        predictor: SAM3SemanticPredictor instance
        device: Inference device (cuda/cpu)
        mask_threshold: Minimum mask coverage to consider item "found" (default 5%)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: Optional[float] = None
    ):
        """
        Initialize SAM 3 verifier.

        Args:
            model_path: Path to sam3.pt weights
            device: Inference device (uses settings if not provided)
            mask_threshold: Minimum mask coverage (uses settings if not provided)
        """
        self.model_path = model_path or settings.sam_model_path
        self.device = device or settings.sam_device
        self.mask_threshold = mask_threshold or settings.sam_mask_threshold

        self.predictor = None
        self._is_loaded = False
        self._use_mock = False

        # Statistics for thesis metrics
        self._stats = {
            "total_verifications": 0,
            "helmets_found": 0,
            "vests_found": 0,
            "total_time_ms": 0.0,
            "errors": 0,
        }

    def load_model(self) -> bool:
        """
        Load SAM 3 model using SAM3SemanticPredictor.

        Returns:
            True if model loaded successfully

        Note:
            Falls back to mock mode if SAM 3 is not available (e.g., CPU dev).
        """
        if self._is_loaded:
            return True

        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.warning(f"SAM 3 weights not found at {model_file}. Using mock mode.")
            self._use_mock = True
            self._is_loaded = True
            print(f"⚠️ SAM 3 weights not found at {model_file}")
            print("📌 Using mock SAM verifier for development")
            return True

        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=0.15,               # Lower conf catches more subtle PPE
                task="segment",
                mode="predict",
                model=str(model_file),
                half=True,               # FP16 for faster GPU inference
                verbose=False,
            )
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            # Warm up the model by setting up internal state
            self.predictor.setup_model()
            self._is_loaded = True
            self._use_mock = False
            print(f"✅ SAM 3 loaded from {model_file} (device={self.device})")
            return True

        except ImportError as e:
            logger.warning(f"SAM 3 not available: {e}")
            print("⚠️ SAM 3 not available. Ensure ultralytics >= 8.3.237")
            print("   pip install -U ultralytics")
            print("   pip uninstall clip -y")
            print("   pip install git+https://github.com/ultralytics/CLIP.git")
            self._use_mock = True
            self._is_loaded = True
            print("📌 Using mock SAM verifier for development")
            return True

        except Exception as e:
            logger.warning(f"Failed to load SAM 3: {e}")
            print(f"⚠️ Failed to load SAM 3: {e}")
            self._use_mock = True
            self._is_loaded = True
            print("📌 Using mock SAM verifier for development")
            return True

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def is_mock(self) -> bool:
        """Check if running in mock mode (no real SAM)."""
        return self._use_mock

    # ── Public verification methods ──────────────────────────────────────────

    def verify_helmet(
        self,
        full_image: np.ndarray,
        person_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Verify helmet presence in HEAD ROI.

        Called for:
        - Path 1 override: Double-check YOLO's no_helmet detection
        - Path 2 (Rescue Head): Vest found, no helmet info

        Args:
            full_image: Full input image (BGR, H×W×C)
            person_bbox: Person bounding box [x_min, y_min, x_max, y_max]

        Returns:
            Verification result dict with helmet_found, confidence, roi_bbox, processing_time_ms
        """
        head_roi = extract_head_roi(person_bbox)
        return self._verify_roi(full_image, head_roi, HELMET_PROMPTS, item_type="helmet")

    def verify_vest(
        self,
        full_image: np.ndarray,
        person_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Verify vest presence in TORSO ROI.

        Called for Path 3 (Rescue Body): Helmet found, no vest info.

        Args:
            full_image: Full input image (BGR, H×W×C)
            person_bbox: Person bounding box [x_min, y_min, x_max, y_max]

        Returns:
            Verification result dict with vest_found, confidence, roi_bbox, processing_time_ms
        """
        torso_roi = extract_torso_roi(person_bbox)
        return self._verify_roi(full_image, torso_roi, VEST_PROMPTS, item_type="vest")

    def verify_both(
        self,
        full_image: np.ndarray,
        person_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Verify both helmet and vest presence.

        Called for Path 4 (Critical): YOLO detected nothing for this person.

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
            "processing_time_ms": total_time,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get SAM verification statistics for thesis metrics."""
        stats = self._stats.copy()
        if stats["total_verifications"] > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_verifications"]
        else:
            stats["avg_time_ms"] = 0.0
        stats["mock_mode"] = self._use_mock
        return stats

    # ── Internal methods ─────────────────────────────────────────────────────

    def _verify_roi(
        self,
        full_image: np.ndarray,
        roi_bbox: List[int],
        prompts: List[str],
        item_type: str
    ) -> Dict[str, Any]:
        """
        Run SAM 3 verification on a cropped ROI.

        CRITICAL: We crop the ROI first, then run SAM on the crop.
        This is MUCH faster than running SAM on the full image.
        This is the "Geometric Prompt Engineering" from the thesis.

        Args:
            full_image: Full input image
            roi_bbox: ROI bounding box to crop
            prompts: Text prompts for SAM 3 concept segmentation
            item_type: "helmet" or "vest"

        Returns:
            Verification result dict
        """
        if not self._is_loaded:
            self.load_model()

        start_time = time.time()

        # CRITICAL: Crop the ROI first — Geometric Prompt Engineering
        roi_crop = crop_roi(full_image, roi_bbox)

        # Validate crop size (too small = unreliable)
        if roi_crop.size == 0 or min(roi_crop.shape[:2]) < 20:
            return {
                f"{item_type}_found": False,
                "confidence": 0.0,
                "roi_bbox": roi_bbox,
                "processing_time_ms": 0.0,
                "error": "ROI too small",
            }

        # Use mock mode for development without GPU/SAM
        if self._use_mock:
            result = self._mock_verification(item_type)
        else:
            result = self._run_sam3_verification(roi_crop, prompts, item_type)

        processing_time = (time.time() - start_time) * 1000
        result["roi_bbox"] = roi_bbox
        result["processing_time_ms"] = processing_time

        # Update stats
        self._stats["total_verifications"] += 1
        self._stats["total_time_ms"] += processing_time
        if result.get(f"{item_type}_found", False):
            if item_type == "helmet":
                self._stats["helmets_found"] += 1
            else:
                self._stats["vests_found"] += 1

        return result

    def _run_sam3_verification(
        self,
        roi_crop: np.ndarray,
        prompts: List[str],
        item_type: str
    ) -> Dict[str, Any]:
        """
        Run SAM 3 concept segmentation on cropped ROI.

        Uses SAM3SemanticPredictor API:
        1. set_image(roi_crop)  — encode image features once
        2. predictor(text=prompts)  — query for each concept
        3. Check mask coverage against threshold

        Args:
            roi_crop: Cropped ROI image (BGR, H×W×C)
            prompts: Text prompts for concept segmentation
            item_type: "helmet" or "vest"

        Returns:
            Verification result with found status and confidence
        """
        try:
            # Ensure minimum size for SAM 3 (resize small crops)
            h, w = roi_crop.shape[:2]
            if max(h, w) < 64:
                scale = 64 / max(h, w)
                roi_crop = cv2.resize(roi_crop, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_LINEAR)

            # Step 1: Set image (encodes features)
            self.predictor.set_image(roi_crop)

            # Step 2: Query with text prompts
            results = self.predictor(text=prompts)

            # Step 3: Analyze masks
            if not results or results[0].masks is None or len(results[0].masks.data) == 0:
                return {
                    f"{item_type}_found": False,
                    "confidence": 0.0,
                }

            # Calculate maximum mask coverage across all returned masks
            max_coverage = 0.0
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy()
                coverage = float(np.sum(mask_np)) / float(mask_np.size)
                max_coverage = max(max_coverage, coverage)

            # Check against threshold
            found = max_coverage > self.mask_threshold

            logger.debug(
                f"SAM3 {item_type}: coverage={max_coverage:.3f}, "
                f"threshold={self.mask_threshold}, found={found}"
            )

            return {
                f"{item_type}_found": found,
                "confidence": float(max_coverage),
            }

        except Exception as e:
            logger.error(f"SAM 3 verification error: {e}")
            self._stats["errors"] += 1
            return {
                f"{item_type}_found": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def _mock_verification(self, item_type: str) -> Dict[str, Any]:
        """
        Mock verification for development without SAM 3.

        Returns random-ish results for testing the pipeline.
        """
        import random

        # Mock: 30% chance of finding the item
        found = random.random() < 0.3
        confidence = random.uniform(0.1, 0.8) if found else random.uniform(0.0, 0.04)

        return {
            f"{item_type}_found": found,
            "confidence": confidence,
            "mock": True,
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_verifier_instance: Optional[SAMVerifier] = None


def get_sam_verifier() -> SAMVerifier:
    """
    Get or create the global SAM 3 verifier instance.

    Returns:
        SAMVerifier instance
    """
    global _verifier_instance

    if _verifier_instance is None:
        _verifier_instance = SAMVerifier()

    return _verifier_instance
