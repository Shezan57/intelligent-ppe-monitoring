"""
Bounding Box Utilities

Functions for ROI extraction and bbox calculations.
Based on thesis specification:
- Head ROI: Top 40% of person bbox
- Torso ROI: 20% to 100% of person bbox
"""

from typing import List, Tuple
import numpy as np

from config.settings import settings


def extract_head_roi(person_bbox: List[float]) -> List[int]:
    """
    Extract head region of interest from person bounding box.
    
    The head ROI is the top 40% of the person bbox, used for
    helmet detection verification with SAM.
    
    Args:
        person_bbox: Person bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        Head ROI [x_min, y_min, x_max, head_y_max]
        
    Example:
        >>> extract_head_roi([100, 50, 200, 350])
        [100, 50, 200, 170]  # Top 40% of height
    """
    x_min, y_min, x_max, y_max = person_bbox
    height = y_max - y_min
    head_y_max = int(y_min + height * settings.head_roi_ratio)
    
    return [int(x_min), int(y_min), int(x_max), head_y_max]


def extract_torso_roi(person_bbox: List[float]) -> List[int]:
    """
    Extract torso region of interest from person bounding box.
    
    The torso ROI is from 20% to 100% of the person bbox, used for
    vest detection verification with SAM.
    
    Args:
        person_bbox: Person bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        Torso ROI [x_min, torso_y_min, x_max, y_max]
        
    Example:
        >>> extract_torso_roi([100, 50, 200, 350])
        [100, 110, 200, 350]  # From 20% to bottom
    """
    x_min, y_min, x_max, y_max = person_bbox
    height = y_max - y_min
    torso_y_min = int(y_min + height * settings.torso_roi_start)
    
    return [int(x_min), torso_y_min, int(x_max), int(y_max)]


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bboxes.
    
    Args:
        bbox1: First bounding box [x_min, y_min, x_max, y_max]
        bbox2: Second bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU value between 0 and 1
    """
    # Get intersection coordinates
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def is_inside_bbox(inner: List[float], outer: List[float], threshold: float = 0.5) -> bool:
    """
    Check if inner bbox is inside outer bbox.
    
    Uses IoU-like calculation to determine if inner bbox
    overlaps significantly with outer bbox.
    
    Args:
        inner: Inner bounding box [x_min, y_min, x_max, y_max]
        outer: Outer bounding box [x_min, y_min, x_max, y_max]
        threshold: Minimum overlap ratio (default 0.5)
        
    Returns:
        True if inner is inside outer with sufficient overlap
    """
    # Get intersection
    x_min = max(inner[0], outer[0])
    y_min = max(inner[1], outer[1])
    x_max = min(inner[2], outer[2])
    y_max = min(inner[3], outer[3])
    
    if x_max <= x_min or y_max <= y_min:
        return False
    
    intersection = (x_max - x_min) * (y_max - y_min)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    
    if inner_area <= 0:
        return False
    
    overlap_ratio = intersection / inner_area
    return overlap_ratio >= threshold


def expand_bbox(
    bbox: List[float], 
    expand_ratio: float = 0.1, 
    image_shape: Tuple[int, int] = None
) -> List[int]:
    """
    Expand bounding box by a ratio.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        expand_ratio: Ratio to expand (default 10%)
        image_shape: Image (height, width) to clip bbox
        
    Returns:
        Expanded bbox [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    # Expand
    x_min = x_min - width * expand_ratio
    y_min = y_min - height * expand_ratio
    x_max = x_max + width * expand_ratio
    y_max = y_max + height * expand_ratio
    
    # Clip to image bounds if provided
    if image_shape is not None:
        img_h, img_w = image_shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def crop_roi(image: np.ndarray, roi_bbox: List[int]) -> np.ndarray:
    """
    Crop region of interest from image.
    
    Args:
        image: Input image (H, W, C)
        roi_bbox: ROI bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        Cropped image region
    """
    x_min, y_min, x_max, y_max = roi_bbox
    
    # Ensure valid bounds
    h, w = image.shape[:2]
    x_min = max(0, min(x_min, w - 1))
    y_min = max(0, min(y_min, h - 1))
    x_max = max(x_min + 1, min(x_max, w))
    y_max = max(y_min + 1, min(y_max, h))
    
    return image[y_min:y_max, x_min:x_max].copy()


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Get center point of bounding box.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        Center point (x, y)
    """
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)


def get_bbox_area(bbox: List[float]) -> float:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        
    Returns:
        Area in pixels squared
    """
    x_min, y_min, x_max, y_max = bbox
    return max(0, (x_max - x_min) * (y_max - y_min))


def get_required_aspect_ratio(
    h: float, w: float,
    strict: float = None,
    lenient: float = None,
    dim_low: int = None,
    dim_high: int = None,
) -> float:
    """
    Dynamically compute the minimum h/w aspect ratio for a person ROI.

    Smaller crops need a stricter (higher) ratio — a tiny nearly-square
    blob is almost never a person, but a large close-up can be.

    Uses settings defaults if parameters not provided.

    Args:
        h: Crop height in pixels
        w: Crop width in pixels
        strict: Required h/w for tiny crops (default from settings)
        lenient: Required h/w for large crops (default from settings)
        dim_low: Below this min_dim → use strict ratio
        dim_high: Above this min_dim → use lenient ratio

    Returns:
        Required minimum h/w aspect ratio
    """
    strict = strict if strict is not None else settings.aspect_ratio_strict
    lenient = lenient if lenient is not None else settings.aspect_ratio_lenient
    dim_low = dim_low if dim_low is not None else settings.aspect_dim_low
    dim_high = dim_high if dim_high is not None else settings.aspect_dim_high

    min_dim = min(h, w)
    if min_dim <= dim_low:
        return strict
    elif min_dim >= dim_high:
        return lenient
    else:
        # Linear interpolation between strict and lenient
        t = (min_dim - dim_low) / (dim_high - dim_low)
        return strict + t * (lenient - strict)


def passes_person_filters(bbox: List[float]) -> tuple:
    """
    Check if a person bbox passes the dynamic aspect ratio and min area filters.

    Args:
        bbox: Person bounding box [x_min, y_min, x_max, y_max]

    Returns:
        (passes: bool, reason: str or None)
        reason is set only when passes=False
    """
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    area = bw * bh
    aspect_ratio = bh / bw if bw > 0 else 0

    # Min area check
    if area < settings.min_person_area:
        return False, f"Too small (area {area:.0f}px² < {settings.min_person_area})"

    # Dynamic aspect ratio check
    required = get_required_aspect_ratio(bh, bw)
    if aspect_ratio < required:
        return False, f"Bad aspect ratio {aspect_ratio:.2f} < {required:.2f} (machine/vehicle)"

    return True, None
