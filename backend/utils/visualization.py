"""
Visualization Utilities

Functions for drawing bounding boxes and annotations on images.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


# Color constants (BGR format for OpenCV)
COLOR_SAFE = (75, 177, 16)       # Green - #10b981
COLOR_VIOLATION = (68, 68, 239)  # Red - #ef4444
COLOR_HELMET = (0, 255, 0)       # Bright green
COLOR_VEST = (255, 255, 0)       # Cyan
COLOR_PERSON = (255, 165, 0)     # Orange
COLOR_NO_HELMET = (0, 0, 255)    # Red


def draw_single_bbox(
    image: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw a single bounding box with label on image.
    
    Args:
        image: Input image (BGR)
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        label: Text label to display
        color: BGR color tuple
        thickness: Line thickness
        font_scale: Font size scale
        
    Returns:
        Annotated image
    """
    img = image.copy()
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    
    # Draw rectangle
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    
    # Draw label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    
    # Position label above bbox
    label_y = max(y_min - 10, text_height + 5)
    
    cv2.rectangle(
        img,
        (x_min, label_y - text_height - 5),
        (x_min + text_width + 10, label_y + 5),
        color,
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        img,
        label,
        (x_min + 5, label_y),
        font,
        font_scale,
        (255, 255, 255),  # White text
        thickness
    )
    
    return img


def draw_detections(
    image: np.ndarray,
    detection_result: Dict,
    show_confidence: bool = True,
    show_decision_path: bool = True
) -> np.ndarray:
    """
    Draw all detections from hybrid detector result.
    
    Color coding:
    - Green (#10b981): Safe (helmet + vest)
    - Red (#ef4444): Violation (missing PPE)
    
    Args:
        image: Input image (BGR)
        detection_result: Result from HybridDetector.detect()
        show_confidence: Show detection confidence
        show_decision_path: Show which decision path was used
        
    Returns:
        Annotated image with all detections
    """
    img = image.copy()
    
    # Get persons from detection result
    persons = detection_result.get("persons", [])
    
    for person in persons:
        bbox = person["bbox"]
        has_helmet = person.get("has_helmet", False)
        has_vest = person.get("has_vest", False)
        confidence = person.get("confidence", 0)
        decision_path = person.get("decision_path", "Unknown")
        
        # Determine status and color
        if has_helmet and has_vest:
            status = "SAFE"
            color = COLOR_SAFE
        else:
            missing = []
            if not has_helmet:
                missing.append("No Helmet")
            if not has_vest:
                missing.append("No Vest")
            status = ", ".join(missing)
            color = COLOR_VIOLATION
        
        # Build label
        label_parts = [status]
        if show_confidence and confidence > 0:
            label_parts.append(f"{confidence:.0%}")
        if show_decision_path:
            label_parts.append(f"[{decision_path}]")
        
        label = " | ".join(label_parts)
        
        # Draw bbox with label
        img = draw_single_bbox(img, bbox, label, color)
    
    return img


def draw_roi_overlay(
    image: np.ndarray,
    person_bbox: List[float],
    head_roi: Optional[List[float]] = None,
    torso_roi: Optional[List[float]] = None,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw ROI overlay on image for visualization.
    
    Useful for debugging SAM verification regions.
    
    Args:
        image: Input image (BGR)
        person_bbox: Person bounding box
        head_roi: Head ROI bbox (if any)
        torso_roi: Torso ROI bbox (if any)
        alpha: Overlay transparency
        
    Returns:
        Image with ROI overlay
    """
    img = image.copy()
    overlay = image.copy()
    
    # Draw person bbox
    x_min, y_min, x_max, y_max = [int(v) for v in person_bbox]
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), COLOR_PERSON, 2)
    
    # Draw head ROI
    if head_roi is not None:
        hx_min, hy_min, hx_max, hy_max = [int(v) for v in head_roi]
        cv2.rectangle(overlay, (hx_min, hy_min), (hx_max, hy_max), (0, 255, 255), -1)
    
    # Draw torso ROI
    if torso_roi is not None:
        tx_min, ty_min, tx_max, ty_max = [int(v) for v in torso_roi]
        cv2.rectangle(overlay, (tx_min, ty_min), (tx_max, ty_max), (255, 0, 255), -1)
    
    # Blend overlay
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    return img


def create_summary_image(
    image: np.ndarray,
    detection_result: Dict,
    padding: int = 20
) -> np.ndarray:
    """
    Create summary image with detection stats.
    
    Args:
        image: Annotated detection image
        detection_result: Detection result dict
        padding: Padding for info panel
        
    Returns:
        Image with info panel
    """
    h, w = image.shape[:2]
    
    # Calculate info panel height
    panel_height = 120
    
    # Create new image with panel
    new_h = h + panel_height
    result = np.zeros((new_h, w, 3), dtype=np.uint8)
    result[:h, :, :] = image
    
    # Draw info panel background
    result[h:, :, :] = (30, 30, 30)  # Dark gray
    
    # Get stats
    total_persons = len(detection_result.get("persons", []))
    violations = sum(
        1 for p in detection_result.get("persons", [])
        if not (p.get("has_helmet", False) and p.get("has_vest", False))
    )
    processing_time = detection_result.get("timing", {}).get("total_ms", 0)
    sam_activations = detection_result.get("stats", {}).get("sam_activations", 0)
    
    # Draw stats
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = h + 30
    
    stats = [
        f"Persons: {total_persons}",
        f"Violations: {violations}",
        f"SAM Activations: {sam_activations}",
        f"Processing: {processing_time:.1f}ms"
    ]
    
    for i, stat in enumerate(stats):
        x = padding + (i * (w // 4))
        cv2.putText(result, stat, (x, y_offset), font, 0.6, (255, 255, 255), 1)
    
    # Draw compliance rate
    if total_persons > 0:
        compliance = ((total_persons - violations) / total_persons) * 100
        compliance_text = f"Compliance: {compliance:.1f}%"
        compliance_color = COLOR_SAFE if compliance >= 80 else COLOR_VIOLATION
    else:
        compliance_text = "Compliance: N/A"
        compliance_color = (128, 128, 128)
    
    cv2.putText(
        result, 
        compliance_text, 
        (padding, y_offset + 40), 
        font, 
        0.8, 
        compliance_color, 
        2
    )
    
    return result
