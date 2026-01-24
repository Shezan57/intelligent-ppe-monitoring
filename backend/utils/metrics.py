"""
Metrics Calculation Utilities

Functions for calculating detection metrics and statistics.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


def calculate_metrics(
    predictions: List[bool],
    ground_truth: List[bool]
) -> DetectionMetrics:
    """
    Calculate classification metrics.
    
    Args:
        predictions: List of predicted values (True = violation)
        ground_truth: List of ground truth values (True = violation)
        
    Returns:
        DetectionMetrics dataclass
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if len(predictions) == 0:
        return DetectionMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            total_predictions=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            true_negatives=0
        )
    
    # Calculate confusion matrix elements
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions)
    
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        total_predictions=len(predictions),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn
    )


def calculate_compliance_rate(
    total_persons: int,
    violations: int
) -> float:
    """
    Calculate PPE compliance rate.
    
    Args:
        total_persons: Total persons detected
        violations: Number of violations
        
    Returns:
        Compliance rate as percentage (0-100)
    """
    if total_persons <= 0:
        return 100.0
    
    compliant = total_persons - violations
    return (compliant / total_persons) * 100


def calculate_path_distribution(
    detection_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate distribution of decision paths used.
    
    Args:
        detection_results: List of detection result dicts
        
    Returns:
        Path distribution with counts and percentages
    """
    path_counts = {
        "Fast Safe": 0,
        "Fast Violation": 0,
        "Rescue Head": 0,
        "Rescue Body": 0,
        "Critical": 0
    }
    
    total = 0
    
    for result in detection_results:
        for person in result.get("persons", []):
            path = person.get("decision_path", "Unknown")
            if path in path_counts:
                path_counts[path] += 1
            total += 1
    
    # Calculate percentages
    distribution = {}
    for path, count in path_counts.items():
        percentage = (count / total * 100) if total > 0 else 0.0
        distribution[path] = {
            "count": count,
            "percentage": percentage
        }
    
    # Add summary stats
    sam_activations = (
        distribution["Rescue Head"]["count"] +
        distribution["Rescue Body"]["count"] +
        distribution["Critical"]["count"]
    )
    bypass_rate = (
        (distribution["Fast Safe"]["count"] + distribution["Fast Violation"]["count"])
        / total * 100
    ) if total > 0 else 0.0
    
    distribution["_summary"] = {
        "total": total,
        "sam_activations": sam_activations,
        "sam_activation_rate": (sam_activations / total * 100) if total > 0 else 0.0,
        "bypass_rate": bypass_rate
    }
    
    return distribution


def calculate_fps(
    processing_times_ms: List[float]
) -> Dict[str, float]:
    """
    Calculate FPS statistics from processing times.
    
    Args:
        processing_times_ms: List of processing times in milliseconds
        
    Returns:
        Dict with avg_fps, min_fps, max_fps
    """
    if not processing_times_ms:
        return {"avg_fps": 0.0, "min_fps": 0.0, "max_fps": 0.0}
    
    # Convert ms to seconds and then to FPS
    fps_values = [1000 / t for t in processing_times_ms if t > 0]
    
    if not fps_values:
        return {"avg_fps": 0.0, "min_fps": 0.0, "max_fps": 0.0}
    
    return {
        "avg_fps": sum(fps_values) / len(fps_values),
        "min_fps": min(fps_values),
        "max_fps": max(fps_values)
    }


def get_violation_breakdown(
    detection_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get breakdown of violation types.
    
    Args:
        detection_results: List of detection result dicts
        
    Returns:
        Breakdown of no_helmet, no_vest, both_missing
    """
    no_helmet = 0
    no_vest = 0
    both_missing = 0
    safe = 0
    
    for result in detection_results:
        for person in result.get("persons", []):
            has_helmet = person.get("has_helmet", False)
            has_vest = person.get("has_vest", False)
            
            if has_helmet and has_vest:
                safe += 1
            elif not has_helmet and not has_vest:
                both_missing += 1
            elif not has_helmet:
                no_helmet += 1
            else:
                no_vest += 1
    
    total_violations = no_helmet + no_vest + both_missing
    total = safe + total_violations
    
    return {
        "safe": safe,
        "no_helmet": no_helmet,
        "no_vest": no_vest,
        "both_missing": both_missing,
        "total_violations": total_violations,
        "total": total,
        "compliance_rate": (safe / total * 100) if total > 0 else 100.0
    }
