"""
Violation Tracker Service

Implements violation de-duplication with cooldown periods.
Prevents repeated alerts for the same worker in continuous monitoring.

Key Features:
- Person tracking via bounding box overlap (IoU)
- Configurable cooldown period per violation type
- Tracks active violations across frames
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from config.settings import settings


@dataclass
class TrackedViolation:
    """Represents a tracked violation for a person."""
    person_id: int
    bbox: List[float]
    violation_type: str
    first_detected: float  # timestamp
    last_seen: float  # timestamp
    stored_to_db: bool = False
    screenshot_taken: bool = False
    
    def is_in_cooldown(self, cooldown_seconds: float) -> bool:
        """Check if this violation is still in cooldown period."""
        return (time.time() - self.first_detected) < cooldown_seconds


@dataclass
class PersonTrack:
    """Tracks a person across multiple frames."""
    track_id: int
    bbox: List[float]
    last_seen: float
    violations: Dict[str, TrackedViolation] = field(default_factory=dict)
    
    def update_bbox(self, new_bbox: List[float]):
        """Update the bounding box with smoothing."""
        # Simple exponential moving average for stability
        alpha = 0.7
        self.bbox = [
            alpha * new + (1 - alpha) * old
            for new, old in zip(new_bbox, self.bbox)
        ]
        self.last_seen = time.time()


class ViolationTracker:
    """
    Tracks violations across frames with cooldown to prevent duplicates.
    
    How it works:
    1. Detects persons in each frame
    2. Matches persons to existing tracks using IoU (bounding box overlap)
    3. For each violation, checks if it's new or within cooldown
    4. Only allows storage/screenshot for NEW violations or after cooldown expires
    
    Example:
        tracker = ViolationTracker(cooldown_seconds=300)  # 5 min cooldown
        
        # Frame 1: New worker without helmet
        should_store = tracker.should_store_violation(person1, "no_helmet")
        # Returns True (new violation)
        
        # Frame 2: Same worker, still no helmet  
        should_store = tracker.should_store_violation(person1, "no_helmet")
        # Returns False (within cooldown)
        
        # 5 minutes later: Same worker, still no helmet
        should_store = tracker.should_store_violation(person1, "no_helmet")
        # Returns True (cooldown expired)
    """
    
    def __init__(
        self,
        cooldown_seconds: float = 300.0,  # 5 minutes default
        iou_threshold: float = 0.3,  # Min overlap to consider same person
        track_timeout_seconds: float = 30.0  # Remove tracks not seen for this long
    ):
        """
        Initialize the violation tracker.
        
        Args:
            cooldown_seconds: Time before re-alerting for same violation type
            iou_threshold: Min IoU to match bounding boxes as same person
            track_timeout_seconds: Remove tracks not seen for this duration
        """
        self.cooldown_seconds = cooldown_seconds
        self.iou_threshold = iou_threshold
        self.track_timeout_seconds = track_timeout_seconds
        
        # Active person tracks: track_id -> PersonTrack
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 0
        
        # Stats for thesis
        self.stats = {
            "total_violations_detected": 0,
            "violations_stored": 0,
            "violations_deduplicated": 0,
            "unique_persons_tracked": 0
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # No intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_to_track(self, bbox: List[float]) -> Optional[int]:
        """
        Find existing track that matches this bounding box.
        
        Args:
            bbox: Person bounding box [x1, y1, x2, y2]
            
        Returns:
            Track ID if matched, None if new person
        """
        best_iou = 0.0
        best_track_id = None
        
        for track_id, track in self.tracks.items():
            iou = self._calculate_iou(bbox, track.bbox)
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_track_id = track_id
        
        return best_track_id
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently."""
        current_time = time.time()
        expired_tracks = [
            track_id for track_id, track in self.tracks.items()
            if (current_time - track.last_seen) > self.track_timeout_seconds
        ]
        
        for track_id in expired_tracks:
            del self.tracks[track_id]
    
    def should_store_violation(
        self,
        person: Dict[str, Any],
        violation_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if this violation should be stored to database.
        
        This is the main entry point for de-duplication logic.
        
        Args:
            person: Person detection dict with 'bbox' key
            violation_type: Type of violation ('no_helmet', 'no_vest', 'both_missing')
            
        Returns:
            Tuple of (should_store: bool, reason: str)
            - (True, "new_violation") - First time seeing this violation
            - (True, "cooldown_expired") - Saw before but cooldown has passed
            - (False, "in_cooldown") - Within cooldown period
        """
        bbox = person.get("bbox", [])
        if not bbox or len(bbox) != 4:
            return True, "invalid_bbox"  # Store it anyway for safety
        
        self.stats["total_violations_detected"] += 1
        
        # Cleanup old tracks
        self._cleanup_old_tracks()
        
        # Try to match to existing track
        track_id = self._match_to_track(bbox)
        
        if track_id is None:
            # New person - create track and allow storage
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id] = PersonTrack(
                track_id=track_id,
                bbox=bbox,
                last_seen=time.time(),
                violations={}
            )
            self.stats["unique_persons_tracked"] += 1
        else:
            # Existing person - update their track
            self.tracks[track_id].update_bbox(bbox)
        
        track = self.tracks[track_id]
        
        # Check if this violation type is already tracked for this person
        if violation_type in track.violations:
            existing = track.violations[violation_type]
            existing.last_seen = time.time()
            existing.bbox = bbox
            
            if existing.is_in_cooldown(self.cooldown_seconds):
                # Still in cooldown - don't store again
                self.stats["violations_deduplicated"] += 1
                return False, "in_cooldown"
            else:
                # Cooldown expired - reset and allow storage
                track.violations[violation_type] = TrackedViolation(
                    person_id=track_id,
                    bbox=bbox,
                    violation_type=violation_type,
                    first_detected=time.time(),
                    last_seen=time.time(),
                    stored_to_db=True
                )
                self.stats["violations_stored"] += 1
                return True, "cooldown_expired"
        else:
            # New violation type for this person
            track.violations[violation_type] = TrackedViolation(
                person_id=track_id,
                bbox=bbox,
                violation_type=violation_type,
                first_detected=time.time(),
                last_seen=time.time(),
                stored_to_db=True
            )
            self.stats["violations_stored"] += 1
            return True, "new_violation"
    
    def process_detection_result(
        self,
        detection_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a full detection result and filter violations.
        
        Args:
            detection_result: Full detection output from HybridDetector
            
        Returns:
            Modified detection result with only NEW violations to store
        """
        persons = detection_result.get("persons", [])
        violations_to_store = []
        all_persons_updated = []
        
        for person in persons:
            is_violation = person.get("is_violation", False)
            violation_type = person.get("violation_type", "unknown")
            
            if is_violation:
                should_store, reason = self.should_store_violation(person, violation_type)
                
                # Add tracking info to person
                person_updated = person.copy()
                person_updated["should_store"] = should_store
                person_updated["storage_reason"] = reason
                
                if should_store:
                    violations_to_store.append(person_updated)
                
                all_persons_updated.append(person_updated)
            else:
                # Not a violation - keep as-is
                person_updated = person.copy()
                person_updated["should_store"] = False
                person_updated["storage_reason"] = "not_violation"
                all_persons_updated.append(person_updated)
        
        # Return modified result
        result = detection_result.copy()
        result["persons"] = all_persons_updated
        result["violations_to_store"] = violations_to_store
        result["tracking_stats"] = {
            "total_persons": len(persons),
            "violations_detected": sum(1 for p in persons if p.get("is_violation")),
            "violations_to_store": len(violations_to_store),
            "violations_deduplicated": sum(
                1 for p in all_persons_updated
                if p.get("storage_reason") == "in_cooldown"
            ),
            "active_tracks": len(self.tracks)
        }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics for thesis metrics."""
        return {
            **self.stats,
            "active_tracks": len(self.tracks),
            "deduplication_rate": (
                self.stats["violations_deduplicated"] / 
                max(self.stats["total_violations_detected"], 1) * 100
            )
        }
    
    def reset(self):
        """Reset all tracking state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.stats = {
            "total_violations_detected": 0,
            "violations_stored": 0,
            "violations_deduplicated": 0,
            "unique_persons_tracked": 0
        }


# Global tracker instance
_violation_tracker: Optional[ViolationTracker] = None


def get_violation_tracker(
    cooldown_seconds: Optional[float] = None
) -> ViolationTracker:
    """
    Get or create global violation tracker.
    
    Args:
        cooldown_seconds: Override default cooldown (uses settings if None)
        
    Returns:
        ViolationTracker instance
    """
    global _violation_tracker
    
    if _violation_tracker is None:
        cooldown = cooldown_seconds or getattr(settings, 'violation_cooldown_seconds', 300.0)
        _violation_tracker = ViolationTracker(cooldown_seconds=cooldown)
    
    return _violation_tracker
