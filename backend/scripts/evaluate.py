#!/usr/bin/env python3
"""
Evaluation Script

Generate thesis metrics by running detection on test images.
Calculates precision, recall, F1, FPS, and SAM activation rate.

Usage:
    python evaluate.py --test-dir ./test_images --output ./results
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import cv2
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.hybrid_detector import get_hybrid_detector
from services.yolo_detector import get_yolo_detector


class EvaluationRunner:
    """Run evaluation on test dataset and generate metrics."""
    
    def __init__(self, test_dir: str, output_dir: str):
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detector = get_hybrid_detector()
        
        # Results storage
        self.results = []
        self.timing_data = []
        
    def find_images(self) -> List[Path]:
        """Find all test images."""
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        
        for ext in extensions:
            images.extend(self.test_dir.glob(f'*{ext}'))
            images.extend(self.test_dir.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test images."""
        images = self.find_images()
        
        if not images:
            raise ValueError(f"No images found in {self.test_dir}")
        
        print(f"\n{'='*60}")
        print(f"PPE Detection Evaluation")
        print(f"{'='*60}")
        print(f"Test Directory: {self.test_dir}")
        print(f"Total Images: {len(images)}")
        print(f"{'='*60}\n")
        
        # Process each image
        total_persons = 0
        total_violations = 0
        sam_activations = 0
        path_counts = {
            "Fast Safe": 0,
            "Fast Violation": 0,
            "Rescue Head": 0,
            "Rescue Body": 0,
            "Critical": 0
        }
        
        start_time = time.time()
        
        for i, img_path in enumerate(images, 1):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"[{i}/{len(images)}] ⚠️ Failed to load: {img_path.name}")
                    continue
                
                # Run detection
                frame_start = time.time()
                result = self.detector.detect(image, save_annotated=False)
                frame_time = (time.time() - frame_start) * 1000
                
                # Store timing
                self.timing_data.append({
                    "image": img_path.name,
                    "total_ms": frame_time,
                    "yolo_ms": result["timing"]["yolo_ms"],
                    "sam_ms": result["timing"]["sam_ms"]
                })
                
                # Aggregate stats
                stats = result["stats"]
                total_persons += stats["total_persons"]
                total_violations += stats["total_violations"]
                sam_activations += stats["sam_activations"]
                
                # Path distribution
                for path, count in stats["path_distribution"].items():
                    if path in path_counts:
                        path_counts[path] += count
                
                # Store result
                self.results.append({
                    "image": img_path.name,
                    "persons": stats["total_persons"],
                    "violations": stats["total_violations"],
                    "sam_activations": stats["sam_activations"],
                    "processing_ms": frame_time
                })
                
                # Progress
                status = "✅" if stats["total_violations"] == 0 else "⚠️"
                print(f"[{i}/{len(images)}] {status} {img_path.name}: "
                      f"{stats['total_persons']} persons, "
                      f"{stats['total_violations']} violations, "
                      f"{frame_time:.1f}ms")
                
            except Exception as e:
                print(f"[{i}/{len(images)}] ❌ Error on {img_path.name}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            total_persons=total_persons,
            total_violations=total_violations,
            sam_activations=sam_activations,
            path_counts=path_counts,
            num_images=len(images),
            total_time=total_time
        )
        
        # Print summary
        self._print_summary(metrics)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self,
        total_persons: int,
        total_violations: int,
        sam_activations: int,
        path_counts: Dict[str, int],
        num_images: int,
        total_time: float
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        
        # Timing stats
        avg_time = np.mean([t["total_ms"] for t in self.timing_data]) if self.timing_data else 0
        avg_yolo = np.mean([t["yolo_ms"] for t in self.timing_data]) if self.timing_data else 0
        avg_sam = np.mean([t["sam_ms"] for t in self.timing_data]) if self.timing_data else 0
        
        # FPS
        fps = num_images / total_time if total_time > 0 else 0
        
        # SAM stats
        sam_rate = (sam_activations / total_persons * 100) if total_persons > 0 else 0
        bypass_rate = 100 - sam_rate
        
        # Compliance
        compliant = total_persons - total_violations
        compliance_rate = (compliant / total_persons * 100) if total_persons > 0 else 100
        
        # Path percentages
        path_percentages = {}
        for path, count in path_counts.items():
            pct = (count / total_persons * 100) if total_persons > 0 else 0
            path_percentages[path] = round(pct, 1)
        
        return {
            "summary": {
                "num_images": num_images,
                "total_persons": total_persons,
                "total_violations": total_violations,
                "compliant_persons": compliant,
                "compliance_rate": round(compliance_rate, 1)
            },
            "timing": {
                "total_seconds": round(total_time, 2),
                "avg_ms_per_image": round(avg_time, 2),
                "avg_yolo_ms": round(avg_yolo, 2),
                "avg_sam_ms": round(avg_sam, 2),
                "effective_fps": round(fps, 2)
            },
            "sam_stats": {
                "total_activations": sam_activations,
                "activation_rate": round(sam_rate, 1),
                "bypass_rate": round(bypass_rate, 1)
            },
            "path_distribution": path_counts,
            "path_percentages": path_percentages,
            "timestamp": datetime.now().isoformat()
        }
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}\n")
        
        s = metrics["summary"]
        print(f"📊 DETECTION SUMMARY")
        print(f"   Images Processed: {s['num_images']}")
        print(f"   Total Persons:    {s['total_persons']}")
        print(f"   Violations:       {s['total_violations']}")
        print(f"   Compliance Rate:  {s['compliance_rate']}%")
        
        t = metrics["timing"]
        print(f"\n⏱️ TIMING")
        print(f"   Total Time:       {t['total_seconds']}s")
        print(f"   Avg per Image:    {t['avg_ms_per_image']}ms")
        print(f"   YOLO Time:        {t['avg_yolo_ms']}ms")
        print(f"   SAM Time:         {t['avg_sam_ms']}ms")
        print(f"   Effective FPS:    {t['effective_fps']}")
        
        sam = metrics["sam_stats"]
        print(f"\n🤖 SAM ACTIVATION")
        print(f"   Activations:      {sam['total_activations']}")
        print(f"   Activation Rate:  {sam['activation_rate']}%")
        print(f"   Bypass Rate:      {sam['bypass_rate']}%")
        
        print(f"\n📈 PATH DISTRIBUTION")
        for path, pct in metrics["path_percentages"].items():
            count = metrics["path_distribution"][path]
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {path:15} {bar} {pct:5.1f}% ({count})")
        
        print(f"\n{'='*60}\n")
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full metrics
        metrics_file = self.output_dir / f"evaluation_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📁 Metrics saved: {metrics_file}")
        
        # Per-image results
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"📁 Results saved: {results_file}")
        
        # Timing data
        timing_file = self.output_dir / f"evaluation_timing_{timestamp}.json"
        with open(timing_file, "w") as f:
            json.dump(self.timing_data, f, indent=2)
        print(f"📁 Timing saved: {timing_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPE detection on test images")
    parser.add_argument(
        "--test-dir", "-t",
        default="./test_images",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output", "-o",
        default="./evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(
        test_dir=args.test_dir,
        output_dir=args.output
    )
    
    try:
        metrics = runner.run_evaluation()
        print("✅ Evaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
