"""
Test script to check YOLO model class names
Run this to see what classes your model uses
"""

from ultralytics import YOLO

# Load model
model = YOLO("models/yolov11m_best.pt")

# Print class names
print("=" * 50)
print("YOLO Model Class Names")
print("=" * 50)

for class_id, class_name in model.names.items():
    print(f"  Class {class_id}: {class_name}")

print("=" * 50)
