# Experiments To Run
# Run these tomorrow and share the outputs — I will refine the thesis with real data.

## Setup
```
cd d:\SHEZAN\AI\intelligent-ppe-monitoring\backend
python -m pip install ultralytics psutil pillow
```

---

## Experiment 1: YOLO26m Inference FPS Benchmark
**What it measures:** Real throughput (frames per second) of the YOLO26m model on your GPU.
**Output needed:** Mean FPS, min/max FPS, GPU model name.

```python
import time
import torch
from ultralytics import YOLO

MODEL_PATH = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\yolo26m_ppe_combined_models_with_images\weights\best.pt"
TEST_IMAGE = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\yolo26m_ppe_combined_models_with_images\val_batch0_pred.jpg"
RUNS = 200  # warm up + timed runs

model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

# Warm up
for _ in range(10):
    model.predict(TEST_IMAGE, device=device, verbose=False, imgsz=640)

# Timed runs
times = []
for i in range(RUNS):
    t0 = time.perf_counter()
    model.predict(TEST_IMAGE, device=device, verbose=False, imgsz=640, conf=0.30)
    times.append(time.perf_counter() - t0)

fps_values = [1/t for t in times]
print(f"\n=== FPS RESULTS ===")
print(f"Mean FPS:   {sum(fps_values)/len(fps_values):.1f}")
print(f"Min FPS:    {min(fps_values):.1f}")
print(f"Max FPS:    {max(fps_values):.1f}")
print(f"Mean ms/frame: {(sum(times)/len(times))*1000:.1f} ms")
```

---

## Experiment 2: Confidence Threshold Sweep (Precision/Recall/F1)
**What it measures:** P, R, F1 at thresholds 0.10, 0.15, 0.20, 0.25, 0.30, 0.50.
**Output needed:** The printed table below.

```python
from ultralytics import YOLO

MODEL_PATH = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\yolo26m_ppe_combined_models_with_images\weights\best.pt"
DATA_YAML  = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\data.yaml"  # Update path if different

model = YOLO(MODEL_PATH)
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]

print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'mAP50':<10} {'mAP50-95'}")
print("-" * 60)
for conf in thresholds:
    results = model.val(data=DATA_YAML, conf=conf, iou=0.45, imgsz=640, verbose=False)
    p  = results.results_dict.get('metrics/precision(B)', 0)
    r  = results.results_dict.get('metrics/recall(B)', 0)
    m50   = results.results_dict.get('metrics/mAP50(B)', 0)
    m5095 = results.results_dict.get('metrics/mAP50-95(B)', 0)
    print(f"{conf:<12} {p:.4f}       {r:.4f}     {m50:.4f}     {m5095:.4f}")
```

---

## Experiment 3: Per-Class AP Breakdown
**What it measures:** AP@50 for each of the 5 classes: helmet, vest, person, no-helmet, no-vest.
**Output needed:** The class names and AP values printed below.

```python
from ultralytics import YOLO

MODEL_PATH = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\yolo26m_ppe_combined_models_with_images\weights\best.pt"
DATA_YAML  = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\data.yaml"

model = YOLO(MODEL_PATH)
results = model.val(data=DATA_YAML, conf=0.30, iou=0.45, imgsz=640, verbose=True)

# Class names and per-class AP
names = model.names
print("\n=== PER-CLASS AP@50 ===")
for i, ap in enumerate(results.box.ap50):
    print(f"  Class {i} ({names[i]}): {ap*100:.2f}%")
print(f"\nOverall mAP@50:    {results.box.map50*100:.2f}%")
print(f"Overall mAP@50-95: {results.box.map*100:.2f}%")
```

---

## Experiment 4: SAM Verification Latency
**What it measures:** Time for SAM to process one cropped ROI image.
**Output needed:** Mean, min, max latency in milliseconds.

```python
import time, torch
from PIL import Image
import numpy as np
# Adjust import based on your SAM version
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM_TYPE = "SAM2"
except:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_TYPE = "SAM1"

# Use any cropped ROI from your dataset as test image
TEST_ROI = r"d:\SHEZAN\AI\intelligent-ppe-monitoring\yolo26m_ppe_combined_models_with_images\val_batch0_pred.jpg"
img = np.array(Image.open(TEST_ROI).convert("RGB"))
# Crop the top-left 200x200 as a simulated ROI
roi = img[:200, :200]

RUNS = 30
print(f"SAM Type: {SAM_TYPE}")
print("Running SAM latency benchmark...")

# NOTE: Load your SAM model the same way as in async_sam_verifier.py
# Then run the mask generation RUNS times and record times
times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    # mask_generator.generate(roi)  <-- replace with your actual SAM call
    times.append(time.perf_counter() - t0)

print(f"\n=== SAM LATENCY RESULTS ===")
print(f"Mean: {(sum(times)/len(times))*1000:.0f} ms")
print(f"Min:  {min(times)*1000:.0f} ms")
print(f"Max:  {max(times)*1000:.0f} ms")
```

---

## Experiment 5: 5-Path Triage Distribution
**What it measures:** How many detections go through each of the 5 paths.
**Output needed:** Count per path from a test video or image batch.

Run your backend's `HybridDetector` on 100+ test images and count how many
detections fall into paths 1–5. Add a print inside `hybrid_detector.py`:

```python
# In HybridDetector._process_person(), before return:
print(f"PATH_TAKEN: {path_number}")  # Add this temporarily
```

Then run:
```bash
cd backend
python -c "
from services.hybrid_detector import HybridDetector
import cv2, glob

detector = HybridDetector()
images = glob.glob(r'd:/SHEZAN/AI/intelligent-ppe-monitoring/yolo26m_ppe_combined_models_with_images/val_batch*.jpg')
for img_path in images:
    frame = cv2.imread(img_path)
    detector.detect(frame)
"
```
Count the PATH_TAKEN outputs for each path number.

---

## Expected Outputs to Share with Me
After running experiments, paste these results:
1. FPS benchmark output (Exp 1)
2. Threshold sweep table (Exp 2)
3. Per-class AP table (Exp 3)
4. SAM latency output (Exp 4)
5. Path distribution counts (Exp 5)
