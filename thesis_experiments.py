# ============================================================================
# THESIS EXPERIMENTS NOTEBOOK — Intelligent PPE Monitoring
# ============================================================================
# Copy this entire file into a Google Colab notebook (one cell per section).
# Runtime > Change runtime type > T4 GPU
#
# Files expected in Google Drive (MyDrive/ppe_models/):
#   - best.pt               (YOLO26m weights)
#   - sam3.pt               (SAM 3 weights)
#   - construction_sites.mp4 (test video)
#   - backend_colab.zip      (backend code)
#
# Outputs: All results are printed AND saved to /content/drive/MyDrive/ppe_models/experiment_results/
# ============================================================================

# %% [markdown]
# # 🧪 Thesis Experiments — Intelligent PPE Monitoring
# Run all 7 experiments to collect real data for thesis Chapter 5.
# **Runtime > Change runtime type > T4 GPU** before running.

# %% Cell 1: GPU Check + Install
import torch
assert torch.cuda.is_available(), '❌ No GPU! Go to Runtime > Change runtime type > T4 GPU'
GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'✅ GPU: {GPU_NAME}')
print(f'   VRAM: {GPU_VRAM:.1f} GB')
print(f'   PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')

# %% Cell 2: Install Dependencies
# !pip uninstall -y pillow pydantic 2>/dev/null
# !pip install -q -U "pillow>=8.0,<12.0" ultralytics opencv-python-headless
# !pip uninstall clip -y 2>/dev/null
# !pip install -q git+https://github.com/ultralytics/CLIP.git
# import ultralytics
# print(f'Ultralytics: {ultralytics.__version__}')

# %% Cell 3: Mount Drive + Setup Paths
from google.colab import drive
import os, shutil, json, time, csv
import numpy as np

drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/ppe_models'
RESULTS_DIR = f'{DRIVE_DIR}/experiment_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = f'{DRIVE_DIR}/best.pt'
VIDEO_PATH = f'{DRIVE_DIR}/construction_sites.mp4'
SAM_PATH   = f'{DRIVE_DIR}/sam3.pt'

# Verify files exist
for f, name in [(MODEL_PATH, 'best.pt'), (VIDEO_PATH, 'construction_sites.mp4')]:
    assert os.path.exists(f), f'❌ {name} not found at {f}'
    mb = os.path.getsize(f) / 1e6
    print(f'✅ {name}: {mb:.1f} MB')

print(f'\n📁 Results will be saved to: {RESULTS_DIR}')

# %% Cell 4: Load YOLO26m Model
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(f'✅ YOLO26m loaded | Classes: {model.names}')

# ============================================================================
# EXPERIMENT 1: YOLO26m FPS Benchmark (Image-Based)
# ============================================================================
# %% Cell 5: Experiment 1 — FPS Benchmark (Image)
print('='*60)
print('EXPERIMENT 1: YOLO26m Inference FPS Benchmark')
print('='*60)

import cv2
# Use a sample frame — create a random 640x640 image to standardize
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

WARMUP = 20
RUNS = 200

# Warm up GPU
print(f'Warming up ({WARMUP} runs)...')
for _ in range(WARMUP):
    model.predict(test_img, device='cuda', verbose=False, imgsz=640, conf=0.30)

# Timed runs
print(f'Benchmarking ({RUNS} runs)...')
times_img = []
for i in range(RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.predict(test_img, device='cuda', verbose=False, imgsz=640, conf=0.30)
    torch.cuda.synchronize()
    times_img.append(time.perf_counter() - t0)

fps_values = [1/t for t in times_img]
mean_fps = sum(fps_values) / len(fps_values)
mean_ms = (sum(times_img) / len(times_img)) * 1000

result_exp1 = {
    'experiment': 'FPS Benchmark (Image)',
    'gpu': GPU_NAME,
    'gpu_vram_gb': round(GPU_VRAM, 1),
    'runs': RUNS,
    'mean_fps': round(mean_fps, 1),
    'min_fps': round(min(fps_values), 1),
    'max_fps': round(max(fps_values), 1),
    'mean_ms_per_frame': round(mean_ms, 1),
    'median_ms': round(sorted(times_img)[len(times_img)//2] * 1000, 1),
}

print(f'\n{"="*40}')
print(f'  GPU:            {GPU_NAME}')
print(f'  Mean FPS:       {result_exp1["mean_fps"]}')
print(f'  Min FPS:        {result_exp1["min_fps"]}')
print(f'  Max FPS:        {result_exp1["max_fps"]}')
print(f'  Mean ms/frame:  {result_exp1["mean_ms_per_frame"]} ms')
print(f'  Median ms:      {result_exp1["median_ms"]} ms')
print(f'{"="*40}')

with open(f'{RESULTS_DIR}/exp1_fps_benchmark.json', 'w') as f:
    json.dump(result_exp1, f, indent=2)
print(f'✅ Saved to exp1_fps_benchmark.json')


# ============================================================================
# EXPERIMENT 2: FPS Benchmark on Video
# ============================================================================
# %% Cell 6: Experiment 2 — FPS Benchmark (Video)
print('\n' + '='*60)
print('EXPERIMENT 2: YOLO26m Video Inference FPS')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Video: {total_frames} frames @ {video_fps:.1f} FPS, {video_w}x{video_h}')

# Process first 300 frames (or all if shorter)
MAX_FRAMES = min(300, total_frames)
times_video = []
detections_per_frame = []

print(f'Processing {MAX_FRAMES} frames...')
for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    torch.cuda.synchronize()
    times_video.append(time.perf_counter() - t0)
    detections_per_frame.append(len(results[0].boxes))
cap.release()

video_fps_values = [1/t for t in times_video]
mean_video_fps = sum(video_fps_values) / len(video_fps_values)

result_exp2 = {
    'experiment': 'FPS Benchmark (Video)',
    'video_file': 'construction_sites.mp4',
    'video_resolution': f'{video_w}x{video_h}',
    'video_native_fps': round(video_fps, 1),
    'frames_processed': len(times_video),
    'mean_fps': round(mean_video_fps, 1),
    'min_fps': round(min(video_fps_values), 1),
    'max_fps': round(max(video_fps_values), 1),
    'mean_ms_per_frame': round((sum(times_video)/len(times_video))*1000, 1),
    'avg_detections_per_frame': round(sum(detections_per_frame)/len(detections_per_frame), 1),
    'max_detections_single_frame': max(detections_per_frame),
}

print(f'\n{"="*40}')
print(f'  Video FPS (inference): {result_exp2["mean_fps"]}')
print(f'  Mean ms/frame:        {result_exp2["mean_ms_per_frame"]} ms')
print(f'  Avg detections/frame: {result_exp2["avg_detections_per_frame"]}')
print(f'  Max detections:       {result_exp2["max_detections_single_frame"]}')
print(f'{"="*40}')

with open(f'{RESULTS_DIR}/exp2_video_fps.json', 'w') as f:
    json.dump(result_exp2, f, indent=2)
print(f'✅ Saved to exp2_video_fps.json')


# ============================================================================
# EXPERIMENT 3: Confidence Threshold Sweep
# ============================================================================
# %% Cell 7: Experiment 3 — Threshold Sweep
print('\n' + '='*60)
print('EXPERIMENT 3: Confidence Threshold Sweep')
print('='*60)
print('⚠️  This experiment requires a data.yaml file.')
print('    If you do NOT have a data.yaml uploaded, skip this cell.')
print('    Instead, the training metrics from results.csv are already in the thesis.\n')

# --- OPTION A: If you have data.yaml on Drive, uncomment and run ---
# DATA_YAML = f'{DRIVE_DIR}/data.yaml'
# thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
# results_sweep = []
#
# print(f'{"Threshold":<12} {"Precision":<12} {"Recall":<10} {"mAP50":<10} {"mAP50-95"}')
# print("-" * 60)
# for conf in thresholds:
#     r = model.val(data=DATA_YAML, conf=conf, iou=0.45, imgsz=640, verbose=False)
#     p  = r.results_dict.get('metrics/precision(B)', 0)
#     rec = r.results_dict.get('metrics/recall(B)', 0)
#     m50 = r.results_dict.get('metrics/mAP50(B)', 0)
#     m5095 = r.results_dict.get('metrics/mAP50-95(B)', 0)
#     row = {'threshold': conf, 'precision': round(p, 4), 'recall': round(rec, 4),
#            'mAP50': round(m50, 4), 'mAP50_95': round(m5095, 4)}
#     results_sweep.append(row)
#     print(f"{conf:<12} {p:.4f}       {rec:.4f}     {m50:.4f}     {m5095:.4f}")
#
# with open(f'{RESULTS_DIR}/exp3_threshold_sweep.json', 'w') as f:
#     json.dump(results_sweep, f, indent=2)
# print(f'✅ Saved to exp3_threshold_sweep.json')

# --- OPTION B: Run on video frames at multiple thresholds (no data.yaml needed) ---
print('Running threshold sweep on video frames...')
cap = cv2.VideoCapture(VIDEO_PATH)
SAMPLE_FRAMES = 100
sample_imgs = []
for i in range(SAMPLE_FRAMES):
    ret, frame = cap.read()
    if not ret: break
    sample_imgs.append(frame)
cap.release()
print(f'Loaded {len(sample_imgs)} sample frames from video.')

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
sweep_results_video = []

print(f'\n{"Threshold":<12} {"Avg Detections":<18} {"Persons":<10} {"Helmets":<10} {"Vests":<10} {"No-Helmet":<12}')
print("-" * 75)

for conf in thresholds:
    total_dets = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # helmet, vest, person, no-helmet, no-vest
    for img in sample_imgs:
        results = model.predict(img, device='cuda', verbose=False, imgsz=640, conf=conf)
        boxes = results[0].boxes
        total_dets += len(boxes)
        for cls_id in boxes.cls.cpu().numpy().astype(int):
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    avg_dets = total_dets / len(sample_imgs)
    row = {
        'threshold': conf,
        'avg_detections': round(avg_dets, 1),
        'total_persons': class_counts.get(2, 0),
        'total_helmets': class_counts.get(0, 0),
        'total_vests': class_counts.get(1, 0),
        'total_no_helmet': class_counts.get(3, 0),
        'total_no_vest': class_counts.get(4, 0),
    }
    sweep_results_video.append(row)
    print(f"{conf:<12} {avg_dets:<18.1f} {class_counts.get(2,0):<10} {class_counts.get(0,0):<10} {class_counts.get(1,0):<10} {class_counts.get(3,0):<12}")

with open(f'{RESULTS_DIR}/exp3_threshold_sweep_video.json', 'w') as f:
    json.dump(sweep_results_video, f, indent=2)
print(f'\n✅ Saved to exp3_threshold_sweep_video.json')


# ============================================================================
# EXPERIMENT 4: Per-Class Detection Analysis on Video
# ============================================================================
# %% Cell 8: Experiment 4 — Per-Class Analysis (Video)
print('\n' + '='*60)
print('EXPERIMENT 4: Per-Class Detection Distribution on Video')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
ANALYZE_FRAMES = min(500, total_frames)
class_names = model.names
class_total = {i: 0 for i in range(len(class_names))}
class_conf_sums = {i: 0.0 for i in range(len(class_names))}
frames_with_violations = 0
total_persons_detected = 0

print(f'Analyzing {ANALYZE_FRAMES} frames...')
for i in range(ANALYZE_FRAMES):
    ret, frame = cap.read()
    if not ret: break
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    boxes = results[0].boxes
    has_violation = False
    for cls_id, conf_val in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
        class_total[cls_id] += 1
        class_conf_sums[cls_id] += conf_val
        if cls_id in [3, 4]:  # no-helmet or no-vest
            has_violation = True
        if cls_id == 2:
            total_persons_detected += 1
    if has_violation:
        frames_with_violations += 1
cap.release()

result_exp4 = {
    'experiment': 'Per-Class Video Analysis',
    'frames_analyzed': ANALYZE_FRAMES,
    'total_persons': total_persons_detected,
    'frames_with_violations': frames_with_violations,
    'violation_frame_rate': round(frames_with_violations / ANALYZE_FRAMES * 100, 1),
    'per_class': {}
}

print(f'\n{"Class":<15} {"Count":<10} {"Avg Conf":<12} {"% of Total"}')
print("-" * 50)
total_all = sum(class_total.values())
for i in range(len(class_names)):
    count = class_total[i]
    avg_conf = class_conf_sums[i] / count if count > 0 else 0
    pct = count / total_all * 100 if total_all > 0 else 0
    result_exp4['per_class'][class_names[i]] = {
        'count': count,
        'avg_confidence': round(avg_conf, 4),
        'percentage': round(pct, 1)
    }
    print(f"{class_names[i]:<15} {count:<10} {avg_conf:<12.4f} {pct:.1f}%")

print(f'\nFrames with violations: {frames_with_violations}/{ANALYZE_FRAMES} ({result_exp4["violation_frame_rate"]}%)')

with open(f'{RESULTS_DIR}/exp4_per_class_video.json', 'w') as f:
    json.dump(result_exp4, f, indent=2)
print(f'✅ Saved to exp4_per_class_video.json')


# ============================================================================
# EXPERIMENT 5: 5-Path Triage Distribution (Simulated)
# ============================================================================
# %% Cell 9: Experiment 5 — 5-Path Triage Distribution
print('\n' + '='*60)
print('EXPERIMENT 5: 5-Path Triage Distribution')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
PATH_FRAMES = min(500, total_frames)

path_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
path_labels = {
    1: 'Fast Safe (helmet + vest)',
    2: 'Fast Violation (no-helmet/no-vest)',
    3: 'Rescue Head (uncertain helmet)',
    4: 'Rescue Body (uncertain vest)',
    5: 'Critical (both uncertain)'
}
total_persons = 0

print(f'Processing {PATH_FRAMES} frames with 5-path triage simulation...')
for i in range(PATH_FRAMES):
    ret, frame = cap.read()
    if not ret: break
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    boxes = results[0].boxes

    # Group detections by spatial proximity (simulate person-PPE association)
    persons = []
    ppe_items = []
    for box, cls_id, conf_val in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy()
    ):
        if cls_id == 2:  # person
            persons.append({'box': box, 'cls': cls_id, 'conf': conf_val})
        else:
            ppe_items.append({'box': box, 'cls': cls_id, 'conf': conf_val})

    for person in persons:
        total_persons += 1
        px1, py1, px2, py2 = person['box']
        p_area = (px2 - px1) * (py2 - py1)
        if p_area <= 0:
            continue

        # Find associated PPE/violation classes via IoU > 0.05 (loose association)
        associated_classes = set()
        for item in ppe_items:
            ix1, iy1, ix2, iy2 = item['box']
            # Compute IoU-like overlap
            xx1 = max(px1, ix1)
            yy1 = max(py1, iy1)
            xx2 = min(px2, ix2)
            yy2 = min(py2, iy2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            if item_area > 0 and inter / item_area > 0.3:
                associated_classes.add(item['cls'])

        has_helmet = 0 in associated_classes
        has_vest = 1 in associated_classes
        has_no_helmet = 3 in associated_classes
        has_no_vest = 4 in associated_classes

        # 5-Path triage logic
        if has_no_helmet or has_no_vest:
            path_counts[2] += 1  # Fast Violation
        elif has_helmet and has_vest:
            path_counts[1] += 1  # Fast Safe
        elif has_helmet and not has_vest:
            path_counts[4] += 1  # Rescue Body
        elif has_vest and not has_helmet:
            path_counts[3] += 1  # Rescue Head
        else:
            path_counts[5] += 1  # Critical

cap.release()

result_exp5 = {
    'experiment': '5-Path Triage Distribution',
    'frames_processed': PATH_FRAMES,
    'total_persons': total_persons,
    'paths': {}
}

print(f'\n{"Path":<8} {"Description":<40} {"Count":<10} {"Percentage":<12} {"SAM?"}')
print("-" * 85)
for p in [1, 2, 3, 4, 5]:
    count = path_counts[p]
    pct = count / total_persons * 100 if total_persons > 0 else 0
    sam = 'No' if p <= 2 else 'Yes'
    result_exp5['paths'][f'path_{p}'] = {
        'description': path_labels[p],
        'count': count,
        'percentage': round(pct, 1),
        'sam_required': p > 2
    }
    print(f"Path {p:<4} {path_labels[p]:<40} {count:<10} {pct:.1f}%{'':<8} {sam}")

bypass_rate = (path_counts[1] + path_counts[2]) / total_persons * 100 if total_persons > 0 else 0
result_exp5['bypass_rate'] = round(bypass_rate, 1)
print(f'\n🎯 SAM Bypass Rate: {bypass_rate:.1f}% (Paths 1+2)')
print(f'   Total persons analyzed: {total_persons}')

with open(f'{RESULTS_DIR}/exp5_triage_distribution.json', 'w') as f:
    json.dump(result_exp5, f, indent=2)
print(f'✅ Saved to exp5_triage_distribution.json')


# ============================================================================
# EXPERIMENT 6: SAM Verification Latency
# ============================================================================
# %% Cell 10: Experiment 6 — SAM Latency (OPTIONAL — requires sam3.pt)
print('\n' + '='*60)
print('EXPERIMENT 6: SAM 3 Verification Latency')
print('='*60)

if os.path.exists(SAM_PATH):
    print(f'Loading SAM 3 from {SAM_PATH}...')
    sam_model = YOLO(SAM_PATH)

    # Generate some test ROIs from video
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Crop 5 simulated ROIs (head regions)
        h, w = frame.shape[:2]
        rois = [
            frame[100:300, 100:300],
            frame[50:250, 200:400],
            frame[150:350, 300:500],
            frame[0:200, 0:200],
            frame[200:400, 100:300],
        ]

        SAM_RUNS = 20
        sam_times = []
        print(f'Running {SAM_RUNS} SAM inference passes...')

        # Warm up
        for _ in range(3):
            sam_model.predict(rois[0], device='cuda', verbose=False)

        for i in range(SAM_RUNS):
            roi = rois[i % len(rois)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sam_model.predict(roi, device='cuda', verbose=False)
            torch.cuda.synchronize()
            sam_times.append(time.perf_counter() - t0)

        result_exp6 = {
            'experiment': 'SAM 3 Latency',
            'runs': SAM_RUNS,
            'mean_ms': round((sum(sam_times)/len(sam_times))*1000, 0),
            'min_ms': round(min(sam_times)*1000, 0),
            'max_ms': round(max(sam_times)*1000, 0),
            'median_ms': round(sorted(sam_times)[len(sam_times)//2]*1000, 0),
        }

        print(f'\n{"="*40}')
        print(f'  Mean SAM latency: {result_exp6["mean_ms"]} ms')
        print(f'  Min:              {result_exp6["min_ms"]} ms')
        print(f'  Max:              {result_exp6["max_ms"]} ms')
        print(f'  Median:           {result_exp6["median_ms"]} ms')
        print(f'{"="*40}')

        with open(f'{RESULTS_DIR}/exp6_sam_latency.json', 'w') as f:
            json.dump(result_exp6, f, indent=2)
        print(f'✅ Saved to exp6_sam_latency.json')
else:
    print(f'⚠️  sam3.pt not found at {SAM_PATH}. Skipping SAM latency test.')
    print('   Upload sam3.pt to Google Drive MyDrive/ppe_models/ to run this experiment.')


# ============================================================================
# EXPERIMENT 7: Video Detection Summary (Full Pipeline Statistics)
# ============================================================================
# %% Cell 11: Experiment 7 — Full Video Detection Summary
print('\n' + '='*60)
print('EXPERIMENT 7: Full Video Detection Summary')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
FULL_FRAMES = min(1000, total_frames)

frame_times = []
all_confs = []
violation_types = {'no-helmet': 0, 'no-vest': 0, 'both': 0}
compliant_count = 0
total_det = 0

print(f'Processing {FULL_FRAMES} frames for full pipeline summary...')
for i in range(FULL_FRAMES):
    ret, frame = cap.read()
    if not ret: break

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    torch.cuda.synchronize()
    frame_times.append(time.perf_counter() - t0)

    boxes = results[0].boxes
    total_det += len(boxes)
    for conf_val in boxes.conf.cpu().numpy():
        all_confs.append(float(conf_val))

    # Count violation types per frame
    cls_ids = set(boxes.cls.cpu().numpy().astype(int))
    if 3 in cls_ids and 4 in cls_ids:
        violation_types['both'] += 1
    elif 3 in cls_ids:
        violation_types['no-helmet'] += 1
    elif 4 in cls_ids:
        violation_types['no-vest'] += 1
    else:
        if 0 in cls_ids or 1 in cls_ids:
            compliant_count += 1

cap.release()

result_exp7 = {
    'experiment': 'Full Video Summary',
    'video_file': 'construction_sites.mp4',
    'frames_processed': len(frame_times),
    'total_detections': total_det,
    'avg_detections_per_frame': round(total_det / len(frame_times), 1),
    'mean_inference_fps': round(len(frame_times) / sum(frame_times), 1),
    'mean_confidence': round(sum(all_confs) / len(all_confs), 4) if all_confs else 0,
    'violation_frames': {
        'no_helmet_only': violation_types['no-helmet'],
        'no_vest_only': violation_types['no-vest'],
        'both_missing': violation_types['both'],
        'compliant': compliant_count,
    },
    'total_processing_time_sec': round(sum(frame_times), 2),
}

print(f'\n{"="*50}')
print(f'  Frames processed:        {result_exp7["frames_processed"]}')
print(f'  Total detections:        {result_exp7["total_detections"]}')
print(f'  Avg detections/frame:    {result_exp7["avg_detections_per_frame"]}')
print(f'  Mean inference FPS:      {result_exp7["mean_inference_fps"]}')
print(f'  Mean confidence:         {result_exp7["mean_confidence"]}')
print(f'  No-helmet frames:        {violation_types["no-helmet"]}')
print(f'  No-vest frames:          {violation_types["no-vest"]}')
print(f'  Both missing frames:     {violation_types["both"]}')
print(f'  Compliant frames:        {compliant_count}')
print(f'  Total time:              {result_exp7["total_processing_time_sec"]}s')
print(f'{"="*50}')

with open(f'{RESULTS_DIR}/exp7_video_summary.json', 'w') as f:
    json.dump(result_exp7, f, indent=2)
print(f'✅ Saved to exp7_video_summary.json')


# ============================================================================
# FINAL SUMMARY
# ============================================================================
# %% Cell 12: Print All Results Summary
print('\n\n')
print('🏁' + '='*58 + '🏁')
print('     ALL EXPERIMENTS COMPLETE — THESIS DATA COLLECTED')
print('🏁' + '='*58 + '🏁')
print(f'\n📁 All results saved to: {RESULTS_DIR}')
print('\nFiles generated:')
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith('.json'):
        size = os.path.getsize(f'{RESULTS_DIR}/{f}')
        print(f'  ✅ {f} ({size} bytes)')

print('\n📋 SUMMARY FOR THESIS:')
print(f'  • YOLO26m Image FPS:    {result_exp1["mean_fps"]} FPS ({result_exp1["mean_ms_per_frame"]} ms/frame)')
print(f'  • YOLO26m Video FPS:    {result_exp2["mean_fps"]} FPS ({result_exp2["mean_ms_per_frame"]} ms/frame)')
print(f'  • SAM Bypass Rate:      {result_exp5.get("bypass_rate", "N/A")}%')
print(f'  • Persons analyzed:     {result_exp5["total_persons"]}')
try:
    print(f'  • SAM Mean Latency:     {result_exp6["mean_ms"]} ms')
except:
    print(f'  • SAM Mean Latency:     (skipped — sam3.pt not available)')
print(f'  • GPU:                  {GPU_NAME}')

print('\n💡 Copy the JSON files or screenshot these results and share with me.')
print('   I will update the thesis Chapter 5 with real measured values.')
