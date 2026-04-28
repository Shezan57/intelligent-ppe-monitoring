# ============================================================================
# THESIS EXPERIMENTS — Intelligent PPE Monitoring System
# ============================================================================
# Paste each cell section into Google Colab. One cell per "# %% Cell N" block.
# Runtime > Change runtime type > T4 GPU
#
# Google Drive structure (MyDrive/ppe_models/):
#   - best.pt                (YOLO26m weights)
#   - sam3.pt                (SAM 3 weights — for Experiment 6)
#   - construction_sites.mp4  (test video)
# ============================================================================


# %% Cell 1: GPU Check
import torch
assert torch.cuda.is_available(), '❌ No GPU! Runtime > Change runtime type > T4 GPU'
GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'✅ GPU: {GPU_NAME}')
print(f'   VRAM: {GPU_VRAM:.1f} GB')
print(f'   PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')


# %% Cell 2: Install Dependencies
# Uncomment and run this cell once
# !pip uninstall -y pillow pydantic 2>/dev/null
# !pip install -q -U "pillow>=8.0,<12.0" ultralytics opencv-python-headless
# !pip uninstall clip -y 2>/dev/null
# !pip install -q git+https://github.com/ultralytics/CLIP.git
# import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')


# %% Cell 3: Mount Drive + Load Model
from google.colab import drive
import os, json, time, cv2
import numpy as np

drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/ppe_models'
RESULTS_DIR = f'{DRIVE_DIR}/experiment_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = f'{DRIVE_DIR}/best.pt'
VIDEO_PATH = f'{DRIVE_DIR}/construction_sites.mp4'
SAM_PATH   = f'{DRIVE_DIR}/sam3.pt'

for f, name in [(MODEL_PATH, 'best.pt'), (VIDEO_PATH, 'construction_sites.mp4')]:
    assert os.path.exists(f), f'❌ {name} not found at {f}'
    mb = os.path.getsize(f) / 1e6
    print(f'✅ {name}: {mb:.1f} MB')

from ultralytics import YOLO
model = YOLO(MODEL_PATH)

# Print actual class names from the trained model
print(f'\n✅ YOLO26m loaded')
print(f'   Class names: {model.names}')
print(f'   Number of classes: {len(model.names)}')

# Build class name lookup (lowercase for safe comparison)
CLASS_NAMES = model.names  # e.g. {0: 'Helmet', 1: 'Vest', 2: 'person', 3: 'no_helmet', 4: 'no_vest'}
NUM_CLASSES = len(CLASS_NAMES)
print(f'\n📁 Results will be saved to: {RESULTS_DIR}')


# ============================================================================
# Helper: convert numpy types to Python native types for JSON serialization
# ============================================================================
def to_python(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data, filename):
    """Save dict to JSON, converting numpy types automatically."""
    path = f'{RESULTS_DIR}/{filename}'
    with open(path, 'w') as f:
        json.dump(to_python(data), f, indent=2)
    print(f'✅ Saved to {filename}')


# ============================================================================
# EXPERIMENT 1: YOLO26m FPS Benchmark (Image — Standardized)
# ============================================================================
# %% Cell 4: Experiment 1 — YOLO FPS on Images
print('='*60)
print('EXPERIMENT 1: YOLO26m Inference FPS (Image)')
print('='*60)

# Use a real frame from the video as the test image
cap = cv2.VideoCapture(VIDEO_PATH)
ret, test_frame = cap.read()
cap.release()
assert ret, 'Could not read a frame from the video'
print(f'Test frame size: {test_frame.shape}')

WARMUP = 20
RUNS = 200

# Warm up GPU
print(f'Warming up ({WARMUP} runs)...')
for _ in range(WARMUP):
    model.predict(test_frame, device='cuda', verbose=False, imgsz=640, conf=0.30)

# Timed runs with CUDA synchronization for accurate timing
print(f'Benchmarking ({RUNS} runs)...')
times_img = []
for i in range(RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.predict(test_frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    torch.cuda.synchronize()
    times_img.append(time.perf_counter() - t0)

fps_values = [1/t for t in times_img]

result_exp1 = {
    'experiment': 'YOLO26m FPS Benchmark (Image)',
    'gpu': GPU_NAME,
    'gpu_vram_gb': round(GPU_VRAM, 1),
    'input_resolution': '640x640',
    'original_frame_size': f'{test_frame.shape[1]}x{test_frame.shape[0]}',
    'warmup_runs': WARMUP,
    'timed_runs': RUNS,
    'mean_fps': round(sum(fps_values)/len(fps_values), 1),
    'min_fps': round(min(fps_values), 1),
    'max_fps': round(max(fps_values), 1),
    'mean_ms_per_frame': round((sum(times_img)/len(times_img))*1000, 1),
    'median_ms': round(sorted(times_img)[len(times_img)//2]*1000, 1),
    'p95_ms': round(sorted(times_img)[int(len(times_img)*0.95)]*1000, 1),
}

print(f'\n{"="*40}')
print(f'  GPU:            {GPU_NAME}')
print(f'  Mean FPS:       {result_exp1["mean_fps"]}')
print(f'  Min FPS:        {result_exp1["min_fps"]}')
print(f'  Max FPS:        {result_exp1["max_fps"]}')
print(f'  Mean ms/frame:  {result_exp1["mean_ms_per_frame"]} ms')
print(f'  Median ms:      {result_exp1["median_ms"]} ms')
print(f'  P95 ms:         {result_exp1["p95_ms"]} ms')
print(f'{"="*40}')

save_json(result_exp1, 'exp1_fps_image.json')


# ============================================================================
# EXPERIMENT 2: YOLO26m FPS on Video Frames (Real-World Throughput)
# ============================================================================
# %% Cell 5: Experiment 2 — YOLO FPS on Video
print('\n' + '='*60)
print('EXPERIMENT 2: YOLO26m Video Inference FPS')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Video: {total_frames} frames @ {video_fps:.1f} FPS, {video_w}x{video_h}')

MAX_FRAMES = min(500, total_frames)
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

result_exp2 = {
    'experiment': 'YOLO26m FPS Benchmark (Video)',
    'video_file': 'construction_sites.mp4',
    'video_resolution': f'{video_w}x{video_h}',
    'video_native_fps': round(video_fps, 1),
    'frames_processed': len(times_video),
    'mean_fps': round(sum(video_fps_values)/len(video_fps_values), 1),
    'min_fps': round(min(video_fps_values), 1),
    'max_fps': round(max(video_fps_values), 1),
    'mean_ms_per_frame': round((sum(times_video)/len(times_video))*1000, 1),
    'median_ms': round(sorted(times_video)[len(times_video)//2]*1000, 1),
    'avg_detections_per_frame': round(sum(detections_per_frame)/len(detections_per_frame), 1),
    'max_detections_single_frame': int(max(detections_per_frame)),
    'total_processing_time_sec': round(sum(times_video), 2),
}

print(f'\n{"="*40}')
print(f'  Mean FPS:             {result_exp2["mean_fps"]}')
print(f'  Mean ms/frame:        {result_exp2["mean_ms_per_frame"]} ms')
print(f'  Avg detections/frame: {result_exp2["avg_detections_per_frame"]}')
print(f'  Max detections:       {result_exp2["max_detections_single_frame"]}')
print(f'  Total time:           {result_exp2["total_processing_time_sec"]}s')
print(f'{"="*40}')

save_json(result_exp2, 'exp2_fps_video.json')


# ============================================================================
# EXPERIMENT 3: Confidence Threshold Sweep on Video Frames
# ============================================================================
# %% Cell 6: Experiment 3 — Threshold Sweep
print('\n' + '='*60)
print('EXPERIMENT 3: Confidence Threshold Sweep (Video Frames)')
print('='*60)

# Load sample frames from video
cap = cv2.VideoCapture(VIDEO_PATH)
SAMPLE_FRAMES = 100
sample_imgs = []
for _ in range(SAMPLE_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    sample_imgs.append(frame)
cap.release()
print(f'Loaded {len(sample_imgs)} sample frames for threshold sweep.')

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
sweep_results = []

# Header using actual class names from model
class_header = '  '.join([f'{CLASS_NAMES[i]:<12}' for i in range(NUM_CLASSES)])
print(f'\n{"Threshold":<12} {"Total Dets":<12} {class_header}')
print("-" * (24 + 12 * NUM_CLASSES))

for conf in thresholds:
    total_dets = 0
    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    for img in sample_imgs:
        results = model.predict(img, device='cuda', verbose=False, imgsz=640, conf=conf)
        boxes = results[0].boxes
        total_dets += len(boxes)
        for cls_id in boxes.cls.cpu().numpy().astype(int):
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    avg_dets = total_dets / len(sample_imgs)
    row = {
        'threshold': conf,
        'total_detections': int(total_dets),
        'avg_detections_per_frame': round(float(avg_dets), 1),
    }
    for i in range(NUM_CLASSES):
        row[CLASS_NAMES[i]] = int(class_counts[i])

    sweep_results.append(row)
    counts_str = '  '.join([f'{int(class_counts[i]):<12}' for i in range(NUM_CLASSES)])
    print(f"{conf:<12} {total_dets:<12} {counts_str}")

save_json(sweep_results, 'exp3_threshold_sweep.json')


# ============================================================================
# EXPERIMENT 4: Per-Class Detection Analysis on Video
# ============================================================================
# %% Cell 7: Experiment 4 — Per-Class Analysis
print('\n' + '='*60)
print('EXPERIMENT 4: Per-Class Detection Distribution on Video')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
ANALYZE_FRAMES = min(500, total_frames)
class_total = {i: 0 for i in range(NUM_CLASSES)}
class_conf_sums = {i: 0.0 for i in range(NUM_CLASSES)}
frames_with_violations = 0
total_all_dets = 0

# Get actual class IDs for violation classes
violation_class_ids = set()
for cid, cname in CLASS_NAMES.items():
    if cname.lower() in ['no_helmet', 'no_vest', 'no-helmet', 'no-vest']:
        violation_class_ids.add(cid)

print(f'Analyzing {ANALYZE_FRAMES} frames...')
print(f'Violation class IDs: {violation_class_ids} ({[CLASS_NAMES[i] for i in violation_class_ids]})')

for i in range(ANALYZE_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    boxes = results[0].boxes

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    has_violation = False
    for cls_id, conf_val in zip(cls_ids, confs):
        class_total[cls_id] += 1
        class_conf_sums[cls_id] += float(conf_val)  # Convert to Python float
        total_all_dets += 1
        if cls_id in violation_class_ids:
            has_violation = True
    if has_violation:
        frames_with_violations += 1
cap.release()

result_exp4 = {
    'experiment': 'Per-Class Video Analysis',
    'frames_analyzed': ANALYZE_FRAMES,
    'total_detections': total_all_dets,
    'frames_with_violations': frames_with_violations,
    'violation_frame_rate_pct': round(frames_with_violations / ANALYZE_FRAMES * 100, 1),
    'per_class': {}
}

print(f'\n{"Class":<15} {"Count":<10} {"Avg Conf":<12} {"% of Total"}')
print("-" * 50)
for i in range(NUM_CLASSES):
    count = int(class_total[i])
    avg_conf = float(class_conf_sums[i] / count) if count > 0 else 0.0
    pct = float(count / total_all_dets * 100) if total_all_dets > 0 else 0.0
    result_exp4['per_class'][CLASS_NAMES[i]] = {
        'count': count,
        'avg_confidence': round(avg_conf, 4),
        'percentage': round(pct, 1)
    }
    print(f"{CLASS_NAMES[i]:<15} {count:<10} {avg_conf:<12.4f} {pct:.1f}%")

print(f'\nFrames with violations: {frames_with_violations}/{ANALYZE_FRAMES} ({result_exp4["violation_frame_rate_pct"]}%)')
print(f'Total detections: {total_all_dets}')

if frames_with_violations == 0:
    print('\n⚠️  NOTE: Zero violations detected. This is valid data — it means the')
    print('   video shows mostly compliant workers. This result is still useful for')
    print('   the thesis (high compliance rate scenario).')

save_json(result_exp4, 'exp4_per_class_video.json')


# ============================================================================
# EXPERIMENT 5: 5-Path Triage Distribution (Simulated from YOLO output)
# ============================================================================
# %% Cell 8: Experiment 5 — 5-Path Triage Distribution
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

# Build normalized class name lookup
def normalize_name(name):
    return name.lower().replace('-', '_')

# Map class IDs to role
presence_ids = set()    # helmet, vest
absence_ids = set()     # no_helmet, no_vest
person_id = None

for cid, cname in CLASS_NAMES.items():
    n = normalize_name(cname)
    if n == 'person':
        person_id = cid
    elif n in ['helmet', 'hard_hat']:
        presence_ids.add(cid)
    elif n in ['vest', 'safety_vest']:
        presence_ids.add(cid)
    elif n in ['no_helmet', 'no_hard_hat']:
        absence_ids.add(cid)
    elif n in ['no_vest', 'no_safety_vest']:
        absence_ids.add(cid)

# Specific ID lookups
helmet_id = next((cid for cid, n in CLASS_NAMES.items() if normalize_name(n) in ['helmet', 'hard_hat']), None)
vest_id = next((cid for cid, n in CLASS_NAMES.items() if normalize_name(n) in ['vest', 'safety_vest']), None)
no_helmet_id = next((cid for cid, n in CLASS_NAMES.items() if normalize_name(n) in ['no_helmet', 'no_hard_hat']), None)
no_vest_id = next((cid for cid, n in CLASS_NAMES.items() if normalize_name(n) in ['no_vest', 'no_safety_vest']), None)

print(f'Class ID mapping:')
print(f'  person={person_id}, helmet={helmet_id}, vest={vest_id}, no_helmet={no_helmet_id}, no_vest={no_vest_id}')

print(f'\nProcessing {PATH_FRAMES} frames with 5-path triage simulation...')

for i in range(PATH_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    boxes = results[0].boxes

    if len(boxes) == 0:
        continue

    all_xyxy = boxes.xyxy.cpu().numpy()
    all_cls = boxes.cls.cpu().numpy().astype(int)
    all_conf = boxes.conf.cpu().numpy()

    # Separate persons from PPE items
    person_indices = [idx for idx, c in enumerate(all_cls) if c == person_id]
    ppe_indices = [idx for idx, c in enumerate(all_cls) if c != person_id]

    # If no person class detected, treat each PPE detection frame as a "virtual person"
    # (The model may not always detect a separate 'person' bbox)
    if len(person_indices) == 0 and len(ppe_indices) > 0:
        # Group all PPE detections in this frame as one "virtual person"
        total_persons += 1
        frame_classes = set(all_cls.tolist())
        has_helmet = helmet_id in frame_classes
        has_vest = vest_id in frame_classes
        has_no_helmet = no_helmet_id in frame_classes if no_helmet_id is not None else False
        has_no_vest = no_vest_id in frame_classes if no_vest_id is not None else False

        if has_no_helmet or has_no_vest:
            path_counts[2] += 1
        elif has_helmet and has_vest:
            path_counts[1] += 1
        elif has_helmet and not has_vest:
            path_counts[4] += 1
        elif has_vest and not has_helmet:
            path_counts[3] += 1
        else:
            path_counts[5] += 1
        continue

    # Normal case: person bboxes detected
    for pidx in person_indices:
        total_persons += 1
        px1, py1, px2, py2 = all_xyxy[pidx]
        p_area = (px2 - px1) * (py2 - py1)
        if p_area <= 0:
            continue

        # Find PPE items overlapping with this person
        associated_classes = set()
        for ppeidx in ppe_indices:
            ix1, iy1, ix2, iy2 = all_xyxy[ppeidx]
            # Check if PPE bbox overlaps with person bbox
            xx1 = max(px1, ix1)
            yy1 = max(py1, iy1)
            xx2 = min(px2, ix2)
            yy2 = min(py2, iy2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            if item_area > 0 and inter / item_area > 0.2:  # 20% overlap threshold
                associated_classes.add(int(all_cls[ppeidx]))

        has_helmet = helmet_id in associated_classes
        has_vest = vest_id in associated_classes
        has_no_helmet = (no_helmet_id in associated_classes) if no_helmet_id is not None else False
        has_no_vest = (no_vest_id in associated_classes) if no_vest_id is not None else False

        # 5-Path triage logic (matches HybridDetector._process_person)
        if has_no_helmet or has_no_vest:
            path_counts[2] += 1         # Fast Violation
        elif has_helmet and has_vest:
            path_counts[1] += 1         # Fast Safe
        elif has_helmet and not has_vest:
            path_counts[4] += 1         # Rescue Body
        elif has_vest and not has_helmet:
            path_counts[3] += 1         # Rescue Head
        else:
            path_counts[5] += 1         # Critical

cap.release()

result_exp5 = {
    'experiment': '5-Path Triage Distribution',
    'frames_processed': PATH_FRAMES,
    'total_persons_evaluated': total_persons,
    'paths': {}
}

print(f'\n{"Path":<8} {"Description":<40} {"Count":<10} {"Percentage":<12} {"SAM?"}')
print("-" * 85)
for p in [1, 2, 3, 4, 5]:
    count = int(path_counts[p])
    pct = float(count / total_persons * 100) if total_persons > 0 else 0.0
    sam = 'No' if p <= 2 else 'Yes'
    result_exp5['paths'][f'path_{p}'] = {
        'description': path_labels[p],
        'count': count,
        'percentage': round(pct, 1),
        'sam_required': p > 2
    }
    print(f"Path {p:<4} {path_labels[p]:<40} {count:<10} {pct:.1f}%{'':<8} {sam}")

bypass_rate = float((path_counts[1] + path_counts[2]) / total_persons * 100) if total_persons > 0 else 0.0
sam_invocation_rate = 100.0 - bypass_rate
result_exp5['bypass_rate_pct'] = round(bypass_rate, 1)
result_exp5['sam_invocation_rate_pct'] = round(sam_invocation_rate, 1)

print(f'\n🎯 SAM Bypass Rate: {bypass_rate:.1f}% (Paths 1+2 — no SAM needed)')
print(f'   SAM Invocation Rate: {sam_invocation_rate:.1f}% (Paths 3+4+5 — SAM required)')
print(f'   Total persons evaluated: {total_persons}')

save_json(result_exp5, 'exp5_triage_distribution.json')


# ============================================================================
# EXPERIMENT 6: SAM 3 Verification Latency
# ============================================================================
# %% Cell 9: Experiment 6 — SAM 3 Latency
print('\n' + '='*60)
print('EXPERIMENT 6: SAM 3 Verification Latency')
print('='*60)

if not os.path.exists(SAM_PATH):
    print(f'⚠️  sam3.pt not found at {SAM_PATH}. Skipping.')
    print('   Upload sam3.pt to MyDrive/ppe_models/ to run this experiment.')
    result_exp6 = {'experiment': 'SAM 3 Latency', 'status': 'SKIPPED', 'reason': 'sam3.pt not found'}
    save_json(result_exp6, 'exp6_sam_latency.json')
else:
    print(f'Loading SAM 3 with SAM3SemanticPredictor...')
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=0.15,
            task="segment",
            mode="predict",
            model=SAM_PATH,
            half=True,
            verbose=False,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        predictor.setup_model()
        print('✅ SAM 3 loaded successfully')

        # Generate test ROIs from the video
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        cap.release()
        assert ret, 'Could not read frame for SAM test'

        h, w = frame.shape[:2]
        # Create realistic ROI crops (head/torso regions)
        rois = []
        for y_start in range(0, min(h, 600), 200):
            for x_start in range(0, min(w, 600), 200):
                roi = frame[y_start:y_start+200, x_start:x_start+200]
                if roi.shape[0] >= 64 and roi.shape[1] >= 64:
                    rois.append(roi)
        rois = rois[:5]  # Use 5 different ROIs
        print(f'Created {len(rois)} test ROIs')

        # Text prompts (same as backend uses)
        HELMET_PROMPTS = ["helmet", "hard hat", "safety helmet", "construction helmet"]
        VEST_PROMPTS = ["safety vest", "high visibility vest", "reflective vest"]

        SAM_RUNS = 20
        sam_times = []

        # Warm up SAM
        print('Warming up SAM (3 runs)...')
        for _ in range(3):
            predictor.set_image(rois[0])
            predictor(text=HELMET_PROMPTS)

        # Timed SAM runs — alternate between helmet and vest prompts
        print(f'Benchmarking SAM ({SAM_RUNS} runs)...')
        for i in range(SAM_RUNS):
            roi = rois[i % len(rois)]
            prompts = HELMET_PROMPTS if i % 2 == 0 else VEST_PROMPTS

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            predictor.set_image(roi)
            results = predictor(text=prompts)
            torch.cuda.synchronize()
            sam_times.append(time.perf_counter() - t0)

        result_exp6 = {
            'experiment': 'SAM 3 Latency',
            'status': 'SUCCESS',
            'runs': SAM_RUNS,
            'mean_ms': round(float(sum(sam_times)/len(sam_times))*1000, 1),
            'min_ms': round(float(min(sam_times))*1000, 1),
            'max_ms': round(float(max(sam_times))*1000, 1),
            'median_ms': round(float(sorted(sam_times)[len(sam_times)//2])*1000, 1),
            'p95_ms': round(float(sorted(sam_times)[int(len(sam_times)*0.95)])*1000, 1),
        }

        print(f'\n{"="*40}')
        print(f'  Mean SAM latency: {result_exp6["mean_ms"]} ms')
        print(f'  Min:              {result_exp6["min_ms"]} ms')
        print(f'  Max:              {result_exp6["max_ms"]} ms')
        print(f'  Median:           {result_exp6["median_ms"]} ms')
        print(f'  P95:              {result_exp6["p95_ms"]} ms')
        print(f'{"="*40}')
        save_json(result_exp6, 'exp6_sam_latency.json')

    except ImportError as e:
        print(f'⚠️  SAM3SemanticPredictor not available: {e}')
        print('   Ensure ultralytics >= 8.3.237 is installed.')
        print('   pip install -U ultralytics')
        result_exp6 = {'experiment': 'SAM 3 Latency', 'status': 'IMPORT_ERROR', 'error': str(e)}
        save_json(result_exp6, 'exp6_sam_latency.json')

    except Exception as e:
        print(f'❌ SAM experiment failed: {e}')
        import traceback
        traceback.print_exc()
        result_exp6 = {'experiment': 'SAM 3 Latency', 'status': 'ERROR', 'error': str(e)}
        save_json(result_exp6, 'exp6_sam_latency.json')


# ============================================================================
# EXPERIMENT 7: Full Video Pipeline Summary
# ============================================================================
# %% Cell 10: Experiment 7 — Full Video Summary
print('\n' + '='*60)
print('EXPERIMENT 7: Full Video Detection Summary')
print('='*60)

cap = cv2.VideoCapture(VIDEO_PATH)
FULL_FRAMES = min(1000, total_frames)

frame_times = []
all_confs = []
violation_frames = 0
compliant_frames = 0
empty_frames = 0
total_det = 0

print(f'Processing {FULL_FRAMES} frames...')
for i in range(FULL_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = model.predict(frame, device='cuda', verbose=False, imgsz=640, conf=0.30)
    torch.cuda.synchronize()
    frame_times.append(time.perf_counter() - t0)

    boxes = results[0].boxes
    n_dets = len(boxes)
    total_det += n_dets

    if n_dets == 0:
        empty_frames += 1
        continue

    for conf_val in boxes.conf.cpu().numpy():
        all_confs.append(float(conf_val))

    cls_ids = set(boxes.cls.cpu().numpy().astype(int).tolist())
    has_violation = bool(cls_ids & violation_class_ids)
    if has_violation:
        violation_frames += 1
    else:
        compliant_frames += 1

cap.release()

result_exp7 = {
    'experiment': 'Full Video Pipeline Summary',
    'video_file': 'construction_sites.mp4',
    'video_resolution': f'{video_w}x{video_h}',
    'frames_processed': len(frame_times),
    'total_detections': total_det,
    'avg_detections_per_frame': round(float(total_det / len(frame_times)), 1),
    'mean_inference_fps': round(float(len(frame_times) / sum(frame_times)), 1),
    'mean_confidence': round(float(sum(all_confs) / len(all_confs)), 4) if all_confs else 0.0,
    'violation_frames': violation_frames,
    'compliant_frames': compliant_frames,
    'empty_frames': empty_frames,
    'total_processing_time_sec': round(float(sum(frame_times)), 2),
}

print(f'\n{"="*50}')
print(f'  Frames processed:        {result_exp7["frames_processed"]}')
print(f'  Total detections:        {result_exp7["total_detections"]}')
print(f'  Avg detections/frame:    {result_exp7["avg_detections_per_frame"]}')
print(f'  Mean inference FPS:      {result_exp7["mean_inference_fps"]}')
print(f'  Mean confidence:         {result_exp7["mean_confidence"]}')
print(f'  Violation frames:        {violation_frames}')
print(f'  Compliant frames:        {compliant_frames}')
print(f'  Empty frames:            {empty_frames}')
print(f'  Total processing time:   {result_exp7["total_processing_time_sec"]}s')
print(f'{"="*50}')

save_json(result_exp7, 'exp7_video_summary.json')


# ============================================================================
# FINAL SUMMARY
# ============================================================================
# %% Cell 11: Final Summary
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

print('\n📋 SUMMARY FOR THESIS CHAPTER 5:')
print(f'  • YOLO26m Image FPS:    {result_exp1["mean_fps"]} FPS ({result_exp1["mean_ms_per_frame"]} ms/frame)')
print(f'  • YOLO26m Video FPS:    {result_exp2["mean_fps"]} FPS ({result_exp2["mean_ms_per_frame"]} ms/frame)')
print(f'  • SAM Bypass Rate:      {result_exp5.get("bypass_rate_pct", "N/A")}%')
print(f'  • Persons evaluated:    {result_exp5["total_persons_evaluated"]}')
try:
    if result_exp6.get("status") == "SUCCESS":
        print(f'  • SAM Mean Latency:     {result_exp6["mean_ms"]} ms')
    else:
        print(f'  • SAM Mean Latency:     ({result_exp6.get("status", "N/A")})')
except:
    print(f'  • SAM Mean Latency:     (not measured)')
print(f'  • GPU:                  {GPU_NAME}')

print('\n💡 Share the JSON files or screenshot these results.')
print('   The thesis Chapter 5 will be updated with these real numbers.')
