# %% [markdown]
# # Intelligent PPE Monitoring - Modular Debugging Pipeline
# This script is designed to be run in Google Colab (or any Jupyter environment).
# Each block `#%%` represents a cell. It breaks down the 5-path detection logic
# so you can visualize exactly what YOLO detects, what decision path the Judge takes,
# and exactly what ROIs SAM receives.

# %% [markdown]
# ## 0. Environment Setup & Imports
# Run this cell to ensure your paths are set properly.

# %%
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# Ensure we are inside the 'backend' folder
if '/content/backend' not in sys.path:
    sys.path.insert(0, '/content/backend')
if os.path.exists('/content/backend'):
    os.chdir('/content/backend')
else:
    print("Warning: Ensure backend folder is set as your working directory.")

try:
    from services.yolo_detector import get_yolo_detector
    from services.sam_verifier import get_sam_verifier
    from services.hybrid_detector import get_hybrid_detector, DecisionPath
    from utils.bbox_utils import extract_head_roi, extract_torso_roi
    from config.settings import settings
except ImportError as e:
    print(f"Error importing internal modules: {e}")

# Helper to draw ROIs in matplotlib
def draw_roi(ax, bbox, color, label):
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, label, color=color, fontsize=10, fontweight='bold', backgroundcolor='black')


# %% [markdown]
# ## 1. Load Models Separately
# This step loads YOLO (Sentry) and SAM (Judge) independently.

# %%
print("Loading YOLO (Sentry)...")
yolo = get_yolo_detector()
yolo.load_model()
print(f"YOLO loaded - device: {yolo.device}")

print("\nLoading SAM 3 (Judge)...")
sam = get_sam_verifier()
sam.load_model()
print(f"SAM 3 loaded - mock mode: {sam.is_mock()}")


# %% [markdown]
# ## 2. TEST CASE: Single Image Modular Debugging
# Upload an image and see exactly what happens inside the pipeline.

# %%
from google.colab import files
print("Upload a test image (JPG/PNG)...")
try:
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    test_image = cv2.imdecode(np.frombuffer(uploaded[fname], np.uint8), cv2.IMREAD_COLOR)
    print(f"Loaded: {fname} ({test_image.shape[1]}x{test_image.shape[0]}px)")
except Exception as e:
    print("Skipping image upload test.")
    test_image = None

# %%
# RUN MODULAR PIPELINE ON IMAGE
if test_image is not None:
    print("========================================")
    print(" STEP 1: YOLO INFERENCE")
    print("========================================")
    yolo_start = time.time()
    
    # Temporarily lower confidence to catch borderline detections (like your other script)
    old_conf = yolo.confidence_threshold
    yolo.confidence_threshold = 0.25
    yolo_results = yolo.detect(test_image)
    yolo.confidence_threshold = old_conf
    
    print(f"YOLO Processing time: {(time.time() - yolo_start) * 1000:.1f}ms")
    
    persons = yolo_results["persons"]
    print(f"Detected {len(persons)} persons.")
    
    detector = get_hybrid_detector() # Use the hybrid detector to simulate path logic
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    img_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title("Modular Debug: YOLO Boxes + SAM ROIs")
    
    # Iterate through persons exactly like Hybrid Detector does
    for i, person in enumerate(persons):
        bbox = person["bbox"]
        conf = person["confidence"]
        helmet_det = person.get("helmet_detected", False)
        vest_det = person.get("vest_detected", False)
        
        print(f"\n--- Person {i+1} ---")
        print(f"YOLO BBox: {bbox}")
        print(f"YOLO Confidence: {conf:.2f}")
        print(f"YOLO says -> Helmet: {helmet_det}, Vest: {vest_det}")
        
        # Calculate expected Decision Path
        if helmet_det and vest_det:
            expected_path = "Fast Safe"
        elif person.get("no_helmet_detected", False) or person.get("no_vest_detected", False):
            expected_path = "Fast Violation"
        elif vest_det and not helmet_det:
            expected_path = "Rescue Head"
        elif helmet_det and not vest_det:
            expected_path = "Rescue Body"
        else:
            expected_path = "Critical"
            
        print(f"Decided Path: {expected_path}")
        
        # Plot Person BBox (Blue)
        draw_roi(ax, bbox, 'blue', f"Person {i+1} ({expected_path})")
        
        # Simulate SAM ROI extraction based on path
        if expected_path in ["Rescue Head", "Critical", "Fast Violation"]:
            # Evaluate Head ROI
            head_roi = extract_head_roi(bbox)
            print(f"  -> Extracted Head ROI: {head_roi}")
            draw_roi(ax, head_roi, 'yellow', 'Head ROI')
            
            # Manual SAM invocation
            sam_res = sam._verify_roi(test_image, head_roi, ["helmet", "hard hat", "safety helmet"], "helmet")
            print(f"  -> SAM Helmet says: {sam_res['helmet_found']} (Conf: {sam_res['confidence']:.3f})")
            
        if expected_path in ["Rescue Body", "Critical", "Fast Violation"]:
            # Evaluate Torso ROI
            torso_roi = extract_torso_roi(bbox)
            print(f"  -> Extracted Torso ROI: {torso_roi}")
            draw_roi(ax, torso_roi, 'orange', 'Torso ROI')
            
            # Manual SAM invocation
            sam_res = sam._verify_roi(test_image, torso_roi, ["safety vest", "high visibility vest", "reflective vest"], "vest")
            print(f"  -> SAM Vest says: {sam_res['vest_found']} (Conf: {sam_res['confidence']:.3f})")

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. TEST CASE: Video Modular Debugging with ROI Tracking
# Uses `cv2.VideoCapture` and explicitly tracks paths and ROIs across frames.

# %%
print("Upload a test video (MP4/AVI)...")
try:
    video_upload = files.upload()
    video_name = list(video_upload.keys())[0]
    video_path = f'/content/{video_name}'
    with open(video_path, 'wb') as f:   
        f.write(video_upload[video_name])
except Exception as e:
    print("Skipping video upload test.")
    video_path = None

# %%
if video_path and os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = '/content/modular_debug_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))
    
    print(f"Processing Video: {video_path}...")
    frame_count = 0
    MAX_FRAMES = 150 # Process first 150 frames to be fast
    
    detector = get_hybrid_detector()
    
    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. YOLO with tracking (MUST use .track() to get IDs, not .predict())
        yolo_results = yolo.model.track(
            frame, 
            conf=0.25, # Explicitly lower to catch borderline detections
            imgsz=settings.yolo_imgsz,
            persist=True, # Required for tracking IDs across frames
            verbose=False
        )
        
        annotated_frame = frame.copy()
        
        if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            cls_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, tid, cls_id in zip(boxes, track_ids, cls_ids):
                if cls_id != 2: continue # Ensure it's a person
                
                # YOLO BBox
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"ID: {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # We will draw the extracted Head & Torso ROIs so you can verify they are placed correctly!
                head_roi = extract_head_roi(box)
                hx1, hy1, hx2, hy2 = map(int, head_roi)
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "Head ROI", (hx1, hy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                torso_roi = extract_torso_roi(box)
                tx1, ty1, tx2, ty2 = map(int, torso_roi)
                cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (0, 165, 255), 2)
                cv2.putText(annotated_frame, "Torso ROI", (tx1, ty1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"Finished. Saved to {out_path}.")
    print("Download and review to see exact tracked ROIs.")

# %% [markdown]
# You can now download `modular_debug_output.mp4` to visualize the tracking and ROI extraction box fidelity.
