# Appendix B: System Requirements and Reproducibility

## B.1 Hardware Requirements

The system was developed and trained using the following hardware configuration. These specifications represent the minimum recommended configuration for full pipeline operation (Sentry + Judge concurrently).

**Table B.1: Hardware Configuration**

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GPU with CUDA support (≥8 GB VRAM recommended) |
| RAM | 16 GB minimum, 32 GB recommended |
| Storage | 50 GB free space (dataset + model weights + database) |
| OS | Windows 10/11 or Ubuntu 20.04+ |
| Camera Input | USB webcam, IP camera (RTSP), or video file |

## B.2 Software Dependencies

**Table B.2: Python Package Requirements**

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| ultralytics | ≥ 8.3.0 | YOLO26m detection |
| torch | ≥ 2.0.0+cu118 | GPU inference |
| torchvision | ≥ 0.15.0 | Image transforms |
| fastapi | ≥ 0.109.0 | REST API backend |
| uvicorn | ≥ 0.27.0 | ASGI server |
| sqlalchemy | ≥ 2.0.0 | Database ORM |
| pydantic-settings | ≥ 2.0.0 | Config management |
| openai | ≥ 1.0.0 | Chatbot API |
| Pillow | ≥ 10.0.0 | Image processing |
| numpy | ≥ 1.24.0 | Array operations |
| opencv-python | ≥ 4.8.0 | Video processing |
| python-multipart | ≥ 0.0.6 | File upload |
| aiofiles | ≥ 23.0.0 | Async file I/O |

**Frontend:**

| Package | Version |
|---------|---------|
| Node.js | ≥ 18.0.0 |
| React | 18.x |
| Vite | 5.x |
| Axios | ≥ 1.6.0 |

## B.3 Model Weights

The trained YOLO26m model weights are located at:
```
yolo26m_ppe_combined_models_with_images/weights/best.pt
```
This file contains the best checkpoint selected based on validation mAP@50 at epoch 50.

## B.4 Reproduction Steps

To reproduce the model training:

```bash
# 1. Install dependencies
pip install ultralytics torch torchvision

# 2. Prepare dataset (structure as per datasets/data.yaml)
# Ensure the 5-class schema: helmet, vest, person, no-helmet, no-vest

# 3. Train the model
yolo train \
  model=yolo26m.pt \
  data=data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  optimizer=AdamW \
  lr0=0.01 \
  cos_lr=True \
  conf=0.30 \
  iou=0.45 \
  name=yolo26m_ppe

# 4. Evaluate on test set
yolo val \
  model=runs/detect/yolo26m_ppe/weights/best.pt \
  data=data.yaml \
  conf=0.30
```

## B.5 Running the Full System

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env   # Add OPENAI_API_KEY to .env
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```
