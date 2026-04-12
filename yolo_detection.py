import os
import glob
from ultralytics import YOLO

# 1. Load your custom model (update this path to your actual best.pt location)
model = YOLO("best.pt")

# 2. Define your uploaded video
# (Make sure you have uploaded a test video to your Colab workspace!)
# video_path = "Video/12098511-hd_1920_1080_50fps.mp4"

# print(f"Starting frame-by-frame processing for: {video_path}...")

# # 3. Run the prediction
# results = model.predict(
#     source=video_path,
#     conf=0.25,       # 25% confidence threshold
#     imgsz=640,       # Forcing your training resolution!
#     save=True,       # Save the output video with bounding boxes
#     stream=True      # Memory-safe processing for videos
# )

# # 4. Execute the stream
# # When using stream=True, you MUST loop through the results to trigger the processing
# for frame_idx, frame_result in enumerate(results):
#     if frame_idx % 30 == 0:  # Print an update every 30 frames
#         print(f"Processed frame {frame_idx}...")

# print("\n✅ Video processing complete!")

# # ============================================================
# # 5. AUTO-DOWNLOAD THE RESULT (If using Google Colab)
# # ============================================================
# try:
#     from google.colab import files

#     # YOLO saves outputs in runs/detect/predict... let's find the newest video file
#     output_videos = glob.glob('runs/detect/predict*/*.avi') + glob.glob('runs/detect/predict*/*.mp4')

#     if output_videos:
#         # Find the most recently created video file
#         latest_video = max(output_videos, key=os.path.getctime)
#         print(f"Downloading annotated video: {latest_video}")
#         files.download(latest_video)
#     else:
#         print("Could not find the saved output video.")
# except ImportError:
#     # If you run this locally instead of Colab, it will just skip the download step
#     print("Check your 'runs/detect/' folder for the saved video!")


# Detect and image


# 2. Run detection on an image
# Replace 'path/to/image.jpg' with your actual image path
results = model.predict(
    source="senior-foreman-performing-inspection-construction-260nw-2502078473.webp",
    conf=0.10,
    save=True
)

# 3. Display or process the results
for result in results:
    result.show()  # Display the image with detections
    print(f"Detected {len(result.boxes)} objects.")


