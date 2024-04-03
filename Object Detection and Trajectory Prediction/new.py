import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8x.pt")
model.conf = 0.01

# Track objects in the video
results = model.track(source='Drone Tracking Video.mp4', show=False, classes=[0, 1, 2],iou=0.5)

# Temporary folder to save frames
output_folder = "YOLO_Track-3"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save frames with bounding boxes
pbar = tqdm(total=len(results), desc="Saving Frames")
for i, det in enumerate(results):
    print(f"Processing result {i}")
    img_path = os.path.join(output_folder, f"{i:04d}.png")
    img = det.orig_img
    bboxes = det.boxes.xyxy
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    cv2.imwrite(img_path, img)
    pbar.update(1)
pbar.close()

# Compile frames into a video
fps = 30  # Assuming a default frame rate for now, adjust as needed
output_video = "YOLO_tracking-3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (results[0].orig_img.shape[1], results[0].orig_img.shape[0]))
for i in range(len(results)):
    img_path = os.path.join(output_folder, f"{i:04d}.png")
    img = cv2.imread(img_path)
    video_writer.write(img)
video_writer.release()

print(f"Video saved as {output_video}")
