import cv2
import os
import json
from tqdm import tqdm
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8x.pt")
model.conf = 0.01

# Track objects in the video
results = model.track(source='cyclist and vehicle Tracking -2.mp4', show=False, classes=[0, 1, 2],iou=0.1)

# Temporary folder to save frames
output_folder = "YOLO_Track-2"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save frames with bounding boxes and detections
detections = []
pbar = tqdm(total=len(results), desc="Saving Frames")
for i, det in enumerate(results):
    print(f"Processing result {i}")
    img_path = os.path.join(output_folder, f"{i:04d}.png")
    img = det.orig_img
    bboxes = det.boxes.xyxy
    frame_detections = []
    for bbox in bboxes:
        # Check if bounding box has enough values
        if len(bbox) == 4:
            # Some bounding boxes might have fewer than 6 values
            # Handle these cases appropriately
            x1, y1, x2, y2 = bbox.tolist()
            conf = None
            class_id = None
        else:
            x1, y1, x2, y2, conf, class_id = bbox.tolist()

        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        detection_info = {
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": int(class_id) if class_id is not None else None
        }
        frame_detections.append(detection_info)
    detections.append(frame_detections)
    cv2.imwrite(img_path, img)
    pbar.update(1)
pbar.close()

# Save detections to a JSON file
json_output_file = "detections(new).json"
with open(json_output_file, "w") as f:
    json.dump(detections, f)

# Compile frames into a video
fps = 30  # Assuming a default frame rate for now, adjust as needed
output_video = "YOLO_tracking-2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (results[0].orig_img.shape[1], results[0].orig_img.shape[0]))
for i in range(len(results)):
    img_path = os.path.join(output_folder, f"{i:04d}.png")
    img = cv2.imread(img_path)
    video_writer.write(img)
video_writer.release()

print(f"Video saved as {output_video}")
print(f"Detections saved to {json_output_file}")
