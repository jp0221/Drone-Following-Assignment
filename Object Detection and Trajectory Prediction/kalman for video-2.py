import cv2
import os
import json
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

# Define Kalman filter parameters
dt = 1.0  # Time step
kalman_filters = {}  # Dictionary to store Kalman filters for each object class

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

        # Kalman filter initialization
        if class_id not in kalman_filters:
            kalman_filters[class_id] = cv2.KalmanFilter(4, 2)  # 4 states, 2 measurements (x, y)
            kalman_filters[class_id].measurementMatrix = np.array([[1, 0, 0, 0],
                                                                   [0, 1, 0, 0]], np.float32)
            kalman_filters[class_id].transitionMatrix = np.array([[1, 0, dt, 0],
                                                                  [0, 1, 0, dt],
                                                                  [0, 0, 1, 0],
                                                                  [0, 0, 0, 1]], np.float32)
            kalman_filters[class_id].processNoiseCov = np.array([[1, 0, 0, 0],
                                                                  [0, 1, 0, 0],
                                                                  [0, 0, 1, 0],
                                                                  [0, 0, 0, 1]], np.float32) * 0.03

            kalman_filters[class_id].statePre = np.array([[x1], [y1], [0], [0]], np.float32)
        
        # Predict next state of the object
        prediction = kalman_filters[class_id].predict()
        predicted_x, predicted_y = prediction[0], prediction[1]

        # Update measurement
        kalman_filters[class_id].correct(np.array([[x1], [y1]], np.float32))

        # Draw Kalman filter predicted position
        img = cv2.circle(img, (int(predicted_x), int(predicted_y)), 5, (255, 0, 0), -1)

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
json_output_file = "detections.json"
with open(json_output_file, "w") as f:
    json.dump(detections, f)

# Compile frames into a video with trajectory
fps = 30  # Assuming a default frame rate for now, adjust as needed
output_video = "YOLO_tracking_with_trajectory-2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (results[0].orig_img.shape[1], results[0].orig_img.shape[0]))
for i in range(len(results)):
    img_path = os.path.join(output_folder, f"{i:04d}.png")
    img = cv2.imread(img_path)
    video_writer.write(img)
    os.remove(img_path)  # Remove temporary image files
video_writer.release()

print(f"Video with trajectory saved as {output_video}")
print(f"Detections saved to {json_output_file}")
