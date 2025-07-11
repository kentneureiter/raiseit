# detect_webcam_mvp.py

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 (person detection)
yolo_model = YOLO("yolov8n.pt")

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# Pose logic: check if either hand is above corresponding shoulder
def is_hand_raised(landmarks):
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    def valid(l): return l.visibility > 0.5  # ignore low-confidence landmarks

    if not (valid(landmarks[LEFT_WRIST]) and valid(landmarks[LEFT_SHOULDER]) and
            valid(landmarks[RIGHT_WRIST]) and valid(landmarks[RIGHT_SHOULDER])):
        return False

    left_raised = landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y
    right_raised = landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y
    return left_raised or right_raised

# Start webcam
cap = cv2.VideoCapture(0)  # Replace 0 with filename for video file

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    results = yolo_model(frame_rgb)[0]

    raised_hand_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Only process "person"
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        person_crop = frame_rgb[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        # Run pose detection
        pose_result = pose.process(person_crop)
        if pose_result.pose_landmarks:
            # Check for raised hand
            if is_hand_raised(pose_result.pose_landmarks.landmark):
                raised_hand_count += 1
                color = (0, 255, 0)  # Green box
            else:
                color = (0, 0, 255)  # Red box

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Log the count
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{timestamp} → {raised_hand_count} student(s) raising hand")

    # Show output
    cv2.putText(frame, f"Hands Raised: {raised_hand_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Raiselt MVP - Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
