import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp

# Load models
yolo_model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' if you have a stronger GPU
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Webcam input
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Couldn't read from webcam.")
        break

    # Run YOLO
    results = yolo_model(frame)[0]
    boxes = results.boxes
    raised_hands = 0

    # Loop over all detected people
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Only process "person" class
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        padding = 60
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2)
        y2 = min(h, y2 + padding)

        # Crop person
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue  # skip if crop fails

        person_img = cv2.resize(person_img, (480, 480))
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb_img)

        if not pose_result.pose_landmarks:
            continue

        # Get keypoints
        landmarks = pose_result.pose_landmarks.landmark
        left_wrist = landmarks[15].y
        left_shoulder = landmarks[11].y
        right_wrist = landmarks[16].y
        right_shoulder = landmarks[12].y

        # Raise hand condition
        if left_wrist < left_shoulder or right_wrist < right_shoulder:
            raised_hands += 1

    # Show result on original frame
    cv2.putText(frame, f"Raised Hands: {raised_hands}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Webcam Hand Raise Detection", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
