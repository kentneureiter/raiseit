import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp
import math  # For distance calculation

# Load models
yolo_model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Webcam input
cap = cv2.VideoCapture(0)

# Simple person tracker
tracked_people = []
FRAME_THRESHOLD = 60  # ~2 seconds at 30 FPS

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Couldn't read from webcam.")
        break

    # Run YOLO
    results = yolo_model(frame)[0]
    boxes = results.boxes

    # Keep list of current detections
    current_detections = []

    # Loop over all detected people
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue  # Only "person" class

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        padding = 60
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2)
        y2 = min(h, y2 + padding)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        current_detections.append({'center': (center_x, center_y), 'hand_raised': False})

        # Crop person
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue

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
            current_detections[-1]['hand_raised'] = True

    # Update tracked people
    for detection in current_detections:
        cx, cy = detection['center']
        hand_raised = detection['hand_raised']

        matched = False
        for person in tracked_people:
            dist = math.hypot(cx - person['center'][0], cy - person['center'][1])
            if dist < 50:
                person['center'] = (cx, cy)
                if hand_raised:
                    person['frames_raised'] += 1
                    person['frames_not_raised'] = 0
                else:
                    person['frames_not_raised'] += 1
                    if person['frames_not_raised'] > 30:
                        person['frames_raised'] = 0
                matched = True
                break

        if not matched:
            tracked_people.append({
                'center': (cx, cy),
                'frames_raised': 1 if hand_raised else 0,
                'frames_not_raised': 0
            })

    # Draw only green circles for confirmed raised hands
    raised_hands = 0
    for person in tracked_people:
        if person['frames_raised'] >= FRAME_THRESHOLD:
            cv2.circle(frame, (int(person['center'][0]), int(person['center'][1])), 20, (0, 255, 0), -1)
            raised_hands += 1

    # Show result
    cv2.putText(frame, f"Raised Hands: {raised_hands}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Webcam Hand Raise Detection", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

"""

"""
