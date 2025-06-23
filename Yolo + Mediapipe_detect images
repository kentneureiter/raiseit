import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp

# Load models
yolo_model = YOLO('yolov8n.pt')  # lightweight model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Create folder to save failed crops
os.makedirs("failed_crops", exist_ok=True)

# Load your image here make sure to change path
image_path = r"C:\Users\HP\Downloads\360_F_303989091_TbyLUUWgKPxzBpyWB17viogkh2rxud2E.jpg" 
frame = cv2.imread(image_path)

# Run YOLO
results = yolo_model(frame)[0]
boxes = results.boxes
raised_hands = 0

# Loop over all detected people
for i, box in enumerate(boxes):
    cls_id = int(box.cls[0])
    if cls_id != 0:  # class 0 = person
        continue

    # Get bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    padding = 60
    h, w, _ = frame.shape
    x1 = max(0, x1)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2)
    y2 = min(h, y2 + padding)

    # Crop and resize person
    person_img = frame[y1:y2, x1:x2]
    person_img = cv2.resize(person_img, (480, 480))
    rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb_img)

    # Handle failed pose detection
    if not pose_result.pose_landmarks:
        print(f"[!] Pose not found in box ({x1}, {y1}, {x2}, {y2})")
        fail_path = f"failed_crops/person_{i}_{x1}_{y1}_{x2}_{y2}.jpg"
        cv2.imwrite(fail_path, person_img)
        continue

    # Get key landmarks
    landmarks = pose_result.pose_landmarks.landmark
    left_wrist = landmarks[15].y
    left_shoulder = landmarks[11].y
    right_wrist = landmarks[16].y
    right_shoulder = landmarks[12].y

    # Check if either hand is raised
    if left_wrist < left_shoulder or right_wrist < right_shoulder:
        raised_hands += 1

# Annotate final image
output = frame.copy()
cv2.putText(output, f"Raised Hands: {raised_hands}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

# Show and save result (this is not working as of now need to figure out and fix)
cv2.imshow("Detected Image", output)
cv2.imwrite(r"C:\Users\HP\Desktop\raised_hand_output.jpg", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
