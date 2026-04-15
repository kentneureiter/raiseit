import cv2
import math
import time
from ultralytics import YOLO
import mediapipe as mp


# How many consecutive frames a hand must be raised before confirmed
FRAME_THRESHOLD = 60

# How many frames of hand-down before a confirmed raise is removed
COUNTDOWN_THRESHOLD = 30

# If no progress for this many seconds, reset that person's count
TIMEOUT_SECONDS = 10


class HandRaiseDetector:

    def __init__(self):
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')

        # Load MediaPipe for pose estimation
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1
        )

        # List that persists between frames — this is the memory of the system
        self.tracked_people = []

    def detect(self, frame):
        """
        Main method — call this on every frame.
        Returns the number of confirmed raised hands.
        """
        current_time = time.time()

        # Step 1: YOLO finds all people in the frame
        results = self.yolo(frame)[0]
        current_detections = []

        for box in results.boxes:
            # Skip anything that isn't a person (class 0)
            if int(box.cls[0]) != 0:
                continue

            # Get bounding box coordinates with padding
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1 - 60)   # padding above head
            x2 = min(w, x2)
            y2 = min(h, y2 + 60)   # padding below feet

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Step 2: crop this person and check their pose
            hand_raised = self._check_pose(frame, x1, y1, x2, y2)
            current_detections.append({
                'center': (center_x, center_y),
                'hand_raised': hand_raised
            })

        # Step 3: update the state machine for each detected person
        self._update_tracking(current_detections, current_time)

        # Step 4: clean up people who have disappeared from frame
        self.tracked_people = [
            p for p in self.tracked_people
            if current_time - p.get('last_seen', current_time) < 5
        ]

        # Step 5: count and return confirmed raises
        confirmed = [p for p in self.tracked_people if p['frames_raised'] >= FRAME_THRESHOLD]
        return len(confirmed)

    def _check_pose(self, frame, x1, y1, x2, y2):
        """
        Crops one person from the frame and checks if their hand is raised.
        Returns True or False.
        """
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return False

        # Resize to standard size — this is what makes distance not matter
        person_crop = cv2.resize(person_crop, (480, 480))
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_crop)

        if not result.pose_landmarks:
            return False

        lm = result.pose_landmarks.landmark

        left_raised  = lm[15].y < lm[11].y   # left wrist above left shoulder
        right_raised = lm[16].y < lm[12].y   # right wrist above right shoulder

        return left_raised or right_raised

    def _update_tracking(self, current_detections, current_time):
        """
        Updates the state machine for each detected person.
        Matches detections to existing tracked people by proximity.
        """
        for detection in current_detections:
            cx, cy = detection['center']
            hand_raised = detection['hand_raised']
            matched = False

            for person in self.tracked_people:
                dist = math.hypot(cx - person['center'][0], cy - person['center'][1])

                if dist < 50:  # same person if within 50 pixels
                    person['center'] = (cx, cy)
                    person['last_seen'] = current_time

                    if hand_raised:
                        if person['frames_raised'] < FRAME_THRESHOLD:
                            person['frames_raised'] += 1
                            person['last_progress_time'] = current_time
                        person['frames_not_raised'] = 0
                        person['state'] = 'confirmed' if person['frames_raised'] >= FRAME_THRESHOLD else 'raising'

                    else:
                        if person['frames_raised'] >= FRAME_THRESHOLD:
                            person['frames_not_raised'] += 1
                            person['state'] = 'countdown'
                            if person['frames_not_raised'] >= COUNTDOWN_THRESHOLD:
                                person['frames_raised'] = 0
                                person['frames_not_raised'] = 0
                                person['state'] = 'idle'
                        else:
                            if current_time - person.get('last_progress_time', current_time) > TIMEOUT_SECONDS:
                                person['frames_raised'] = 0
                                person['state'] = 'idle'
                            person['frames_not_raised'] += 1

                    matched = True
                    break

            if not matched:
                # First time seeing this person
                self.tracked_people.append({
                    'center': (cx, cy),
                    'frames_raised': 1 if hand_raised else 0,
                    'frames_not_raised': 0,
                    'state': 'raising' if hand_raised else 'idle',
                    'last_progress_time': current_time,
                    'last_seen': current_time
                })