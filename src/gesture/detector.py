import cv2
import time
import numpy as np
from ultralytics import YOLO
import supervision as sv

FRAME_THRESHOLD = 60
COUNTDOWN_THRESHOLD = 30
TIMEOUT_SECONDS = 10

# Padding added above/below each person crop (pixels, pre-scale)
CROP_PADDING_Y = 60
# Target size for MediaPipe input — tall rectangle to preserve body proportions
POSE_INPUT_SIZE = (256, 512)  # (width, height) — 1:2 aspect matches standing person
# Wrist must be this many normalized units above shoulder to count as raised
# Reduces false positives from people sitting upright
RAISE_MARGIN = 0.05


class HandRaiseDetector:

    def __init__(self):
        self.yolo = YOLO('yolov8n-pose.pt')  # downloads automatically on first run
        self.tracker = sv.ByteTrack()
        self.tracked_people = {}

        # ByteTrack: handles occlusion, re-entry, and crowded scenes
        self.tracker = sv.ByteTrack()

        # Keyed by ByteTrack tracker_id (stable across frames)
        self.tracked_people = {}

    #  Public API                                                          

    def detect(self, frame):
        current_time = time.time()

        raw_results = self.yolo(frame, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(raw_results)
        detections = detections[detections.class_id == 0]
        detections = self.tracker.update_with_detections(detections)

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            x1, y1, x2, y2 = map(int, detections.xyxy[i])

            h, w = frame.shape[:2]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # keypoints come directly from the pose model result
            kps = raw_results.keypoints
            keypoints = kps.data[i].cpu().numpy() if kps is not None and i < len(kps.data) else None

            hand_raised = self._check_pose(keypoints)

            if tracker_id not in self.tracked_people:
                self.tracked_people[tracker_id] = self._new_person(
                    center_x, center_y, hand_raised, current_time
                )
            else:
                self._update_person(
                    self.tracked_people[tracker_id],
                    center_x, center_y,
                    hand_raised, current_time
                )

        self.tracked_people = {
            tid: p for tid, p in self.tracked_people.items()
            if current_time - p['last_seen'] < 5
        }

        confirmed = [p for p in self.tracked_people.values()
                 if p['frames_raised'] >= FRAME_THRESHOLD]
        return len(confirmed)


    #  Pose check - aspect-ratio-preserving crop                          
    def _check_pose(self, keypoints):
    # keypoints shape: (17, 3) — x, y, confidence per landmark
    # COCO indices: 5=left shoulder, 6=right shoulder, 9=left wrist, 10=right wrist
        if keypoints is None or len(keypoints) < 11:
            return False

        ls_conf  = keypoints[5][2]
        rs_conf  = keypoints[6][2]
        lw_conf  = keypoints[9][2]
        rw_conf  = keypoints[10][2]

        if max(ls_conf, rs_conf) < 0.3 or max(lw_conf, rw_conf) < 0.3:
            return False

        ls_y = keypoints[5][1]
        rs_y = keypoints[6][1]
        lw_y = keypoints[9][1]
        rw_y = keypoints[10][1]

        left_raised  = lw_y < ls_y - RAISE_MARGIN * (ls_y)
        right_raised = rw_y < rs_y - RAISE_MARGIN * (rs_y)

        return left_raised or right_raised

    #  State machine helpers                                               
    def _new_person(self, cx, cy, hand_raised, current_time):
        return {
            'center': (cx, cy),
            'frames_raised': 1 if hand_raised else 0,
            'frames_not_raised': 0,
            'state': 'raising' if hand_raised else 'idle',
            'last_progress_time': current_time,
            'last_seen': current_time,
        }

    def _update_person(self, person, cx, cy, hand_raised, current_time):
        person['center'] = (cx, cy)
        person['last_seen'] = current_time

        if hand_raised:
            if person['frames_raised'] < FRAME_THRESHOLD:
                person['frames_raised'] += 1
                person['last_progress_time'] = current_time
            person['frames_not_raised'] = 0
            person['state'] = (
                'confirmed' if person['frames_raised'] >= FRAME_THRESHOLD
                else 'raising'
            )
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