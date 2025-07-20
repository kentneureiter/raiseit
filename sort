import cv2
import math
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# --------------------------
# SORT Tracker Implementation
# --------------------------
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + 
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:4] = pos
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, dets, trks):
        if(len(trks) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)
        matched_indices = linear_assignment(-iou_matrix)
        unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < self.iou_threshold):
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if(len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

# --------------------------
# Detection + Pose + Tracking
# --------------------------
FRAME_THRESHOLD = 30
COUNTDOWN_THRESHOLD = 20
TIMEOUT_SECONDS = 5

yolo_model = YOLO('yolov8n.pt')
pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
cap = cv2.VideoCapture(0)
tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)
tracked_people = {}

def is_hand_raised(landmarks):
    left_wrist = landmarks[15]
    left_elbow = landmarks[13]
    left_shoulder = landmarks[11]
    right_wrist = landmarks[16]
    right_elbow = landmarks[14]
    right_shoulder = landmarks[12]
    left_raised = left_wrist.y < left_shoulder.y and left_wrist.y < left_elbow.y
    right_raised = right_wrist.y < right_shoulder.y and right_wrist.y < right_elbow.y
    return left_raised or right_raised

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Couldn't read from webcam.")
        break

    current_time = time.time()
    results = yolo_model(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append([x1, y1, x2, y2, float(box.conf[0])])

    dets = np.array(boxes)
    if len(dets) == 0:
        dets = np.empty((0, 5))

    tracked_objects = tracker.update(dets)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        person_img = frame[y1:y2, x1:x2]
        hand_raised = False
        if person_img.size > 0:
            rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb_img)
            if pose_result.pose_landmarks:
                hand_raised = is_hand_raised(pose_result.pose_landmarks.landmark)

        if obj_id not in tracked_people:
            tracked_people[obj_id] = {'frames_raised': 0, 'frames_not_raised': 0, 'last_seen': current_time, 'center': (center_x, center_y)}

        person = tracked_people[obj_id]
        person['center'] = (center_x, center_y)
        person['last_seen'] = current_time

        if hand_raised:
            person['frames_raised'] = min(FRAME_THRESHOLD, person['frames_raised'] + 1)
            person['frames_not_raised'] = 0
        else:
            person['frames_not_raised'] += 1
            if person['frames_not_raised'] > COUNTDOWN_THRESHOLD:
                person['frames_raised'] = 0

    tracked_people = {pid: p for pid, p in tracked_people.items() if current_time - p['last_seen'] < TIMEOUT_SECONDS}

    raised_hands = 0
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        if tracked_people[obj_id]['frames_raised'] >= FRAME_THRESHOLD:
            raised_hands += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"ID {obj_id} - RAISED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Raised Hands: {raised_hands}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("YOLO + SORT Hand Raise Detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
