import cv2
import time
import numpy as np
import onnxruntime as ort
import supervision as sv

FRAME_THRESHOLD = 30
COUNTDOWN_THRESHOLD = 30
TIMEOUT_SECONDS = 10
RAISE_MARGIN = 0.05
CONF_THRESHOLD = 0.3
INPUT_SIZE = 320    #input resolution (once upgrade to jetson --> raise to 640)


class HandRaiseDetector:

    def __init__(self, model_path='yolov8n-pose.onnx'):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.tracker = sv.ByteTrack()
        self.tracked_people = {}

    def detect(self, frame):
        current_time = time.time()

        # Preprocess
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        raw = outputs[0][0]  # shape: (8400, 56)

        # Parse detections
        boxes, scores, keypoints_list = self._parse_outputs(
            raw, scale, pad_x, pad_y, frame.shape
        )

        if len(boxes) == 0:
            self._expire_tracks(current_time)
            return len([p for p in self.tracked_people.values()
                        if p['frames_raised'] >= FRAME_THRESHOLD])

        # Feed into ByteTrack
        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(scores),
            class_id=np.zeros(len(boxes), dtype=int)
        )
        detections = self.tracker.update_with_detections(detections)

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Match keypoints to this detection by closest box center
            kps = self._match_keypoints(
                detections.xyxy[i], boxes, keypoints_list
            )
            hand_raised = self._check_pose(kps)

            if tracker_id not in self.tracked_people:
                self.tracked_people[tracker_id] = self._new_person(
                    center_x, center_y, hand_raised, current_time
                )
            else:
                self._update_person(
                    self.tracked_people[tracker_id],
                    center_x, center_y, hand_raised, current_time
                )

        self._expire_tracks(current_time)
        return len([p for p in self.tracked_people.values()
                    if p['frames_raised'] >= FRAME_THRESHOLD])

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        pad_x = (INPUT_SIZE - new_w) // 2
        pad_y = (INPUT_SIZE - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]
        return tensor, scale, pad_x, pad_y

    def _parse_outputs(self, raw, scale, pad_x, pad_y, frame_shape):
        # raw shape: (56, 8400) — transpose to (8400, 56)
        if raw.shape[0] == 56:
            raw = raw.T

        boxes, scores, keypoints_list = [], [], []
        fh, fw = frame_shape[:2]

        for det in raw:
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            conf = det[4]
            if conf < CONF_THRESHOLD:
                continue

            # Convert from padded input coords back to frame coords
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale

            x1 = max(0, min(fw, x1))
            y1 = max(0, min(fh, y1))
            x2 = max(0, min(fw, x2))
            y2 = max(0, min(fh, y2))

            # Keypoints: det[5:] = 17 * 3 values (x, y, conf)
            kps_raw = det[5:].reshape(17, 3)
            kps = np.zeros((17, 3))
            for k in range(17):
                kps[k, 0] = (kps_raw[k, 0] - pad_x) / scale
                kps[k, 1] = (kps_raw[k, 1] - pad_y) / scale
                kps[k, 2] = kps_raw[k, 2]

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            keypoints_list.append(kps)

        return boxes, scores, keypoints_list

    def _match_keypoints(self, target_box, all_boxes, keypoints_list):
        if not all_boxes:
            return None
        tx1, ty1, tx2, ty2 = target_box
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        best_idx, best_dist = 0, float('inf')
        for i, (bx1, by1, bx2, by2) in enumerate(all_boxes):
            dist = ((tcx - (bx1 + bx2) / 2) ** 2 +
                    (tcy - (by1 + by2) / 2) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_idx = dist, i
        return keypoints_list[best_idx]

    def _check_pose(self, keypoints):
        if keypoints is None:
            return False
        # COCO: 5=left shoulder, 6=right shoulder, 9=left wrist, 10=right wrist
        ls_y, ls_c = keypoints[5][1], keypoints[5][2]
        rs_y, rs_c = keypoints[6][1], keypoints[6][2]
        lw_y, lw_c = keypoints[9][1], keypoints[9][2]
        rw_y, rw_c = keypoints[10][1], keypoints[10][2]

        if max(ls_c, rs_c) < 0.3 or max(lw_c, rw_c) < 0.3:
            return False

        left_raised  = lw_y < ls_y * (1 - RAISE_MARGIN)
        right_raised = rw_y < rs_y * (1 - RAISE_MARGIN)
        return left_raised or right_raised

    def _expire_tracks(self, current_time):
        self.tracked_people = {
            tid: p for tid, p in self.tracked_people.items()
            if current_time - p['last_seen'] < 5
        }

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