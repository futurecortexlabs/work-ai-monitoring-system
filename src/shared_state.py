import threading
from collections import deque


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()

        self.frame = None
        self.detections = []
        self.features = {
            "has_hand": False,
            "has_pose": False,
            "hands": [],
            "pose": {},
        }
        self.posture = {}
        self.lstm_label = "unknown"
        self.ng_status = {
            "is_ng": False,
            "reason": "not_checked",
            "expected_action": "unknown",
        }

        self.running = True
        self.is_recording = False
        self.current_label = "unknown"

        self.sequence = deque(maxlen=30)

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def update_detections(self, detections):
        with self.lock:
            self.detections = detections or []

    def update_pose(self, features, posture):
        features = features or {
            "has_hand": False,
            "has_pose": False,
            "hands": [],
            "pose": {},
        }
        posture = posture or {}

        with self.lock:
            self.features = features
            self.posture = posture
            self.sequence.append({
                "features": features,
                "posture": posture,
            })

    def update_lstm_label(self, label):
        with self.lock:
            self.lstm_label = label or "unknown"

    def update_ng_status(self, ng_status):
        with self.lock:
            self.ng_status = ng_status or {
                "is_ng": False,
                "reason": "not_checked",
                "expected_action": "unknown",
            }

    def set_label(self, label):
        with self.lock:
            self.current_label = label

    def set_recording(self, value):
        with self.lock:
            self.is_recording = bool(value)

    def stop(self):
        with self.lock:
            self.running = False

    def snapshot(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return {
                "frame": frame,
                "detections": list(self.detections),
                "features": dict(self.features),
                "posture": dict(self.posture),
                "lstm_label": self.lstm_label,
                "ng_status": dict(self.ng_status),
                "is_recording": self.is_recording,
                "current_label": self.current_label,
                "running": self.running,
            }
