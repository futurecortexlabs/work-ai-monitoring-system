import threading


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
        self.gantt_result = {
            "result": "WAIT",
            "reason": "not_started",
        }

        self.current_label = "unknown"
        self.is_recording = False
        self.running = True

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def update_ai(self, detections, features, posture, gantt_result):
        with self.lock:
            self.detections = detections or []
            self.features = features or self.features
            self.posture = posture or {}
            self.gantt_result = gantt_result or self.gantt_result

    def snapshot(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return {
                "frame": frame,
                "detections": self.detections,
                "features": self.features,
                "posture": self.posture,
                "gantt_result": self.gantt_result,
                "current_label": self.current_label,
                "is_recording": self.is_recording,
            }