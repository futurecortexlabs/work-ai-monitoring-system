import csv
import os
from datetime import datetime


class FeatureLogger:
    def __init__(self, log_dir="/app/logs", filename="pose_features.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.file_path = os.path.join(log_dir, filename)
        self.header_written = os.path.exists(self.file_path)

    def new_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(
            self.log_dir,
            f"pose_features_{timestamp}.csv",
        )
        self.header_written = False
        return self.file_path

    def save(self, features, posture=None, label="unknown", extra=None):
        pose = features.get("pose", {})
        posture = posture or {}
        extra = extra or {}

        row = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "has_hand": features.get("has_hand", False),
            "has_pose": features.get("has_pose", False),
            "hand_count": len(features.get("hands", [])),
        }

        row.update(pose)
        row.update(posture)
        row.update(extra)

        file_exists = os.path.exists(self.file_path)

        with open(self.file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not self.header_written or not file_exists:
                writer.writeheader()
                self.header_written = True

            writer.writerow(row)
