import csv
import os
from datetime import datetime


class FeatureLogger:
    def __init__(self, log_dir="logs", filename="pose_features.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, filename)
        self.header_written = os.path.exists(self.file_path)

    def save(self, features, posture=None, label="unknown"):
        pose = features.get("pose", {})
        posture = posture or {}

        row = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "has_hand": features.get("has_hand", False),
            "has_pose": features.get("has_pose", False),
            "hand_count": len(features.get("hands", [])),
        }

        row.update(pose)
        row.update(posture)

        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            writer.writerow(row)