import csv
import os
from datetime import datetime


class NGLogger:
    def __init__(self, log_dir="logs", filename="ng_log.csv"):
        os.makedirs(log_dir, exist_ok=True)

        self.file_path = os.path.join(log_dir, filename)
        self.header_written = os.path.exists(self.file_path)

    def save(self, elapsed_sec, gantt_result, actual_action, posture_label, detected_objects):
        with open(self.file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not self.header_written:
                writer.writerow([
                    "timestamp",
                    "elapsed_sec",
                    "result",
                    "reason",
                    "expected_step",
                    "expected_action",
                    "required_object",
                    "allowed_posture",
                    "actual_action",
                    "posture_label",
                    "detected_objects",
                ])
                self.header_written = True

            expected = gantt_result.get("expected") or {}

            writer.writerow([
                datetime.now().isoformat(),
                round(elapsed_sec, 2),
                gantt_result.get("result"),
                gantt_result.get("reason"),
                expected.get("step"),
                expected.get("expected_action"),
                expected.get("required_object"),
                expected.get("allowed_posture"),
                actual_action,
                posture_label,
                ",".join(detected_objects or []),
            ])