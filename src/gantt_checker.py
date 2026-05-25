import csv
import os


class GanttChecker:
    """
    Compare current AI action against a standard work timeline.

    CSV format:
    step,start_sec,end_sec,expected_action,required_object,allowed_posture
    1,0,10,normal,none,standing
    2,10,25,bending,none,working_pose
    """

    def __init__(self, csv_path="/app/config/standard_work.csv"):
        self.csv_path = csv_path
        self.steps = []
        self.load()

    def load(self):
        if not os.path.exists(self.csv_path):
            self.steps = []
            return

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.steps = list(reader)

        for step in self.steps:
            step["start_sec"] = float(step["start_sec"])
            step["end_sec"] = float(step["end_sec"])

    def get_expected_step(self, elapsed_sec):
        for step in self.steps:
            if step["start_sec"] <= elapsed_sec < step["end_sec"]:
                return step
        return None

    def check(self, elapsed_sec, actual_action, detections, posture):
        step = self.get_expected_step(elapsed_sec)

        if step is None:
            return {
                "is_ng": False,
                "reason": "no_standard_step",
                "expected_action": "unknown",
            }

        expected_action = step.get("expected_action", "unknown")
        required_object = step.get("required_object", "none")
        allowed_posture = step.get("allowed_posture", "any")

        if expected_action != "any" and actual_action != expected_action:
            return {
                "is_ng": True,
                "reason": f"action_mismatch expected={expected_action} actual={actual_action}",
                "expected_action": expected_action,
            }

        if required_object and required_object != "none":
            detected_names = {d.get("class_name") for d in detections}
            if required_object not in detected_names:
                return {
                    "is_ng": True,
                    "reason": f"required_object_missing object={required_object}",
                    "expected_action": expected_action,
                }

        if allowed_posture == "standing" and not posture.get("is_standing"):
            return {
                "is_ng": True,
                "reason": "posture_mismatch expected=standing",
                "expected_action": expected_action,
            }

        if allowed_posture == "squatting" and not posture.get("is_squatting"):
            return {
                "is_ng": True,
                "reason": "posture_mismatch expected=squatting",
                "expected_action": expected_action,
            }

        if allowed_posture == "working_pose" and not posture.get("is_working_pose"):
            return {
                "is_ng": True,
                "reason": "posture_mismatch expected=working_pose",
                "expected_action": expected_action,
            }

        return {
            "is_ng": False,
            "reason": "ok",
            "expected_action": expected_action,
        }
