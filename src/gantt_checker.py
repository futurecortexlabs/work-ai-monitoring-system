import csv


class GanttChecker:
    def __init__(self, csv_path="config/standard_work.csv"):
        self.steps = []
        self.load(csv_path)

    def load(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.steps.append({
                    "step": row["step"],
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "expected_action": row["expected_action"],
                    "required_object": row["required_object"],
                    "allowed_posture": row["allowed_posture"],
                })

    def get_expected_step(self, elapsed_sec):
        for step in self.steps:
            if step["start_sec"] <= elapsed_sec < step["end_sec"]:
                return step
        return None

    def check(self, elapsed_sec, actual_action, detected_objects, posture_label):
        expected = self.get_expected_step(elapsed_sec)

        if expected is None:
            return {
                "result": "NG",
                "reason": "standard_step_not_found",
                "expected": None,
                "actual_action": actual_action,
            }

        reasons = []

        if actual_action != expected["expected_action"]:
            reasons.append(
                f"action_mismatch expected={expected['expected_action']} actual={actual_action}"
            )

        required_object = expected["required_object"]

        if required_object != "none":
            if required_object not in detected_objects:
                reasons.append(
                    f"object_missing required={required_object}"
                )

        allowed_posture = expected["allowed_posture"]

        if allowed_posture != "any":
            if posture_label != allowed_posture:
                reasons.append(
                    f"posture_mismatch expected={allowed_posture} actual={posture_label}"
                )

        if reasons:
            result = "NG"
        else:
            result = "OK"

        return {
            "result": result,
            "reason": " / ".join(reasons) if reasons else "ok",
            "expected": expected,
            "actual_action": actual_action,
            "detected_objects": detected_objects,
            "posture_label": posture_label,
        }