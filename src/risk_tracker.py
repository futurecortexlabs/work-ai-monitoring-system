import time


class RiskTracker:
    def __init__(self):
        self.start_times = {}

    def update(self, posture):
        now = time.time()

        watch_items = [
            "is_forward_bending",
            "is_squatting",
            "is_both_arms_up",
            "is_leaning_left",
            "is_leaning_right",
        ]

        result = {}

        for key in watch_items:
            if posture.get(key):
                if key not in self.start_times:
                    self.start_times[key] = now

                duration = now - self.start_times[key]
            else:
                self.start_times.pop(key, None)
                duration = 0

            result[f"{key}_duration_sec"] = round(duration, 2)

        result["is_long_forward_bending"] = (
            result["is_forward_bending_duration_sec"] >= 3
        )
        result["is_long_squatting"] = (
            result["is_squatting_duration_sec"] >= 3
        )
        result["is_long_arms_up"] = (
            result["is_both_arms_up_duration_sec"] >= 5
        )

        return result
