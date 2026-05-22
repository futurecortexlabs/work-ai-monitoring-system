class PostureAnalyzer:
    def analyze(self, features):
        pose = features.get("pose", {})

        result = {
            "is_standing": False,
            "is_sitting_like": False,
            "is_forward_bending": False,
            "is_backward_bending": False,
            "is_squatting": False,

            "is_left_arm_up": False,
            "is_right_arm_up": False,
            "is_both_arms_up": False,
            "is_left_arm_down": False,
            "is_right_arm_down": False,

            "is_hand_below_hip": False,
            "is_hand_above_shoulder": False,

            "is_leaning_left": False,
            "is_leaning_right": False,

            "is_facing_left_like": False,
            "is_facing_right_like": False,

            "is_working_pose": False,

            "risk_score": 0,
            "risk_level": "low",
        }

        if not features.get("has_pose"):
            return result

        def y(name):
            return pose.get(f"{name}_y")

        def x(name):
            return pose.get(f"{name}_x")

        required = [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_wrist", "right_wrist",
            "left_elbow", "right_elbow",
        ]

        for name in required:
            if x(name) is None or y(name) is None:
                return result

        lsx, lsy = x("left_shoulder"), y("left_shoulder")
        rsx, rsy = x("right_shoulder"), y("right_shoulder")
        lhx, lhy = x("left_hip"), y("left_hip")
        rhx, rhy = x("right_hip"), y("right_hip")
        lkx, lky = x("left_knee"), y("left_knee")
        rkx, rky = x("right_knee"), y("right_knee")
        lwx, lwy = x("left_wrist"), y("left_wrist")
        rwx, rwy = x("right_wrist"), y("right_wrist")
        lex, ley = x("left_elbow"), y("left_elbow")
        rex, rey = x("right_elbow"), y("right_elbow")

        shoulder_y = (lsy + rsy) / 2
        hip_y = (lhy + rhy) / 2
        knee_y = (lky + rky) / 2

        shoulder_x = (lsx + rsx) / 2
        hip_x = (lhx + rhx) / 2

        shoulder_width = abs(lsx - rsx)
        hip_width = abs(lhx - rhx)

        # MediaPipeは上が0、下が1
        result["is_standing"] = hip_y < knee_y - 0.10
        result["is_sitting_like"] = abs(hip_y - knee_y) < 0.12
        result["is_squatting"] = abs(hip_y - knee_y) < 0.15

        # 前傾・後傾っぽさ
        result["is_forward_bending"] = abs(shoulder_y - hip_y) < 0.13
        result["is_backward_bending"] = shoulder_y < hip_y - 0.35

        # 腕の上げ下げ
        result["is_left_arm_up"] = lwy < lsy
        result["is_right_arm_up"] = rwy < rsy
        result["is_both_arms_up"] = result["is_left_arm_up"] and result["is_right_arm_up"]

        result["is_left_arm_down"] = lwy > lhy
        result["is_right_arm_down"] = rwy > rhy

        # 手の高さ
        result["is_hand_below_hip"] = lwy > lhy or rwy > rhy
        result["is_hand_above_shoulder"] = lwy < lsy or rwy < rsy

        # 体の左右傾き
        result["is_leaning_left"] = shoulder_x < hip_x - 0.05
        result["is_leaning_right"] = shoulder_x > hip_x + 0.05

        # 体の向きっぽさ
        # 肩幅が腰幅より極端に狭い場合、横向きの可能性あり
        if shoulder_width < hip_width * 0.8:
            result["is_facing_left_like"] = rsx < lsx
            result["is_facing_right_like"] = lsx < rsx

        # 作業姿勢っぽい条件
        result["is_working_pose"] = (
            result["is_forward_bending"]
            or result["is_squatting"]
            or result["is_hand_below_hip"]
            or result["is_hand_above_shoulder"]
        )

        # 危険姿勢スコア
        risk_score = 0

        if result["is_forward_bending"]:
            risk_score += 2

        if result["is_squatting"]:
            risk_score += 2

        if result["is_both_arms_up"]:
            risk_score += 2

        if result["is_leaning_left"] or result["is_leaning_right"]:
            risk_score += 1

        if result["is_hand_below_hip"]:
            risk_score += 1

        result["risk_score"] = risk_score

        if risk_score >= 5:
            result["risk_level"] = "high"
        elif risk_score >= 3:
            result["risk_level"] = "middle"
        else:
            result["risk_level"] = "low"

        return result

        return result