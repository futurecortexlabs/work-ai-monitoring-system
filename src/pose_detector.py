import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, draw_landmarks=False):
        self.draw_landmarks = draw_landmarks

        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb)
        pose_results = self.pose.process(rgb)

        features = {
            "hands": [],
            "pose": {},
            "has_hand": False,
            "has_pose": False,
        }

        if hand_results.multi_hand_landmarks:
            features["has_hand"] = True

            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_points = []

                for lm in hand_landmarks.landmark:
                    hand_points.extend([lm.x, lm.y, lm.z])

                features["hands"].append(hand_points)

                if self.draw_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

        if pose_results.pose_landmarks:
            features["has_pose"] = True

            landmarks = pose_results.pose_landmarks.landmark

            target_points = {
                "left_shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                "right_shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                "left_elbow": self.mp_pose.PoseLandmark.LEFT_ELBOW,
                "right_elbow": self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                "left_wrist": self.mp_pose.PoseLandmark.LEFT_WRIST,
                "right_wrist": self.mp_pose.PoseLandmark.RIGHT_WRIST,
                "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP,
                "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP,
                "left_knee": self.mp_pose.PoseLandmark.LEFT_KNEE,
                "right_knee": self.mp_pose.PoseLandmark.RIGHT_KNEE,
            }

            pose_dict = {}

            for name, point in target_points.items():
                lm = landmarks[point.value]
                pose_dict[f"{name}_x"] = lm.x
                pose_dict[f"{name}_y"] = lm.y
                pose_dict[f"{name}_z"] = lm.z
                pose_dict[f"{name}_visibility"] = lm.visibility

            features["pose"] = pose_dict

            if self.draw_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                )

        return frame, features
