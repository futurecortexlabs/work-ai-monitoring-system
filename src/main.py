import cv2
import yaml
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from risk_tracker import RiskTracker
from yolo_detector import YoloDetector
from camera import Camera
from pose_detector import PoseDetector
from feature_logger import FeatureLogger
from posture_analyzer import PostureAnalyzer


CONFIG_PATH = Path("config/sample.yaml")

app = FastAPI()

feature_logger = FeatureLogger()
posture_analyzer = PostureAnalyzer()
risk_tracker = RiskTracker()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_status_panel(frame, features, posture, detections=None):
    h, w, _ = frame.shape
    panel_w = 520
    panel_h = max(h, 720)

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    y = 30
    line_h = 24

    cv2.putText(
        panel,
        "Recognition Status",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )
    y += 38

    status_items = {
        "has_hand": features.get("has_hand", False),
        "has_pose": features.get("has_pose", False),
        "hand_count": len(features.get("hands", [])),
        "yolo_count": len(detections) if detections else 0,
    }

    status_items.update(posture)

    for key, value in status_items.items():
        if value is True:
            color = (0, 255, 0)
        elif value is False:
            color = (120, 120, 120)
        else:
            color = (255, 255, 255)

        text = f"{key}: {value}"

        cv2.putText(
            panel,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
        )

        y += line_h

    if frame.shape[0] < panel_h:
        pad_h = panel_h - frame.shape[0]
        frame = cv2.copyMakeBorder(
            frame,
            0,
            pad_h,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    combined = np.hstack((frame, panel))
    return combined


config = load_config()
camera_config = config["camera"]

camera = Camera(
    source=camera_config["source"],
    width=camera_config["width"],
    height=camera_config["height"],
    fps=camera_config["fps"],
)

detector = YoloDetector(
    model_path=config["models"]["yolo_model_path"],
    confidence=config["detection"]["confidence_threshold"],
)

pose_detector = PoseDetector()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Work AI Monitoring System</title>
        </head>
        <body>
            <h1>Work AI Monitoring System</h1>
            <h2>Live Camera</h2>
            <img src="/video_feed" width="1280">
        </body>
    </html>
    """


def generate_frames():
    if camera.cap is None or not camera.cap.isOpened():
        camera.open()

    frame_count = 0
    last_detections = []

    try:
        while True:
            frame = camera.read()

            if frame is None:
                continue

            frame_count += 1

            if frame_count % 5 == 0:
                frame, detections = detector.predict(frame)
                last_detections = detections
            else:
                detections = last_detections

            frame, features = pose_detector.detect(frame)

            posture = posture_analyzer.analyze(features)
            risk_state = risk_tracker.update(posture)
            posture.update(risk_state)

            if frame_count % 10 == 0:
                feature_logger.save(
                    features,
                    posture=posture,
                    label="unknown",
                )

            frame = create_status_panel(
                frame,
                features,
                posture,
                detections=detections,
            )

            ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    except GeneratorExit:
        pass


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    print("===================================")
    print("Work AI Monitoring System Started")
    print("===================================")
    print("Open: http://localhost:8000")
    print("===================================")

    uvicorn.run(app, host="0.0.0.0", port=8000)