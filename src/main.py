import cv2
import yaml
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from yolo_detector import YoloDetector
from camera import Camera
from pose_detector import PoseDetector


CONFIG_PATH = Path("config/sample.yaml")

app = FastAPI()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
            <img src="/video_feed" width="640">
        </body>
    </html>
    """


def generate_frames():
    camera.open()

    try:
        while True:
            frame = camera.read()

            if frame is None:
                continue

            frame, detections = detector.predict(frame)
            frame, features = pose_detector.detect(frame)

            cv2.putText(
                frame,
                f"Hand: {features['has_hand']} Pose: {features['has_pose']}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if detections:
                print(detections)
            
            ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        camera.release()


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