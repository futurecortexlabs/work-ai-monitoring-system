import cv2
import yaml
import time
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from shared_state import SharedState
from camera import Camera
from yolo_detector import YoloDetector
from pose_detector import PoseDetector
from posture_analyzer import PostureAnalyzer
from risk_tracker import RiskTracker
from feature_logger import FeatureLogger
from gantt_checker import GanttChecker


CONFIG_PATH = Path("config/sample.yaml")

app = FastAPI()
state = SharedState()

feature_logger = FeatureLogger()
posture_analyzer = PostureAnalyzer()
risk_tracker = RiskTracker()
gantt_checker = GanttChecker()

threads_started = False
start_time = time.time()


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
camera_config = config["camera"]

camera = Camera(
    source=camera_config["source"],
    width=camera_config["width"],
    height=camera_config["height"],
    fps=camera_config["fps"],
)

yolo_detector = YoloDetector(
    model_path=config["models"]["yolo_model_path"],
    confidence=config["detection"]["confidence_threshold"],
)

pose_detector = PoseDetector(draw_landmarks=False)


def camera_loop():
    print("Camera loop started")

    try:
        camera.open()
        print("Camera opened")
    except Exception as e:
        print(f"Camera open error: {e}")
        state.stop()
        return

    while state.snapshot()["running"]:
        try:
            frame = camera.read()

            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                state.update_frame(frame)

        except Exception as e:
            print(f"Camera read error: {e}")
            time.sleep(0.1)

        time.sleep(0.001)

    camera.release()
    print("Camera loop stopped")


def yolo_loop():
    print("YOLO loop started")

    while state.snapshot()["running"]:
        frame = state.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            _, detections = yolo_detector.predict(frame, draw=False)
            state.update_detections(detections)
        except Exception as e:
            print(f"YOLO error: {e}")
            state.update_detections([])

        # YOLO is intentionally slower than camera.
        time.sleep(0.20)


def pose_loop():
    print("Pose loop started")

    while state.snapshot()["running"]:
        frame = state.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            _, features = pose_detector.detect(frame)

            posture = posture_analyzer.analyze(features)
            risk_state = risk_tracker.update(posture)
            posture.update(risk_state)

            state.update_pose(features, posture)

        except Exception as e:
            print(f"Pose error: {e}")

        # MediaPipe is intentionally slower than camera.
        time.sleep(0.10)


def lstm_loop():
    print("LSTM loop started")

    while state.snapshot()["running"]:
        snapshot = state.snapshot()
        posture = snapshot["posture"]

        # Temporary rule-based classifier.
        # Replace this block with a real LSTM model later.
        if posture.get("is_squatting"):
            label = "squat"
        elif posture.get("is_forward_bending"):
            label = "bending"
        elif posture.get("is_both_arms_up"):
            label = "arms_up"
        else:
            label = "normal"

        state.update_lstm_label(label)

        time.sleep(0.20)


def gantt_loop():
    print("Gantt checker loop started")

    while state.snapshot()["running"]:
        snapshot = state.snapshot()

        elapsed_sec = time.time() - start_time
        actual_action = snapshot["lstm_label"]
        detections = snapshot["detections"]
        posture = snapshot["posture"]

        ng_status = gantt_checker.check(
            elapsed_sec=elapsed_sec,
            actual_action=actual_action,
            detections=detections,
            posture=posture,
        )

        state.update_ng_status(ng_status)

        time.sleep(0.20)


def logger_loop():
    print("Logger loop started")

    while state.snapshot()["running"]:
        snapshot = state.snapshot()

        if snapshot["is_recording"]:
            feature_logger.save(
                snapshot["features"],
                posture=snapshot["posture"],
                label=snapshot["current_label"],
                extra={
                    "lstm_label": snapshot["lstm_label"],
                    "is_ng": snapshot["ng_status"].get("is_ng"),
                    "ng_reason": snapshot["ng_status"].get("reason"),
                    "expected_action": snapshot["ng_status"].get("expected_action"),
                },
            )

        time.sleep(1.0)


def draw_overlay(frame, snapshot):
    ng = snapshot["ng_status"]
    is_ng = ng.get("is_ng", False)

    status_text = (
        f"REC:{snapshot['is_recording']} "
        f"TEACH:{snapshot['current_label']} "
        f"AI:{snapshot['lstm_label']} "
        f"NG:{is_ng}"
    )

    color = (0, 0, 255) if is_ng else (0, 255, 255)

    cv2.putText(
        frame,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )

    if is_ng:
        cv2.putText(
            frame,
            ng.get("reason", "NG"),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

    return frame


def start_threads():
    global threads_started

    if threads_started:
        return

    threads_started = True

    workers = [
        camera_loop,
        yolo_loop,
        pose_loop,
        lstm_loop,
        gantt_loop,
        logger_loop,
    ]

    for worker in workers:
        t = threading.Thread(target=worker, daemon=True)
        t.start()


@app.on_event("startup")
def startup_event():
    start_threads()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Work AI Monitoring System</title>
        </head>
        <body style="background:#111;color:white;font-family:sans-serif;padding:20px;">

            <h1>Work AI Monitoring System</h1>

            <h2 id="recordingStatus" style="color:#ff5555;">REC: OFF</h2>
            <h2 id="currentLabel" style="color:#00ccff;">TEACH LABEL: unknown</h2>

            <div style="margin-bottom:20px;">
                <button onclick="setLabel('unknown')">unknown</button>
                <button onclick="setLabel('normal')">normal</button>
                <button onclick="setLabel('squat')">squat</button>
                <button onclick="setLabel('bending')">bending</button>
                <button onclick="setLabel('arms_up')">arms_up</button>

                <br><br>

                <button onclick="startRecording()" style="background:#00aa44;color:white;padding:12px 20px;">
                    Start Recording
                </button>

                <button onclick="stopRecording()" style="background:#aa0000;color:white;padding:12px 20px;">
                    Stop Recording
                </button>

                <button onclick="newSession()" style="background:#666;color:white;padding:12px 20px;">
                    New Session
                </button>

                <button onclick="releaseCamera()" style="background:#4444aa;color:white;padding:12px 20px;">
                    Release Camera
                </button>
            </div>

            <img src="/video_feed" width="640">

            <script>
                async function setLabel(label) {
                    await fetch(`/set_label/${label}`);
                    document.getElementById("currentLabel").innerText = "TEACH LABEL: " + label;
                }

                async function startRecording() {
                    await fetch('/start_recording');
                    document.getElementById("recordingStatus").innerText = "REC: ON";
                    document.getElementById("recordingStatus").style.color = "#00ff66";
                }

                async function stopRecording() {
                    await fetch('/stop_recording');
                    document.getElementById("recordingStatus").innerText = "REC: OFF";
                    document.getElementById("recordingStatus").style.color = "#ff5555";
                }

                async function newSession() {
                    const res = await fetch('/new_session');
                    const data = await res.json();
                    alert('New log file: ' + data.log_file);
                }

                async function releaseCamera() {
                    await fetch('/release_camera');
                    document.getElementById("recordingStatus").innerText = "CAMERA RELEASED";
                    document.getElementById("recordingStatus").style.color = "#ffaa00";
                }
            </script>
        </body>
    </html>
    """


@app.get("/set_label/{label}")
def set_label(label: str):
    state.set_label(label)
    return {"current_label": label}


@app.get("/start_recording")
def start_recording():
    state.set_recording(True)
    return {"is_recording": True}


@app.get("/stop_recording")
def stop_recording():
    state.set_recording(False)
    return {"is_recording": False}


@app.get("/new_session")
def new_session():
    path = feature_logger.new_session()
    return {"log_file": path}


@app.get("/release_camera")
def release_camera():
    state.stop()
    camera.release()
    return {"camera": "released"}


@app.get("/status")
def status():
    snapshot = state.snapshot()
    snapshot["frame"] = snapshot["frame"] is not None
    return snapshot


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def generate_frames():
    while True:
        snapshot = state.snapshot()

        if not snapshot["running"]:
            break

        frame = snapshot["frame"]

        if frame is None:
            time.sleep(0.03)
            continue

        frame = draw_overlay(frame, snapshot)

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )

        if not ret:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        time.sleep(0.01)


if __name__ == "__main__":
    import uvicorn

    print("===================================")
    print("Work AI Monitoring System Started")
    print("===================================")
    print("Open: http://localhost:8000")
    print("===================================")

    uvicorn.run(app, host="0.0.0.0", port=8000)
