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

current_label = "unknown"
is_recording = False


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_status_panel(frame, features, posture, detections=None, label="unknown", is_recording=False):
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
        "label": label,
        "is_recording": is_recording,
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

        <body style="
            background:#111;
            color:white;
            font-family:sans-serif;
            padding:20px;
        ">

            <h1>Work AI Monitoring System</h1>

            <h2 id="recordingStatus" style="color:#ff5555;">
                REC: OFF
            </h2>

            <h2 id="currentLabel" style="color:#00ccff;">
                LABEL: unknown
            </h2>

            <div style="margin-bottom:20px;">

                <button id="btn_unknown"
                    onclick="setLabel('unknown')"
                    style="padding:12px 20px;font-size:18px;">
                    unknown
                </button>

                <button id="btn_normal"
                    onclick="setLabel('normal')"
                    style="padding:12px 20px;font-size:18px;">
                    normal
                </button>

                <button id="btn_squat"
                    onclick="setLabel('squat')"
                    style="padding:12px 20px;font-size:18px;">
                    squat
                </button>

                <button id="btn_bending"
                    onclick="setLabel('bending')"
                    style="padding:12px 20px;font-size:18px;">
                    bending
                </button>

                <button id="btn_arms_up"
                    onclick="setLabel('arms_up')"
                    style="padding:12px 20px;font-size:18px;">
                    arms_up
                </button>

                <br><br>

                <button onclick="startRecording()"
                    style="
                        background:#00aa44;
                        color:white;
                        padding:14px 24px;
                        font-size:20px;
                    ">
                    Start Recording
                </button>

                <button onclick="stopRecording()"
                    style="
                        background:#aa0000;
                        color:white;
                        padding:14px 24px;
                        font-size:20px;
                    ">
                    Stop Recording
                </button>

            </div>

            <img src="/video_feed" width="1280">

            <script>

                function resetButtons() {

                    const buttons = [
                        "unknown",
                        "normal",
                        "squat",
                        "bending",
                        "arms_up"
                    ];

                    buttons.forEach(name => {

                        const btn = document.getElementById("btn_" + name);

                        btn.style.background = "#eeeeee";
                        btn.style.color = "black";
                    });
                }

                async function setLabel(label) {

                    await fetch(`/set_label/${label}`);

                    document.getElementById("currentLabel").innerText =
                        "LABEL: " + label;

                    resetButtons();

                    const activeBtn =
                        document.getElementById("btn_" + label);

                    activeBtn.style.background = "#00ccff";
                    activeBtn.style.color = "white";

                    console.log("label changed:", label);
                }

                async function startRecording() {

                    await fetch('/start_recording');

                    document.getElementById("recordingStatus").innerText =
                        "REC: ON";

                    document.getElementById("recordingStatus").style.color =
                        "#00ff66";
                }

                async function stopRecording() {

                    await fetch('/stop_recording');

                    document.getElementById("recordingStatus").innerText =
                        "REC: OFF";

                    document.getElementById("recordingStatus").style.color =
                        "#ff5555";
                }

            </script>

        </body>
    </html>
    """

@app.get("/set_label/{label}")
def set_label(label: str):
    global current_label
    current_label = label
    return {"current_label": current_label}

@app.get("/start_recording")
def start_recording():
    global is_recording
    is_recording = True
    return {"is_recording": is_recording}


@app.get("/stop_recording")
def stop_recording():
    global is_recording
    is_recording = False
    return {"is_recording": is_recording}


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

            if is_recording and frame_count % 10 == 0:
                feature_logger.save(
                    features,
                    posture=posture,
                    label=current_label,
                )

            frame = create_status_panel(
                frame,
                features,
                posture,
                detections=detections,
                label=current_label,
                is_recording=is_recording,
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