import cv2
import yaml
import time
import threading
from pathlib import Path
import csv
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from shared_state import SharedState
from camera import Camera
from yolo_detector import YoloDetector
from pose_detector import PoseDetector
from posture_analyzer import PostureAnalyzer
from feature_logger import FeatureLogger
from gantt_checker import GanttChecker
from ng_logger import NGLogger


CONFIG_PATH = Path("config/sample.yaml")

app = FastAPI()
state = SharedState()

feature_logger = FeatureLogger()
posture_analyzer = PostureAnalyzer()
gantt_checker = GanttChecker()
ng_logger = NGLogger()

last_ng_reason = None
work_start_time = None
work_running = False


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

yolo_detector = YoloDetector(
    model_path=config["models"]["yolo_model_path"],
    confidence=config["detection"]["confidence_threshold"],
)

pose_detector = PoseDetector()


def get_posture_label(posture):
    if posture.get("is_squatting"):
        return "squatting"
    if posture.get("is_forward_bending"):
        return "bending"
    if posture.get("is_working_pose"):
        return "working_pose"
    if posture.get("is_standing"):
        return "standing"
    return "unknown"


def get_actual_action(posture):
    if posture.get("is_squatting"):
        return "squat"
    if posture.get("is_forward_bending"):
        return "bending"
    if posture.get("is_both_arms_up"):
        return "arms_up"
    if posture.get("is_working_pose"):
        return "use_tool"
    return "standby"


def get_detected_objects(detections):
    objects = []

    for d in detections or []:
        if isinstance(d, dict):
            name = d.get("class") or d.get("name") or d.get("label")
            if name:
                objects.append(name)

    return objects


def camera_loop():
    print("Camera thread started")

    try:
        camera.open()
    except Exception as e:
        print(f"Camera open error: {e}")
        return

    while state.running:
        frame = camera.read()

        if frame is not None:
            frame = cv2.resize(frame, (640, 480))
            state.set_frame(frame)

        time.sleep(0.001)

    camera.release()


def ai_loop():
    global last_ng_reason
    global work_start_time
    global work_running

    print("AI thread started")

    frame_count = 0
    last_detections = []

    while state.running:
        frame = state.get_frame()

        if frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1
        detections = last_detections

        if frame_count % 5 == 0:
            try:
                _, detections = yolo_detector.predict(frame)
                last_detections = detections or []
            except Exception as e:
                print(f"YOLO error: {e}")
                detections = last_detections

        try:
            _, features = pose_detector.detect(frame)

            if features is None:
                features = {
                    "has_hand": False,
                    "has_pose": False,
                    "hands": [],
                    "pose": {},
                }

        except Exception as e:
            print(f"Pose error: {e}")
            features = {
                "has_hand": False,
                "has_pose": False,
                "hands": [],
                "pose": {},
            }

        try:
            posture = posture_analyzer.analyze(features)
            if posture is None:
                posture = {}
        except Exception as e:
            print(f"Posture error: {e}")
            posture = {}

        posture_label = get_posture_label(posture)
        actual_action = get_actual_action(posture)
        detected_objects = get_detected_objects(detections)

        if not work_running or work_start_time is None:
            elapsed_sec = 0
            gantt_result = {
                "result": "WAIT",
                "reason": "work_not_started",
            }
            last_ng_reason = None
        else:
            elapsed_sec = time.time() - work_start_time

            try:
                gantt_result = gantt_checker.check(
                    elapsed_sec=elapsed_sec,
                    actual_action=actual_action,
                    detected_objects=detected_objects,
                    posture_label=posture_label,
                )
            except Exception as e:
                print(f"Gantt error: {e}")
                gantt_result = {
                    "result": "ERROR",
                    "reason": str(e),
                }

            if gantt_result.get("result") == "NG":
                reason = gantt_result.get("reason")

                if reason != last_ng_reason:
                    ng_logger.save(
                        elapsed_sec=elapsed_sec,
                        gantt_result=gantt_result,
                        actual_action=actual_action,
                        posture_label=posture_label,
                        detected_objects=detected_objects,
                    )
                    last_ng_reason = reason
            else:
                last_ng_reason = None

        state.update_ai(
            detections=detections,
            features=features,
            posture={
                **posture,
                "elapsed_sec": round(elapsed_sec, 2),
                "actual_action": actual_action,
                "posture_label": posture_label,
                "detected_objects": ",".join(detected_objects),
            },
            gantt_result=gantt_result,
        )

        snapshot = state.snapshot()

        if snapshot["is_recording"]:
            feature_logger.save(
                features,
                posture={
                    **posture,
                    "elapsed_sec": round(elapsed_sec, 2),
                    "actual_action": actual_action,
                    "posture_label": posture_label,
                    "detected_objects": ",".join(detected_objects),
                    "gantt_result": gantt_result.get("result"),
                    "gantt_reason": gantt_result.get("reason"),
                },
                label=snapshot["current_label"],
            )

        time.sleep(0.1)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=ai_loop, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Work AI Monitoring System</title>
            <style>
                body {
                    background: #101014;
                    color: white;
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }

                .layout {
                    display: flex;
                    gap: 24px;
                    align-items: flex-start;
                }

                .video-box {
                    background: #000;
                    padding: 10px;
                    border-radius: 12px;
                }

                .panel {
                    width: 430px;
                    background: #1c1c24;
                    padding: 18px;
                    border-radius: 12px;
                    box-shadow: 0 0 12px rgba(0,0,0,0.4);
                }

                .status-main {
                    font-size: 32px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }

                .status-sub {
                    font-size: 16px;
                    color: #ddd;
                    margin-bottom: 8px;
                }

                button {
                    padding: 12px 16px;
                    margin: 4px;
                    font-size: 15px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                }

                .btn-label {
                    background: #e8e8e8;
                    color: black;
                }

                .btn-work {
                    background: #0066cc;
                    color: white;
                }

                .btn-rec {
                    background: #00aa44;
                    color: white;
                }

                .btn-stop {
                    background: #aa2222;
                    color: white;
                }

                .btn-other {
                    background: #555;
                    color: white;
                }

                pre {
                    background: #111;
                    color: #ffcccc;
                    padding: 12px;
                    height: 220px;
                    overflow: auto;
                    border-radius: 8px;
                    font-size: 12px;
                }

                .ok {
                    color: #00ff66;
                }

                .ng {
                    color: #ff4444;
                }

                .wait {
                    color: #ffff66;
                }
            </style>
        </head>

        <body>
            <h1>Work AI Monitoring System</h1>

            <div class="layout">
                <div class="video-box">
                    <img src="/video_feed" width="640">
                </div>

                <div class="panel">
                    <div id="judgementStatus" class="status-main wait">
                        JUDGEMENT: WAIT
                    </div>

                    <div id="judgementReason" class="status-sub">
                        REASON: work_not_started
                    </div>

                    <div id="elapsedTime" class="status-sub">
                        TIME: 0 sec
                    </div>

                    <div id="actionStatus" class="status-sub">
                        ACTION: unknown
                    </div>

                    <div id="postureStatus" class="status-sub">
                        POSTURE: unknown
                    </div>

                    <hr>

                    <h3>Work Control</h3>
                    <button class="btn-work" onclick="startWork()">Start Work</button>
                    <button class="btn-stop" onclick="stopWork()">Stop Work</button>
                    <button class="btn-other" onclick="resetWork()">Reset Timer</button>

                    <h3>Recording</h3>
                    <div id="recordingStatus" class="status-sub">
                        REC: OFF
                    </div>
                    <button class="btn-rec" onclick="startRecording()">Start Recording</button>
                    <button class="btn-stop" onclick="stopRecording()">Stop Recording</button>

                    <h3>Manual Label</h3>
                    <div id="currentLabel" class="status-sub">
                        LABEL: unknown
                    </div>
                    <button class="btn-label" onclick="setLabel('unknown')">unknown</button>
                    <button class="btn-label" onclick="setLabel('normal')">normal</button>
                    <button class="btn-label" onclick="setLabel('squat')">squat</button>
                    <button class="btn-label" onclick="setLabel('bending')">bending</button>
                    <button class="btn-label" onclick="setLabel('arms_up')">arms_up</button>

                    <h3>NG History</h3>
                    <button class="btn-other" onclick="loadNgHistory()">Refresh NG History</button>
                    <pre id="ngHistory"></pre>

                    <hr>
                </div>
            </div>

            <script>
                async function setLabel(label) {
                    await fetch(`/set_label/${label}`);
                    document.getElementById("currentLabel").innerText = "LABEL: " + label;
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

                async function startWork() {
                    await fetch('/start_work');
                }

                async function stopWork() {
                    await fetch('/stop_work');
                }

                async function resetWork() {
                    await fetch('/reset_work');
                }

                async function loadNgHistory() {
                    const res = await fetch('/ng_history');
                    const data = await res.json();

                    document.getElementById("ngHistory").innerText =
                        JSON.stringify(data, null, 2);
                }

                async function updateStatus() {
                    const res = await fetch('/status');
                    const data = await res.json();

                    const judgement = data.gantt_result.result;
                    const reason = data.gantt_result.reason;
                    const posture = data.posture || {};

                    const judgementStatus = document.getElementById("judgementStatus");

                    judgementStatus.innerText = "JUDGEMENT: " + judgement;
                    judgementStatus.className = "status-main";

                    if (judgement === "OK") {
                        judgementStatus.classList.add("ok");
                    } else if (judgement === "NG") {
                        judgementStatus.classList.add("ng");
                    } else {
                        judgementStatus.classList.add("wait");
                    }

                    document.getElementById("judgementReason").innerText =
                        "REASON: " + reason;

                    document.getElementById("elapsedTime").innerText =
                        "TIME: " + (posture.elapsed_sec || 0) + " sec";

                    document.getElementById("actionStatus").innerText =
                        "ACTION: " + (posture.actual_action || "unknown");

                    document.getElementById("postureStatus").innerText =
                        "POSTURE: " + (posture.posture_label || "unknown");
                }

                setInterval(updateStatus, 500);
            </script>
        </body>
    </html>
    """


@app.get("/set_label/{label}")
def set_label(label: str):
    with state.lock:
        state.current_label = label
    return {"current_label": label}


@app.get("/start_recording")
def start_recording():
    with state.lock:
        state.is_recording = True
    return {"is_recording": True}


@app.get("/stop_recording")
def stop_recording():
    with state.lock:
        state.is_recording = False
    return {"is_recording": False}


@app.get("/start_work")
def start_work():
    global work_start_time
    global work_running
    global last_ng_reason

    work_start_time = time.time()
    work_running = True
    last_ng_reason = None

    return {
        "work_running": work_running,
        "work_start_time": work_start_time,
    }


@app.get("/stop_work")
def stop_work():
    global work_running
    work_running = False
    return {"work_running": work_running}


@app.get("/reset_work")
def reset_work():
    global work_start_time
    global last_ng_reason

    work_start_time = time.time()
    last_ng_reason = None

    return {"work_start_time": work_start_time}


@app.get("/status")
def status():
    snapshot = state.snapshot()
    snapshot["frame"] = "available" if snapshot["frame"] is not None else None
    snapshot["work_running"] = work_running
    return snapshot


@app.get("/ng_history")
def ng_history():
    path = "logs/ng_log.csv"

    if not os.path.exists(path):
        return []

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows[-20:]


def generate_frames():
    while True:
        snapshot = state.snapshot()
        frame = snapshot["frame"]

        if frame is None:
            time.sleep(0.03)
            continue

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        time.sleep(0.01)


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