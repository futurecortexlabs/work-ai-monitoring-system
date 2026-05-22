# Work AI Monitoring System
# 作業AI監視システム

---

# Overview / 概要

Work AI Monitoring System is an offline-first edge AI platform for factory work monitoring and anomaly detection.

Work AI Monitoring System は、工場向けのオフライン対応エッジAI作業監視システムです。

This system uses computer vision and time-series AI models to:

本システムは画像認識および時系列AIを使用して以下を実現します。

- Detect tools and objects
- Track hand and body movements
- Classify work processes
- Detect abnormal work behavior
- Visualize work timelines
- Output anomaly signals to PLC systems

---

# Main Features / 主な機能

## 1. Object Detection / 物体検出

Using YOLO-based AI models:

YOLOベースのAIモデルを使用して：

- Tool detection
- Workpiece detection
- Part detection
- Human detection

を行います。

---

## 2. Hand & Pose Tracking / 手・姿勢追跡

Using MediaPipe:

MediaPipeを使用して：

- Hand tracking
- Finger tracking
- Body pose estimation
- Motion feature extraction

を行います。

---

## 3. Work Classification / 作業分類

Using LSTM time-series models:

LSTM時系列モデルを使用して：

- Work step classification
- Process transition analysis
- Work duration estimation

を行います。

Example:

例：

```text
Pick Part
↓
Set Part
↓
Tighten Screw
↓
Inspection
↓
Remove Part
```

---

## 4. Anomaly Detection / 異常検知

The system detects:

本システムは以下を検知します。

- Incorrect work sequence
- Missing operations
- Unusual work duration
- Unexpected motion patterns
- Abnormal work behavior

---

## 5. Gantt Chart Visualization / ガントチャート表示

The system generates:

本システムは以下を生成します。

- Work timelines
- Process history
- Worker activity logs
- Abnormality history

Example:

例：

```text
08:00 - Pick Part
08:05 - Set Part
08:10 - Tighten Screw
08:20 - Inspection
```

---

## 6. PLC Output / PLC出力

When anomalies are detected:

異常検知時：

- PLC ON/OFF signal output
- Warning lamp activation
- Buzzer activation
- Machine stop support

を行います。

Supported protocols:

対応予定プロトコル：

- Modbus TCP
- OPC UA
- EtherNet/IP
- MQTT

---

# System Architecture / システム構成

## AI Pipeline / AI処理構成

```text
Camera Input
↓
YOLO Object Detection
↓
MediaPipe Hand/Pose Tracking
↓
Feature Generation
↓
LSTM Work Classification
↓
Anomaly Detection
↓
Gantt Chart Visualization
↓
PLC Output
```

## 日本語構成

```text
カメラ入力
↓
YOLO物体検出
↓
MediaPipe 手・姿勢追跡
↓
特徴量生成
↓
LSTM 作業分類
↓
異常検知
↓
ガントチャート表示
↓
PLC出力
```

---

# Development Environment / 開発環境

| Item | Environment |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Container | Docker / Docker Compose |
| Language | Python 3.10 |
| AI Framework | PyTorch |
| Object Detection | YOLO |
| Pose Tracking | MediaPipe |
| Image Processing | OpenCV |
| Database | PostgreSQL / SQLite |
| Dashboard | Streamlit / FastAPI |

---

# Offline Architecture / オフライン構成

This system is designed for offline factory environments.

本システムはオフライン工場環境向けに設計されています。

Features:

特徴：

- No internet required
- Local AI inference
- Local database
- Local dashboard
- Edge AI deployment

---

# Project Structure / プロジェクト構成

```text
work-ai-monitoring-system/
├─ src/
├─ config/
├─ scripts/
├─ docs/
├─ models/
├─ logs/
├─ db/
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

---

# GitHub Policy / GitHub運用方針

## Managed in GitHub / GitHub管理対象

- Source code
- Docker configuration
- Documentation
- Scripts
- Configuration templates

## NOT Managed in GitHub / GitHub管理対象外

- AI trained models
- Videos
- Factory logs
- Databases
- Local configuration files

---

# Future Roadmap / 今後のロードマップ

## Phase 1

- Camera input
- YOLO integration
- MediaPipe integration

## Phase 2

- LSTM work classification
- Gantt chart visualization

## Phase 3

- Anomaly detection
- PLC integration

## Phase 4

- Multi-camera support
- Factory deployment
- Dashboard enhancement

---

# Target Use Cases / 想定用途

- Factory work monitoring
- Assembly line monitoring
- Manual process analysis
- Work anomaly detection
- Manufacturing DX
- Worker behavior analysis

---

# License / ライセンス

Private development project.
非公開開発プロジェクト。