# Work AI Monitoring System
# 作業AI監視システム

---

## Overview / 概要

AI-based work monitoring system for factories.

工場向けのAI作業監視システムです。

This system uses AI to:

このシステムはAIを使用して以下を実現します。

- Detect tools and objects using YOLO  
  YOLOによる工具・物体検出

- Track hands and poses using MediaPipe  
  MediaPipeによる手・姿勢追跡

- Classify work processes using LSTM  
  LSTMによる作業分類

- Detect abnormal work behavior  
  異常作業検知

- Generate Gantt chart visualizations  
  ガントチャート生成

- Output ON/OFF signals to PLC systems  
  PLCへのON/OFF出力

---

## Architecture / システム構成

### English

```text
Camera
↓
YOLO Object Detection
↓
MediaPipe Hand/Pose Tracking
↓
LSTM Work Classification
↓
Anomaly Detection
↓
Gantt Chart Visualization
↓
PLC ON/OFF Output
```

### 日本語

```text
カメラ
↓
YOLO物体検出
↓
MediaPipe 手・姿勢追跡
↓
LSTM 作業分類
↓
異常検知
↓
ガントチャート表示
↓
PLC ON/OFF出力
```

---

## Environment / 開発環境

| Item | Environment |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Container | Docker / Docker Compose |
| Language | Python 3.10 |
| AI Framework | PyTorch |
| Object Detection | YOLO11 |
| Pose Tracking | MediaPipe |
| Image Processing | OpenCV |
| Database | PostgreSQL / SQLite |
| Dashboard | Streamlit / FastAPI |

---

## Development Policy / 開発方針

- Offline-first architecture  
  オフライン優先構成

- Edge AI deployment  
  エッジAI構成

- Factory deployment ready  
  工場導入前提

- Docker-based environment  
  Dockerベース環境

- GitHub source management  
  GitHubによるソース管理

---

## Future Features / 今後の機能

- Multi-camera support  
  複数カメラ対応

- Real-time anomaly alerts  
  リアルタイム異常通知

- Historical work analysis  
  作業履歴分析

- PLC integration  
  PLC連携

- Factory dashboard  
  工場ダッシュボード

- AI model update system  
  AIモデル更新機能