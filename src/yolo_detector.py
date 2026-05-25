from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.5, device="cuda"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def predict(self, frame, draw=False):
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )

        annotated_frame = results[0].plot() if draw else frame

        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = results[0].names[cls_id]

            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": conf,
            })

        return annotated_frame, detections
