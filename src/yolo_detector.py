from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def predict(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        annotated_frame = results[0].plot()
        return annotated_frame