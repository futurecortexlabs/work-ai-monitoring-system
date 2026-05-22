import cv2


class Camera:
    def __init__(self, source="/dev/video0", width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(
                self.source,
                cv2.CAP_V4L2
            )

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera is not opened")

        for _ in range(10):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame

        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()