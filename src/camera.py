import cv2
import time


class Camera:
    """
    OpenCV camera wrapper.

    Important:
    - This class itself does NOT start a thread.
    - main.py runs camera_loop() in a dedicated thread.
    - read() returns one fresh frame from the device.
    """

    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self):
        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")

        # Reduce latency. Not every camera honors all settings, but they are safe.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        time.sleep(0.2)

        # Discard old frames after opening.
        for _ in range(5):
            self.cap.grab()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera is not opened")

        ret, frame = self.cap.read()

        if ret and frame is not None:
            return frame

        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
