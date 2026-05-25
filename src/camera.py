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

        self.cap = cv2.VideoCapture(
            self.source,
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():

            print("Retry camera open without V4L2")

            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera source: {self.source}"
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            self.fps
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

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
