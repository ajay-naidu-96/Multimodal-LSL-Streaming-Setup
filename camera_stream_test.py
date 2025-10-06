import cv2
import threading
import time

class CameraStream:
    def __init__(self, src, name="Camera"):
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            time.sleep(0.01)  # avoid hogging CPU

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Start two camera streams
cam0 = CameraStream(0, "Camera 0")
cam1 = CameraStream(1, "Camera 1")

time.sleep(1)  # Give threads time to start

try:
    while True:
        ret0, frame0 = cam0.read()
        ret1, frame1 = cam1.read()

        if ret0:
            cv2.imshow("Camera 0", frame0)
        if ret1:
            cv2.imshow("Camera 1", frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
