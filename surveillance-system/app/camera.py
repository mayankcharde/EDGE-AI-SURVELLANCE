# import cv2
# import time
# import logging
# import threading

# logger = logging.getLogger(__name__)


# class CameraStream:
#     """Threaded RTSP reader that always returns the latest frame."""

#     def __init__(self, url):
#         self.url = url
#         self.cap = None
#         self.frame = None
#         self.lock = threading.Lock()
#         self.running = False
#         self.thread = None

#     def start(self):
#         self.running = True
#         self.thread = threading.Thread(target=self._update, daemon=True)
#         self.thread.start()
#         # Wait for first frame
#         for _ in range(50):
#             if self.frame is not None:
#                 break
#             time.sleep(0.1)
#         return self

#     def _open(self):
#         # If URL is numeric string, treat as webcam index
#         if self.url.isdigit():
#             cap = cv2.VideoCapture(int(self.url))
#         else:
#             cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
#         return cap

#     def _update(self):
#         while self.running:
#             if self.cap is None or not self.cap.isOpened():
#                 self.cap = self._open()
#                 if not self.cap.isOpened():
#                     logger.warning("Camera not opened, retrying in 3s...")
#                     time.sleep(3)
#                     continue
#             ok, frame = self.cap.read()
#             if not ok:
#                 logger.warning("Frame read failed, reconnecting...")
#                 self.cap.release()
#                 self.cap = None
#                 time.sleep(2)
#                 continue
#             with self.lock:
#                 self.frame = frame
#             time.sleep(0.001)

#     def read(self):
#         with self.lock:
#             if self.frame is None:
#                 return False, None
#             return True, self.frame.copy()

#     def stop(self):
#         self.running = False
#         if self.cap:
#             self.cap.release()




import cv2
import time
import os
import logging
import threading

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Threaded video source that supports:
      - webcam: source=int index
      - rtsp:   source="rtsp://..."
      - file:   source="path/to/video.mp4"

    For video files, it loops when it reaches the end.
    """

    def __init__(self, source, is_file=False, loop_file=True):
        self.source = source
        self.is_file = is_file
        self.loop_file = loop_file

        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0

    # ---------- Open ----------
    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source)
        elif str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source))
        elif self.is_file or os.path.exists(str(self.source)):
            cap = cv2.VideoCapture(str(self.source))
        else:
            cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        return cap

    # ---------- Start ----------
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        # Wait for first frame
        for _ in range(60):
            if self.frame is not None:
                break
            time.sleep(0.1)
        return self

    # ---------- Reader loop ----------
    def _update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open()
                if not self.cap.isOpened():
                    logger.warning(f"Cannot open source: {self.source}. Retry in 3s.")
                    time.sleep(3)
                    continue

                # Read metadata for file playback
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                logger.info(f"Source opened: {self.source} (fps={self.fps:.1f}, frames={self.total_frames})")

            ok, frame = self.cap.read()

            # End of file → loop or stop
            if not ok:
                if self.is_file:
                    if self.loop_file:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame = 0
                        continue
                    else:
                        logger.info("Video file finished.")
                        self.running = False
                        break
                else:
                    logger.warning("Frame read failed, reconnecting...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(2)
                    continue

            with self.lock:
                self.frame = frame
                self.current_frame += 1

            # Throttle file playback to match real FPS
            if self.is_file and self.fps > 0:
                time.sleep(1.0 / self.fps)
            else:
                time.sleep(0.001)

    # ---------- Read ----------
    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    # ---------- Progress (for file mode) ----------
    def get_progress(self):
        if not self.is_file or self.total_frames == 0:
            return None
        return {
            "current": self.current_frame,
            "total": self.total_frames,
            "percent": round(100 * self.current_frame / self.total_frames, 1),
        }

    # ---------- Stop ----------
    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass