# import os
# import cv2
# import time
# import math
# import struct
# import wave
# import logging
# from datetime import datetime
# from collections import defaultdict, deque

# # ── Alert Sound (pygame) ────────────────────────────────────────────
# try:
#     os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
#     import pygame
#     pygame.mixer.pre_init(44100, -16, 1, 512)
#     pygame.mixer.init()

#     alert_path = os.path.join('data', 'alert.wav')
#     if not os.path.exists(alert_path):
#         # Two-tone police-siren pattern: alternating 880 Hz and 660 Hz
#         sample_rate = 44100
#         tones  = [880, 660, 880, 660]   # Hz for each 0.25 s burst
#         frames = []
#         for freq in tones:
#             for i in range(int(sample_rate * 0.25)):
#                 val = int(28000 * math.sin(2 * math.pi * freq * i / sample_rate))
#                 frames.append(struct.pack('<h', val))
#         os.makedirs('data', exist_ok=True)
#         with wave.open(alert_path, 'w') as wf:
#             wf.setnchannels(1)
#             wf.setsampwidth(2)
#             wf.setframerate(sample_rate)
#             wf.writeframes(b''.join(frames))

#     alert_sound = pygame.mixer.Sound(alert_path)
#     alert_sound.set_volume(1.0)          # max volume
# except Exception as _e:
#     alert_sound = None

# from config import (
#     SOURCE_TYPE, WEBCAM_INDEX, RTSP_URL, VIDEO_PATH, SNAPSHOT_DIR,
#     FACE_RECOGNITION_EVERY_N_FRAMES, CAMERA_NAME
# )
# from app.camera import CameraStream
# from app.detector import PersonDetector
# from app.face_recognizer import FaceRecognizer
# from app.behavior import BehaviorAnalyzer
# from app.database import log_event
# from app.alerts import dispatch_alert
# from app import xai

# logger = logging.getLogger(__name__)

# # ── Per-track ID colours (up to 20 unique persons) ──────────────────
# _TRACK_COLOURS = [
#     (0, 255,   0),   # green  — #0  (known person default)
#     (255, 180,  0),  # cyan-ish
#     (255,   0, 200), # magenta
#     (0, 200, 255),   # amber
#     (180, 255,   0), # lime
#     (255, 120, 120), # light blue
#     (0, 255, 200),   # aqua
#     (200,   0, 255), # purple
#     (128, 255, 128), # light green
#     (255, 200,   0), # sky blue
# ]
# _UNKNOWN_COLOR = (0, 60, 255)   # solid red for unknown
# _UNKNOWN_BG    = (0, 40, 200)


# def _track_colour(tid: int) -> tuple:
#     return _TRACK_COLOURS[tid % len(_TRACK_COLOURS)]


# def build_source():
#     """Return (source, is_file) based on config."""
#     if SOURCE_TYPE == 'webcam':
#         return WEBCAM_INDEX, False
#     if SOURCE_TYPE == 'video':
#         return VIDEO_PATH, True
#     if SOURCE_TYPE == 'rtsp':
#         return RTSP_URL, False
#     return None, False


# # ── Per-track recognition vote buffer ────────────────────────────────
# # For each track_id we keep a rolling window of recent recognition
# # results. We pick the majority-vote winner so that a single bad frame
# # does not flip the label.
# _VOTE_WINDOW = 6      # how many recent recognition frames to consider


# class _TrackVoter:
#     """Majority-vote name across the last _VOTE_WINDOW recognition frames."""
#     def __init__(self):
#         self.votes: deque = deque(maxlen=_VOTE_WINDOW)
#         self.conf_acc: deque = deque(maxlen=_VOTE_WINDOW)
#         self.last_seen_frame: int = 0

#     def push(self, name: str, conf: float, frame_idx: int):
#         self.votes.append(name)
#         self.conf_acc.append(conf)
#         self.last_seen_frame = frame_idx

#     def result(self) -> tuple[str, float]:
#         """Return (majority_name, avg_conf_for_that_name)."""
#         if not self.votes:
#             return 'Unknown', 0.0

#         # Count votes per name
#         counts: dict[str, int]   = defaultdict(int)
#         confs:  dict[str, list]  = defaultdict(list)
#         for name, conf in zip(self.votes, self.conf_acc):
#             counts[name] += 1
#             confs[name].append(conf)

#         # Winner = most votes; if tie, prefer non-Unknown
#         best_name = max(
#             counts,
#             key=lambda n: (counts[n], 0 if n == 'Unknown' else 1)
#         )
#         avg_conf = sum(confs[best_name]) / len(confs[best_name])
#         return best_name, avg_conf


# class SurveillancePipeline:
#     def __init__(self, source=None, is_file=None, camera_name=CAMERA_NAME):
#         if source is None:
#             source, is_file = build_source()
#         self.source      = source
#         self.is_file     = is_file if is_file is not None else False
#         self.camera_name = camera_name

#         if self.source is not None and str(self.source) != "":
#             self.camera = CameraStream(source, is_file=self.is_file, loop_file=True).start()
#         else:
#             self.camera = None
#             logger.info("No initial video source configured. Waiting for user to select one.")
            
#         self.detector   = PersonDetector()
#         self.recognizer = FaceRecognizer()
#         self.behavior   = BehaviorAnalyzer()
#         self.frame_count = 0
#         self.running     = False

#         # Per-track voters — stale entries (not seen for N frames) are pruned
#         self._voters: dict[int, _TrackVoter] = {}
#         self._STALE_FRAMES = 30   # prune tracks not seen for 30 frames

#         os.makedirs(SNAPSHOT_DIR, exist_ok=True)

#     def switch_source(self, source, is_file=False):
#         """Hot-swap the video source (used when user uploads a demo video)."""
#         logger.info(f'Switching source → {source} (is_file={is_file})')
#         if self.camera:
#             try:
#                 self.camera.stop()
#             except Exception:
#                 pass
#         self.source      = source
#         self.is_file     = is_file
#         self.behavior    = BehaviorAnalyzer()
#         self._voters     = {}
#         self.frame_count = 0
#         self.camera = CameraStream(source, is_file=is_file, loop_file=True).start()

#     def _save_snapshot(self, frame, event_type):
#         ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#         path = os.path.join(SNAPSHOT_DIR, f'{event_type}_{ts}.jpg')
#         cv2.imwrite(path, frame)
#         return path

#     # ── Drawing ─────────────────────────────────────────────────────
#     def _draw_box(self, frame, x1, y1, x2, y2, name, tid, conf):
#         """Draw a highly visible bounding box with filled label."""
#         is_known = name != 'Unknown'

#         if is_known:
#             color = _track_colour(tid)
#             bg    = tuple(max(0, c - 60) for c in color)
#         else:
#             color = _UNKNOWN_COLOR
#             bg    = _UNKNOWN_BG

#         # Thick border
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

#         # Corner L-brackets for extra pop
#         L = 22; T = 6
#         for (px, py, dx, dy) in [
#             (x1, y1,  L,  0), (x1, y1,  0,  L),
#             (x2, y1, -L,  0), (x2, y1,  0,  L),
#             (x1, y2,  L,  0), (x1, y2,  0, -L),
#             (x2, y2, -L,  0), (x2, y2,  0, -L),
#         ]:
#             cv2.line(frame, (px, py), (px + dx, py + dy), color, T)

#         # Label
#         label = f'{name}  #{tid}'
#         if is_known and conf > 0:
#             label += f'  {conf*100:.0f}%'
#         elif not is_known:
#             label += '  ⚠ UNKNOWN'

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         fs   = 0.62
#         ft   = 2
#         (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
#         lx = max(0, x1)
#         ly = y1 - 12 if y1 - 12 > th + 6 else y1 + th + 12

#         cv2.rectangle(frame,
#                       (lx - 2,      ly - th - bl - 4),
#                       (lx + tw + 4, ly + bl),
#                       bg, -1)
#         cv2.putText(frame, label, (lx, ly - bl),
#                     font, fs, (255, 255, 255), ft, cv2.LINE_AA)

#     # ── Core processing ─────────────────────────────────────────────
#     def process_frame(self, frame):
#         detections   = self.detector.track(frame)
#         self.frame_count += 1
#         active_tids  = set()

#         # Run recognition every 3rd frame per detection to balance speed vs accuracy
#         do_face = (self.frame_count % 3 == 0)

#         for det in detections:
#             tid  = int(det['track_id'])
#             box  = det['box']
#             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#             active_tids.add(tid)

#             # Ensure voter exists for this track
#             if tid not in self._voters:
#                 self._voters[tid] = _TrackVoter()

#             voter = self._voters[tid]

#             if do_face:
#                 name, face_conf = self.recognizer.recognize(frame, box)
#                 voter.push(name, face_conf, self.frame_count)

#             # Majority-vote result for stable label
#             name, confidence = voter.result()
#             voter.last_seen_frame = self.frame_count

#             # Draw box
#             self._draw_box(frame, x1, y1, x2, y2, name, tid, confidence)

#             h_frame, w_frame = frame.shape[:2]
#             events = self.behavior.analyze(
#                 tid, box, name,
#                 frame_size=(w_frame, h_frame),
#                 det_confidence=det['confidence'],
#             )
#             for ev in events:
#                 if ev['type'] == 'UNKNOWN_PERSON' and alert_sound:
#                     if not pygame.mixer.get_busy():
#                         alert_sound.play()

#                 snapshot = self._save_snapshot(frame, ev['type'])
#                 ev['metadata']['xai'] = xai.structured_explanation(ev)

#                 event_id = log_event(
#                     person=ev['person'],
#                     event=ev['type'],
#                     severity=ev['severity'],
#                     confidence=ev['confidence'],
#                     explanation=ev['explanation'],
#                     snapshot_path=snapshot,
#                     camera=self.camera_name,
#                     metadata=ev['metadata'],
#                 )
#                 logger.info(f"[EVENT #{event_id}] {ev['type']} — {ev['explanation']}")
#                 try:
#                     dispatch_alert(event_id, ev['person'], ev['type'],
#                                    ev['explanation'], snapshot)
#                 except Exception as e:
#                     logger.error(f'Alert dispatch failed: {e}')

#         # Prune stale voter entries (tracks that disappeared)
#         stale = [
#             tid for tid, v in self._voters.items()
#             if self.frame_count - v.last_seen_frame > self._STALE_FRAMES
#             and tid not in active_tids
#         ]
#         for tid in stale:
#             self._voters.pop(tid, None)
#             self.behavior.forget_track(tid)

#         return frame

#     def run(self, show_window=True):
#         self.running = True
#         logger.info(f'Pipeline started (source={self.source}, is_file={self.is_file})')
#         self.latest_frame = None
#         while self.running:
#             if self.camera is None:
#                 time.sleep(1)
#                 continue
            
#             ok, frame = self.camera.read()
#             if not ok:
#                 time.sleep(0.05)
#                 continue
#             try:
#                 frame = self.process_frame(frame)
#                 self.latest_frame = frame.copy()
#             except Exception as e:
#                 logger.exception(f'Frame processing error: {e}')

#             if show_window:
#                 cv2.imshow('Surveillance', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.running = False
#         if show_window:
#             cv2.destroyAllWindows()

#     def stop(self):
#         self.running = False
#         self.camera.stop()









import os
import cv2
import time
import math
import struct
import wave
import logging
from datetime import datetime
from collections import defaultdict, deque

# ── Alert Sound (pygame) ────────────────────────────────────────────
try:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
    import pygame
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.mixer.init()

    alert_path = os.path.join('data', 'alert.wav')
    if not os.path.exists(alert_path):
        # Two-tone police-siren pattern: alternating 880 Hz and 660 Hz
        sample_rate = 44100
        tones  = [880, 660, 880, 660]   # Hz for each 0.25 s burst
        frames = []
        for freq in tones:
            for i in range(int(sample_rate * 0.25)):
                val = int(28000 * math.sin(2 * math.pi * freq * i / sample_rate))
                frames.append(struct.pack('<h', val))
        os.makedirs('data', exist_ok=True)
        with wave.open(alert_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

    alert_sound = pygame.mixer.Sound(alert_path)
    alert_sound.set_volume(1.0)

    # Dedicated channel so critical alarms are never skipped
    _alert_channel = pygame.mixer.Channel(0)
except Exception as _e:
    alert_sound = None
    _alert_channel = None

from config import (
    SOURCE_TYPE, WEBCAM_INDEX, RTSP_URL, VIDEO_PATH, SNAPSHOT_DIR,
    FACE_RECOGNITION_EVERY_N_FRAMES, CAMERA_NAME
)
from app.camera import CameraStream
from app.detector import PersonDetector
from app.face_recognizer import FaceRecognizer
from app.behavior import BehaviorAnalyzer
from app.database import log_event
from app.alerts import dispatch_alert
from app import xai

logger = logging.getLogger(__name__)

# ── Which events trigger the audible alarm ──────────────────────────
ALARM_EVENTS = {"INTRUSION", "FALL", "UNKNOWN_PERSON", "RUNNING"}

# ── Per-track ID colours (up to 20 unique persons) ──────────────────
_TRACK_COLOURS = [
    (0, 255,   0),   # green  — #0  (known person default)
    (255, 180,  0),  # cyan-ish
    (255,   0, 200), # magenta
    (0, 200, 255),   # amber
    (180, 255,   0), # lime
    (255, 120, 120), # light blue
    (0, 255, 200),   # aqua
    (200,   0, 255), # purple
    (128, 255, 128), # light green
    (255, 200,   0), # sky blue
]
_UNKNOWN_COLOR = (0, 60, 255)   # solid red for unknown
_UNKNOWN_BG    = (0, 40, 200)


def _track_colour(tid: int) -> tuple:
    return _TRACK_COLOURS[tid % len(_TRACK_COLOURS)]


def build_source():
    """Return (source, is_file) based on config."""
    if SOURCE_TYPE == 'webcam':
        return WEBCAM_INDEX, False
    if SOURCE_TYPE == 'video':
        return VIDEO_PATH, True
    if SOURCE_TYPE == 'rtsp':
        return RTSP_URL, False
    return None, False


# ── Per-track recognition vote buffer ────────────────────────────────
_VOTE_WINDOW = 6


class _TrackVoter:
    """Majority-vote name across the last _VOTE_WINDOW recognition frames."""
    def __init__(self):
        self.votes: deque = deque(maxlen=_VOTE_WINDOW)
        self.conf_acc: deque = deque(maxlen=_VOTE_WINDOW)
        self.last_seen_frame: int = 0

    def push(self, name: str, conf: float, frame_idx: int):
        self.votes.append(name)
        self.conf_acc.append(conf)
        self.last_seen_frame = frame_idx

    def result(self) -> tuple[str, float]:
        """Return (majority_name, avg_conf_for_that_name)."""
        if not self.votes:
            return 'Unknown', 0.0

        counts: dict[str, int]   = defaultdict(int)
        confs:  dict[str, list]  = defaultdict(list)
        for name, conf in zip(self.votes, self.conf_acc):
            counts[name] += 1
            confs[name].append(conf)

        best_name = max(
            counts,
            key=lambda n: (counts[n], 0 if n == 'Unknown' else 1)
        )
        avg_conf = sum(confs[best_name]) / len(confs[best_name])
        return best_name, avg_conf


class SurveillancePipeline:
    def __init__(self, source=None, is_file=None, camera_name=CAMERA_NAME):
        if source is None:
            source, is_file = build_source()
        self.source      = source
        self.is_file     = is_file if is_file is not None else False
        self.camera_name = camera_name

        if self.source is not None and str(self.source) != "":
            self.camera = CameraStream(source, is_file=self.is_file,
                                       loop_file=True).start()
        else:
            self.camera = None
            logger.info("No initial video source configured. "
                        "Waiting for user to select one.")

        self.detector   = PersonDetector()
        self.recognizer = FaceRecognizer()
        self.behavior   = BehaviorAnalyzer()
        self.frame_count = 0
        self.running     = False
        self.latest_frame  = None      # exposed via get_latest_frame()
        self._latest_jpeg  = None      # cached JPEG for streaming

        self._voters: dict[int, _TrackVoter] = {}
        self._STALE_FRAMES = 30

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # ── Source hot-swap ─────────────────────────────────────────────
    def switch_source(self, source, is_file=False):
        """Hot-swap the video source (used when user uploads a demo video)."""
        logger.info(f'Switching source → {source} (is_file={is_file})')
        if self.camera:
            try:
                self.camera.stop()
            except Exception:
                pass
        self.source      = source
        self.is_file     = is_file
        self.behavior    = BehaviorAnalyzer()
        self._voters     = {}
        self.frame_count = 0
        self.latest_frame = None
        self._latest_jpeg = None
        self.camera = CameraStream(source, is_file=is_file,
                                   loop_file=True).start()

    # ── Frame access for the web dashboard ──────────────────────────
    def get_latest_frame(self):
        """Return a copy of the most recent processed frame (BGR ndarray)."""
        return None if self.latest_frame is None else self.latest_frame.copy()

    def get_latest_jpeg(self, quality: int = 80):
        """Encode the latest frame as JPEG bytes (for MJPEG streaming)."""
        if self.latest_frame is None:
            return None
        ok, buf = cv2.imencode(
            '.jpg', self.latest_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        return buf.tobytes() if ok else None

    # ── Snapshot helper ─────────────────────────────────────────────
    def _save_snapshot(self, frame, event_type):
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(SNAPSHOT_DIR, f'{event_type}_{ts}.jpg')
        cv2.imwrite(path, frame)
        return path

    # ── Alarm ───────────────────────────────────────────────────────
    def _play_alarm(self, event_type: str):
        """Play alert sound for high-priority events.

        Uses a dedicated pygame channel so a critical FALL alert is
        never skipped just because an UNKNOWN_PERSON alert is still
        ringing. If the channel is busy, we restart it — important
        events should always be heard.
        """
        if alert_sound is None or _alert_channel is None:
            return
        if event_type not in ALARM_EVENTS:
            return
        try:
            _alert_channel.play(alert_sound, loops=0)
        except Exception as e:
            logger.debug(f'Alarm play failed: {e}')

    # ── Drawing ─────────────────────────────────────────────────────
    def _draw_box(self, frame, x1, y1, x2, y2, name, tid, conf):
        """Draw a highly visible bounding box with filled label."""
        is_known = name != 'Unknown'

        if is_known:
            color = _track_colour(tid)
            bg    = tuple(max(0, c - 60) for c in color)
        else:
            color = _UNKNOWN_COLOR
            bg    = _UNKNOWN_BG

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        L = 22; T = 6
        for (px, py, dx, dy) in [
            (x1, y1,  L,  0), (x1, y1,  0,  L),
            (x2, y1, -L,  0), (x2, y1,  0,  L),
            (x1, y2,  L,  0), (x1, y2,  0, -L),
            (x2, y2, -L,  0), (x2, y2,  0, -L),
        ]:
            cv2.line(frame, (px, py), (px + dx, py + dy), color, T)

        label = f'{name}  #{tid}'
        if is_known and conf > 0:
            label += f'  {conf*100:.0f}%'
        elif not is_known:
            label += '  ⚠ UNKNOWN'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = 0.62
        ft   = 2
        (tw, th), bl = cv2.getTextSize(label, font, fs, ft)
        lx = max(0, x1)
        ly = y1 - 12 if y1 - 12 > th + 6 else y1 + th + 12

        cv2.rectangle(frame,
                      (lx - 2,      ly - th - bl - 4),
                      (lx + tw + 4, ly + bl),
                      bg, -1)
        cv2.putText(frame, label, (lx, ly - bl),
                    font, fs, (255, 255, 255), ft, cv2.LINE_AA)

    # ── Core processing ─────────────────────────────────────────────
    def process_frame(self, frame):
        detections   = self.detector.track(frame)
        self.frame_count += 1
        active_tids  = set()
        h_frame, w_frame = frame.shape[:2]

        # Recognition every N frames (from config)
        do_face = (self.frame_count % FACE_RECOGNITION_EVERY_N_FRAMES == 0) \
                  or (self.frame_count % 3 == 0)

        # Dedup snapshots by event type within this frame
        snapshots_this_frame: dict[str, str] = {}

        for det in detections:
            tid  = int(det['track_id'])
            box  = det['box']
            conf = det['confidence']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            active_tids.add(tid)

            if tid not in self._voters:
                self._voters[tid] = _TrackVoter()
            voter = self._voters[tid]

            if do_face:
                name, face_conf = self.recognizer.recognize(frame, box)
                voter.push(name, face_conf, self.frame_count)

            # Majority-vote result for stable label
            name, confidence = voter.result()

            self._draw_box(frame, x1, y1, x2, y2, name, tid, confidence)

            # ── Behavior analysis (keyword args — matches new signature) ──
            events = self.behavior.analyze(
                track_id=tid,
                box=box,
                person_name=name,
                frame_size=(w_frame, h_frame),
                det_confidence=conf,
            )

            for ev in events:
                etype = ev['type']

                # Audible alarm for high-priority events
                self._play_alarm(etype)

                # Save one snapshot per event type per frame
                if etype not in snapshots_this_frame:
                    snapshots_this_frame[etype] = self._save_snapshot(frame, etype)
                snapshot = snapshots_this_frame[etype]

                # Preserve behavior.py's XAI if present; otherwise generate
                if 'xai' not in ev.get('metadata', {}):
                    ev['metadata']['xai'] = xai.structured_explanation(ev)

                event_id = log_event(
                    person=ev['person'],
                    event=etype,
                    severity=ev['severity'],
                    confidence=ev['confidence'],
                    explanation=ev['explanation'],
                    snapshot_path=snapshot,
                    camera=self.camera_name,
                    metadata=ev['metadata'],
                )
                logger.info(f"[EVENT #{event_id}] {etype} — {ev['explanation']}")
                try:
                    dispatch_alert(event_id, ev['person'], etype,
                                   ev['explanation'], snapshot)
                except Exception as e:
                    logger.error(f'Alert dispatch failed: {e}')

        # Prune stale voter entries
        stale = [
            tid for tid, v in self._voters.items()
            if self.frame_count - v.last_seen_frame > self._STALE_FRAMES
            and tid not in active_tids
        ]
        for tid in stale:
            self._voters.pop(tid, None)
            self.behavior.forget_track(tid)

        return frame

    def run(self, show_window=True):
        self.running = True
        logger.info(f'Pipeline started (source={self.source}, '
                    f'is_file={self.is_file})')
        self.latest_frame = None

        while self.running:
            if self.camera is None:
                time.sleep(1)
                continue

            ok, frame = self.camera.read()
            if not ok:
                # Video file finished → wait for a new source
                if self.is_file and not self.camera.running:
                    logger.info('Video file finished. Waiting for new source.')
                    self.camera = None
                    continue
                time.sleep(0.05)
                continue

            try:
                frame = self.process_frame(frame)
                self.latest_frame = frame.copy()
            except Exception as e:
                logger.exception(f'Frame processing error: {e}')

            if show_window:
                cv2.imshow('Surveillance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

        if show_window:
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception as e:
                logger.debug(f'Camera stop error: {e}')