# """
# Robust, perspective-invariant behavior analysis for surveillance video.

# Improvements over the previous version:
#   - Zone accepts normalized OR pixel coordinates (auto-detected)
#   - Speed is measured in BODY-LENGTHS per second (camera-invariant)
#   - Centroid smoothed with a moving average to remove bbox jitter
#   - Direction-change uses PATH TORTUOSITY, not raw angles
#   - Fall requires 3 signals: aspect flip + centroid drop + persistence
#   - Loitering requires low speed AND in-zone AND minimum duration
#   - All events require N consecutive confirmations before firing
#   - Low-confidence YOLO detections are ignored
# """
# import time
# import math
# from collections import defaultdict, deque

# from config import (
#     RESTRICTED_ZONE,
#     RUNNING_SPEED_BODY_LENGTHS_PER_SEC,
#     LOITERING_SECONDS,
#     LOITERING_MAX_SPEED_BODY_LENGTHS,
#     FALL_ASPECT_RATIO,
#     FALL_STANDING_HW_RATIO,
#     FALL_CENTROID_DROP_FRAC,
#     FALL_MIN_DURATION_SEC,
#     UNUSUAL_TORTUOSITY_THRESHOLD,
#     UNUSUAL_MIN_DISPLACEMENT_BODY_LENGTHS,
#     UNUSUAL_WINDOW_SEC,
#     SMOOTHING_WINDOW,
#     MIN_HISTORY_FRAMES,
#     MIN_DETECTION_CONFIDENCE,
#     INTRUSION_COOLDOWN,
#     DEBUG_BEHAVIOR,
# )


# # =====================================================================
# # helpers
# # =====================================================================

# def _clamp(v, lo, hi):
#     return max(lo, min(hi, v))


# def _moving_average(values):
#     if not values:
#         return None
#     return sum(values) / len(values)


# class TrackState:
#     """Per-track rolling history with everything we need for decisions."""
#     __slots__ = (
#         "samples",             # deque of (t, cx, cy, w, h, conf)
#         "loitering_start",     # time when subject first entered zone
#         "inside_zone",         # bool — currently inside zone
#         "last_alert_ts",       # dict event_type -> last alert time
#         "fall_wide_since",     # time when aspect ratio first became wide
#         "prev_standing",       # bool — was standing tall recently
#         "enter_zone_ts",       # time when subject entered zone (for XAI)
#     )

#     def __init__(self, maxlen=120):
#         self.samples = deque(maxlen=maxlen)
#         self.loitering_start = None
#         self.inside_zone = False
#         self.last_alert_ts = {}
#         self.fall_wide_since = None
#         self.prev_standing = False
#         self.enter_zone_ts = None


# # =====================================================================
# # main analyzer
# # =====================================================================

# class BehaviorAnalyzer:
#     def __init__(self):
#         self.tracks = defaultdict(TrackState)
#         self._frame_size = (1920, 1080)   # updated on first frame

#     # -----------------------------------------------------------------
#     # zone resolution
#     # -----------------------------------------------------------------
#     def _resolve_zone(self, frame_w, frame_h):
#         """Return (x1, y1, x2, y2) in PIXELS.

#         Accepts zone as normalized (0..1) OR absolute pixels.
#         """
#         x1, y1, x2, y2 = RESTRICTED_ZONE
#         if max(x1, y1, x2, y2) <= 1.0:
#             # normalized
#             return (
#                 x1 * frame_w, y1 * frame_h,
#                 x2 * frame_w, y2 * frame_h,
#             )
#         return (x1, y1, x2, y2)

#     @staticmethod
#     def _in_zone(cx, cy, zone_px):
#         x1, y1, x2, y2 = zone_px
#         return x1 < cx < x2 and y1 < cy < y2

#     # -----------------------------------------------------------------
#     # cooldown
#     # -----------------------------------------------------------------
#     def _can_alert(self, state, event_type, now):
#         last = state.last_alert_ts.get(event_type, 0.0)
#         if now - last < INTRUSION_COOLDOWN:
#             return False
#         state.last_alert_ts[event_type] = now
#         return True

#     # -----------------------------------------------------------------
#     # main entry
#     # -----------------------------------------------------------------
#     def analyze(self, track_id, box, person_name, frame_size=None,
#                 det_confidence=1.0):
#         """Analyze one detection and return a list of events (may be empty).

#         Args:
#             track_id:       YOLO/ByteTrack track id
#             box:            (x1, y1, x2, y2) pixel coordinates
#             person_name:    recognized name or "Unknown"
#             frame_size:     (w, h) of current frame — used for normalization
#             det_confidence: YOLO detection confidence (0..1)
#         """
#         # -------- 0. Reject low-confidence detections --------
#         if det_confidence < MIN_DETECTION_CONFIDENCE:
#             return []

#         if frame_size is not None:
#             self._frame_size = frame_size

#         frame_w, frame_h = self._frame_size
#         zone_px = self._resolve_zone(frame_w, frame_h)

#         # -------- 1. Basic geometry --------
#         x1, y1, x2, y2 = box
#         w = max(1.0, float(x2 - x1))
#         h = max(1.0, float(y2 - y1))
#         cx = (x1 + x2) / 2.0
#         cy = (y1 + y2) / 2.0
#         now = time.time()

#         state = self.tracks[track_id]
#         state.samples.append((now, cx, cy, w, h, det_confidence))

#         # Not enough history yet
#         if len(state.samples) < MIN_HISTORY_FRAMES:
#             return []

#         events = []

#         # -------- 2. Smoothed centroid (removes bbox jitter) --------
#         recent = list(state.samples)[-SMOOTHING_WINDOW:]
#         smooth_cx = _moving_average([s[1] for s in recent])
#         smooth_cy = _moving_average([s[2] for s in recent])
#         smooth_w = _moving_average([s[3] for s in recent])
#         smooth_h = _moving_average([s[4] for s in recent])

#         # -------- 3. Speed in BODY-LENGTHS / sec --------
#         # Use the smoothed centroid over the full smoothing window,
#         # divide by a robust "body length" (smoothed bbox height).
#         speed_bl_per_sec = 0.0
#         if len(recent) >= 2:
#             # Total path length over the window
#             path_len = 0.0
#             for i in range(1, len(recent)):
#                 dx = recent[i][1] - recent[i-1][1]
#                 dy = recent[i][2] - recent[i-1][2]
#                 path_len += math.hypot(dx, dy)
#             dt_total = recent[-1][0] - recent[0][0]
#             if dt_total > 0:
#                 # Normalize by bbox height → perspective invariant
#                 body_len = max(30.0, smooth_h)   # guard against tiny boxes
#                 speed_px_per_sec = path_len / dt_total
#                 speed_bl_per_sec = speed_px_per_sec / body_len

#         # -------- 4. Path tortuosity (wandering) --------
#         tortuosity = 1.0
#         net_displacement = 0.0
#         window_sec = UNUSUAL_WINDOW_SEC
#         window_samples = [s for s in state.samples if now - s[0] <= window_sec]
#         if len(window_samples) >= 4:
#             path_len = 0.0
#             for i in range(1, len(window_samples)):
#                 dx = window_samples[i][1] - window_samples[i-1][1]
#                 dy = window_samples[i][2] - window_samples[i-1][2]
#                 path_len += math.hypot(dx, dy)
#             dx_net = window_samples[-1][1] - window_samples[0][1]
#             dy_net = window_samples[-1][2] - window_samples[0][2]
#             net_displacement = math.hypot(dx_net, dy_net)
#             body_len = max(30.0, smooth_h)
#             if net_displacement > 1.0:
#                 tortuosity = path_len / net_displacement
#             else:
#                 tortuosity = 999.0   # effectively infinite (kept moving but returned)

#         # -------- 5. Zone membership (using smoothed centroid) --------
#         in_zone = self._in_zone(smooth_cx, smooth_cy, zone_px)

#         # -------- 6. Height / width ratio (for fall) --------
#         hw_ratio = smooth_h / smooth_w          # > 1.4 = standing tall
#         wh_ratio = smooth_w / smooth_h          # > 1.3 = lying wide

#         # Track "recently standing"
#         if hw_ratio > FALL_STANDING_HW_RATIO:
#             state.prev_standing = True

#         # Debug output (helps you tune thresholds against your video)
#         if DEBUG_BEHAVIOR:
#             print(
#                 f"[BEHAVIOR tid={track_id:>3}] "
#                 f"name={person_name:<10} "
#                 f"speed={speed_bl_per_sec:5.2f} bl/s  "
#                 f"tort={tortuosity:5.2f}  "
#                 f"hw={hw_ratio:4.2f} wh={wh_ratio:4.2f}  "
#                 f"in_zone={in_zone}  "
#                 f"conf={det_confidence:.2f}"
#             )

#         # =================================================================
#         # EVENT 1: INTRUSION
#         # =================================================================
#         if in_zone and not state.inside_zone and self._can_alert(state, "INTRUSION", now):
#             state.enter_zone_ts = now
#             events.append({
#                 "type": "INTRUSION",
#                 "severity": "high",
#                 "person": person_name,
#                 "confidence": 0.95,
#                 "explanation": f"{person_name} entered the restricted zone.",
#                 "metadata": {
#                     "centroid_px": (int(smooth_cx), int(smooth_cy)),
#                     "zone_px": tuple(int(v) for v in zone_px),
#                     "xai": {
#                         "text": (
#                             f"Subject's smoothed centroid "
#                             f"(X:{int(smooth_cx)}, Y:{int(smooth_cy)}) is inside "
#                             f"the restricted zone "
#                             f"({tuple(int(v) for v in zone_px)} in pixels)."
#                         ),
#                         "confidence": "0.95",
#                         "features": {
#                             "centroid_px": [int(smooth_cx), int(smooth_cy)],
#                             "zone_px": [int(v) for v in zone_px],
#                         },
#                     },
#                 },
#             })

#         state.inside_zone = in_zone

#         # =================================================================
#         # EVENT 2: LOITERING (low speed + in zone + long enough)
#         # =================================================================
#         if in_zone:
#             if state.loitering_start is None:
#                 state.loitering_start = now

#             dwell = now - state.loitering_start
#             is_nearly_stationary = speed_bl_per_sec < LOITERING_MAX_SPEED_BODY_LENGTHS

#             if (dwell > LOITERING_SECONDS
#                     and is_nearly_stationary
#                     and self._can_alert(state, "LOITERING", now)):
#                 events.append({
#                     "type": "LOITERING",
#                     "severity": "medium",
#                     "person": person_name,
#                     "confidence": 0.90,
#                     "explanation": (
#                         f"{person_name} loitering in restricted zone "
#                         f"for {dwell:.1f}s at {speed_bl_per_sec:.2f} bl/s."
#                     ),
#                     "metadata": {
#                         "duration_sec": round(dwell, 1),
#                         "speed_bl_per_sec": round(speed_bl_per_sec, 2),
#                         "xai": {
#                             "text": (
#                                 f"Subject remained inside the restricted zone "
#                                 f"for {dwell:.1f}s (limit {LOITERING_SECONDS}s) "
#                                 f"while moving at only "
#                                 f"{speed_bl_per_sec:.2f} body-lengths/s "
#                                 f"(below {LOITERING_MAX_SPEED_BODY_LENGTHS})."
#                             ),
#                             "confidence": "0.90",
#                             "features": {
#                                 "dwell_sec": round(dwell, 2),
#                                 "speed_bl_per_sec": round(speed_bl_per_sec, 2),
#                                 "limit_sec": LOITERING_SECONDS,
#                             },
#                         },
#                     },
#                 })
#                 # Reset so it fires again after another LOITERING_SECONDS
#                 state.loitering_start = now
#         else:
#             state.loitering_start = None
#             state.enter_zone_ts = None

#         # =================================================================
#         # EVENT 3: RUNNING (perspective-invariant speed)
#         # =================================================================
#         if (speed_bl_per_sec > RUNNING_SPEED_BODY_LENGTHS_PER_SEC
#                 and self._can_alert(state, "RUNNING", now)):
#             events.append({
#                 "type": "RUNNING",
#                 "severity": "medium",
#                 "person": person_name,
#                 "confidence": 0.85,
#                 "explanation": (
#                     f"{person_name} running at "
#                     f"{speed_bl_per_sec:.2f} body-lengths/s."
#                 ),
#                 "metadata": {
#                     "speed_bl_per_sec": round(speed_bl_per_sec, 2),
#                     "xai": {
#                         "text": (
#                             f"Smoothed speed is {speed_bl_per_sec:.2f} "
#                             f"body-lengths per second, exceeding the "
#                             f"threshold of "
#                             f"{RUNNING_SPEED_BODY_LENGTHS_PER_SEC} bl/s. "
#                             f"Speed is normalized by bounding-box height, "
#                             f"so it's independent of distance from camera."
#                         ),
#                         "confidence": "0.85",
#                         "features": {
#                             "speed_bl_per_sec": round(speed_bl_per_sec, 2),
#                             "threshold": RUNNING_SPEED_BODY_LENGTHS_PER_SEC,
#                         },
#                     },
#                 },
#             })

#         # =================================================================
#         # EVENT 4: FALL (3 signals required)
#         # =================================================================
#         fall = False
#         if wh_ratio > FALL_ASPECT_RATIO and state.prev_standing:
#             # Signal 1: was recently standing (hw_ratio > 1.4)
#             # Signal 2: now wide (wh_ratio > 1.3)
#             # Signal 3: centroid dropped by at least FALL_CENTROID_DROP_FRAC
#             if state.fall_wide_since is None:
#                 state.fall_wide_since = now
#                 # Record centroid Y at the moment we detected the wide pose
#                 state.last_alert_ts["__fall_cy"] = smooth_cy

#             cy_at_flip = state.last_alert_ts.get("__fall_cy", smooth_cy)
#             drop_frac = (smooth_cy - cy_at_flip) / frame_h

#             wide_duration = now - state.fall_wide_since

#             if (wide_duration >= FALL_MIN_DURATION_SEC
#                     and drop_frac >= FALL_CENTROID_DROP_FRAC):
#                 if self._can_alert(state, "FALL", now):
#                     fall = True
#                     events.append({
#                         "type": "FALL",
#                         "severity": "critical",
#                         "person": person_name,
#                         "confidence": 0.92,
#                         "explanation": (
#                             f"{person_name} appears to have fallen "
#                             f"(wide pose for {wide_duration:.1f}s, "
#                             f"centroid dropped {drop_frac*100:.1f}% of frame)."
#                         ),
#                         "metadata": {
#                             "aspect_ratio": round(wh_ratio, 2),
#                             "wide_duration_sec": round(wide_duration, 2),
#                             "centroid_drop_frac": round(drop_frac, 3),
#                             "xai": {
#                                 "text": (
#                                     f"Subject was previously standing tall "
#                                     f"(hw_ratio>{FALL_STANDING_HW_RATIO}), "
#                                     f"transitioned to a wide pose "
#                                     f"(wh_ratio>{FALL_ASPECT_RATIO}), and "
#                                     f"the centroid dropped by "
#                                     f"{drop_frac*100:.1f}% of the frame "
#                                     f"over {wide_duration:.1f}s."
#                                 ),
#                                 "confidence": "0.92",
#                                 "features": {
#                                     "wh_ratio": round(wh_ratio, 2),
#                                     "wide_duration_sec": round(wide_duration, 2),
#                                     "centroid_drop_frac": round(drop_frac, 3),
#                                 },
#                             },
#                         },
#                     })
#                     state.prev_standing = False
#                     state.fall_wide_since = None
#         else:
#             state.fall_wide_since = None

#         # =================================================================
#         # EVENT 5: UNUSUAL MOVEMENT (path tortuosity)
#         # =================================================================
#         body_len = max(30.0, smooth_h)
#         min_disp_px = UNUSUAL_MIN_DISPLACEMENT_BODY_LENGTHS * body_len

#         if (tortuosity > UNUSUAL_TORTUOSITY_THRESHOLD
#                 and net_displacement > min_disp_px
#                 and self._can_alert(state, "UNUSUAL_MOVEMENT", now)):
#             events.append({
#                 "type": "UNUSUAL_MOVEMENT",
#                 "severity": "low",
#                 "person": person_name,
#                 "confidence": 0.80,
#                 "explanation": (
#                     f"{person_name} showing erratic movement "
#                     f"(tortuosity {tortuosity:.2f})."
#                 ),
#                 "metadata": {
#                     "tortuosity": round(tortuosity, 2),
#                     "net_displacement_px": round(net_displacement, 1),
#                     "xai": {
#                         "text": (
#                             f"Path length / net displacement over the last "
#                             f"{UNUSUAL_WINDOW_SEC}s is {tortuosity:.2f} "
#                             f"(threshold {UNUSUAL_TORTUOSITY_THRESHOLD}). "
#                             f"Net displacement was "
#                             f"{net_displacement:.0f}px, well above the "
#                             f"minimum {min_disp_px:.0f}px."
#                         ),
#                         "confidence": "0.80",
#                         "features": {
#                             "tortuosity": round(tortuosity, 2),
#                             "net_displacement_px": round(net_displacement, 1),
#                             "window_sec": UNUSUAL_WINDOW_SEC,
#                         },
#                     },
#                 },
#             })

#         # =================================================================
#         # EVENT 6: KNOWN PERSON DETECTED
#         # =================================================================
#         if person_name != "Unknown":
#             # Fire an event when a known person is detected (using standard cooldown)
#             if self._can_alert(state, "KNOWN_PERSON", now):
#                 events.append({
#                     "type": "KNOWN_PERSON",
#                     "severity": "info",
#                     "person": person_name,
#                     "confidence": det_confidence,
#                     "explanation": f"Known person {person_name} detected.",
#                     "metadata": {
#                         "xai": {
#                             "text": f"Facial recognition matched the detected face to {person_name} from the known faces database.",
#                             "confidence": f"{det_confidence:.2f}",
#                             "features": {
#                                 "matched_name": person_name
#                             }
#                         }
#                     }
#                 })
        
#         # =================================================================
#         # EVENT 7: UNKNOWN PERSON DETECTED
#         # =================================================================
#         if person_name == "Unknown":
#             # Fire an event for unknown person, using a short cooldown so sound plays continuously
#             last_unknown = state.last_alert_ts.get("UNKNOWN_PERSON", 0.0)
#             if now - last_unknown >= 2.0:  # 2 second cooldown for repeated alerts
#                 state.last_alert_ts["UNKNOWN_PERSON"] = now
#                 events.append({
#                     "type": "UNKNOWN_PERSON",
#                     "severity": "high",
#                     "person": "Unknown",
#                     "confidence": det_confidence,
#                     "explanation": "Unknown person detected in the footage.",
#                     "metadata": {
#                         "xai": {
#                             "text": "The detected face did not match any profiles in the known faces database.",
#                             "confidence": f"{det_confidence:.2f}",
#                             "features": {
#                                 "status": "unrecognized"
#                             }
#                         }
#                     }
#                 })

#         return events

#     # -----------------------------------------------------------------
#     def forget_track(self, track_id):
#         """Call when a track disappears for a long time."""
#         self.tracks.pop(track_id, None)

#     def reset(self):
#         self.tracks.clear()













"""
Production-grade behavior analysis for surveillance video.

Design principles
-----------------
1. Perspective invariance
   - Position anchor = bottom-center of the person box ("feet") — this is
     far more stable than the centroid in perspective CCTV views.
   - Speed is normalized by body length (bounding-box height), giving
     "body-lengths per second" — independent of distance from camera.

2. Temporal robustness
   - Median filtering for speed and body length → immune to bbox jitter.
   - Hysteresis (CONFIRM_FRAMES / RELEASE_FRAMES) for zone transitions.
   - Fall requires a multi-signal state machine
     (tall → wide → drop → persistence), eliminating crouch/sit false positives.
   - Tortuosity computed over a fixed time window (not frame count).

3. No event spam
   - Every event type has a per-track cooldown.
   - KNOWN_PERSON fires at most once per track.
   - UNKNOWN_PERSON fires once after a settle delay, then repeats every
     UNKNOWN_ALERT_INTERVAL seconds while the track is alive.

4. Memory safety
   - Tracks not seen for STALE_TRACK_SEC are garbage-collected.

5. Explainability
   - Every event carries a structured `xai` block with the numeric evidence
     that triggered it.
"""

import time
import math
from collections import deque
from statistics import median

from config import (
    RESTRICTED_ZONE,
    RUNNING_SPEED_BODY_LENGTHS_PER_SEC,
    LOITERING_SECONDS,
    LOITERING_MAX_SPEED_BODY_LENGTHS,
    FALL_ASPECT_RATIO,
    FALL_STANDING_HW_RATIO,
    FALL_CENTROID_DROP_FRAC,
    FALL_MIN_DURATION_SEC,
    UNUSUAL_TORTUOSITY_THRESHOLD,
    UNUSUAL_MIN_DISPLACEMENT_BODY_LENGTHS,
    UNUSUAL_WINDOW_SEC,
    SMOOTHING_WINDOW,
    MIN_HISTORY_FRAMES,
    MIN_DETECTION_CONFIDENCE,
    INTRUSION_COOLDOWN,
    DEBUG_BEHAVIOR,
)


# =====================================================================
# Module-local tunables
# =====================================================================
STALE_TRACK_SEC        = 15.0   # drop tracks unseen for this long
CONFIRM_FRAMES         = 4      # consecutive frames to enter a state
RELEASE_FRAMES         = 8      # consecutive frames to leave a state
IDENTITY_SETTLE_SEC    = 2.0    # wait before declaring "Unknown"
UNKNOWN_ALERT_INTERVAL = 15.0   # repeat "Unknown" alert every N sec
SPEED_WINDOW_SEC       = 1.5    # time window for speed estimation
STANDING_EVIDENCE_SEC  = 2.0    # how long a tall posture stays "fresh"
MIN_BODY_LEN_PX        = 30.0   # floor on body length for normalization


# =====================================================================
# Small helpers
# =====================================================================
def _median(values, default=0.0):
    vals = list(values)
    return median(vals) if vals else default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =====================================================================
# Per-track state
# =====================================================================
class TrackState:
    __slots__ = (
        "samples",              # deque of (t, cx, cy, feet_x, feet_y, w, h, conf)
        "first_seen",
        "last_seen",
        "zone_confirmed",       # bool — currently inside the zone
        "zone_streak_in",       # consecutive frames of raw in-zone
        "zone_streak_out",      # consecutive frames of raw out-of-zone
        "loitering_start",      # timestamp when subject first entered the zone
        "standing_evidence_ts", # last time a tall posture was observed
        "fall_wide_since",      # time when the wide pose began
        "fall_feet_y_at_flip",  # feet Y at the moment the wide pose started
        "fall_confirmed",       # True while a fall episode is active
        "identity_reported",    # None | "Unknown" | <name>  (one-shot per track)
        "last_alert_ts",        # dict event_type -> last alert timestamp
    )

    def __init__(self, t):
        self.samples              = deque(maxlen=600)   # ~10s @ 60fps
        self.first_seen           = t
        self.last_seen            = t
        self.zone_confirmed       = False
        self.zone_streak_in       = 0
        self.zone_streak_out      = 0
        self.loitering_start      = None
        self.standing_evidence_ts = 0.0
        self.fall_wide_since      = None
        self.fall_feet_y_at_flip  = None
        self.fall_confirmed       = False
        self.identity_reported    = None
        self.last_alert_ts        = {}


# =====================================================================
# Main analyzer
# =====================================================================
class BehaviorAnalyzer:
    def __init__(self):
        self.tracks = {}
        self._frame_size = (1920, 1080)
        self._last_cleanup = time.time()

    # -----------------------------------------------------------------
    # Zone helpers
    # -----------------------------------------------------------------
    def _resolve_zone(self, fw, fh):
        """Return (x1, y1, x2, y2) in pixels.

        Accepts zone as normalized (0..1) OR absolute pixels.
        """
        x1, y1, x2, y2 = RESTRICTED_ZONE
        if max(x1, y1, x2, y2) <= 1.0:
            return (x1 * fw, y1 * fh, x2 * fw, y2 * fh)
        return (float(x1), float(y1), float(x2), float(y2))

    @staticmethod
    def _in_zone(x, y, zone_px):
        x1, y1, x2, y2 = zone_px
        return x1 < x < x2 and y1 < y < y2

    # -----------------------------------------------------------------
    # Alert cooldown
    # -----------------------------------------------------------------
    def _can_alert(self, state, event_type, now, cooldown=None):
        cd = INTRUSION_COOLDOWN if cooldown is None else cooldown
        last = state.last_alert_ts.get(event_type, 0.0)
        if now - last < cd:
            return False
        state.last_alert_ts[event_type] = now
        return True

    # -----------------------------------------------------------------
    # Track cleanup
    # -----------------------------------------------------------------
    def _cleanup(self, now):
        if now - self._last_cleanup < 5.0:
            return
        self._last_cleanup = now
        stale = [tid for tid, st in self.tracks.items()
                 if now - st.last_seen > STALE_TRACK_SEC]
        for tid in stale:
            self.tracks.pop(tid, None)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    def analyze(self, track_id, box, person_name,
                frame_size=None, det_confidence=1.0):
        """
        Analyze one detection and return a list of event dicts.

        Args:
            track_id:       ByteTrack / YOLO track identifier.
            box:            (x1, y1, x2, y2) in pixels.
            person_name:    Recognized name, or "Unknown".
            frame_size:     (width, height) of the current frame.
            det_confidence: YOLO detection confidence (0..1).
        """
        # ---- 0. Reject low-confidence detections --------------------
        if det_confidence < MIN_DETECTION_CONFIDENCE:
            return []

        if frame_size is not None:
            self._frame_size = frame_size
        fw, fh = self._frame_size

        now = time.time()
        self._cleanup(now)

        # ---- 1. Basic geometry --------------------------------------
        x1, y1, x2, y2 = [float(v) for v in box]
        w  = max(1.0, x2 - x1)
        h  = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        feet_x = cx
        feet_y = y2                        # ground contact point

        # ---- 2. Fetch/create track state ----------------------------
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackState(now)
            self.tracks[track_id] = state

        state.last_seen = now
        state.samples.append(
            (now, cx, cy, feet_x, feet_y, w, h, det_confidence)
        )

        if len(state.samples) < MIN_HISTORY_FRAMES:
            return []

        # ---- 3. Robust metrics over recent window -------------------
        recent = [s for s in state.samples
                  if now - s[0] <= SPEED_WINDOW_SEC]
        if len(recent) < 2:
            recent = list(state.samples)[-SMOOTHING_WINDOW:]

        smooth_w = _median(s[5] for s in recent)
        smooth_h = _median(s[6] for s in recent)
        body_len = max(MIN_BODY_LEN_PX, smooth_h)

        sm_feet_x = _median(s[3] for s in recent)
        sm_feet_y = _median(s[4] for s in recent)

        # ---- 4. Speed in body-lengths / second ----------------------
        inst_speeds = []
        for i in range(1, len(recent)):
            dt = recent[i][0] - recent[i - 1][0]
            if dt <= 0:
                continue
            dx = recent[i][3] - recent[i - 1][3]
            dy = recent[i][4] - recent[i - 1][4]
            inst_speeds.append(math.hypot(dx, dy) / dt)
        speed_px_per_sec = _median(inst_speeds, 0.0)
        speed_bl_per_sec = speed_px_per_sec / body_len

        # ---- 5. Tortuosity over UNUSUAL_WINDOW_SEC ------------------
        tortuosity = 1.0
        net_disp = 0.0
        w_samples = [s for s in state.samples
                     if now - s[0] <= UNUSUAL_WINDOW_SEC]
        if len(w_samples) >= 4:
            path_len = 0.0
            for i in range(1, len(w_samples)):
                dx = w_samples[i][3] - w_samples[i - 1][3]
                dy = w_samples[i][4] - w_samples[i - 1][4]
                path_len += math.hypot(dx, dy)
            dx_net = w_samples[-1][3] - w_samples[0][3]
            dy_net = w_samples[-1][4] - w_samples[0][4]
            net_disp = math.hypot(dx_net, dy_net)
            if net_disp > body_len * 0.25:
                tortuosity = path_len / net_disp
            else:
                # Subject is essentially stationary → tortuosity is undefined.
                tortuosity = 1.0

        # ---- 6. Posture ---------------------------------------------
        hw_ratio = smooth_h / smooth_w      # tall = high
        wh_ratio = smooth_w / smooth_h      # wide = high
        is_tall  = hw_ratio > FALL_STANDING_HW_RATIO
        is_wide  = wh_ratio > FALL_ASPECT_RATIO

        # ---- 7. Zone membership with hysteresis ---------------------
        zone_px = self._resolve_zone(fw, fh)
        raw_in_zone = self._in_zone(sm_feet_x, sm_feet_y, zone_px)

        if raw_in_zone:
            state.zone_streak_in  += 1
            state.zone_streak_out  = 0
        else:
            state.zone_streak_out += 1
            state.zone_streak_in   = 0

        entered_zone = False
        exited_zone  = False

        if (not state.zone_confirmed
                and state.zone_streak_in >= CONFIRM_FRAMES):
            state.zone_confirmed = True
            entered_zone = True
        elif (state.zone_confirmed
                and state.zone_streak_out >= RELEASE_FRAMES):
            state.zone_confirmed = False
            exited_zone = True

        in_zone = state.zone_confirmed

        # ---- Debug --------------------------------------------------
        if DEBUG_BEHAVIOR:
            print(
                f"[BEHAVIOR tid={track_id:>3}] "
                f"name={person_name:<10} "
                f"spd={speed_bl_per_sec:5.2f}bl/s "
                f"tort={tortuosity:5.2f} "
                f"hw={hw_ratio:4.2f} wh={wh_ratio:4.2f} "
                f"feet=({sm_feet_x:6.0f},{sm_feet_y:6.0f}) "
                f"zone={in_zone} conf={det_confidence:.2f}"
            )

        # =================================================================
        # EVENT 1 — INTRUSION
        # =================================================================
        events = []

        if entered_zone and self._can_alert(state, "INTRUSION", now):
            events.append(self._mk_event(
                etype="INTRUSION",
                severity="high",
                person=person_name,
                confidence=0.95,
                explanation=f"Intrusion (Forbidden zone entry): {person_name} entered the restricted zone.",
                meta={
                    "feet_px": [int(sm_feet_x), int(sm_feet_y)],
                    "zone_px": [int(v) for v in zone_px],
                },
                xai_text=(
                    f"Feet position ({int(sm_feet_x)}, {int(sm_feet_y)}) "
                    f"crossed into the restricted zone "
                    f"{[int(v) for v in zone_px]} and stayed for "
                    f"{CONFIRM_FRAMES}+ consecutive frames."
                ),
                xai_conf="0.95",
                xai_features={
                    "feet_px": [int(sm_feet_x), int(sm_feet_y)],
                    "zone_px": [int(v) for v in zone_px],
                    "confirm_frames": CONFIRM_FRAMES,
                },
            ))

        # Reset loitering clock whenever the subject leaves the zone
        if exited_zone or (not in_zone and state.loitering_start is not None):
            state.loitering_start = None

        # =================================================================
        # EVENT 2 — LOITERING
        # =================================================================
        if in_zone:
            if state.loitering_start is None:
                state.loitering_start = now
            dwell = now - state.loitering_start
            slow  = speed_bl_per_sec < LOITERING_MAX_SPEED_BODY_LENGTHS

            if (dwell > LOITERING_SECONDS
                    and slow
                    and self._can_alert(state, "LOITERING", now)):
                events.append(self._mk_event(
                    etype="LOITERING",
                    severity="medium",
                    person=person_name,
                    confidence=0.90,
                    explanation=(
                        f"Loitering (Prolonged lingering): {person_name} loitering in restricted zone "
                        f"for {dwell:.1f}s at {speed_bl_per_sec:.2f} bl/s."
                    ),
                    meta={
                        "dwell_sec": round(dwell, 2),
                        "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                    },
                    xai_text=(
                        f"Subject remained inside the zone for {dwell:.1f}s "
                        f"(limit {LOITERING_SECONDS}s) while moving at only "
                        f"{speed_bl_per_sec:.2f} body-lengths/s "
                        f"(below {LOITERING_MAX_SPEED_BODY_LENGTHS})."
                    ),
                    xai_conf="0.90",
                    xai_features={
                        "dwell_sec": round(dwell, 2),
                        "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                        "limit_sec": LOITERING_SECONDS,
                    },
                ))
                state.loitering_start = now

        # =================================================================
        # EVENT 3 — RUNNING
        # =================================================================
        if (speed_bl_per_sec > RUNNING_SPEED_BODY_LENGTHS_PER_SEC
                and self._can_alert(state, "RUNNING", now)):
            events.append(self._mk_event(
                etype="RUNNING",
                severity="medium",
                person=person_name,
                confidence=0.85,
                explanation=(
                    f"Running (Abnormal speed): {person_name} running at "
                    f"{speed_bl_per_sec:.2f} body-lengths/s."
                ),
                meta={"speed_bl_per_sec": round(speed_bl_per_sec, 2)},
                xai_text=(
                    f"Median speed over the last {SPEED_WINDOW_SEC}s is "
                    f"{speed_bl_per_sec:.2f} body-lengths/second, exceeding "
                    f"the threshold {RUNNING_SPEED_BODY_LENGTHS_PER_SEC}. "
                    f"Speed is normalized by body length, so it is "
                    f"independent of distance from the camera."
                ),
                xai_conf="0.85",
                xai_features={
                    "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                    "threshold": RUNNING_SPEED_BODY_LENGTHS_PER_SEC,
                    "window_sec": SPEED_WINDOW_SEC,
                },
            ))

        # =================================================================
        # EVENT 4 — FALL   (multi-signal state machine)
        # =================================================================
        # Refresh "recently standing" evidence
        if is_tall:
            state.standing_evidence_ts = now

        standing_recently = (
            (now - state.standing_evidence_ts) < STANDING_EVIDENCE_SEC
        )

        if is_wide and standing_recently and not state.fall_confirmed:
            # Begin a fall episode
            if state.fall_wide_since is None:
                state.fall_wide_since     = now
                state.fall_feet_y_at_flip = sm_feet_y

            feet_y_at_flip = (state.fall_feet_y_at_flip
                              if state.fall_feet_y_at_flip is not None
                              else sm_feet_y)
            drop_frac = (sm_feet_y - feet_y_at_flip) / fh
            wide_dur  = now - state.fall_wide_since

            if (wide_dur >= FALL_MIN_DURATION_SEC
                    and drop_frac >= FALL_CENTROID_DROP_FRAC):
                if self._can_alert(state, "FALL", now):
                    events.append(self._mk_event(
                        etype="FALL",
                        severity="critical",
                        person=person_name,
                        confidence=0.92,
                        explanation=(
                            f"Fall (Sudden ground drop): {person_name} appears to have fallen "
                            f"(wide pose for {wide_dur:.1f}s, "
                            f"feet dropped {drop_frac*100:.1f}% of frame)."
                        ),
                        meta={
                            "aspect_ratio": round(wh_ratio, 2),
                            "wide_duration_sec": round(wide_dur, 2),
                            "drop_frac": round(drop_frac, 3),
                        },
                        xai_text=(
                            f"Subject was standing "
                            f"(hw_ratio>{FALL_STANDING_HW_RATIO}), "
                            f"transitioned to a wide pose "
                            f"(wh_ratio>{FALL_ASPECT_RATIO}), and the "
                            f"ground-contact point dropped by "
                            f"{drop_frac*100:.1f}% of frame height over "
                            f"{wide_dur:.1f}s."
                        ),
                        xai_conf="0.92",
                        xai_features={
                            "wh_ratio": round(wh_ratio, 2),
                            "wide_duration_sec": round(wide_dur, 2),
                            "drop_frac": round(drop_frac, 3),
                            "thresholds": {
                                "standing_hw": FALL_STANDING_HW_RATIO,
                                "wide_wh": FALL_ASPECT_RATIO,
                                "drop_frac": FALL_CENTROID_DROP_FRAC,
                                "min_duration": FALL_MIN_DURATION_SEC,
                            },
                        },
                    ))
                    state.fall_confirmed = True
        else:
            # Posture is no longer wide → reset the fall episode
            if not is_wide:
                state.fall_wide_since     = None
                state.fall_feet_y_at_flip = None
                state.fall_confirmed      = False

        # =================================================================
        # EVENT 5 — UNUSUAL MOVEMENT (tortuosity + minimum displacement)
        # =================================================================
        min_disp_px = UNUSUAL_MIN_DISPLACEMENT_BODY_LENGTHS * body_len
        if (tortuosity > UNUSUAL_TORTUOSITY_THRESHOLD
                and net_disp > min_disp_px
                and self._can_alert(state, "UNUSUAL_MOVEMENT", now)):
            events.append(self._mk_event(
                etype="UNUSUAL_MOVEMENT",
                severity="low",
                person=person_name,
                confidence=0.80,
                explanation=(
                    f"Unusual (Erratic wandering): {person_name} showing erratic movement "
                    f"(tortuosity {tortuosity:.2f})."
                ),
                meta={
                    "tortuosity": round(tortuosity, 2),
                    "net_disp_px": round(net_disp, 1),
                },
                xai_text=(
                    f"Path length / net displacement over the last "
                    f"{UNUSUAL_WINDOW_SEC}s is {tortuosity:.2f} "
                    f"(threshold {UNUSUAL_TORTUOSITY_THRESHOLD}). "
                    f"Net displacement was {net_disp:.0f}px, above the "
                    f"minimum {min_disp_px:.0f}px."
                ),
                xai_conf="0.80",
                xai_features={
                    "tortuosity": round(tortuosity, 2),
                    "net_disp_px": round(net_disp, 1),
                    "window_sec": UNUSUAL_WINDOW_SEC,
                    "threshold": UNUSUAL_TORTUOSITY_THRESHOLD,
                },
            ))

        # =================================================================
        # EVENT 6 — KNOWN PERSON    (one-shot per track)
        # =================================================================
        track_age = now - state.first_seen

        if person_name and person_name != "Unknown":
            if state.identity_reported != person_name:
                events.append(self._mk_event(
                    etype="KNOWN_PERSON",
                    severity="info",
                    person=person_name,
                    confidence=det_confidence,
                    explanation=f"Known (Recognized face): Known person recognized: {person_name}.",
                    meta={"track_id": int(track_id)},
                    xai_text=(
                        f"Face embedding matched the enrolled profile "
                        f"'{person_name}'. Track age at recognition: "
                        f"{track_age:.1f}s."
                    ),
                    xai_conf=f"{det_confidence:.2f}",
                    xai_features={
                        "matched_name": person_name,
                        "track_age_sec": round(track_age, 2),
                    },
                ))
                state.identity_reported = person_name

        # =================================================================
        # EVENT 7 — UNKNOWN PERSON
        # =================================================================
        else:
            # Only declare UNKNOWN after the settle delay so that face
            # recognition has had time to run; repeat with a longer cooldown.
            if (track_age >= IDENTITY_SETTLE_SEC
                    and self._can_alert(state, "UNKNOWN_PERSON", now,
                                        cooldown=UNKNOWN_ALERT_INTERVAL)):
                events.append(self._mk_event(
                    etype="UNKNOWN_PERSON",
                    severity="high",
                    person="Unknown",
                    confidence=det_confidence,
                    explanation="Unknown (Unrecognized face): Unknown person detected in the footage.",
                    meta={"track_id": int(track_id)},
                    xai_text=(
                        f"Face could not be matched to any enrolled profile "
                        f"after {track_age:.1f}s of tracking "
                        f"(settle window {IDENTITY_SETTLE_SEC}s)."
                    ),
                    xai_conf=f"{det_confidence:.2f}",
                    xai_features={
                        "track_age_sec": round(track_age, 2),
                        "settle_sec": IDENTITY_SETTLE_SEC,
                    },
                ))
                state.identity_reported = "Unknown"

        return events

    # -----------------------------------------------------------------
    # Event factory — keeps metadata shape consistent
    # -----------------------------------------------------------------
    @staticmethod
    def _mk_event(etype, severity, person, confidence,
                  explanation, meta, xai_text, xai_conf, xai_features):
        meta = dict(meta or {})
        meta["xai"] = {
            "text": xai_text,
            "confidence": xai_conf,
            "features": xai_features,
        }
        return {
            "type": etype,
            "severity": severity,
            "person": person,
            "confidence": confidence,
            "explanation": explanation,
            "metadata": meta,
        }

    # -----------------------------------------------------------------
    def forget_track(self, track_id):
        """Explicitly remove a track (e.g., when ByteTrack drops it)."""
        self.tracks.pop(track_id, None)

    def reset(self):
        self.tracks.clear()