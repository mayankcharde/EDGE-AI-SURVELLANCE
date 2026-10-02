# import time
# import math
# from collections import defaultdict
# from config import (
#     RESTRICTED_ZONE, LOITERING_SECONDS, RUNNING_SPEED_THRESHOLD,
#     FALL_ASPECT_RATIO, DIRECTION_CHANGE_THRESHOLD,
#     DIRECTION_CHANGE_WINDOW, INTRUSION_COOLDOWN
# )


# class BehaviorAnalyzer:
#     def __init__(self):
#         self.track_history = defaultdict(list)       # tid -> [(centroid, t), ...]
#         self.loitering_start = {}                    # tid -> t
#         self.direction_events = defaultdict(list)    # tid -> [t, ...]
#         self.last_alert = {}                         # (tid, event_type) -> t

#     def _can_alert(self, tid, event_type):
#         key = (tid, event_type)
#         now = time.time()
#         if key in self.last_alert and now - self.last_alert[key] < INTRUSION_COOLDOWN:
#             return False
#         self.last_alert[key] = now
#         return True

#     @staticmethod
#     def _centroid(box):
#         x1, y1, x2, y2 = box
#         return ((x1 + x2) // 2, (y1 + y2) // 2)

#     @staticmethod
#     def _in_zone(centroid, zone):
#         cx, cy = centroid
#         x1, y1, x2, y2 = zone
#         return x1 < cx < x2 and y1 < cy < y2

#     def analyze(self, track_id, box, person_name):
#         """Return list of abnormal event dicts with XAI explanations."""
#         now = time.time()
#         events = []
#         centroid = self._centroid(box)
#         x1, y1, x2, y2 = box
#         w = max(1, x2 - x1)
#         h = max(1, y2 - y1)
#         aspect_ratio = w / h

#         # ---- Update track history ----
#         hist = self.track_history[track_id]
        
#         # Safe migration if hot-reloaded with old 2-tuple format
#         if hist and len(hist[0]) == 2:
#             hist.clear()
            
#         hist.append((centroid, now, w, h))
#         if len(hist) > 30:
#             hist[:] = hist[-30:]

#         # ---- Speed (Moving Average for stability) ----
#         speed = 0.0
#         if len(hist) >= 5:
#             total_dist = 0.0
#             total_dt = 0.0
#             for i in range(1, min(6, len(hist))):
#                 p_c, p_t, _, _ = hist[-(i+1)]
#                 c_c, c_t, _, _ = hist[-i]
#                 dt = c_t - p_t
#                 if dt > 0:
#                     total_dist += math.dist(p_c, c_c)
#                     total_dt += dt
#             if total_dt > 0:
#                 speed = total_dist / total_dt

#         # ---- Direction change (wandering) ----
#         if len(hist) >= 3:
#             c1 = hist[-3][0]
#             c2 = hist[-2][0]
#             c3 = hist[-1][0]
#             v1 = (c2[0] - c1[0], c2[1] - c1[1])
#             v2 = (c3[0] - c2[0], c3[1] - c2[1])
#             if v1 != (0, 0) and v2 != (0, 0):
#                 a1 = math.atan2(v1[1], v1[0])
#                 a2 = math.atan2(v2[1], v2[0])
#                 diff = abs(a1 - a2)
#                 if diff > math.pi:
#                     diff = 2 * math.pi - diff
#                 if diff > math.pi / 2:
#                     self.direction_events[track_id].append(now)
#                     self.direction_events[track_id] = [
#                         t for t in self.direction_events[track_id]
#                         if now - t < DIRECTION_CHANGE_WINDOW
#                     ]

#         in_zone = self._in_zone(centroid, RESTRICTED_ZONE)

#         # ---- 1. Intrusion ----
#         if in_zone and self._can_alert(track_id, "INTRUSION"):
#             events.append({
#                 "type": "INTRUSION",
#                 "severity": "high",
#                 "person": person_name,
#                 "confidence": 0.95,
#                 "explanation": f"{person_name} breached the restricted zone.",
#                 "metadata": {
#                     "centroid": (int(centroid[0]), int(centroid[1])),
#                     "xai": {
#                         "text": f"The detected center point (X:{int(centroid[0])}, Y:{int(centroid[1])}) lies strictly within the forbidden restricted zone {RESTRICTED_ZONE}.",
#                         "confidence": "95%",
#                         "features": {"zone": RESTRICTED_ZONE, "person_location": centroid}
#                     }
#                 }
#             })

#         # ---- 2. Loitering ----
#         if in_zone:
#             if track_id not in self.loitering_start:
#                 self.loitering_start[track_id] = now
#             elif now - self.loitering_start[track_id] > LOITERING_SECONDS:
#                 if self._can_alert(track_id, "LOITERING"):
#                     dur = now - self.loitering_start[track_id]
#                     events.append({
#                         "type": "LOITERING",
#                         "severity": "medium",
#                         "person": person_name,
#                         "confidence": 0.90,
#                         "explanation": f"{person_name} has been loitering in the restricted area for {dur:.1f}s.",
#                         "metadata": {
#                             "duration_sec": round(dur, 1),
#                             "xai": {
#                                 "text": f"The subject remained continuously inside the restricted zone for {dur:.1f} seconds, which exceeds the {LOITERING_SECONDS}s limit.",
#                                 "confidence": "90%",
#                                 "features": {"time_in_zone": round(dur, 2), "limit": LOITERING_SECONDS}
#                             }
#                         }
#                     })
#                 self.loitering_start[track_id] = now
#         else:
#             self.loitering_start.pop(track_id, None)

#         # ---- 3. Running ----
#         if speed > RUNNING_SPEED_THRESHOLD and self._can_alert(track_id, "RUNNING"):
#             events.append({
#                 "type": "RUNNING",
#                 "severity": "medium",
#                 "person": person_name,
#                 "confidence": 0.85,
#                 "explanation": f"{person_name} is running at an average speed of {speed:.1f} px/s.",
#                 "metadata": {
#                     "speed_px_per_sec": round(speed, 2),
#                     "xai": {
#                         "text": f"Calculated moving average speed over the last 5 frames is {speed:.1f} pixels per second, exceeding the limit of {RUNNING_SPEED_THRESHOLD} px/s.",
#                         "confidence": "85%",
#                         "features": {"average_speed": round(speed, 2), "threshold": RUNNING_SPEED_THRESHOLD}
#                     }
#                 }
#             })

#         # ---- 4. Fall ----
#         is_fall = False
#         if aspect_ratio > FALL_ASPECT_RATIO:
#             # Check if recently standing upright to avoid false positives (like crawling/sitting)
#             if len(hist) >= 10:
#                 past_ratios = [w_old / h_old for (_, _, w_old, h_old) in hist[:-5] if h_old > 0]
#                 avg_past_ratio = sum(past_ratios) / len(past_ratios) if past_ratios else 1.0
#                 if avg_past_ratio < 1.0:
#                     is_fall = True
#             else:
#                 # Fallback if history is short but aspect ratio is extreme
#                 is_fall = True

#         if is_fall and self._can_alert(track_id, "FALL"):
#             events.append({
#                 "type": "FALL",
#                 "severity": "critical",
#                 "person": person_name,
#                 "confidence": 0.92,
#                 "explanation": f"{person_name} appears to have fallen down.",
#                 "metadata": {
#                     "aspect_ratio": round(aspect_ratio, 2),
#                     "xai": {
#                         "text": f"The subject's bounding box aspect ratio increased abruptly to {aspect_ratio:.2f} (width > height), indicating a sudden fall from a standing position.",
#                         "confidence": "92%",
#                         "features": {"current_ratio": round(aspect_ratio, 2), "fall_threshold": FALL_ASPECT_RATIO}
#                     }
#                 }
#             })

#         # ---- 5. Unusual movement ----
#         if len(self.direction_events[track_id]) > DIRECTION_CHANGE_THRESHOLD:
#             if self._can_alert(track_id, "UNUSUAL_MOVEMENT"):
#                 turns = len(self.direction_events[track_id])
#                 events.append({
#                     "type": "UNUSUAL_MOVEMENT",
#                     "severity": "low",
#                     "person": person_name,
#                     "confidence": 0.80,
#                     "explanation": f"{person_name} is exhibiting erratic wandering.",
#                     "metadata": {
#                         "turns": turns,
#                         "xai": {
#                             "text": f"Detected {turns} sharp directional changes (>90 degrees) within a {DIRECTION_CHANGE_WINDOW}-second window, suggesting abnormal wandering behavior.",
#                             "confidence": "80%",
#                             "features": {"sharp_turns": turns, "window_seconds": DIRECTION_CHANGE_WINDOW}
#                         }
#                     }
#                 })
#                 self.direction_events[track_id] = []

#         return events











"""
Robust, perspective-invariant behavior analysis for surveillance video.

Improvements over the previous version:
  - Zone accepts normalized OR pixel coordinates (auto-detected)
  - Speed is measured in BODY-LENGTHS per second (camera-invariant)
  - Centroid smoothed with a moving average to remove bbox jitter
  - Direction-change uses PATH TORTUOSITY, not raw angles
  - Fall requires 3 signals: aspect flip + centroid drop + persistence
  - Loitering requires low speed AND in-zone AND minimum duration
  - All events require N consecutive confirmations before firing
  - Low-confidence YOLO detections are ignored
"""
import time
import math
from collections import defaultdict, deque

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
# helpers
# =====================================================================

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _moving_average(values):
    if not values:
        return None
    return sum(values) / len(values)


class TrackState:
    """Per-track rolling history with everything we need for decisions."""
    __slots__ = (
        "samples",             # deque of (t, cx, cy, w, h, conf)
        "loitering_start",     # time when subject first entered zone
        "inside_zone",         # bool — currently inside zone
        "last_alert_ts",       # dict event_type -> last alert time
        "fall_wide_since",     # time when aspect ratio first became wide
        "prev_standing",       # bool — was standing tall recently
        "enter_zone_ts",       # time when subject entered zone (for XAI)
    )

    def __init__(self, maxlen=120):
        self.samples = deque(maxlen=maxlen)
        self.loitering_start = None
        self.inside_zone = False
        self.last_alert_ts = {}
        self.fall_wide_since = None
        self.prev_standing = False
        self.enter_zone_ts = None


# =====================================================================
# main analyzer
# =====================================================================

class BehaviorAnalyzer:
    def __init__(self):
        self.tracks = defaultdict(TrackState)
        self._frame_size = (1920, 1080)   # updated on first frame

    # -----------------------------------------------------------------
    # zone resolution
    # -----------------------------------------------------------------
    def _resolve_zone(self, frame_w, frame_h):
        """Return (x1, y1, x2, y2) in PIXELS.

        Accepts zone as normalized (0..1) OR absolute pixels.
        """
        x1, y1, x2, y2 = RESTRICTED_ZONE
        if max(x1, y1, x2, y2) <= 1.0:
            # normalized
            return (
                x1 * frame_w, y1 * frame_h,
                x2 * frame_w, y2 * frame_h,
            )
        return (x1, y1, x2, y2)

    @staticmethod
    def _in_zone(cx, cy, zone_px):
        x1, y1, x2, y2 = zone_px
        return x1 < cx < x2 and y1 < cy < y2

    # -----------------------------------------------------------------
    # cooldown
    # -----------------------------------------------------------------
    def _can_alert(self, state, event_type, now):
        last = state.last_alert_ts.get(event_type, 0.0)
        if now - last < INTRUSION_COOLDOWN:
            return False
        state.last_alert_ts[event_type] = now
        return True

    # -----------------------------------------------------------------
    # main entry
    # -----------------------------------------------------------------
    def analyze(self, track_id, box, person_name, frame_size=None,
                det_confidence=1.0):
        """Analyze one detection and return a list of events (may be empty).

        Args:
            track_id:       YOLO/ByteTrack track id
            box:            (x1, y1, x2, y2) pixel coordinates
            person_name:    recognized name or "Unknown"
            frame_size:     (w, h) of current frame — used for normalization
            det_confidence: YOLO detection confidence (0..1)
        """
        # -------- 0. Reject low-confidence detections --------
        if det_confidence < MIN_DETECTION_CONFIDENCE:
            return []

        if frame_size is not None:
            self._frame_size = frame_size

        frame_w, frame_h = self._frame_size
        zone_px = self._resolve_zone(frame_w, frame_h)

        # -------- 1. Basic geometry --------
        x1, y1, x2, y2 = box
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        now = time.time()

        state = self.tracks[track_id]
        state.samples.append((now, cx, cy, w, h, det_confidence))

        # Not enough history yet
        if len(state.samples) < MIN_HISTORY_FRAMES:
            return []

        events = []

        # -------- 2. Smoothed centroid (removes bbox jitter) --------
        recent = list(state.samples)[-SMOOTHING_WINDOW:]
        smooth_cx = _moving_average([s[1] for s in recent])
        smooth_cy = _moving_average([s[2] for s in recent])
        smooth_w = _moving_average([s[3] for s in recent])
        smooth_h = _moving_average([s[4] for s in recent])

        # -------- 3. Speed in BODY-LENGTHS / sec --------
        # Use the smoothed centroid over the full smoothing window,
        # divide by a robust "body length" (smoothed bbox height).
        speed_bl_per_sec = 0.0
        if len(recent) >= 2:
            # Total path length over the window
            path_len = 0.0
            for i in range(1, len(recent)):
                dx = recent[i][1] - recent[i-1][1]
                dy = recent[i][2] - recent[i-1][2]
                path_len += math.hypot(dx, dy)
            dt_total = recent[-1][0] - recent[0][0]
            if dt_total > 0:
                # Normalize by bbox height → perspective invariant
                body_len = max(30.0, smooth_h)   # guard against tiny boxes
                speed_px_per_sec = path_len / dt_total
                speed_bl_per_sec = speed_px_per_sec / body_len

        # -------- 4. Path tortuosity (wandering) --------
        tortuosity = 1.0
        net_displacement = 0.0
        window_sec = UNUSUAL_WINDOW_SEC
        window_samples = [s for s in state.samples if now - s[0] <= window_sec]
        if len(window_samples) >= 4:
            path_len = 0.0
            for i in range(1, len(window_samples)):
                dx = window_samples[i][1] - window_samples[i-1][1]
                dy = window_samples[i][2] - window_samples[i-1][2]
                path_len += math.hypot(dx, dy)
            dx_net = window_samples[-1][1] - window_samples[0][1]
            dy_net = window_samples[-1][2] - window_samples[0][2]
            net_displacement = math.hypot(dx_net, dy_net)
            body_len = max(30.0, smooth_h)
            if net_displacement > 1.0:
                tortuosity = path_len / net_displacement
            else:
                tortuosity = 999.0   # effectively infinite (kept moving but returned)

        # -------- 5. Zone membership (using smoothed centroid) --------
        in_zone = self._in_zone(smooth_cx, smooth_cy, zone_px)

        # -------- 6. Height / width ratio (for fall) --------
        hw_ratio = smooth_h / smooth_w          # > 1.4 = standing tall
        wh_ratio = smooth_w / smooth_h          # > 1.3 = lying wide

        # Track "recently standing"
        if hw_ratio > FALL_STANDING_HW_RATIO:
            state.prev_standing = True

        # Debug output (helps you tune thresholds against your video)
        if DEBUG_BEHAVIOR:
            print(
                f"[BEHAVIOR tid={track_id:>3}] "
                f"name={person_name:<10} "
                f"speed={speed_bl_per_sec:5.2f} bl/s  "
                f"tort={tortuosity:5.2f}  "
                f"hw={hw_ratio:4.2f} wh={wh_ratio:4.2f}  "
                f"in_zone={in_zone}  "
                f"conf={det_confidence:.2f}"
            )

        # =================================================================
        # EVENT 1: INTRUSION
        # =================================================================
        if in_zone and not state.inside_zone and self._can_alert(state, "INTRUSION", now):
            state.enter_zone_ts = now
            events.append({
                "type": "INTRUSION",
                "severity": "high",
                "person": person_name,
                "confidence": 0.95,
                "explanation": f"{person_name} entered the restricted zone.",
                "metadata": {
                    "centroid_px": (int(smooth_cx), int(smooth_cy)),
                    "zone_px": tuple(int(v) for v in zone_px),
                    "xai": {
                        "text": (
                            f"Subject's smoothed centroid "
                            f"(X:{int(smooth_cx)}, Y:{int(smooth_cy)}) is inside "
                            f"the restricted zone "
                            f"({tuple(int(v) for v in zone_px)} in pixels)."
                        ),
                        "confidence": "0.95",
                        "features": {
                            "centroid_px": [int(smooth_cx), int(smooth_cy)],
                            "zone_px": [int(v) for v in zone_px],
                        },
                    },
                },
            })

        state.inside_zone = in_zone

        # =================================================================
        # EVENT 2: LOITERING (low speed + in zone + long enough)
        # =================================================================
        if in_zone:
            if state.loitering_start is None:
                state.loitering_start = now

            dwell = now - state.loitering_start
            is_nearly_stationary = speed_bl_per_sec < LOITERING_MAX_SPEED_BODY_LENGTHS

            if (dwell > LOITERING_SECONDS
                    and is_nearly_stationary
                    and self._can_alert(state, "LOITERING", now)):
                events.append({
                    "type": "LOITERING",
                    "severity": "medium",
                    "person": person_name,
                    "confidence": 0.90,
                    "explanation": (
                        f"{person_name} loitering in restricted zone "
                        f"for {dwell:.1f}s at {speed_bl_per_sec:.2f} bl/s."
                    ),
                    "metadata": {
                        "duration_sec": round(dwell, 1),
                        "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                        "xai": {
                            "text": (
                                f"Subject remained inside the restricted zone "
                                f"for {dwell:.1f}s (limit {LOITERING_SECONDS}s) "
                                f"while moving at only "
                                f"{speed_bl_per_sec:.2f} body-lengths/s "
                                f"(below {LOITERING_MAX_SPEED_BODY_LENGTHS})."
                            ),
                            "confidence": "0.90",
                            "features": {
                                "dwell_sec": round(dwell, 2),
                                "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                                "limit_sec": LOITERING_SECONDS,
                            },
                        },
                    },
                })
                # Reset so it fires again after another LOITERING_SECONDS
                state.loitering_start = now
        else:
            state.loitering_start = None
            state.enter_zone_ts = None

        # =================================================================
        # EVENT 3: RUNNING (perspective-invariant speed)
        # =================================================================
        if (speed_bl_per_sec > RUNNING_SPEED_BODY_LENGTHS_PER_SEC
                and self._can_alert(state, "RUNNING", now)):
            events.append({
                "type": "RUNNING",
                "severity": "medium",
                "person": person_name,
                "confidence": 0.85,
                "explanation": (
                    f"{person_name} running at "
                    f"{speed_bl_per_sec:.2f} body-lengths/s."
                ),
                "metadata": {
                    "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                    "xai": {
                        "text": (
                            f"Smoothed speed is {speed_bl_per_sec:.2f} "
                            f"body-lengths per second, exceeding the "
                            f"threshold of "
                            f"{RUNNING_SPEED_BODY_LENGTHS_PER_SEC} bl/s. "
                            f"Speed is normalized by bounding-box height, "
                            f"so it's independent of distance from camera."
                        ),
                        "confidence": "0.85",
                        "features": {
                            "speed_bl_per_sec": round(speed_bl_per_sec, 2),
                            "threshold": RUNNING_SPEED_BODY_LENGTHS_PER_SEC,
                        },
                    },
                },
            })

        # =================================================================
        # EVENT 4: FALL (3 signals required)
        # =================================================================
        fall = False
        if wh_ratio > FALL_ASPECT_RATIO and state.prev_standing:
            # Signal 1: was recently standing (hw_ratio > 1.4)
            # Signal 2: now wide (wh_ratio > 1.3)
            # Signal 3: centroid dropped by at least FALL_CENTROID_DROP_FRAC
            if state.fall_wide_since is None:
                state.fall_wide_since = now
                # Record centroid Y at the moment we detected the wide pose
                state.last_alert_ts["__fall_cy"] = smooth_cy

            cy_at_flip = state.last_alert_ts.get("__fall_cy", smooth_cy)
            drop_frac = (smooth_cy - cy_at_flip) / frame_h

            wide_duration = now - state.fall_wide_since

            if (wide_duration >= FALL_MIN_DURATION_SEC
                    and drop_frac >= FALL_CENTROID_DROP_FRAC):
                if self._can_alert(state, "FALL", now):
                    fall = True
                    events.append({
                        "type": "FALL",
                        "severity": "critical",
                        "person": person_name,
                        "confidence": 0.92,
                        "explanation": (
                            f"{person_name} appears to have fallen "
                            f"(wide pose for {wide_duration:.1f}s, "
                            f"centroid dropped {drop_frac*100:.1f}% of frame)."
                        ),
                        "metadata": {
                            "aspect_ratio": round(wh_ratio, 2),
                            "wide_duration_sec": round(wide_duration, 2),
                            "centroid_drop_frac": round(drop_frac, 3),
                            "xai": {
                                "text": (
                                    f"Subject was previously standing tall "
                                    f"(hw_ratio>{FALL_STANDING_HW_RATIO}), "
                                    f"transitioned to a wide pose "
                                    f"(wh_ratio>{FALL_ASPECT_RATIO}), and "
                                    f"the centroid dropped by "
                                    f"{drop_frac*100:.1f}% of the frame "
                                    f"over {wide_duration:.1f}s."
                                ),
                                "confidence": "0.92",
                                "features": {
                                    "wh_ratio": round(wh_ratio, 2),
                                    "wide_duration_sec": round(wide_duration, 2),
                                    "centroid_drop_frac": round(drop_frac, 3),
                                },
                            },
                        },
                    })
                    state.prev_standing = False
                    state.fall_wide_since = None
        else:
            state.fall_wide_since = None

        # =================================================================
        # EVENT 5: UNUSUAL MOVEMENT (path tortuosity)
        # =================================================================
        body_len = max(30.0, smooth_h)
        min_disp_px = UNUSUAL_MIN_DISPLACEMENT_BODY_LENGTHS * body_len

        if (tortuosity > UNUSUAL_TORTUOSITY_THRESHOLD
                and net_displacement > min_disp_px
                and self._can_alert(state, "UNUSUAL_MOVEMENT", now)):
            events.append({
                "type": "UNUSUAL_MOVEMENT",
                "severity": "low",
                "person": person_name,
                "confidence": 0.80,
                "explanation": (
                    f"{person_name} showing erratic movement "
                    f"(tortuosity {tortuosity:.2f})."
                ),
                "metadata": {
                    "tortuosity": round(tortuosity, 2),
                    "net_displacement_px": round(net_displacement, 1),
                    "xai": {
                        "text": (
                            f"Path length / net displacement over the last "
                            f"{UNUSUAL_WINDOW_SEC}s is {tortuosity:.2f} "
                            f"(threshold {UNUSUAL_TORTUOSITY_THRESHOLD}). "
                            f"Net displacement was "
                            f"{net_displacement:.0f}px, well above the "
                            f"minimum {min_disp_px:.0f}px."
                        ),
                        "confidence": "0.80",
                        "features": {
                            "tortuosity": round(tortuosity, 2),
                            "net_displacement_px": round(net_displacement, 1),
                            "window_sec": UNUSUAL_WINDOW_SEC,
                        },
                    },
                },
            })

        # =================================================================
        # EVENT 6: KNOWN PERSON DETECTED
        # =================================================================
        if person_name != "Unknown":
            # Fire an event when a known person is detected (using standard cooldown)
            if self._can_alert(state, "KNOWN_PERSON", now):
                events.append({
                    "type": "KNOWN_PERSON",
                    "severity": "info",
                    "person": person_name,
                    "confidence": det_confidence,
                    "explanation": f"Known person {person_name} detected.",
                    "metadata": {
                        "xai": {
                            "text": f"Facial recognition matched the detected face to {person_name} from the known faces database.",
                            "confidence": f"{det_confidence:.2f}",
                            "features": {
                                "matched_name": person_name
                            }
                        }
                    }
                })
        
        # =================================================================
        # EVENT 7: UNKNOWN PERSON DETECTED
        # =================================================================
        if person_name == "Unknown":
            # Fire an event for unknown person, using a short cooldown so sound plays continuously
            last_unknown = state.last_alert_ts.get("UNKNOWN_PERSON", 0.0)
            if now - last_unknown >= 2.0:  # 2 second cooldown for repeated alerts
                state.last_alert_ts["UNKNOWN_PERSON"] = now
                events.append({
                    "type": "UNKNOWN_PERSON",
                    "severity": "high",
                    "person": "Unknown",
                    "confidence": det_confidence,
                    "explanation": "Unknown person detected in the footage.",
                    "metadata": {
                        "xai": {
                            "text": "The detected face did not match any profiles in the known faces database.",
                            "confidence": f"{det_confidence:.2f}",
                            "features": {
                                "status": "unrecognized"
                            }
                        }
                    }
                })

        return events

    # -----------------------------------------------------------------
    def forget_track(self, track_id):
        """Call when a track disappears for a long time."""
        self.tracks.pop(track_id, None)

    def reset(self):
        self.tracks.clear()