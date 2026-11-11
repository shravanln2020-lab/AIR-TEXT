"""
camera.py
=========
Background camera thread for AIR TEXT.

Uses OpenCV for live video capture and MediaPipe Tasks HandLandmarker
for real-time hand skeleton landmark tracking and index fingertip tracing.
Decoupled from Tkinter using thread-safe queues.

Improvements:
-------------
1. Clearly defined Virtual Writing Area (ROI) with clean HUD guides.
2. Rotation-invariant finger gesture detection using knuckle-relative distances:
   - Index Extended + other fingers curled = WRITING (Pen Down)
   - Open hand / 2+ fingers raised = HOVER (Pen Up)
3. Smooth exponential moving average trajectory to eliminate hand tremor.
4. Preview generator for finished characters.
5. Immediate status notifications for hand detection / loss.
"""

import threading
import queue
import time
import os
import traceback
from typing import List, Tuple, Optional
import numpy as np
import cv2

# Import MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import HandLandmarksConnections
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

# ── Visual Styling Constants ────────────────────────────────────────────────
TRAIL_COLOR       = (255, 80,  20)   # BGR: vivid blue trail
FINGER_INNER      = (0,  220,  30)   # BGR: green fingertip dot
FINGER_OUTER      = (0,  255, 100)   # BGR: light green concentric ring
LM_COLOR          = (30,  30, 220)   # BGR: red landmarks
CONN_COLOR        = (30,  30, 220)   # BGR: red skeleton connections

TRAIL_THICKNESS   = 6
MIN_MOVE_PX       = 4                # Minimum movement threshold to record points

# Virtual Writing Zone coordinates on 640x480 frame
ZONE_X1, ZONE_Y1  = 60, 45
ZONE_X2, ZONE_Y2  = 580, 435


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class CameraThread(threading.Thread):
    """
    Background camera processing thread.
    Captures live webcam frames, performs hand tracking, renders landmarks
    and multi-stroke blue trails, and delivers frames to GUI queue.
    """

    def __init__(self, frame_queue: queue.Queue, status_queue: queue.Queue,
                 camera_index: int = 0):
        super().__init__(daemon=True)
        self.frame_queue   = frame_queue
        self.status_queue  = status_queue
        self.camera_index  = camera_index

        self._stop_event   = threading.Event()
        self._writing      = False
        self._pen_down     = False
        self._lock         = threading.Lock()

        # Stroke data
        self._strokes: List[List[Tuple[int, int]]] = []
        self._stroke_times: List[Tuple[float, float]] = []
        self._current_stroke_start_time: Optional[float] = None
        self._current: List[Tuple[int, int]] = []
        self._trail: List[List[Tuple[int, int]]] = []

        # Hand detection status
        self._hand_present = False
        self._last_smooth_pt: Optional[Tuple[float, float]] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.detector = None
        self.is_simulated = False
        self._down_count   = 0
        self._up_count     = 0

    def start_writing(self):
        with self._lock:
            self._writing    = True
            self._pen_down   = False
            self._down_count = 0
            self._up_count   = 0
            self._strokes    = []
            self._stroke_times = []
            self._current_stroke_start_time = None
            self._current    = []
            self._trail      = []
            self._last_smooth_pt = None

    def _commit_current_stroke(self, now: float):
        if self._current and len(self._current) >= 2:
            diffs = [np.hypot(self._current[i][0]-self._current[i-1][0], self._current[i][1]-self._current[i-1][1]) for i in range(1, len(self._current))]
            if sum(diffs) >= 8.0 or len(self._current) >= 3:
                self._strokes.append(list(self._current))
                start_t = self._current_stroke_start_time if self._current_stroke_start_time is not None else (now - 0.25)
                self._stroke_times.append((start_t, now))
        self._current = []
        self._current_stroke_start_time = None

    def finish_writing(self) -> List[List[Tuple[int, int]]]:
        """Commit active stroke and return snapshot of all strokes."""
        with self._lock:
            self._writing = False
            self._commit_current_stroke(time.time())
            self._pen_down   = False
            self._down_count = 0
            self._up_count   = 0
            self._last_smooth_pt = None
            result = [list(s) for s in self._strokes if len(s) >= 2]
        return result

    def finish_writing_with_times(self) -> List[dict]:
        """Commit active stroke and return list of dicts with pts, start_time, end_time."""
        with self._lock:
            self._writing = False
            self._commit_current_stroke(time.time())
            self._pen_down   = False
            self._down_count = 0
            self._up_count   = 0
            self._last_smooth_pt = None
            result = []
            for i, s in enumerate(self._strokes):
                if len(s) >= 2:
                    t_start, t_end = self._stroke_times[i] if i < len(self._stroke_times) else (0.0, 0.3)
                    result.append({
                        'pts': np.array(s, dtype=float),
                        'start_time': t_start,
                        'end_time': t_end,
                    })
        return result

    def clear_strokes(self):
        with self._lock:
            self._strokes    = []
            self._stroke_times = []
            self._current_stroke_start_time = None
            self._current    = []
            self._trail      = []
            self._pen_down   = False
            self._down_count = 0
            self._up_count   = 0
            self._last_smooth_pt = None

    def get_trajectory_preview(self, width: int = 120, height: int = 120) -> Optional[np.ndarray]:
        """Renders the captured trajectory into a crisp thumbnail image."""
        with self._lock:
            strokes = [list(s) for s in self._strokes if len(s) >= 2]
            if self._current and len(self._current) >= 2:
                strokes.append(list(self._current))

        if not strokes:
            return None

        img = np.full((height, width, 3), 255, dtype=np.uint8)
        all_pts = np.vstack([np.array(s) for s in strokes])
        mn = all_pts.min(axis=0)
        mx = all_pts.max(axis=0)
        span = max(float(np.ptp(all_pts, axis=0).max()), 1e-4)

        margin = 15
        target_w = width - 2 * margin
        target_h = height - 2 * margin

        for s in strokes:
            s_arr = np.array(s, dtype=float)
            # Scale and center
            sc_x = (s_arr[:, 0] - mn[0]) / span * target_w + margin
            sc_y = (s_arr[:, 1] - mn[1]) / span * target_h + margin
            pts = np.column_stack([sc_x, sc_y]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (220, 60, 10), 3, cv2.LINE_AA)

        return img

    def add_manual_point(self, x: int, y: int, is_new_stroke: bool = False):
        """Allows mouse drawing on canvas for testing without webcam."""
        with self._lock:
            if not self._writing:
                return
            now = time.time()
            pt = (int(x), int(y))
            if is_new_stroke or not self._current:
                self._commit_current_stroke(now)
                self._current = [pt]
                self._current_stroke_start_time = now
                self._trail.append([pt])
            else:
                last = self._current[-1]
                if _distance(pt, last) >= MIN_MOVE_PX:
                    self._current.append(pt)
                    if self._trail:
                        self._trail[-1].append(pt)
                    else:
                        self._trail.append([pt])

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self._run_internal()
        except Exception as e:
            traceback.print_exc()
            self.status_queue.put(("error", f"Camera error: {e}"))

    def _run_internal(self):
        # ── Step 1: Open Webcam ───────────────────────────────────────
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        # Check if frames can be read
        ret, test_frame = False, None
        if self.cap.isOpened():
            ret, test_frame = self.cap.read()

        if not self.cap.isOpened() or not ret:
            # Physical camera unavailable -> simulated mode
            self.is_simulated = True
            self.status_queue.put(("warning", "Webcam unavailable. Interactive canvas draw mode active."))
            self._run_simulated_camera()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # ── Step 2: Initialize MediaPipe HandLandmarker ───────────────
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if MEDIAPIPE_OK and os.path.exists(model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.40,
                    min_tracking_confidence=0.40,
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[Warning] Failed to initialize MediaPipe HandLandmarker: {e}")
                self.detector = None
        else:
            self.detector = None

        self.status_queue.put(("ready", "Camera ready."))

        last_pt_time = time.time()
        last_hand_seen_time = time.time()

        # ── Step 3: Main Video Loop ───────────────────────────────────
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)   # Mirror for natural air writing
            h, w = frame.shape[:2]

            fingertip_px: Optional[Tuple[int, int]] = None
            is_pen_down = False
            hand_found = False

            # Detect hand landmarks
            if self.detector is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    detection_result = self.detector.detect(mp_image)

                    if detection_result.hand_landmarks:
                        hand_found = True
                        last_hand_seen_time = time.time()
                        for hand_lms in detection_result.hand_landmarks:
                            # Draw skeleton connections (red)
                            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                                pt1 = (int(hand_lms[conn.start].x * w), int(hand_lms[conn.start].y * h))
                                pt2 = (int(hand_lms[conn.end].x * w), int(hand_lms[conn.end].y * h))
                                cv2.line(frame, pt1, pt2, CONN_COLOR, 2)

                            # Draw joint landmarks (red)
                            for lm in hand_lms:
                                pt = (int(lm.x * w), int(lm.y * h))
                                cv2.circle(frame, pt, 4, LM_COLOR, -1)

                            # Raw index fingertip (landmark 8)
                            raw_ix = hand_lms[8].x * w
                            raw_iy = hand_lms[8].y * h

                            # Exponential smoothing (alpha = 0.70) to eliminate micro-jitter
                            if self._last_smooth_pt is None:
                                sm_x, sm_y = raw_ix, raw_iy
                            else:
                                sm_x = 0.70 * raw_ix + 0.30 * self._last_smooth_pt[0]
                                sm_y = 0.70 * raw_iy + 0.30 * self._last_smooth_pt[1]

                            self._last_smooth_pt = (sm_x, sm_y)
                            fingertip_px = (int(round(sm_x)), int(round(sm_y)))

                            # ── Knuckle-relative gesture recognition (rotation-invariant) ──
                            wrist = np.array([hand_lms[0].x, hand_lms[0].y])
                            mid_mcp = np.array([hand_lms[9].x, hand_lms[9].y])
                            palm_scale = max(float(np.linalg.norm(mid_mcp - wrist)), 0.05)

                            # Distance from fingertips to their MCP joints
                            idx_tip = np.array([hand_lms[8].x, hand_lms[8].y])
                            idx_mcp = np.array([hand_lms[5].x, hand_lms[5].y])
                            idx_ext = float(np.linalg.norm(idx_tip - idx_mcp)) / palm_scale

                            mid_tip = np.array([hand_lms[12].x, hand_lms[12].y])
                            mid_mcp = np.array([hand_lms[9].x, hand_lms[9].y])
                            mid_ext = float(np.linalg.norm(mid_tip - mid_mcp)) / palm_scale

                            rng_tip = np.array([hand_lms[16].x, hand_lms[16].y])
                            rng_mcp = np.array([hand_lms[13].x, hand_lms[13].y])
                            rng_ext = float(np.linalg.norm(rng_tip - rng_mcp)) / palm_scale

                            pnk_tip = np.array([hand_lms[20].x, hand_lms[20].y])
                            pnk_mcp = np.array([hand_lms[17].x, hand_lms[17].y])
                            pnk_ext = float(np.linalg.norm(pnk_tip - pnk_mcp)) / palm_scale

                            # Index pointing condition:
                            # Index must be clearly extended (> 0.60)
                            # At least two other fingers must be curled/folded (< 0.75)
                            # Open hand (all fingers extended) = HOVER mode
                            others_folded = sum([mid_ext < 0.75, rng_ext < 0.70, pnk_ext < 0.70])
                            is_pointing = (idx_ext > 0.60) and (others_folded >= 2)

                            # Check if fingertip is inside the Virtual Writing Zone
                            in_zone = (ZONE_X1 <= fingertip_px[0] <= ZONE_X2 and
                                       ZONE_Y1 <= fingertip_px[1] <= ZONE_Y2)

                            raw_pen_down = is_pointing and in_zone

                            # 2-frame debouncing to eliminate flicker
                            if raw_pen_down:
                                self._down_count += 1
                                self._up_count = 0
                            else:
                                self._up_count += 1
                                self._down_count = 0

                            if self._down_count >= 2:
                                is_pen_down = True
                            elif self._up_count >= 2:
                                is_pen_down = False
                            else:
                                is_pen_down = self._pen_down

                            # Draw fingertip feedback
                            if is_pen_down:
                                cv2.circle(frame, fingertip_px, 14, FINGER_OUTER, 2, cv2.LINE_AA)
                                cv2.circle(frame, fingertip_px, 8, FINGER_INNER, -1, cv2.LINE_AA)
                            else:
                                # Hover ring (amber) indicates pen is lifted
                                cv2.circle(frame, fingertip_px, 12, (0, 215, 255), 2, cv2.LINE_AA)
                                cv2.circle(frame, fingertip_px, 4, (0, 215, 255), -1, cv2.LINE_AA)

                except Exception as ex:
                    print(f"[Warning] Frame landmark error: {ex}")

            # Notify GUI of hand loss
            if not hand_found and (time.time() - last_hand_seen_time > 1.2):
                if self._hand_present:
                    self._hand_present = False
                    self.status_queue.put(("warning", "No hand detected. Raise your hand inside the camera frame."))
            elif hand_found:
                if not self._hand_present:
                    self._hand_present = True
                    self.status_queue.put(("ready", "Hand detected. Ready to write in the air."))

            # ── Step 4: Stroke Recording & Segmentation ───────────────
            with self._lock:
                writing = self._writing

            now = time.time()
            if writing and fingertip_px is not None:
                with self._lock:
                    if is_pen_down:
                        if not self._pen_down or not self._current:
                            self._commit_current_stroke(now)
                            self._current = [fingertip_px]
                            self._current_stroke_start_time = now
                            self._trail.append([fingertip_px])
                            self._pen_down = True
                        else:
                            last = self._current[-1]
                            if _distance(fingertip_px, last) >= MIN_MOVE_PX:
                                self._current.append(fingertip_px)
                                if self._trail:
                                    self._trail[-1].append(fingertip_px)
                                else:
                                    self._trail.append([fingertip_px])
                        last_pt_time = now
                    else:
                        # Pen lifted -> commit stroke if it meets minimum path length
                        if self._pen_down:
                            self._commit_current_stroke(now)
                            self._pen_down = False
            elif writing and fingertip_px is None:
                if now - last_pt_time > 0.35:
                    with self._lock:
                        self._commit_current_stroke(now)
                        self._pen_down = False

            # ── Step 5: Draw Virtual Writing Zone & HUD ───────────────
            with self._lock:
                trail_strokes = [list(s) for s in self._trail if len(s) >= 2]
                currently_writing = self._writing
                currently_pen_down = self._pen_down

            # Draw Virtual Writing Zone outline
            zone_col = (0, 220, 30) if currently_pen_down else (180, 180, 180)
            # Corner brackets for clean modern aesthetic
            cor_len = 25
            # Top-Left
            cv2.line(frame, (ZONE_X1, ZONE_Y1), (ZONE_X1 + cor_len, ZONE_Y1), zone_col, 2)
            cv2.line(frame, (ZONE_X1, ZONE_Y1), (ZONE_X1, ZONE_Y1 + cor_len), zone_col, 2)
            # Top-Right
            cv2.line(frame, (ZONE_X2, ZONE_Y1), (ZONE_X2 - cor_len, ZONE_Y1), zone_col, 2)
            cv2.line(frame, (ZONE_X2, ZONE_Y1), (ZONE_X2, ZONE_Y1 + cor_len), zone_col, 2)
            # Bottom-Left
            cv2.line(frame, (ZONE_X1, ZONE_Y2), (ZONE_X1 + cor_len, ZONE_Y2), zone_col, 2)
            cv2.line(frame, (ZONE_X1, ZONE_Y2), (ZONE_X1, ZONE_Y2 - cor_len), zone_col, 2)
            # Bottom-Right
            cv2.line(frame, (ZONE_X2, ZONE_Y2), (ZONE_X2 - cor_len, ZONE_Y2), zone_col, 2)
            cv2.line(frame, (ZONE_X2, ZONE_Y2), (ZONE_X2, ZONE_Y2 - cor_len), zone_col, 2)

            # Zone Label
            cv2.putText(frame, "AIR WRITING ZONE", (ZONE_X1 + 10, ZONE_Y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            # Draw Blue Trajectory Trails
            for stroke in trail_strokes:
                pts = np.array(stroke, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, TRAIL_COLOR, TRAIL_THICKNESS, cv2.LINE_AA)

            # On-Screen Status HUD
            if currently_writing:
                if currently_pen_down:
                    cv2.putText(frame, "● WRITING: INDEX POINTING", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 30), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "○ HOVERING: OPEN HAND / PINCH", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

            # Push frame to GUI display queue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        # ── Cleanup ───────────────────────────────────────────────────
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
        self.status_queue.put(("stopped", "Camera stopped."))

    def _run_simulated_camera(self):
        """Simulated canvas mode when webcam hardware is unavailable."""
        self.status_queue.put(("ready", "Simulated Camera ready. Draw with mouse or gestures."))

        while not self._stop_event.is_set():
            frame = np.full((480, 640, 3), (242, 240, 240), dtype=np.uint8)

            # Subtle gridlines
            for gx in range(0, 640, 40):
                cv2.line(frame, (gx, 0), (gx, 480), (230, 225, 225), 1)
            for gy in range(0, 480, 40):
                cv2.line(frame, (0, gy), (640, gy), (230, 225, 225), 1)

            # Draw Zone
            cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (200, 200, 200), 1)
            cv2.putText(frame, "VIRTUAL WRITING CANVAS", (ZONE_X1 + 10, ZONE_Y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

            with self._lock:
                trail_strokes = [list(s) for s in self._trail if len(s) >= 2]
                writing = self._writing

            for stroke in trail_strokes:
                pts = np.array(stroke, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, TRAIL_COLOR, TRAIL_THICKNESS, cv2.LINE_AA)

            if writing:
                cv2.putText(frame, "WRITING ACTIVE", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 180, 20), 2, cv2.LINE_AA)
                cv2.putText(frame, "Click & drag inside zone to draw letter", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "READY", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (140, 40, 40), 2, cv2.LINE_AA)
                cv2.putText(frame, "Click 'Start Writing' to draw a character", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1, cv2.LINE_AA)

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            time.sleep(0.033)

        self.status_queue.put(("stopped", "Camera stopped."))
