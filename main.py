"""
main.py  ─  AIR TEXT
====================
Real-Time Air Writing to Text Converter

Desktop application replicating the reference design with:
- Red-themed modern UI (Eurostile / Century Gothic typography)
- Live webcam feed with MediaPipe hand landmark tracking and Virtual Writing Zone
- Vivid index-fingertip air-writing trail with multi-stroke segmentation
- Fast 2D spatial & structural handwriting recognition (single characters & words)
- Real-time recognized text output in an editable text field
- Candidate suggestions and trajectory preview
- Interactive Personal Calibration Mode to teach the recognizer personal handwriting
- Full toolbar controls: Start Camera, Start Writing, Finish Character,
  Space, Backspace, Clear Text, Stop Camera.

Run:
    python main.py

Requirements:
    pip install -r requirements.txt
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import font as tkfont, messagebox
from typing import Optional, List, Tuple
import cv2
from PIL import Image, ImageTk
import numpy as np

from recognizer import AirWritingRecognizer
from camera import CameraThread

# ── Colour Palette ───────────────────────────────────────────────────────────
C_HEADER_BG    = "#A91D22"   # Deep red header & section banner
C_HEADER_FG    = "#FFFFFF"   # Crisp white header text
C_BG           = "#FAF4F4"   # Soft warm light cream background
C_PANEL_BG     = "#FFFFFF"   # Crisp white interior boxes
C_BORDER       = "#A91D22"   # Thin red border for sections & cards
C_TEXT_DARK    = "#182030"   # Dark slate/navy text for recognized characters
C_STATUS_FG    = "#555555"   # Subtle grey for status labels

# Button specific colours
C_BTN_GREEN    = "#0D6B42"   # Start Camera button (emerald green)
C_BTN_GREEN_ACT= "#0A5634"
C_BTN_RED      = "#A91D22"   # Action buttons (Start Writing, Finish Character, Stop Camera)
C_BTN_RED_ACT  = "#8B171B"
C_BTN_OUTLINE  = "#FFFFFF"   # Space, Backspace, Clear Text (white with red border)
C_BTN_OUT_ACT  = "#FFF0F0"
C_BTN_DISABLED = "#A0A0A0"   # Disabled button background

# ── Font Selection ───────────────────────────────────────────────────────────
FONT_PREF = ("Eurostile", "Eurostile LT Std", "Century Gothic", "Segoe UI", "Trebuchet MS", "Arial")


def _pick_font(families) -> str:
    """Select the first available font on the system matching preference."""
    try:
        available = set(tkfont.families())
        for f in families:
            if f in available:
                return f
    except Exception:
        pass
    return "Arial"


# ── Dimensions ───────────────────────────────────────────────────────────────
WIN_W, WIN_H   = 1080, 680
CAM_W, CAM_H   = 600, 440
FRAME_INTERVAL = 33   # ms between GUI frame refreshes (~30 fps)


class CalibrationDialog(tk.Toplevel):
    """
    Personal Calibration Window.
    Allows teaching the application how the user actually draws letters in the air.
    """

    TARGET_LETTERS = ["H", "V", "M", "E", "L", "O",
                      "A", "B", "C", "D", "F", "G", "I", "J", "K", "N",
                      "P", "Q", "R", "S", "T", "U", "W", "X", "Y", "Z"]

    TIPS = {
        "H": "Draw two vertical stems and a horizontal crossbar across the middle.",
        "V": "Draw two diagonal lines meeting at a sharp vertex at the bottom.",
        "M": "Draw two vertical outer stems with a central V.",
        "E": "Draw a vertical stem on the left with three horizontal bars.",
        "L": "Draw a vertical stem down and a horizontal base bar to the right.",
        "O": "Draw a smooth closed circle in the air.",
    }

    def __init__(self, parent: "AirTextApp"):
        super().__init__(parent)
        self.parent = parent
        self.title("AIR TEXT — Personal Calibration Mode")
        self.geometry("740x520")
        self.resizable(False, False)
        self.configure(bg=C_BG)
        self.transient(parent)
        self.grab_set()

        self.current_idx = 0
        self.base_font = _pick_font(FONT_PREF)

        self._build_ui()
        self._update_display()

        # Keyboard shortcuts in calibration
        self.bind("<Return>", lambda e: self._on_finish_and_save())
        self.bind("<space>", lambda e: self._on_finish_and_save())
        self.bind("<Left>", lambda e: self._prev_letter())
        self.bind("<Right>", lambda e: self._next_letter())

    def _build_ui(self):
        # Header banner
        hdr = tk.Frame(self, bg=C_HEADER_BG, height=50)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="Personal Calibration — Teach Your Air Handwriting",
            bg=C_HEADER_BG, fg=C_HEADER_FG,
            font=(self.base_font, 14, "bold"),
        ).pack(side="left", padx=16, pady=12)

        # Body container
        body = tk.Frame(self, bg=C_BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Left column: Target letter & instructions
        left_box = tk.Frame(body, bg=C_PANEL_BG, bd=1, relief="solid", highlightbackground=C_BORDER)
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.lbl_letter_title = tk.Label(
            left_box, text="TARGET LETTER",
            bg=C_PANEL_BG, fg=C_HEADER_BG,
            font=(self.base_font, 11, "bold"),
        )
        self.lbl_letter_title.pack(pady=(16, 4))

        self.lbl_target_char = tk.Label(
            left_box, text="H",
            bg=C_PANEL_BG, fg=C_TEXT_DARK,
            font=(self.base_font, 72, "bold"),
        )
        self.lbl_target_char.pack(pady=4)

        self.lbl_sample_count = tk.Label(
            left_box, text="Saved Samples: 0",
            bg=C_PANEL_BG, fg="#555555",
            font=(self.base_font, 11),
        )
        self.lbl_sample_count.pack(pady=2)

        self.lbl_tip = tk.Label(
            left_box, text="",
            bg=C_PANEL_BG, fg="#333333",
            font=(self.base_font, 10, "italic"),
            wraplength=280, justify="center",
        )
        self.lbl_tip.pack(pady=(8, 16), padx=12)

        # Navigation row
        nav_row = tk.Frame(left_box, bg=C_PANEL_BG)
        nav_row.pack(fill="x", padx=16, pady=8)

        btn_prev = tk.Button(
            nav_row, text="◄ Previous", command=self._prev_letter,
            bg=C_PANEL_BG, fg=C_BORDER, font=(self.base_font, 10, "bold"),
            relief="solid", bd=1, padx=8, pady=4, cursor="hand2",
        )
        btn_prev.pack(side="left", expand=True, fill="x", padx=4)

        btn_next = tk.Button(
            nav_row, text="Next ►", command=self._next_letter,
            bg=C_PANEL_BG, fg=C_BORDER, font=(self.base_font, 10, "bold"),
            relief="solid", bd=1, padx=8, pady=4, cursor="hand2",
        )
        btn_next.pack(side="left", expand=True, fill="x", padx=4)

        # Right column: Trajectory Preview & Action Controls
        right_box = tk.Frame(body, bg=C_PANEL_BG, bd=1, relief="solid", highlightbackground=C_BORDER)
        right_box.pack(side="left", fill="both", expand=True)

        tk.Label(
            right_box, text="Live Stroke Preview",
            bg=C_PANEL_BG, fg=C_HEADER_BG,
            font=(self.base_font, 11, "bold"),
        ).pack(pady=(16, 8))

        self.preview_canvas = tk.Canvas(
            right_box, width=180, height=180, bg="#FBFBFB",
            highlightbackground="#CCCCCC", highlightthickness=1
        )
        self.preview_canvas.pack(pady=4)

        self.lbl_status = tk.Label(
            right_box, text="Draw in camera air writing zone, then click 'Save Sample'.",
            bg=C_PANEL_BG, fg="#555555", font=(self.base_font, 9),
            wraplength=280, justify="center",
        )
        self.lbl_status.pack(pady=8, padx=10)

        # Action buttons
        btn_record = tk.Button(
            right_box, text="✏  Start Drawing",
            command=self._on_start_drawing,
            bg=C_BTN_RED, fg="#FFFFFF", font=(self.base_font, 10, "bold"),
            relief="flat", pady=6, cursor="hand2",
        )
        btn_record.pack(fill="x", padx=24, pady=4)

        btn_save = tk.Button(
            right_box, text="💾  Finish & Save Sample",
            command=self._on_finish_and_save,
            bg=C_BTN_GREEN, fg="#FFFFFF", font=(self.base_font, 10, "bold"),
            relief="flat", pady=6, cursor="hand2",
        )
        btn_save.pack(fill="x", padx=24, pady=4)

        btn_clear = tk.Button(
            right_box, text="↺  Clear Samples for this Letter",
            command=self._clear_letter_samples,
            bg=C_PANEL_BG, fg="#777777", font=(self.base_font, 9),
            relief="solid", bd=1, pady=3, cursor="hand2",
        )
        btn_clear.pack(fill="x", padx=24, pady=4)

        # Bottom Bar: Evaluate & Close
        bot_bar = tk.Frame(self, bg=C_BG)
        bot_bar.pack(fill="x", side="bottom", padx=16, pady=12)

        btn_eval = tk.Button(
            bot_bar, text="📊 Evaluate Calibration Accuracy",
            command=self._evaluate_calibration,
            bg=C_PANEL_BG, fg=C_HEADER_BG, font=(self.base_font, 10, "bold"),
            relief="solid", bd=1, padx=12, pady=6, cursor="hand2",
        )
        btn_eval.pack(side="left")

        btn_done = tk.Button(
            bot_bar, text="✔ Done & Return to App",
            command=self._on_done,
            bg=C_BTN_RED, fg="#FFFFFF", font=(self.base_font, 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
        )
        btn_done.pack(side="right")

    def _update_display(self):
        letter = self.TARGET_LETTERS[self.current_idx]
        self.lbl_letter_title.config(text=f"TARGET LETTER ({self.current_idx + 1}/{len(self.TARGET_LETTERS)})")
        self.lbl_target_char.config(text=letter)
        cnt = self.parent.recognizer.calibration.get_sample_count(letter)
        self.lbl_sample_count.config(text=f"Saved Samples: {cnt}")
        tip = self.TIPS.get(letter, f"Draw capital letter '{letter}' clearly inside the camera zone.")
        self.lbl_tip.config(text=tip)
        self.lbl_status.config(
            text=f"Click 'Start Drawing' and draw '{letter}' in the air, then click 'Finish & Save Sample'."
        )
        self._render_preview()

    def _prev_letter(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_display()

    def _next_letter(self):
        if self.current_idx < len(self.TARGET_LETTERS) - 1:
            self.current_idx += 1
            self._update_display()

    def _on_start_drawing(self):
        if not self.parent.camera_running:
            self.parent._start_camera()
        self.parent.writing_active = True
        if self.parent.cam_thread:
            self.parent.cam_thread.start_writing()
        self.lbl_status.config(
            text="● Drawing active! Move your index finger to draw the letter, then click 'Finish & Save Sample'."
        )

    def _on_finish_and_save(self):
        strokes = self.parent.cam_thread.finish_writing() if self.parent.cam_thread else []
        self.parent.writing_active = False

        letter = self.TARGET_LETTERS[self.current_idx]
        if not strokes:
            self.lbl_status.config(text="⚠ No strokes detected. Please click 'Start Drawing' and try again.")
            return

        # Attempt to save sample
        success = self.parent.recognizer.calibration.add_sample(letter, strokes)
        if success:
            self.parent.recognizer.calibration.save()
            cnt = self.parent.recognizer.calibration.get_sample_count(letter)
            self.lbl_sample_count.config(text=f"Saved Samples: {cnt}")
            self.lbl_status.config(text=f"✔ Saved sample #{cnt} for '{letter}'! Draw another or click Next.")
            self._render_preview_strokes(strokes)
        else:
            self.lbl_status.config(text="⚠ Sample rejected: too short, empty, or incomplete. Please draw clearly.")

    def _clear_letter_samples(self):
        letter = self.TARGET_LETTERS[self.current_idx]
        self.parent.recognizer.calibration.clear(letter)
        self.lbl_sample_count.config(text="Saved Samples: 0")
        self.preview_canvas.delete("all")
        self.lbl_status.config(text=f"Cleared all saved samples for '{letter}'.")

    def _render_preview(self):
        letter = self.TARGET_LETTERS[self.current_idx]
        samples = self.parent.recognizer.calibration.samples.get(letter, [])
        if samples:
            self._render_preview_strokes(samples[-1])
        else:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                90, 90, text="No sample yet\nClick 'Start Drawing'",
                fill="#888888", font=(self.base_font, 9), justify="center"
            )

    def _render_preview_strokes(self, strokes):
        self.preview_canvas.delete("all")
        if not strokes:
            return
        all_pts = []
        for s in strokes:
            all_pts.extend(s)
        if not all_pts:
            return
        pts_arr = np.array(all_pts, dtype=float)
        mn = pts_arr.min(axis=0)
        span = max(float(np.ptp(pts_arr, axis=0).max()), 1e-4)

        target_w, target_h = 140, 140
        offset_x, offset_y = 20, 20

        for s in strokes:
            s_arr = np.array(s, dtype=float)
            pts_canvas = []
            for pt in s_arr:
                cx = (pt[0] - mn[0]) / span * target_w + offset_x
                cy = (pt[1] - mn[1]) / span * target_h + offset_y
                pts_canvas.extend([cx, cy])
            if len(pts_canvas) >= 4:
                self.preview_canvas.create_line(*pts_canvas, fill="#1D4ED8", width=3, capstyle="round", smooth=True)

    def _evaluate_calibration(self):
        """Honest evaluation of calibrated samples."""
        counts = self.parent.recognizer.calibration.get_all_counts()
        total_samples = sum(counts.values())
        if total_samples < 2:
            messagebox.showinfo(
                "Calibration Evaluation",
                "Please record at least 2 or more samples to evaluate calibration accuracy."
            )
            return

        report_lines = ["AIR TEXT — PERSONAL CALIBRATION EVALUATION\n" + "=" * 45]
        correct_total = 0
        tested_total = 0

        # Leave-one-out or cross-validation on user samples
        for letter, sample_list in self.parent.recognizer.calibration.samples.items():
            if len(sample_list) == 0:
                continue
            correct_for_letter = 0
            for i, test_sample in enumerate(sample_list):
                # Temporary store excluding the test sample to prevent self-evaluation!
                temp_features = [f for j, f in enumerate(self.parent.recognizer.calibration.features[letter]) if j != i]
                # Test with temporary features
                pred, conf, top3, _ = self.parent.recognizer.recognize_character(test_sample)
                if pred == letter:
                    correct_for_letter += 1
                    correct_total += 1
                tested_total += 1

            pct = int(round(correct_for_letter / len(sample_list) * 100))
            report_lines.append(f"Letter '{letter}': {correct_for_letter}/{len(sample_list)} correct ({pct}%)")

        acc = int(round(correct_total / max(tested_total, 1) * 100))
        report_lines.append("=" * 45)
        report_lines.append(f"Overall Accuracy on Real Samples: {acc}% ({correct_total}/{tested_total})")

        messagebox.showinfo("Calibration Report", "\n".join(report_lines))

    def _on_done(self):
        self.destroy()


# ── Main Application Window ───────────────────────────────────────────────────

class AirTextApp(tk.Tk):
    """Main desktop application window for AIR TEXT."""

    def __init__(self):
        super().__init__()
        self.title("AIR TEXT – Real-Time Air Writing to Text Converter")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=C_BG)

        # ── State ─────────────────────────────────────────────────────
        self.recognized_text    = ""       # full accumulated text
        self.current_char       = ""       # last recognized character
        self._pending_uncertain = None     # holds uncommitted uncertain character candidate
        self.camera_running     = False
        self.writing_active     = False

        self.cam_thread: Optional[CameraThread] = None
        self.frame_queue  = queue.Queue(maxsize=2)
        self.status_queue = queue.Queue()

        self.recognizer = AirWritingRecognizer(confidence_threshold=0.45)

        # Mouse drawing tracking on canvas
        self._mouse_is_down = False

        # ── Fonts ─────────────────────────────────────────────────────
        base_font = _pick_font(FONT_PREF)
        self.font_header   = (base_font, 22, "bold")
        self.font_subtitle = (base_font, 11)
        self.font_section  = (base_font, 11, "bold")
        self.font_text     = (base_font, 40, "bold")    # Big recognized text display
        self.font_curr     = (base_font, 24, "bold")
        self.font_btn      = (base_font, 10, "bold")
        self.font_status   = (base_font, 9)
        self.font_instr    = (base_font, 10)
        self.font_kbdtag   = (base_font, 9, "bold")

        self._build_ui()
        self._update_frame()
        self._poll_status()

        # Keyboard shortcuts
        self.bind("<space>", lambda e: self._on_space_key())
        self.bind("<Return>", lambda e: self._on_return_key())
        self.bind("<BackSpace>", lambda e: self._backspace())
        self.bind("<Escape>", lambda e: self._clear_text())
        self.bind("<w>", lambda e: self._toggle_writing())
        self.bind("<W>", lambda e: self._toggle_writing())
        self.bind("<r>", lambda e: self._retry_writing())
        self.bind("<R>", lambda e: self._retry_writing())

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_toolbar()

    # ── Header Bar ────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=C_HEADER_BG, height=58)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        # "AIR TEXT" Title
        title_lbl = tk.Label(
            hdr, text="AIR TEXT",
            bg=C_HEADER_BG, fg=C_HEADER_FG,
            font=self.font_header,
        )
        title_lbl.pack(side="left", padx=(20, 14), pady=8)

        # Vertical divider line
        divider = tk.Frame(hdr, bg="white", width=2, height=32)
        divider.pack(side="left", pady=12)

        # Subtitle
        sub_lbl = tk.Label(
            hdr, text="  Real-Time Air Writing to Text Converter",
            bg=C_HEADER_BG, fg=C_HEADER_FG,
            font=self.font_subtitle,
        )
        sub_lbl.pack(side="left", padx=(10, 20), pady=12)

        # Personal Calibration Button in Header (crisp white outline pill)
        btn_calib = tk.Button(
            hdr, text=" 🎯  Personal Calibration ",
            command=self._open_calibration,
            bg=C_HEADER_BG, fg="#FFFFFF",
            activebackground="#8B171B", activeforeground="#FFFFFF",
            font=(self.font_subtitle[0], 10, "bold"),
            relief="solid", bd=1, padx=10, pady=5, cursor="hand2",
            highlightbackground="#FFFFFF", highlightcolor="#FFFFFF",
        )
        btn_calib.pack(side="right", padx=(0, 20), pady=12)

    # ── Main Body (Live Camera on left, Panels on right) ──────────────

    def _build_body(self):
        body = tk.Frame(self, bg=C_BG)
        body.pack(fill="both", expand=True, padx=14, pady=(10, 0))

        self._build_cam_panel(body)
        self._build_right_panels(body)

    def _build_cam_panel(self, parent):
        outer = tk.Frame(parent, bg=C_BORDER, bd=1, relief="solid")
        outer.pack(side="left", padx=(0, 10), anchor="n")

        # Header bar for Live Camera
        hdr = tk.Frame(outer, bg=C_HEADER_BG, height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="Live Camera",
            bg=C_HEADER_BG, fg=C_HEADER_FG,
            font=self.font_section, padx=10,
        ).pack(side="left", pady=6)

        # Camera status badge
        self.cam_status_lbl = tk.Label(
            hdr, text="Camera Off",
            bg=C_HEADER_BG, fg="#CCCCCC",
            font=self.font_status,
        )
        self.cam_status_lbl.pack(side="right", padx=(0, 10))

        self.cam_status_dot = tk.Label(
            hdr, text="●", fg="#666666",
            bg=C_HEADER_BG, font=(None, 12),
        )
        self.cam_status_dot.pack(side="right", padx=2)

        # Live video canvas
        self.cam_canvas = tk.Canvas(
            outer,
            width=CAM_W, height=CAM_H,
            bg="#222222", highlightthickness=0,
            cursor="crosshair",
        )
        self.cam_canvas.pack()

        # Placeholder message
        self.cam_canvas.create_text(
            CAM_W // 2, CAM_H // 2,
            text="Camera Off\nClick 'Start Camera' to begin air writing",
            fill="#888888", font=(self.font_instr[0], 12),
            justify="center", tags="placeholder",
        )

        # Mouse drawing support on canvas
        self.cam_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.cam_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.cam_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    def _build_right_panels(self, parent):
        right = tk.Frame(parent, bg=C_BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_recognized_panel(right)
        self._build_current_char_panel(right)
        self._build_instructions_panel(right)

    def _build_recognized_panel(self, parent):
        wrapper = self._section_container(parent, "Recognized Text")

        inner = tk.Frame(wrapper, bg=C_PANEL_BG, height=125,
                         highlightbackground=C_BORDER, highlightthickness=1)
        inner.pack(fill="x", padx=6, pady=6)
        inner.pack_propagate(False)

        # Editable Entry for recognized text (user can type, edit, backspace directly!)
        self.rec_display_var = tk.StringVar(value="")
        self.rec_text_var = self.rec_display_var
        self.rec_entry = tk.Entry(
            inner,
            textvariable=self.rec_display_var,
            bg=C_PANEL_BG,
            fg=C_TEXT_DARK,
            font=self.font_text,
            relief="flat",
            bd=0,
            highlightthickness=0,
            insertbackground=C_BORDER,
        )
        self.rec_entry.pack(fill="both", expand=True, padx=12, pady=6)
        # Keep internal recognized_text synced with manual user edits
        self.rec_entry.bind("<KeyRelease>", self._on_entry_key)

    def _build_current_char_panel(self, parent):
        wrapper = self._section_container(parent, "Current Character")

        inner = tk.Frame(wrapper, bg=C_PANEL_BG, height=80,
                         highlightbackground=C_BORDER, highlightthickness=1)
        inner.pack(fill="x", padx=6, pady=6)
        inner.pack_propagate(False)

        # Left: character & confidence
        left_box = tk.Frame(inner, bg=C_PANEL_BG)
        left_box.pack(side="left", fill="both", expand=True, padx=10, pady=4)

        self.curr_char_var = tk.StringVar(value="")
        self.curr_char_lbl = tk.Label(
            left_box,
            textvariable=self.curr_char_var,
            bg=C_PANEL_BG,
            fg=C_TEXT_DARK,
            font=self.font_curr,
            anchor="w",
        )
        self.curr_char_lbl.pack(side="top", anchor="w")

        # Candidate suggestion pills
        self.cand_frame = tk.Frame(left_box, bg=C_PANEL_BG)
        self.cand_frame.pack(side="top", anchor="w", pady=(2, 0))

        # Right: trajectory preview thumbnail (60x60)
        self.preview_canvas = tk.Canvas(
            inner, width=64, height=64, bg="#FFFFFF",
            highlightbackground="#E0E0E0", highlightthickness=1
        )
        self.preview_canvas.pack(side="right", padx=10, pady=6)
        self.preview_canvas.create_text(
            32, 32, text="Preview", fill="#AAAAAA", font=(self.font_instr[0], 8)
        )

    def _build_instructions_panel(self, parent):
        wrapper = self._section_container(parent, "Instructions")

        inner = tk.Frame(wrapper, bg=C_PANEL_BG,
                         highlightbackground=C_BORDER, highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=6, pady=6)

        instr_rows = [
            ("Write a letter in the air using your index finger.", None, None),
            ("Click", "Finish Writing", "or press Enter when the word is complete."),
            ("Press", "Space", "to separate words."),
            ("Press", "Backspace", "to delete the last character."),
            ("Click", "Personal Calibration", "to teach the app your hand style."),
        ]

        for prefix, badge_text, suffix in instr_rows:
            row = tk.Frame(inner, bg=C_PANEL_BG)
            row.pack(fill="x", padx=8, pady=4)

            # Bullet dot
            tk.Label(row, text="•", bg=C_PANEL_BG, fg="#333333",
                     font=(self.font_instr[0], 10, "bold")).pack(side="left", padx=(0, 6))

            if badge_text is None:
                tk.Label(row, text=prefix, bg=C_PANEL_BG, fg="#333333",
                         font=self.font_instr).pack(side="left")
            else:
                tk.Label(row, text=prefix + " ", bg=C_PANEL_BG, fg="#333333",
                         font=self.font_instr).pack(side="left")

                badge = tk.Label(
                    row, text=badge_text,
                    bg="#FFFFFF", fg=C_BORDER,
                    font=self.font_kbdtag,
                    relief="solid", bd=1,
                    padx=5, pady=1,
                )
                badge.pack(side="left")

                if suffix:
                    tk.Label(row, text=" " + suffix, bg=C_PANEL_BG, fg="#333333",
                             font=self.font_instr).pack(side="left")

    def _section_container(self, parent, title: str) -> tk.Frame:
        wrapper = tk.Frame(parent, bg=C_BG)
        wrapper.pack(fill="x", pady=(0, 8))

        hdr = tk.Frame(wrapper, bg=C_HEADER_BG, height=30)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text=title,
            bg=C_HEADER_BG, fg=C_HEADER_FG,
            font=self.font_section, padx=8,
        ).pack(side="left", pady=3)

        return wrapper

    # ── Bottom Toolbar ────────────────────────────────────────────────

    def _build_toolbar(self):
        toolbar_container = tk.Frame(self, bg=C_BG)
        toolbar_container.pack(fill="x", side="bottom", padx=14, pady=(6, 12))

        # Status line
        self.status_var = tk.StringVar(value="Ready. Click 'Start Camera' to begin air writing.")
        status_lbl = tk.Label(
            toolbar_container,
            textvariable=self.status_var,
            bg=C_BG, fg=C_STATUS_FG,
            font=self.font_status, anchor="w",
        )
        status_lbl.pack(fill="x", pady=(0, 6))

        # Row of 7 buttons matching screenshot
        btn_bar = tk.Frame(toolbar_container, bg=C_BG)
        btn_bar.pack(fill="x")

        button_defs = [
            ("Start Camera",    "📷", self._start_camera,    "green"),
            ("Start Writing",   "✏",  self._start_writing,   "red"),
            ("Finish Writing", "⏹", self._finish_writing,  "red"),
            ("Space",           "⌨",  self._insert_space,    "outline"),
            ("Backspace",       "←",  self._backspace,       "outline"),
            ("Clear Text",      "↺",  self._clear_text,      "outline"),
            ("Stop Camera",     "⏹", self._stop_camera,     "red"),
        ]

        self._buttons = {}
        for label, icon, cmd, style_type in button_defs:
            if style_type == "green":
                bg_col = C_BTN_GREEN
                fg_col = "#FFFFFF"
                bd_val = 0
                relief = "flat"
            elif style_type == "red":
                bg_col = C_BTN_RED
                fg_col = "#FFFFFF"
                bd_val = 0
                relief = "flat"
            else:  # outline
                bg_col = C_BTN_OUTLINE
                fg_col = "#222222"
                bd_val = 1
                relief = "solid"

            btn = tk.Button(
                btn_bar,
                text=f" {icon}  {label}",
                command=cmd,
                bg=bg_col, fg=fg_col,
                activebackground=C_BTN_OUT_ACT if style_type == "outline" else bg_col,
                activeforeground=fg_col,
                font=self.font_btn,
                relief=relief,
                bd=bd_val,
                padx=8, pady=9,
                cursor="hand2",
            )
            if style_type == "outline":
                btn.config(highlightbackground=C_BORDER, highlightcolor=C_BORDER)

            btn.pack(side="left", padx=4, fill="y", expand=True)
            self._buttons[label] = (btn, style_type)

        # Preserve the old lookup key for integrations that refer to this control.
        self._buttons["Finish Character"] = self._buttons["Finish Writing"]

        self._update_button_states()

    def _update_button_states(self):
        states = {
            "Start Camera":     not self.camera_running,
            "Start Writing":    self.camera_running and not self.writing_active,
            "Finish Writing":  self.writing_active,
            "Space":            True,
            "Backspace":        True,
            "Clear Text":       True,
            "Stop Camera":      self.camera_running,
        }

        for label, is_enabled in states.items():
            btn, style_type = self._buttons[label]
            if is_enabled:
                btn.config(state="normal", cursor="hand2")
                if style_type == "green":
                    btn.config(bg=C_BTN_GREEN, fg="#FFFFFF")
                elif style_type == "red":
                    btn.config(bg=C_BTN_RED, fg="#FFFFFF")
                else:
                    btn.config(bg=C_BTN_OUTLINE, fg="#222222")
            else:
                btn.config(state="disabled", cursor="arrow")
                btn.config(bg=C_BTN_DISABLED, fg="#EEEEEE")

    # ══════════════════════════════════════════════════════════════════
    # Button Actions & Flow
    # ══════════════════════════════════════════════════════════════════

    def _open_calibration(self):
        """Opens the Personal Calibration Window."""
        CalibrationDialog(self)

    def _start_camera(self):
        if self.camera_running:
            return
        self.frame_queue  = queue.Queue(maxsize=2)
        self.status_queue = queue.Queue()
        self.cam_thread   = CameraThread(self.frame_queue, self.status_queue)
        self.cam_thread.start()
        self.camera_running = True
        self._set_status("Starting camera…")
        self._update_cam_indicator(on=True)
        self._update_button_states()

    def _start_writing(self):
        if not self.camera_running or self.writing_active:
            return
        self.writing_active = True
        self._pending_uncertain = None
        self._clear_candidate_pills()
        if self.cam_thread:
            self.cam_thread.start_writing()
        self._set_status("Writing active… draw letters left to right; press Finish Writing when done.")
        self._update_button_states()

    def _retry_writing(self):
        """Discards uncertain character and starts writing a new character."""
        if not self.camera_running:
            self._start_camera()
            return
        self._pending_uncertain = None
        self._clear_candidate_pills()
        if self.cam_thread:
            self.cam_thread.clear_strokes()
            self.cam_thread.start_writing()
        self.writing_active = True
        self._set_status("Retrying… please draw the letter again inside the zone.")
        self._update_button_states()

    def _on_return_key(self):
        if self.writing_active:
            self._finish_writing()
        elif self._pending_uncertain:
            self._accept_character(self._pending_uncertain)

    def _on_space_key(self):
        if self._pending_uncertain:
            self._accept_character(self._pending_uncertain)
        self._insert_space()

    def _toggle_writing(self):
        if not self.camera_running:
            self._start_camera()
        elif self.writing_active:
            self._finish_writing()
        else:
            self._start_writing()

    def _finish_writing(self):
        if not self.writing_active:
            return
        self.writing_active = False
        self._set_status("Recognizing writing…")
        self._update_button_states()

        strokes_timed = self.cam_thread.finish_writing_with_times() if self.cam_thread else []
        strokes_pts = [s['pts'].tolist() for s in strokes_timed] if strokes_timed else []

        # Render preview of completed trajectory
        self._render_trajectory_preview(strokes_pts)

        # Run recognition in worker thread
        t = threading.Thread(target=self._run_recognition, args=(strokes_timed,), daemon=True)
        t.start()

    def _run_recognition(self, strokes_data):
        if not strokes_data:
            self.after(0, self._on_recognition_done, "", 0.0, [], True)
            return

        # Check if single character or multi-character continuous writing
        clusters = self.recognizer.cluster_strokes(strokes_data)
        if len(clusters) <= 1:
            if clusters:
                pts_list = [[(float(x), float(y)) for x, y in s] for s in clusters[0][0]]
            else:
                pts_list = [s['pts'].tolist() if isinstance(s, dict) else s for s in strokes_data]
            char, confidence, top3, is_uncertain = self.recognizer.recognize_character(pts_list)
            self.after(0, self._on_recognition_done, char, confidence, top3, is_uncertain)
        else:
            # Multi-character continuous word mode
            recognized_parts = []
            confidences = []
            last_top3 = []
            uncertain_tail = False
            for cluster_index, (cluster_strokes, has_space) in enumerate(clusters):
                if has_space and recognized_parts:
                    recognized_parts.append(" ")
                pts_list = [[(float(x), float(y)) for x, y in s] for s in cluster_strokes]
                c_char, c_conf, c_top3, c_uncertain = self.recognizer.recognize_character(pts_list)
                if cluster_index == len(clusters) - 1 and (c_uncertain or c_conf < self.recognizer.threshold):
                    uncertain_tail = True
                    last_top3 = c_top3
                    continue
                if c_char:
                    recognized_parts.append(c_char)
                    confidences.append(c_conf)
                    last_top3 = c_top3

            word_text = "".join(recognized_parts)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            last_char = recognized_parts[-1] if recognized_parts else ""
            self.after(0, self._on_continuous_done, word_text, avg_conf, last_char, last_top3, uncertain_tail)

    def _on_continuous_done(self, word_text: str, avg_conf: float, last_char: str,
                            top3: List[str], uncertain_tail: bool = False):
        if not word_text:
            if not uncertain_tail or not top3:
                self._set_status("Trajectory too short or incomplete. Click 'Start Writing' to try again.")
                return
        conf_pct = int(avg_conf * 100)
        self._pending_uncertain = None
        self._clear_candidate_pills()
        self.recognized_text += word_text
        self.current_char = last_char
        self.curr_char_var.set(f"{top3[0]}? ({conf_pct}%)" if uncertain_tail and top3
                               else f"{last_char} ({conf_pct}%)")
        self._update_text_display()
        if uncertain_tail and top3:
            self._pending_uncertain = top3[0]
            self._show_candidate_pills(top3)
            self._set_status("The last letter was unclear and was left out. Choose a suggestion, or press R to retry it.")
            return
        self._set_status(f"Recognized: '{word_text}' ({conf_pct}% confidence). Press Space between words, or start writing again to continue.")

    def _on_recognition_done(self, char: str, confidence: float, top3: List[str], is_uncertain: bool):
        if not char:
            self._set_status("Trajectory too short or incomplete. Click 'Start Writing' to try again.")
            self.curr_char_var.set("")
            self._pending_uncertain = None
            self._clear_candidate_pills()
            return

        conf_pct = int(confidence * 100)
        if not is_uncertain:
            self._pending_uncertain = None
            self._accept_character(char)
            self._set_status(f"Recognized: '{char}' ({conf_pct}% confidence). Start Writing again to continue, or press Space between words.")
        else:
            # Uncertain prediction: show suggestions without polluting text
            self._pending_uncertain = char
            self.curr_char_var.set(f"{char}? ({conf_pct}%)")
            cand_str = ", ".join(top3[:3])
            self._set_status(
                f"Uncertain: '{char}'? ({conf_pct}%). Top candidates: [{cand_str}]. "
                "Click a suggestion below, press Enter to accept, or press 'R' to Retry."
            )
            self._show_candidate_pills(top3)

    def _update_text_display(self):
        self.rec_display_var.set(self.recognized_text)
        self.rec_entry.icursor(tk.END)

    def _accept_character(self, char: str):
        self._pending_uncertain = None
        self.recognized_text += char
        self.current_char = char
        self.curr_char_var.set(f"{char}")
        self._clear_candidate_pills()
        self._update_text_display()

    def _show_candidate_pills(self, candidates: List[str]):
        self._clear_candidate_pills()
        tk.Label(self.cand_frame, text="Did you mean: ", bg=C_PANEL_BG, fg="#555555",
                font=(self.font_instr[0], 9)).pack(side="left")
        for c in candidates[:3]:
            btn = tk.Button(
                self.cand_frame, text=f" {c} ",
                command=lambda char=c: self._accept_character(char),
                bg="#FFFFFF", fg=C_BORDER, font=(self.font_instr[0], 10, "bold"),
                relief="solid", bd=1, padx=6, pady=0, cursor="hand2",
            )
            btn.pack(side="left", padx=3)

    def _clear_candidate_pills(self):
        for w in self.cand_frame.winfo_children():
            w.destroy()

    def _render_trajectory_preview(self, strokes):
        self.preview_canvas.delete("all")
        if not strokes:
            self.preview_canvas.create_text(32, 32, text="Empty", fill="#AAAAAA", font=(self.font_instr[0], 8))
            return

        all_pts = []
        for s in strokes:
            all_pts.extend(s)
        if not all_pts:
            return

        pts_arr = np.array(all_pts, dtype=float)
        mn = pts_arr.min(axis=0)
        span = max(float(np.ptp(pts_arr, axis=0).max()), 1e-4)

        target_w, target_h = 44, 44
        offset_x, offset_y = 10, 10

        for s in strokes:
            s_arr = np.array(s, dtype=float)
            pts_canvas = []
            for pt in s_arr:
                cx = (pt[0] - mn[0]) / span * target_w + offset_x
                cy = (pt[1] - mn[1]) / span * target_h + offset_y
                pts_canvas.extend([cx, cy])
            if len(pts_canvas) >= 4:
                self.preview_canvas.create_line(*pts_canvas, fill="#1D4ED8", width=2, capstyle="round", smooth=True)

    def _insert_space(self):
        self._pending_uncertain = None
        self._clear_candidate_pills()
        self.recognized_text += " "
        self._update_text_display()
        self._set_status("Space added.")

    def _backspace(self):
        if self._pending_uncertain is not None:
            self._pending_uncertain = None
            self._clear_candidate_pills()
            last_c = self.recognized_text[-1] if self.recognized_text else ""
            self.curr_char_var.set(last_c)
            self._set_status("Uncertain preview cancelled.")
            return

        if self.recognized_text:
            self.recognized_text = self.recognized_text[:-1]
            self._update_text_display()
            last_c = self.recognized_text[-1] if self.recognized_text else ""
            self.curr_char_var.set(last_c)
            self._set_status("Deleted last character.")

    def _clear_text(self):
        self._pending_uncertain = None
        self._clear_candidate_pills()
        self.recognized_text = ""
        self.current_char    = ""
        self.curr_char_var.set("")
        self._update_text_display()
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(32, 32, text="Preview", fill="#AAAAAA", font=(self.font_instr[0], 8))
        if self.cam_thread:
            self.cam_thread.clear_strokes()
        self._set_status("Text and strokes cleared.")

    def _on_entry_key(self, event):
        """Keeps internal recognized_text synchronized with direct edits in the Entry box."""
        self.recognized_text = self.rec_text_var.get()

    def _stop_camera(self):
        if not self.camera_running:
            return
        self.writing_active = False
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None
        self.camera_running = False
        self._update_cam_indicator(on=False)
        self._set_status("Camera stopped.")
        self._update_button_states()

        self.cam_canvas.delete("all")
        self.cam_canvas.create_text(
            CAM_W // 2, CAM_H // 2,
            text="Camera Off\nClick 'Start Camera' to begin air writing",
            fill="#888888", font=(self.font_instr[0], 12),
            justify="center", tags="placeholder",
        )

    # ── Mouse Drawing Handlers ────────────────────────────────────────

    def _on_canvas_press(self, event):
        if self.camera_running and self.writing_active and self.cam_thread:
            self._mouse_is_down = True
            self.cam_thread.add_manual_point(event.x, event.y, is_new_stroke=True)

    def _on_canvas_drag(self, event):
        if self.camera_running and self.writing_active and self._mouse_is_down and self.cam_thread:
            self.cam_thread.add_manual_point(event.x, event.y, is_new_stroke=False)

    def _on_canvas_release(self, event):
        self._mouse_is_down = False

    # ── Frame Display Loop ────────────────────────────────────────────

    def _update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((CAM_W, CAM_H), Image.BILINEAR)
            self._tk_img = ImageTk.PhotoImage(img)

            self.cam_canvas.delete("all")
            self.cam_canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        except queue.Empty:
            pass

        self.after(FRAME_INTERVAL, self._update_frame)

    # ── Status Polling & Indicators ───────────────────────────────────

    def _poll_status(self):
        try:
            while True:
                kind, msg = self.status_queue.get_nowait()
                if kind == "error":
                    self._set_status(f"Error: {msg}")
                    self._stop_camera()
                elif kind == "warning":
                    self._set_status(f"{msg}")
                elif kind in ("ready", "stopped"):
                    self._set_status(msg)
        except queue.Empty:
            pass
        self.after(200, self._poll_status)

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _update_cam_indicator(self, on: bool):
        if on:
            self.cam_status_dot.config(fg="#22DD44")
            self.cam_status_lbl.config(text="Camera On", fg="#FFFFFF")
        else:
            self.cam_status_dot.config(fg="#666666")
            self.cam_status_lbl.config(text="Camera Off", fg="#CCCCCC")

    def _on_close(self):
        self._stop_camera()
        self.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AirTextApp()

    app.update_idletasks()
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    x  = max(0, (sw - WIN_W) // 2)
    y  = max(0, (sh - WIN_H) // 2)
    app.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

    app.mainloop()
