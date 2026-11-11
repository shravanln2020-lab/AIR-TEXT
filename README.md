# AIR TEXT — Real-Time Air Writing to Text Converter

A robust, beginner-friendly Python desktop application that tracks your hand through a live webcam, captures letters and words written in the air using your index finger, recognizes the writing via fast oriented point-cloud matching, and converts it into digital text in real time.

Designed for resume showcases, portfolio demonstrations, and computer vision / human-computer interaction (HCI) projects.

---

## Visual Design & Red UI Layout

The application preserves the clean, modern red-themed desktop interface:
- **Deep Red Header** (`#A91D22`): Title `AIR TEXT` with subtitle `Real-Time Air Writing to Text Converter` and a **Personal Calibration** button.
- **Left Panel (Live Camera)**:
  - 640×480 live video stream with mirrored display for natural writing.
  - MediaPipe 21-joint red hand skeleton and landmark overlay.
  - Defined **Air Writing Zone** with corner brackets and HUD status.
  - Green concentric tracking ring on the index fingertip when drawing (Pen Down), and amber hover ring when resting (Pen Up).
  - Vivid blue trajectory trail (`#1450FF`) tracing multi-stroke air handwriting.
- **Right Panels**:
  - **Recognized Text**: Large editable text field displaying recognized text with cursor navigation.
  - **Current Character**: Shows the most recent letter and confidence score, plus clickable candidate suggestion pills (`[H]`, `[M]`, etc.) and a real-time trajectory preview thumbnail.
  - **Instructions**: Quick-reference guide with keyboard shortcut badges (`Space`, `Backspace`, `Enter`, `R`).
- **Bottom Toolbar**: 7 evenly spaced control buttons:
  - `📷 Start Camera` (Emerald Green)
  - `✏ Start Writing` (Deep Red)
  - `⏹ Finish Writing` (Deep Red)
  - `⌨ Space` (White with thin red border)
  - `← Backspace` (White with thin red border)
  - `↺ Clear Text` (White with thin red border)
  - `⏹ Stop Camera` (Deep Red)

---

## Core Technologies & Architecture

1. **Knuckle-Relative Hand Tracking (MediaPipe Hands)**:
   - Tracks 21 hand landmarks at 30 FPS using MediaPipe Tasks Vision.
   - **Rotation-Invariant Gesture Detection**: Measures fingertip extension relative to palm scale ($d_{\text{wrist} \to \text{mcp}}$).
     - **Pen Down (Writing)**: Index finger extended ($> 0.60$), at least 2 other fingers folded.
     - **Pen Up (Hover)**: Open hand or multiple fingers raised.
   - **2-Frame Temporal Debouncing**: Eliminates accidental stroke breaks from tracking flickers.
   - **Exponential Moving Average Smoothing**: $\alpha = 0.70$ smoothing on fingertip coordinates removes micro-tremor while preserving sharp letter corners.

2. **Hybrid ML & Oriented Point Cloud Recognition Architecture**:
   - **Dual-Engine Fusion**: Integrates a 76-dimensional feature embedding with Dynamic Oriented Point Cloud Matching:
     - 34 Chamfer distance embeddings against canonical geometric templates.
     - Soft 8-Bin Orientation Histogram with circular smoothing.
     - 16-bin horizontal and vertical projection profiles.
     - Aspect ratio preservation ($w / \max(h, 10^{-4})$).
     - Area-based closed loop topology check (Shoelace formula).
     - Parallel boundary stem detection (`has_two_stems`).
   - **Pretrained Logistic Regression Classifier**: Fast inference ($2.60\text{ ms}$ per character) trained on multi-stroke permutations and realistic human tremor deformations.
   - **Zero Hardcoded Dictionaries**: Operates purely on kinematic, topological, and spatial trajectory geometry.
   - **Spatio-Temporal Continuous Writing Segmenter**:
     - Automatically clusters multi-stroke letters (`H`, `E`, `F`, `T`, etc.) and parallel stems.
     - Distinguishes intra-character pen-ups from inter-character spatial progression and word-level spaces.
     - Supports both letter-by-letter mode and seamless continuous word writing.

3. **Personal Calibration System**:
   - Built-in interactive calibration dialog to record labeled user handwriting exemplars.
   - Saves personal writing styles to `calibration_data/user_profile.json`.
   - Calibration exemplars receive priority matching with cross-validation evaluation.

4. **Thread-Safe Concurrency**:
   - Camera acquisition, MediaPipe inference, and OpenCV rendering execute on a background daemon thread (`CameraThread`).
   - Non-blocking frame queues (`queue.Queue`) deliver frames to Tkinter without GUI lag.

---

## Project Structure

```
AIR TEXT/
├── main.py                     # Tkinter GUI application, calibration UI & event handling
├── camera.py                   # CameraThread, MediaPipe hand tracking & trail rendering
├── recognizer.py               # Oriented point cloud recognizer & CalibrationStore
├── calibration_data/           # Directory storing personal handwriting calibration
│   └── user_profile.json       # Labeled user stroke samples
├── hand_landmarker.task        # MediaPipe HandLandmarker pretrained model asset
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and usage guide
├── test_recognizer.py          # Canonical template self-tests (34/34 = 100%)
├── test_app_flow.py            # End-to-end integration and UI controls test
├── test_validation_suite.py     # 7-phase honest validation suite
├── test_real_airwriting.py     # Real-input benchmark under hand tremor
└── test_hybrid_recognizer.py   # Arbitrary word recognition test
```

---

## Installation & Setup (Windows)

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (or Windows Python 3.14).
- A standard USB or built-in webcam.

### 2. Setup Virtual Environment

Open PowerShell in the project directory:

```powershell
cd "c:\Users\shrav\Desktop\AIR TEXT"
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

*(Packages: `opencv-python`, `mediapipe`, `numpy`, `Pillow`, `scipy`)*

### 4. Run the Application

```powershell
python main.py
```

---

## How to Use AIR TEXT

### Continuous Air Writing Workflow

1. Click **`Start Camera`** (or press keyboard shortcut `w`). The live feed appears on the left with the green camera badge.
2. Click **`Start Writing`** once for the word or phrase.
3. Position your hand inside the **Air Writing Zone**:
   - **To Draw (Pen Down)**: Point clearly with your **index finger** while folding your other fingers into a light fist. Move your hand to draw the letter. A vivid blue line will follow your fingertip.
   - **To Reposition / Pen Up**: Open your hand or lift your middle finger. The tracking dot turns amber (**Hovering**), allowing you to reposition your hand without leaving a mark (essential for multi-stroke letters like `H`, `E`, `T`, `A`).
4. Draw letters from left to right in the same writing session. Lift your index finger between strokes to reposition without drawing a connecting line.
5. Press **`Space`** while writing to separate words. Press **Enter** or click **`Finish Writing`** once the word or phrase is complete.
6. The recognized text is appended to the **Recognized Text** field. If the last letter is unclear, it is left out and candidate suggestions appear; choose one or press **`R`** to retry.
7. Use **`Backspace`** to delete a character, or click inside the text field to edit directly.

### Using Personal Calibration Mode

If your personal drawing style differs from standard print templates:
1. Click the **`Personal Calibration`** button in the top-right of the header.
2. Select a target letter (starts with `H`, `V`, `M`, `E`, `L`, `O`).
3. Click **`Start Drawing`**, draw the letter in the air, and click **`Finish & Save Sample`**.
4. Record 2–3 samples per letter.
5. Click **`Evaluate Calibration`** to run a held-out test validating that your samples generalize accurately.
6. Calibration profiles are automatically saved to `calibration_data/user_profile.json` and persist across restarts.

### Interactive Canvas Mode (No Webcam Required)
If running in an environment without a physical webcam:
1. Click **`Start Camera`** — the application automatically enters **Canvas Draw Mode**.
2. Click **`Start Writing`**.
3. Click and drag the mouse inside the video panel to draw letters directly on the canvas.
4. Click **`Finish Writing`** to recognize.

---

## Automated Verification & Test Commands

Run all automated test suites using the regression runner:

```powershell
python scratch/run_all_tests.py
```

Or execute individual test suites:

```powershell
# 1. Canonical Template Self-Test (verifies all 34 reference exemplars)
python test_recognizer.py

# 2. End-to-End Application Flow & UI State Machine Test
python test_app_flow.py

# 3. 7-Phase Rigorous Validation Suite (Normalization, Multi-stroke, H/V/M confusion, HELLO entry, Calibration)
python test_validation_suite.py

# 4. Realistic Noise & Continuous Sentence Benchmark
python test_real_airwriting.py

# 5. Hybrid Arbitrary Word Recognition Test
python test_hybrid_recognizer.py

# 6. Comprehensive Benchmark (Confusion Matrix, CER, WER, Latency)
python test_comprehensive_eval.py
```

### Measured Benchmark Results

| Test Suite | Metric | Result |
| :--- | :--- | :--- |
| `test_recognizer.py` | Canonical Template Self-Recognition | **34/34 (100.0%)** |
| `test_app_flow.py` | UI Controls, CameraThread & Multi-stroke | **100% PASS** |
| `test_validation_suite.py` | Multi-stroke Invariance (Order/Direction) | **100% PASS** |
| `test_validation_suite.py` | H vs V vs M Confusion Matrix (N=30) | **Zero Cross-Confusion (100% accuracy)** |
| `test_validation_suite.py` | Letter-by-Letter "HELLO" Entry | **"HELLO" Assembled Correctly** |
| `test_validation_suite.py` | Personal Calibration Held-Out Generalization | **100% PASS** |
| `test_real_airwriting.py` | Arbitrary Words (CAT, DOG, SUN, AIR, CODE, PYTHON) | **100% EXACT MATCH** |
| `test_real_airwriting.py` | Continuous Sentence ("AIR TEXT") | **EXACT MATCH** |
| `test_comprehensive_eval.py` | 26-Letter Macro Accuracy / F1-Score (N=520) | **98.1% Accuracy / 98.1% F1** |
| `test_comprehensive_eval.py` | Acceptance Words CER (9 Words) | **0.00% Error (100% Exact Match)** |
| `test_comprehensive_eval.py` | Acceptance Sentences WER (4 Sentences) | **0.00% Error (100% Exact Match)** |
| `test_comprehensive_eval.py` | Inference Latency | **2.60 ms per character (~385 chars/s)** |

---

## Real-World Usage Notes & Limitations

- **Lighting**: Ensure adequate, even lighting on your hand. Heavy backlighting (e.g. sitting in front of a bright window) may reduce MediaPipe landmark confidence.
- **Hand Positioning**: Keep your hand approximately 1.5 to 3 feet from the webcam so the entire hand remains visible inside the **Air Writing Zone**.
- **Movement Pace**: Air writing is most reliable when drawn at a natural, steady pace rather than extremely fast flicks.
- **Headless Sandbox Limitation**: In automated CI/CD or headless testing environments without a physical camera or video display, automated tests use synthetic realistic trajectories or simulated canvas drawing. Physical webcam trials should be tested locally on your machine with `python main.py`.
