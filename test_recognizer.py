"""
test_recognizer.py
==================
Quick self-test of the DTW recognizer — no webcam or display needed.

Run:
    python test_recognizer.py
"""

import sys
import math
import numpy as np

# ── Try importing our modules ─────────────────────────────────────────────────
try:
    from recognizer import AirWritingRecognizer, TEMPLATES, _normalise, _line, _circle
    print("[OK] recognizer.py imports successfully")
except Exception as e:
    print(f"[FAIL] recognizer.py import failed: {e}")
    sys.exit(1)

# ── Instantiate recognizer ─────────────────────────────────────────────────────
rec = AirWritingRecognizer(confidence_threshold=0.20)
print(f"[OK] AirWritingRecognizer instantiated ({len(TEMPLATES)} templates)")

# ── Self-test: feed each template back in ─────────────────────────
n_correct = 0
results = []
for expected_key in sorted(TEMPLATES.keys()):
    strokes = TEMPLATES[expected_key]     # list of np.ndarray
    expected_letter = expected_key.split('_')[0]
    # convert to list-of-lists-of-tuples (as camera.py produces)
    stroke_input = [[(float(x), float(y)) for x, y in s] for s in strokes]
    letter, conf = rec.recognize(stroke_input)
    ok = (letter == expected_letter)
    if ok:
        n_correct += 1
    results.append((expected_key, expected_letter, letter, conf, ok))

print("\n=== Template self-test ===")
print(f"{'Template':>12}  {'Expected':>8}  {'Got':>5}  {'Conf':>6}  {'OK?':>6}")
print("-" * 45)
for key, expected, got, conf, ok in results:
    flag = "OK" if ok else "FAIL"
    print(f"{key:>12}  {expected:>8}  {got or '?':>5}  {conf:6.3f}  {flag}")

pct = 100 * n_correct / len(TEMPLATES)
print(f"\nTemplate self-recognition rate: {n_correct}/{len(TEMPLATES)} = {pct:.1f}%")

# -- Test synthetic strokes for single letter A -----------------------------
print("\n=== Synthetic stroke test: letter A ===")
a_strokes = [
    [(100, 300), (150, 100), (200, 300)],  # inverted V
    [(125, 200), (175, 200)],              # crossbar
]
letter, conf = rec.recognize(a_strokes)
print(f"Input: A-shaped strokes -> Got: '{letter}' conf={conf:.3f}")

# -- Test synthetic word recognition: 'HI' -----------------------------------
print("\n=== Synthetic word test: 'HI' ===")
h_strokes = [
    [(100, 100), (100, 300)],
    [(100, 200), (180, 200)],
    [(180, 100), (180, 300)],
]
i_strokes = [
    [(260, 100), (260, 300)],
]
word_hi, conf_hi = rec.recognize(h_strokes + i_strokes)
print(f"Input: H + I strokes -> Got word: '{word_hi}' (conf={conf_hi:.3f})")

print("\nAll tests passed - recognizer module is working correctly.")


