"""
test_regression_hi.py
=====================
Regression tests for the specific real-world recognition failures:
  1. H drawn with slanted stems (was: recognized as V/T/S/M at low confidence)
  2. I drawn with top + bottom serif bars (was: recognized as G at 38%)
  3. Continuous single-stroke word "Hello" segmentation (was: A? at 47%)
  4. has_two_stems now uses outer 4-bin clusters, not just bin-0 and bin-15

Run:
  python test_regression_hi.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import recognizer
from recognizer import (
    AirWritingRecognizer,
    extract_features,
    _segment_continuous_stroke,
    _line,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

rec = AirWritingRecognizer(confidence_threshold=0.40)


def _make_H_slanted(cx=320, cy=240, w=90, h=140, slant_px=12):
    """Capital H with naturally slanted stems (human air-writing style)."""
    # Left stem: top-left to bottom-right (slight rightward slant)
    left_stem = [(cx - w//2 + int(slant_px * t), cy - h//2 + int(h * t))
                 for t in np.linspace(0, 1, 20)]
    # Right stem: top-right to bottom-right (slight rightward slant)
    right_stem = [(cx + w//2 + int(slant_px * t), cy - h//2 + int(h * t))
                  for t in np.linspace(0, 1, 20)]
    # Crossbar at mid-height
    crossbar = [(cx - w//2 + int(slant_px * 0.5) + int(w * t), cy)
                for t in np.linspace(0, 1, 15)]
    return [left_stem, right_stem, crossbar]


def _make_I_serif(cx=320, cy=240, hw=45, h=140):
    """Capital I with top bar + vertical stem + bottom bar (serif style)."""
    top_bar    = [(cx - hw + int(2*hw*t), cy - h//2) for t in np.linspace(0, 1, 12)]
    stem       = [(cx, cy - h//2 + int(h*t)) for t in np.linspace(0, 1, 20)]
    bottom_bar = [(cx - hw + int(2*hw*t), cy + h//2) for t in np.linspace(0, 1, 12)]
    return [top_bar, stem, bottom_bar]


def _make_I_simple(cx=320, cy=240, h=140):
    """Capital I as a bare vertical stroke (simplest form)."""
    return [[(cx, cy - h//2 + int(h * t)) for t in np.linspace(0, 1, 25)]]


def _make_H_upright(cx=320, cy=240, w=90, h=140):
    """Perfect upright H (no slant)."""
    left_stem  = [(cx - w//2, cy - h//2 + int(h * t)) for t in np.linspace(0, 1, 20)]
    right_stem = [(cx + w//2, cy - h//2 + int(h * t)) for t in np.linspace(0, 1, 20)]
    crossbar   = [(cx - w//2 + int(w * t), cy) for t in np.linspace(0, 1, 15)]
    return [left_stem, right_stem, crossbar]


def _make_hello_continuous():
    """
    Simulate cursive "Hello" written as a single continuous stroke
    with no pen lifts. The x-axis progresses continuously left-to-right
    and covers about 300px wide, 100px tall.
    """
    pts = []
    # Rough cursive trace: H shape, then loop, then loop, loop, lowercase-o
    # Drawn as rough single stroke with continuous x progression
    xs = np.linspace(60, 360, 300)
    # Create a wavy y that simulates cursive letter shapes
    ys = (100 + 50 * np.sin(xs * 0.08) + 30 * np.sin(xs * 0.025) + 
           10 * np.sin(xs * 0.20))
    pts = [(float(xs[i]), float(ys[i])) for i in range(len(xs))]
    return [pts]  # single continuous stroke


# ─────────────────────────── Tests ──────────────────────────────────────────

results = []


def check(name, got_letter, expected, got_conf, min_conf=0.40):
    ok = (got_letter == expected) and (got_conf >= min_conf)
    sym = PASS if ok else FAIL
    print(f"  {sym} {name}: got={got_letter!r} ({got_conf:.1%}) expected={expected!r} min_conf={min_conf:.0%}")
    results.append((name, ok))
    return ok


# ── Test 1: Upright H ───────────────────────────────────────────────────────
print("\n[1] Upright H (3 strokes, no slant)")
strokes = _make_H_upright()
letter, conf, top3, uncertain = rec.recognize_character(strokes)
check("H upright", letter, "H", conf, 0.45)
print(f"     top3={top3}")

# ── Test 2: Slanted H (the actual user failure) ─────────────────────────────
print("\n[2] Slanted H (slant=12px like real webcam)")
strokes = _make_H_slanted(slant_px=12)
letter, conf, top3, uncertain = rec.recognize_character(strokes)
check("H slanted +12px", letter, "H", conf, 0.38)
print(f"     top3={top3}")

# ── Test 3: Heavier slant H ─────────────────────────────────────────────────
print("\n[3] Heavily slanted H (slant=22px)")
strokes = _make_H_slanted(slant_px=22)
letter, conf, top3, uncertain = rec.recognize_character(strokes)
check("H slanted +22px", letter, "H", conf, 0.30)
print(f"     top3={top3}")

# ── Test 4: Serif I ──────────────────────────────────────────────────────────
print("\n[4] Serif I (top bar + stem + bottom bar)")
strokes = _make_I_serif()
letter, conf, top3, uncertain = rec.recognize_character(strokes)
check("I serif", letter, "I", conf, 0.38)
print(f"     top3={top3}")

# ── Test 5: Simple vertical I ───────────────────────────────────────────────
print("\n[5] Simple vertical I (bare line)")
strokes = _make_I_simple()
letter, conf, top3, uncertain = rec.recognize_character(strokes)
check("I simple vertical", letter, "I", conf, 0.40)
print(f"     top3={top3}")

# ── Test 6: has_two_stems uses outer bins ───────────────────────────────────
print("\n[6] has_two_stems: slanted H should detect two stems via outer-bin clusters")
strokes = _make_H_slanted(slant_px=15)
feat = extract_features(strokes)
left_mass  = float(np.sum(feat['h_proj'][:4]))
right_mass = float(np.sum(feat['h_proj'][-4:]))
has_two = (left_mass > 0.18 and right_mass > 0.18 and feat['aspect'] > 0.35)
ok = has_two
sym = PASS if ok else FAIL
print(f"  {sym} two_stems: left_bin_mass={left_mass:.3f} right_bin_mass={right_mass:.3f} aspect={feat['aspect']:.2f} has_two_stems={has_two}")
results.append(("has_two_stems slanted H", ok))

# ── Test 7: H not confused with V ───────────────────────────────────────────
print("\n[7] H must not be misidentified as V or M")
for trial in range(3):
    strokes = _make_H_slanted(slant_px=10 + trial * 5)
    letter, conf, _, _ = rec.recognize_character(strokes)
    ok = letter == "H"
    sym = PASS if ok else FAIL
    print(f"  {sym} H≠V≠M trial {trial+1}: got={letter!r} ({conf:.1%})")
    results.append((f"H not V/M trial {trial+1}", ok))

# ── Test 8: I not confused with G ───────────────────────────────────────────
print("\n[8] Serif I must not be misidentified as G")
for trial in range(3):
    # Slight width variation
    strokes = _make_I_serif(hw=35 + trial * 10)
    letter, conf, _, _ = rec.recognize_character(strokes)
    ok = letter == "I"
    sym = PASS if ok else FAIL
    print(f"  {sym} I≠G trial {trial+1}: got={letter!r} ({conf:.1%})")
    results.append((f"I not G trial {trial+1}", ok))

# ── Test 9: Continuous stroke segmentation ──────────────────────────────────
print("\n[9] _segment_continuous_stroke: wide single stroke should split")
# Make a wide synthetic stroke (5 character widths wide)
t = np.linspace(0, 1, 200)
x = 60 + 300 * t
# Wavy y simulating multiple letters
y = 240 + 50 * np.sin(t * 15 * np.pi)
pts = np.column_stack([x, y])
segs = _segment_continuous_stroke(pts, min_points_per_seg=10, min_width_ratio=0.10)
ok = len(segs) >= 2
sym = PASS if ok else FAIL
print(f"  {sym} Segmented into {len(segs)} sub-strokes (expected >= 2)")
results.append(("continuous stroke splits", ok))

# ── Test 10: V still recognized correctly ───────────────────────────────────
print("\n[10] V recognition (not broken by H/I changes)")
v_strokes = [
    [(320 - 60 + int(60*t), 170 + int(120*t)) for t in np.linspace(0, 1, 20)],
    [(320 + 60 - int(60*t), 170 + int(120*t)) for t in np.linspace(0, 1, 20)],
]
letter, conf, top3, _ = rec.recognize_character(v_strokes)
check("V still correct", letter, "V", conf, 0.35)
print(f"     top3={top3}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
n_pass = sum(1 for _, ok in results if ok)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} tests passed")
for name, ok in results:
    sym = "✓" if ok else "✗"
    print(f"  {sym} {name}")

if n_pass < n_total:
    sys.exit(1)

