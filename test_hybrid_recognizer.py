"""
test_hybrid_recognizer.py
=========================
Prototype and benchmark of the Hybrid Recognizer:
- Direction-invariant DTW (handles forward and backward writing)
- Point Cloud / Chamfer distance (invariant to stroke order & direction)
- Aspect ratio and topology penalties (loop vs non-loop, aspect ratio)
- Removal of any hardcoded phrase matching
- Evaluation on arbitrary words and letters with realistic human hand noise
"""

import math
import numpy as np
from scipy.spatial.distance import cdist

def _line(x0, y0, x1, y1, n=20):
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])

def _circle(cx, cy, r, start=0, end=2 * math.pi, n=40, clockwise=False):
    angles = np.linspace(start, end, n)
    if clockwise:
        angles = angles[::-1]
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _resample(pts: np.ndarray, n: int = 48) -> np.ndarray:
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) == 1 else np.array([0.0, 0.0]), (n, 1))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total = seg_lens.sum()
    if total < 1e-9:
        return np.tile(pts[0], (n, 1))
    cumlen = np.concatenate([[0], np.cumsum(seg_lens)])
    target = np.linspace(0, total, n)
    xs = np.interp(target, cumlen, pts[:, 0])
    ys = np.interp(target, cumlen, pts[:, 1])
    return np.column_stack([xs, ys])

def _normalise(strokes, n_points=48):
    if not strokes:
        return np.zeros((n_points, 2))
    combined = np.vstack(strokes)
    mn = combined.min(axis=0)
    mx = combined.max(axis=0)
    rng = mx - mn
    max_rng = max(rng[0], rng[1], 1e-6)

    # Aspect ratio check
    aspect = rng[0] / max_rng
    if rng[0] < 0.20 * rng[1]:
        scale_x = max_rng
        scale_y = max_rng
    elif rng[1] < 0.20 * rng[0]:
        scale_x = max_rng
        scale_y = max_rng
    else:
        scale_x = max(rng[0], 1e-6)
        scale_y = max(rng[1], 1e-6)

    normed = np.zeros_like(combined, dtype=float)
    normed[:, 0] = (combined[:, 0] - mn[0]) / scale_x * 2.0 - 1.0
    normed[:, 1] = (combined[:, 1] - mn[1]) / scale_y * 2.0 - 1.0
    return _resample(normed, n_points), aspect

def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    d = cdist(a, b)
    n, m = d.shape
    dp = np.zeros((n, m), dtype=float)
    dp[0, 0] = d[0, 0]
    for i in range(1, n): dp[i, 0] = dp[i - 1, 0] + d[i, 0]
    for j in range(1, m): dp[0, j] = dp[0, j - 1] + d[0, j]
    for i in range(1, n):
        r, pr = dp[i], dp[i - 1]
        for j in range(1, m):
            r[j] = d[i, j] + min(pr[j], r[j - 1], pr[j - 1])
    return float(dp[-1, -1])

def _point_cloud_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    d = cdist(pts1, pts2)
    return float((d.min(axis=1).mean() + d.min(axis=0).mean()) / 2.0)

# Build baseline templates A-Z
TEMPLATES = {}
TEMPLATES['A'] = [np.vstack([_line(0, 1, 0.5, -1, 15), _line(0.5, -1, 1, 1, 15)]), _line(0.2, 0.1, 0.8, 0.1, 10)]
TEMPLATES['B'] = [_line(0, -1, 0, 1, 15), np.vstack([_circle(0.5, -0.5, 0.5, -math.pi/2, math.pi/2, 15), _circle(0.6, 0.5, 0.5, -math.pi/2, math.pi/2, 15)])]
TEMPLATES['C'] = [_circle(0.5, 0, 0.9, math.pi * 0.25, math.pi * 1.75, 40)]
TEMPLATES['D'] = [_line(0, -1, 0, 1, 15), _circle(0.3, 0, 0.95, -math.pi/2, math.pi/2, 30)]
TEMPLATES['E'] = [_line(0, -1, 0, 1, 15), _line(0, -1, 0.9, -1, 10), _line(0, 0, 0.7, 0, 10), _line(0, 1, 0.9, 1, 10)]
TEMPLATES['F'] = [_line(0, -1, 0, 1, 15), _line(0, -1, 0.9, -1, 10), _line(0, 0, 0.7, 0, 10)]
TEMPLATES['G'] = [np.vstack([_circle(0.5, 0, 0.9, math.pi * 0.25, math.pi * 1.75, 35), _line(0.5 + 0.9 * math.cos(math.pi * 0.25), 0.9 * math.sin(math.pi * 0.25), 0.5, 0, 10)])]
TEMPLATES['H'] = [_line(0, -1, 0, 1, 15), _line(0, 0, 1, 0, 10), _line(1, -1, 1, 1, 15)]
TEMPLATES['I'] = [_line(0.5, -1, 0.5, 1, 20)]
TEMPLATES['J'] = [np.vstack([_line(0.8, -1, 0.8, 0.6, 15), _circle(0.3, 0.6, 0.5, 0, math.pi, 20)])]
TEMPLATES['K'] = [_line(0, -1, 0, 1, 15), _line(0, 0, 1, -1, 12), _line(0, 0, 1, 1, 12)]
TEMPLATES['L'] = [np.vstack([_line(0, -1, 0, 1, 15), _line(0, 1, 1, 1, 12)])]
TEMPLATES['M'] = [np.vstack([_line(0, 1, 0, -1, 12), _line(0, -1, 0.5, 0.2, 12), _line(0.5, 0.2, 1, -1, 12), _line(1, -1, 1, 1, 12)])]
TEMPLATES['N'] = [np.vstack([_line(0, 1, 0, -1, 15), _line(0, -1, 1, 1, 15), _line(1, 1, 1, -1, 15)])]
TEMPLATES['O'] = [_circle(0.5, 0, 0.9, 0, 2 * math.pi, 45)]
TEMPLATES['P'] = [_line(0, -1, 0, 1, 15), _circle(0.5, -0.5, 0.5, -math.pi/2, math.pi/2, 20)]
TEMPLATES['Q'] = [_circle(0.5, 0, 0.85, 0, 2 * math.pi, 40), _line(0.7, 0.4, 1.1, 1.0, 10)]
TEMPLATES['R'] = [_line(0, -1, 0, 1, 15), _circle(0.5, -0.5, 0.5, -math.pi/2, math.pi/2, 20), _line(0.3, 0.1, 1, 1, 12)]
TEMPLATES['S'] = [np.vstack([_circle(0.5, -0.4, 0.5, math.pi * 0.1, math.pi * 1.1, 20), _circle(0.5, 0.4, 0.5, math.pi * 1.1, math.pi * 2.1, 20)])]
TEMPLATES['T'] = [_line(0, -1, 1, -1, 12), _line(0.5, -1, 0.5, 1, 15)]
TEMPLATES['U'] = [np.vstack([_line(0, -1, 0, 0.6, 12), _circle(0.5, 0.6, 0.5, math.pi, 2 * math.pi, 20), _line(1, 0.6, 1, -1, 12)])]
TEMPLATES['V'] = [np.vstack([_line(0, -1, 0.5, 1, 15), _line(0.5, 1, 1, -1, 15)])]
TEMPLATES['W'] = [np.vstack([_line(0, -1, 0.25, 1, 12), _line(0.25, 1, 0.5, 0, 12), _line(0.5, 0, 0.75, 1, 12), _line(0.75, 1, 1, -1, 12)])]
TEMPLATES['X'] = [_line(0, -1, 1, 1, 15), _line(1, -1, 0, 1, 15)]
TEMPLATES['Y'] = [np.vstack([_line(0, -1, 0.5, 0, 12), _line(1, -1, 0.5, 0, 12)]), _line(0.5, 0, 0.5, 1, 12)]
TEMPLATES['Z'] = [np.vstack([_line(0, -1, 1, -1, 12), _line(1, -1, 0, 1, 15), _line(0, 1, 1, 1, 12)])]

NORM_TEMPLATES = {k: _normalise(v)[0] for k, v in TEMPLATES.items()}

# Closed loop letters
LOOP_LETTERS = set(['O', 'D', 'P', 'B', 'Q', 'A', 'R'])

def evaluate_candidate(cand_strokes):
    cand_norm, cand_aspect = _normalise(cand_strokes)

    # Check candidate closed-loop feature
    has_loop = False
    for s in cand_strokes:
        if len(s) >= 8:
            dist_ends = np.linalg.norm(s[0] - s[-1])
            bbox = np.ptp(s, axis=0)
            diag = max(np.linalg.norm(bbox), 1e-6)
            if dist_ends / diag < 0.28:
                has_loop = True
                break

    scores = {}
    for letter, tmpl_pts in NORM_TEMPLATES.items():
        # 1. Point Cloud Distance (scale [0, 2])
        pc_dist = _point_cloud_distance(cand_norm, tmpl_pts)

        # 2. Direction-invariant DTW (forward and reverse)
        dtw_fwd = _dtw_distance(cand_norm, tmpl_pts) / 35.0
        dtw_rev = _dtw_distance(cand_norm, tmpl_pts[::-1]) / 35.0
        dtw_score = min(dtw_fwd, dtw_rev)

        # Combine PC and DTW
        cost = 0.55 * pc_dist + 0.45 * dtw_score

        # 3. Topology / loop penalty
        letter_has_loop = letter in LOOP_LETTERS
        if has_loop and not letter_has_loop:
            cost += 0.25
        elif not has_loop and letter in ('O', 'D', 'Q', 'B'):
            cost += 0.18

        # 4. Aspect ratio penalty for I
        if letter == 'I' and cand_aspect > 0.45:
            cost += 0.40
        elif letter != 'I' and cand_aspect < 0.22:
            cost += 0.35

        scores[letter] = cost

    sorted_letters = sorted(scores, key=scores.get)
    best_l = sorted_letters[0]
    best_cost = scores[best_l]
    second_cost = scores[sorted_letters[1]]

    # Confidence calculation based on cost and margin to 2nd candidate
    margin = (second_cost - best_cost) / max(best_cost, 1e-6)
    conf = max(0.0, min(1.0, 1.0 - best_cost * 1.4 + 0.2 * min(margin, 1.0)))

    return best_l, conf, sorted_letters[:3]

# Test with realistic human noisy input for arbitrary letters
np.random.seed(42)
test_letters = ['C', 'A', 'T', 'D', 'O', 'G', 'S', 'U', 'N', 'P', 'Y', 'T', 'H', 'O', 'N']
results = []
for target in test_letters:
    strokes = [s + np.random.normal(0, 0.05, s.shape) for s in TEMPLATES[target]]
    got, conf, top3 = evaluate_candidate(strokes)
    ok = (got == target)
    results.append((target, got, conf, ok, top3))

print("=== Evaluation of Arbitrary Words with Realistic Noisy Input ===")
print(f"{'Target':>6}  {'Recognized':>10}  {'Conf':>6}  {'OK?':>5}  {'Top-3 Candidates'}")
print("-" * 55)
for t, g, c, ok, top3 in results:
    flag = "OK" if ok else "FAIL"
    print(f"{t:>6}  {g:>10}  {c:6.2f}  {flag:>5}  {top3}")

acc = sum(r[3] for r in results) / len(results) * 100
print(f"\nAccuracy across arbitrary letters with noise: {acc:.1f}%")
