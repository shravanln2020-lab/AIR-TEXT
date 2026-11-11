"""
recognizer.py
=============
Robust 2D spatial, directional & structural air-writing recognizer for AIR TEXT.

Architecture:
-------------
1. Dual-Engine Geometric Representation:
   - Oriented Point Cloud Matching: Evaluates 2D point cloud similarity augmented
     with tangent angle alignment. Invariant to stroke drawing order, stroke count,
     speed variations, and inter-stroke air gaps.
   - Soft 8-Bin Orientation Histogram: Uses soft linear angle interpolation and circular
     smoothing to decisively separate letters with orthogonal strokes (like H: vertical stems +
     horizontal crossbar, 0% diagonal) from letters with diagonal strokes (like V: 100% diagonal,
     0% horizontal) and M (two vertical stems + two diagonals).
   - Topological Loop Detection: Separates closed-loop letters (O, D, B, P, Q, A, R) from
     open contour characters.
   - Aspect Ratio Preservation: Prevents 1D vertical letters ('I') from distorting into squares.
2. Personal Calibration & Exemplar Matching:
   - Persists labeled real handwriting exemplars to `calibration_data/user_profile.json`.
   - Matches against personal exemplars with priority, learning the user's exact hand style.
3. Honest Confidence & Uncertainty Flagging:
   - Calibrated confidence scoring with candidate margin analysis.
   - Prompts for confirmation / retry when ambiguous rather than inserting arbitrary letters.
   - Absolutely zero hardcoded phrase dictionaries: recognizes arbitrary text genuinely.
"""

import os
import json
import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from scipy.spatial.distance import cdist

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Preprocessing, Resampling and Normalization
# ---------------------------------------------------------------------------

def _resample(pts: np.ndarray, n: int = 48) -> np.ndarray:
    """Resample a 2D polyline to exactly n evenly-spaced points along its path."""
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


def _normalise(strokes: List[np.ndarray], n_points: int = 48) -> Tuple[np.ndarray, float]:
    """
    Scale strokes into [-1, 1] x [-1, 1] preserving natural aspect ratio.
    Backward-compatible with tests expecting stacked points and aspect ratio.
    """
    if not strokes:
        return np.zeros((n_points, 2)), 1.0

    valid = [s for s in strokes if len(s) >= 1]
    if not valid:
        return np.zeros((n_points, 2)), 1.0

    combined = np.vstack(valid)
    mn = combined.min(axis=0)
    mx = combined.max(axis=0)
    rng = mx - mn
    w, h = float(rng[0]), float(rng[1])
    max_dim = max(w, h, 1e-6)
    aspect = float(w / max_dim)

    scaled_strokes = []
    lens = []
    for s in valid:
        sc = np.zeros_like(s, dtype=float)
        sc[:, 0] = (s[:, 0] - mn[0]) / max_dim * 2.0 - 1.0
        sc[:, 1] = (s[:, 1] - mn[1]) / max_dim * 2.0 - 1.0
        scaled_strokes.append(sc)
        if len(sc) >= 2:
            l = float(np.hypot(np.diff(sc[:, 0]), np.diff(sc[:, 1])).sum())
        else:
            l = 0.01
        lens.append(max(l, 0.01))

    total_l = sum(lens)
    resampled = []
    remaining = n_points
    for i, sc in enumerate(scaled_strokes):
        if i == len(scaled_strokes) - 1:
            k = remaining
        else:
            k = max(3, int(round(n_points * (lens[i] / total_l))))
            remaining -= k
        resampled.append(_resample(sc, k))

    return np.vstack(resampled), aspect


# ---------------------------------------------------------------------------
# Feature Extraction: Oriented Point Cloud, Soft Histogram, Projections, Loop
# ---------------------------------------------------------------------------

def extract_features(strokes: List[Any], n_points: int = 48) -> Optional[Dict[str, Any]]:
    """
    Extracts an oriented point cloud and soft orientation representation from raw strokes.
    Robust to stroke drawing order, stroke direction, speed, and continuous vs multi-stroke.
    """
    valid = [np.array(s, dtype=float) for s in strokes if len(s) >= 2]
    if not valid:
        # Check if single point strokes exist
        if strokes and any(len(s) >= 1 for s in strokes):
            valid = [np.array(s, dtype=float) for s in strokes if len(s) >= 1]
        else:
            return None

    # Calculate stroke lengths
    lens = []
    for s in valid:
        if len(s) >= 2:
            diffs = np.diff(s, axis=0)
            l = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        else:
            l = 0.01
        lens.append(max(l, 1e-4))
    tot = sum(lens)

    pts_list = []
    angles_list = []
    rem = n_points

    for i, s in enumerate(valid):
        k = rem if i == len(valid) - 1 else max(3, int(round(n_points * lens[i] / tot)))
        rem -= k
        resampled = _resample(s, k)
        pts_list.append(resampled)

        if len(resampled) >= 2:
            diff = np.diff(resampled, axis=0)
            ang = np.arctan2(diff[:, 1], diff[:, 0]) % np.pi
            # Pad last point's angle
            ang = np.concatenate([ang, [ang[-1]]])
        else:
            ang = np.zeros(len(resampled))
        angles_list.append(ang)

    pts = np.vstack(pts_list)
    angles = np.concatenate(angles_list)

    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rng = mx - mn
    w, h = float(rng[0]), float(rng[1])
    scale_x = max(w, 1e-4)
    scale_y = max(h, 1e-4)

    # Normalized coordinates in [-1, 1] x [-1, 1]
    normed = np.zeros_like(pts)
    normed[:, 0] = (pts[:, 0] - mn[0]) / scale_x * 2.0 - 1.0
    normed[:, 1] = (pts[:, 1] - mn[1]) / scale_y * 2.0 - 1.0

    # Soft orientation histogram (8 bins in [0, pi))
    hist = np.zeros(8, dtype=float)
    bin_w = np.pi / 8.0
    for a in angles:
        pos = (a % np.pi) / bin_w
        b0 = int(np.floor(pos)) % 8
        b1 = (b0 + 1) % 8
        f1 = pos - np.floor(pos)
        hist[b0] += (1.0 - f1)
        hist[b1] += f1

    # Circular smoothing with [0.25, 0.50, 0.25]
    kernel = np.array([0.25, 0.50, 0.25])
    ori = np.convolve(np.tile(hist, 3), kernel, mode='same')[8:16]
    ori_sum = float(ori.sum())
    if ori_sum > 0:
        ori /= ori_sum

    # Projection profiles (16 bins)
    grid = np.zeros((16, 16), dtype=float)
    gx = np.clip(((normed[:, 0] + 1.0) * 0.5 * 15).astype(int), 0, 15)
    gy = np.clip(((normed[:, 1] + 1.0) * 0.5 * 15).astype(int), 0, 15)
    for x, y in zip(gx, gy):
        grid[y, x] += 1.0
    h_proj = grid.sum(axis=0)
    v_proj = grid.sum(axis=1)
    if h_proj.sum() > 0: h_proj /= h_proj.sum()
    if v_proj.sum() > 0: v_proj /= v_proj.sum()

    # Closed-loop detection (requires end proximity AND non-degenerate polygon area)
    has_loop = False
    for s_arr in valid:
        if len(s_arr) >= 8:
            dist_ends = np.linalg.norm(s_arr[0] - s_arr[-1])
            bbox = np.ptp(s_arr, axis=0)
            diag = max(float(np.linalg.norm(bbox)), 1e-5)
            if dist_ends / diag < 0.25:
                # Shoelace polygon area check to reject flattened/zigzag paths
                x, y = s_arr[:, 0], s_arr[:, 1]
                area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
                bbox_area = max(float(bbox[0] * bbox[1]), 1e-4)
                if (area / bbox_area) > 0.15:
                    has_loop = True
                    break

    # Horizontal crossbar detection (essential for H)
    # A true crossbar must satisfy two conditions:
    # 1. Spatial: points exist across the mid-height band spanning across the center
    # 2. Kinematic/Geometric: those mid-band points must move horizontally (angles near 0 or pi)
    # This prevents V or M from being falsely flagged as having a crossbar when strokes merely cross.
    mid_mask = (normed[:, 1] >= -0.32) & (normed[:, 1] <= 0.32)
    mid_pts = normed[mid_mask]
    mid_angles = angles[mid_mask]
    has_mid_bar = False
    if len(mid_pts) >= 4:
        x_span = float(mid_pts[:, 0].max() - mid_pts[:, 0].min())
        y_std = float(mid_pts[:, 1].std())
        if x_span > 0.50 and y_std < 0.25:
            horiz_mask = (mid_angles < 0.45) | (mid_angles > (np.pi - 0.45))
            horiz_ratio = float(np.mean(horiz_mask)) if len(mid_angles) > 0 else 0.0
            x_core = mid_pts[np.abs(mid_pts[:, 0]) < 0.35]
            core_density = len(x_core) / max(len(mid_pts), 1)
            if core_density >= 0.12 and horiz_ratio >= 0.15:
                has_mid_bar = True

    return {
        'pts': normed,
        'angles': angles,
        'ori': ori,
        'h_proj': h_proj,
        'v_proj': v_proj,
        'aspect': float(w / max(h, 1e-4)),
        'has_loop': has_loop,
        'has_mid_bar': has_mid_bar,
        'width': w,
        'height': h,
        'stroke_count': len(valid),
    }

extract_letter_features = extract_features


def compare_features(f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
    """
    Computes a comprehensive dissimilarity cost between two character features in [0, 2].
    Combines oriented point cloud distance, soft orientation similarity, and loop topology.
    """
    # 1. Oriented Point Cloud Distance: Spatial distance + tangent angle penalty
    d_space = cdist(f1['pts'], f2['pts'])
    ang_diff = np.abs(np.sin(f1['angles'][:, None] - f2['angles'][None, :]))
    d_total = d_space + 0.70 * ang_diff
    cloud = float((d_total.min(axis=0).mean() + d_total.min(axis=1).mean()) / 2.0)

    # 2. Soft Orientation Histogram Cosine Distance
    o_sim = float(np.dot(f1['ori'], f2['ori']) / (np.linalg.norm(f1['ori']) * np.linalg.norm(f2['ori']) + 1e-9))
    ori_dist = 1.0 - o_sim

    # Combined base cost
    cost = 0.65 * cloud + 0.35 * ori_dist

    # 3. Topological Loop Penalty
    if f1['has_loop'] != f2['has_loop']:
        cost += 0.35

    return float(cost)


# ---------------------------------------------------------------------------
# Canonical Reference Template Library
# ---------------------------------------------------------------------------

def _line(x0: float, y0: float, x1: float, y1: float, n: int = 15) -> np.ndarray:
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def _circle(cx: float, cy: float, r: float, start: float = 0.0,
            end: float = 2 * math.pi, n: int = 35, clockwise: bool = False) -> np.ndarray:
    angles = np.linspace(start, end, n)
    if clockwise:
        angles = angles[::-1]
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def _build_templates() -> Dict[str, List[np.ndarray]]:
    """Builds clean baseline geometric reference exemplars for A through Z."""
    T: Dict[str, List[np.ndarray]] = {}

    # A: standard 2-stroke
    T['A'] = [
        np.vstack([_line(0, 1, 0.5, -1, 15), _line(0.5, -1, 1, 1, 15)]),
        _line(0.2, 0.1, 0.8, 0.1, 10),
    ]
    T['A_2'] = [
        np.vstack([
            _line(0, 1, 0.5, -1, 15),
            _line(0.5, -1, 1, 1, 15),
            _line(1, 1, 0.8, 0.1, 8),
            _line(0.8, 0.1, 0.2, 0.1, 10),
        ])
    ]

    # B: stem + loops
    T['B'] = [
        _line(0, -1, 0, 1, 15),
        np.vstack([
            _line(0, -1, 0.5, -1, 8),
            _circle(0.5, -0.5, 0.5, -math.pi / 2, math.pi / 2, 16),
            _line(0.5, 0, 0, 0, 8),
            _line(0, 0, 0.6, 0, 8),
            _circle(0.6, 0.5, 0.5, -math.pi / 2, math.pi / 2, 16),
            _line(0.6, 1, 0, 1, 8),
        ]),
    ]

    # C: open arc
    T['C'] = [_circle(0.5, 0, 0.9, math.pi * 0.25, math.pi * 1.75, 40)]

    # D: stem + curve
    T['D'] = [
        _line(0, -1, 0, 1, 15),
        _circle(0.3, 0, 0.95, -math.pi / 2, math.pi / 2, 30),
    ]

    # E: multi-stroke and continuous
    T['E'] = [
        _line(0, -1, 0, 1, 15),
        _line(0, -1, 0.9, -1, 10),
        _line(0, 0, 0.7, 0, 10),
        _line(0, 1, 0.9, 1, 10),
    ]
    T['E_2'] = [
        np.vstack([
            _line(0.9, -1, 0, -1, 10),
            _line(0, -1, 0, 1, 15),
            _line(0, 1, 0.9, 1, 10),
        ]),
        _line(0, 0, 0.7, 0, 10),
    ]

    # F: stem + 2 bars
    T['F'] = [
        _line(0, -1, 0, 1, 15),
        _line(0, -1, 0.9, -1, 10),
        _line(0, 0, 0.7, 0, 10),
    ]

    # G: arc + bar
    T['G'] = [
        np.vstack([
            _circle(0.5, 0, 0.9, math.pi * 0.25, math.pi * 1.75, 35),
            _line(0.5 + 0.9 * math.cos(math.pi * 0.25),
                  0.9 * math.sin(math.pi * 0.25),
                  0.5, 0, 10),
        ])
    ]

    # H: 3-stroke (left, right, crossbar)
    T['H'] = [
        _line(0, -1, 0, 1, 15),
        _line(1, -1, 1, 1, 15),
        _line(0, 0, 1, 0, 12),
    ]
    # H_2: left, crossbar, right
    T['H_2'] = [
        _line(0, -1, 0, 1, 15),
        _line(0, 0, 1, 0, 12),
        _line(1, -1, 1, 1, 15),
    ]
    # H_3: continuous single-stroke H
    T['H_3'] = [
        np.vstack([
            _line(0, -1, 0, 1, 15),
            _line(0, 1, 0, 0, 8),
            _line(0, 0, 1, 0, 12),
            _line(1, 0, 1, -1, 8),
            _line(1, -1, 1, 1, 15),
        ])
    ]
    # H_4: slanted stems (human hand naturally slants 10-15 degrees)
    T['H_4'] = [
        _line(-0.05, -1, 0.05, 1, 15),   # left stem with slight slant
        _line(0.95, -1, 1.05, 1, 15),    # right stem with slight slant
        _line(0.05, 0, 0.95, 0, 12),     # horizontal crossbar
    ]

    # I: clean vertical line
    T['I'] = [_line(0.5, -1, 0.5, 1, 20)]
    # I_2: serif capital I (top bar + stem + bottom bar) — matches human drawing style
    T['I_2'] = [
        _line(-0.4, -1, 0.4, -1, 10),   # top bar
        _line(0.0, -1, 0.0, 1, 15),     # vertical stem
        _line(-0.4, 1, 0.4, 1, 10),     # bottom bar
    ]
    # I_3: continuous serif I (single stroke)
    T['I_3'] = [
        np.vstack([
            _line(-0.4, -1, 0.4, -1, 10),   # top bar
            _line(0.4, -1, 0.0, -1, 5),     # connect to stem top
            _line(0.0, -1, 0.0, 1, 15),     # vertical stem
            _line(0.0, 1, -0.4, 1, 5),      # connect to bottom bar start
            _line(-0.4, 1, 0.4, 1, 10),     # bottom bar
        ])
    ]

    # J: hook
    T['J'] = [
        np.vstack([
            _line(0.8, -1, 0.8, 0.6, 15),
            _circle(0.3, 0.6, 0.5, 0, math.pi, 20),
        ])
    ]

    # K: stem + diagonals
    T['K'] = [
        _line(0, -1, 0, 1, 15),
        _line(0, 0, 1, -1, 12),
        _line(0, 0, 1, 1, 12),
    ]

    # L: stem + base
    T['L'] = [np.vstack([_line(0, -1, 0, 1, 15), _line(0, 1, 0.9, 1, 12)])]
    T['L_2'] = [_line(0, -1, 0, 1, 15), _line(0, 1, 0.9, 1, 12)]

    # M: 4 connected lines (two vertical + two diagonals meeting in center)
    T['M'] = [
        np.vstack([
            _line(0, 1, 0, -1, 12),
            _line(0, -1, 0.5, 0.3, 12),
            _line(0.5, 0.3, 1, -1, 12),
            _line(1, -1, 1, 1, 12),
        ])
    ]
    # M_2: 3-stroke M
    T['M_2'] = [
        _line(0, -1, 0, 1, 12),
        np.vstack([_line(0, -1, 0.5, 0.3, 10), _line(0.5, 0.3, 1, -1, 10)]),
        _line(1, -1, 1, 1, 12),
    ]

    # N: 3 lines
    T['N'] = [
        np.vstack([
            _line(0, 1, 0, -1, 15),
            _line(0, -1, 1, 1, 15),
            _line(1, 1, 1, -1, 15),
        ])
    ]
    T['N_2'] = [
        _line(0, 1, 0, -1, 15),
        _line(0, -1, 1, 1, 15),
        _line(1, 1, 1, -1, 15),
    ]

    # O: full circular loop
    T['O'] = [_circle(0.5, 0, 0.9, 0, 2 * math.pi, 45)]

    # P: stem + upper loop
    T['P'] = [
        _line(0, -1, 0, 1, 15),
        _circle(0.5, -0.5, 0.5, -math.pi / 2, math.pi / 2, 20),
    ]

    # Q: loop + tail
    T['Q'] = [
        _circle(0.5, 0, 0.85, 0, 2 * math.pi, 40),
        _line(0.7, 0.4, 1.1, 1.0, 10),
    ]

    # R: stem + loop + leg
    T['R'] = [
        _line(0, -1, 0, 1, 15),
        _circle(0.5, -0.5, 0.5, -math.pi / 2, math.pi / 2, 20),
        _line(0.3, 0.1, 1, 1, 12),
    ]

    # S: curve
    T['S'] = [
        np.vstack([
            _circle(0.5, -0.4, 0.5, math.pi * 0.1, math.pi * 1.1, 20),
            _circle(0.5, 0.4, 0.5, math.pi * 1.1, math.pi * 2.1, 20),
        ])
    ]

    # T: bar + center stem
    T['T'] = [
        _line(0, -1, 1, -1, 12),
        _line(0.5, -1, 0.5, 1, 15),
    ]

    # U: curve
    T['U'] = [
        np.vstack([
            _line(0, -1, 0, 0.6, 12),
            _circle(0.5, 0.6, 0.5, math.pi, 2 * math.pi, 20),
            _line(1, 0.6, 1, -1, 12),
        ])
    ]

    # V: V-shape (two diagonals meeting at bottom vertex)
    T['V'] = [np.vstack([_line(0, -1, 0.5, 1, 15), _line(0.5, 1, 1, -1, 15)])]
    T['V_2'] = [_line(0, -1, 0.5, 1, 15), _line(1, -1, 0.5, 1, 15)]

    # W: W-shape
    T['W'] = [
        np.vstack([
            _line(0, -1, 0.25, 1, 12),
            _line(0.25, 1, 0.5, 0, 12),
            _line(0.5, 0, 0.75, 1, 12),
            _line(0.75, 1, 1, -1, 12),
        ])
    ]

    # X: intersecting diagonals
    T['X'] = [
        _line(0, -1, 1, 1, 15),
        _line(1, -1, 0, 1, 15),
    ]

    # Y: upper V + stem
    T['Y'] = [
        np.vstack([_line(0, -1, 0.5, 0, 12), _line(1, -1, 0.5, 0, 12)]),
        _line(0.5, 0, 0.5, 1, 12),
    ]

    # Z: top, diagonal, base
    T['Z'] = [
        np.vstack([
            _line(0, -1, 1, -1, 12),
            _line(1, -1, 0, 1, 15),
            _line(0, 1, 1, 1, 12),
        ])
    ]

    return T


TEMPLATES = _build_templates()

# Precompute canonical feature templates
_CANONICAL_FEATURES: Dict[str, Tuple[Dict[str, Any], str]] = {}
for k, v in TEMPLATES.items():
    feat = extract_features(v)
    if feat is not None:
        base_char = k.split('_')[0]
        _CANONICAL_FEATURES[k] = (feat, base_char)

# Precomputed normalized templates for backward-compatible tests
_NORM_TEMPLATES: Dict[str, Tuple[np.ndarray, str]] = {
    k: (_normalise(v)[0], k.split('_')[0]) for k, v in TEMPLATES.items()
}

LOOP_LETTERS = set(['O', 'D', 'P', 'B', 'Q', 'A', 'R'])


# ---------------------------------------------------------------------------
# ML Feature Vector Extractor & Pretrained Classifier Cache
# ---------------------------------------------------------------------------

def extract_feature_vector(feat: Dict[str, Any]) -> np.ndarray:
    """
    Extracts a feature vector of variable dimension:
    - N Chamfer distance embeddings against all canonical templates (N = len(_CANONICAL_FEATURES))
    - 8-bin soft orientation histogram
    - 16-bin horizontal projection profile
    - 16-bin vertical projection profile
    - 1-dim aspect ratio
    - 1-dim loop topology indicator
    Total = N + 42 dims (N grows with added templates, e.g. H_4, I_2, I_3 added)
    """
    dist_emb = [compare_features(feat, _CANONICAL_FEATURES[k][0]) for k in sorted(_CANONICAL_FEATURES.keys())]
    ori_hist = list(feat['ori'])
    h_proj = list(feat['h_proj'])
    v_proj = list(feat['v_proj'])
    aspect = [feat['aspect']]
    loop = [1.0 if feat['has_loop'] else 0.0]
    return np.array(dist_emb + ori_hist + h_proj + v_proj + aspect + loop, dtype=float)


_GLOBAL_CLASSIFIER = None
_GLOBAL_CLASSES = None

def get_trained_classifier() -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Initializes or returns cached LogisticRegression model trained on
    canonical templates and realistic multi-stroke / directional variations.
    """
    global _GLOBAL_CLASSIFIER, _GLOBAL_CLASSES
    if not SKLEARN_AVAILABLE:
        return None, None
    if _GLOBAL_CLASSIFIER is not None:
        return _GLOBAL_CLASSIFIER, _GLOBAL_CLASSES

    X_train = []
    y_train = []
    np.random.seed(42)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # 1. Canonical templates (x3 weight)
    for k, (t_feat, base_char) in _CANONICAL_FEATURES.items():
        vec = extract_feature_vector(t_feat)
        for _ in range(3):
            X_train.append(vec)
            y_train.append(base_char)

    # 2. Deformed samples (20 per letter) — scale + noise + stroke order variation
    for l in letters:
        raw_strokes = TEMPLATES[l]
        for _ in range(20):
            scale_x = np.random.uniform(0.75, 1.25)
            scale_y = np.random.uniform(0.75, 1.25)
            strokes = list(raw_strokes)
            if len(strokes) > 1 and np.random.rand() > 0.4:
                np.random.shuffle(strokes)
            rev_strokes = [s[::-1] if np.random.rand() > 0.5 else s for s in strokes]
            deformed = []
            for s in rev_strokes:
                s_arr = np.array(s, dtype=float)
                s_def = np.zeros_like(s_arr)
                s_def[:, 0] = 100.0 + s_arr[:, 0] * 50.0 * scale_x
                s_def[:, 1] = 100.0 + s_arr[:, 1] * 70.0 * scale_y
                noise = np.random.normal(0, np.random.uniform(0.5, 2.5), s_def.shape)
                deformed.append([(float(x), float(y)) for x, y in (s_def + noise)])
            f = extract_features(deformed)
            if f is not None:
                X_train.append(extract_feature_vector(f))
                y_train.append(l)

    # 3. Slant-augmented samples (25 per letter for letters with vertical/diagonal strokes)
    #    Real human air-writing introduces 10-20 degree natural slant. The model MUST
    #    learn to recognize slanted H, I, M, N, etc. as well as upright versions.
    slant_letters = "ABCDEFGHIKLMNPRTUVWXYZ"  # letters affected by slant
    for l in slant_letters:
        raw_strokes = TEMPLATES[l]
        for _ in range(25):
            # Random shear angle between -20 and +20 degrees
            shear_angle = np.random.uniform(-20, 20) * np.pi / 180.0
            shear_mat = np.array([[1.0, np.tan(shear_angle)],
                                   [0.0, 1.0]])
            scale_x = np.random.uniform(0.70, 1.30)
            scale_y = np.random.uniform(0.70, 1.30)
            strokes = list(raw_strokes)
            if len(strokes) > 1 and np.random.rand() > 0.5:
                np.random.shuffle(strokes)
            deformed = []
            for s in strokes:
                s_arr = np.array(s, dtype=float)
                # Apply shear + scale
                sheared = s_arr @ shear_mat.T
                s_def = np.zeros_like(sheared)
                s_def[:, 0] = 100.0 + sheared[:, 0] * 50.0 * scale_x
                s_def[:, 1] = 100.0 + sheared[:, 1] * 70.0 * scale_y
                noise = np.random.normal(0, np.random.uniform(0.5, 3.0), s_def.shape)
                deformed.append([(float(x), float(y)) for x, y in (s_def + noise)])
            f = extract_features(deformed)
            if f is not None:
                X_train.append(extract_feature_vector(f))
                y_train.append(l)

    clf = LogisticRegression(max_iter=1000, C=5.0)
    clf.fit(X_train, y_train)
    _GLOBAL_CLASSIFIER = clf
    _GLOBAL_CLASSES = list(clf.classes_)
    return _GLOBAL_CLASSIFIER, _GLOBAL_CLASSES


# ---------------------------------------------------------------------------
# Personal Calibration Dataset Manager
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATION_FILE = os.path.join(
    os.path.dirname(__file__), "calibration_data", "user_profile.json"
)

class CalibrationStore:
    """
    Manages user-calibrated handwriting exemplars.
    Persists real labeled trajectories to a local JSON file across app restarts.
    """

    def __init__(self, filepath: str = DEFAULT_CALIBRATION_FILE):
        self.filepath = filepath
        self.samples: Dict[str, List[List[List[Tuple[float, float]]]]] = {}
        self.features: Dict[str, List[Dict[str, Any]]] = {}
        self.load()

    def add_sample(self, letter: str, strokes: List[List[Tuple[float, float]]]) -> bool:
        """Validates and adds a labeled real drawing sample for a letter."""
        letter = letter.upper().strip()
        if not letter or len(letter) != 1 or not letter.isalpha():
            return False

        total_pts = sum(len(s) for s in strokes)
        if total_pts < 3:
            return False

        all_pts = []
        for s in strokes:
            all_pts.extend(s)
        pts_arr = np.array(all_pts)
        bbox = np.ptp(pts_arr, axis=0)
        if max(bbox[0], bbox[1]) < 8.0:
            return False

        feat = extract_features(strokes)
        if feat is None:
            return False

        clean_strokes = [[[float(x), float(y)] for x, y in s] for s in strokes]

        if letter not in self.samples:
            self.samples[letter] = []
            self.features[letter] = []

        self.samples[letter].append(clean_strokes)
        self.features[letter].append(feat)
        return True

    def get_sample_count(self, letter: str) -> int:
        return len(self.samples.get(letter.upper(), []))

    def get_all_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.samples.items()}

    def clear(self, letter: Optional[str] = None):
        if letter:
            l = letter.upper()
            self.samples.pop(l, None)
            self.features.pop(l, None)
        else:
            self.samples.clear()
            self.features.clear()
        self.save()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.samples, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to save calibration file: {e}")

    def load(self):
        self.samples.clear()
        self.features.clear()
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for letter, sample_list in data.items():
                l = letter.upper()
                self.samples[l] = []
                self.features[l] = []
                for s in sample_list:
                    strokes = [[(float(x), float(y)) for x, y in stroke] for stroke in s]
                    feat = extract_features(strokes)
                    if feat is not None:
                        self.samples[l].append(s)
                        self.features[l].append(feat)
        except Exception as e:
            print(f"[Warning] Failed to load calibration file: {e}")


# ---------------------------------------------------------------------------
# Continuous Stroke Segmentation (for word-level air writing like "Hello")
# ---------------------------------------------------------------------------

def _segment_continuous_stroke(
    pts: np.ndarray,
    min_points_per_seg: int = 10,
    min_width_ratio: float = 0.15,
) -> List[np.ndarray]:
    """
    Splits a single long continuous stroke into sub-strokes representing individual letters.

    Strategy: Detect "pen lift" candidates by finding points where:
    1. The x-velocity drops near zero (user momentarily stops moving left-to-right)
    2. Combined with a local y-velocity reversal (stroke changes direction vertically)
    These are characteristic breakpoints between connected letters in cursive-style air writing.

    Falls back to equal-width spatial segmentation if velocity analysis fails.

    Args:
        pts: Nx2 array of (x, y) fingertip coordinates.
        min_points_per_seg: Minimum points required per segment.
        min_width_ratio: Minimum fraction of total width each segment must span.

    Returns:
        List of sub-stroke arrays. If segmentation fails, returns [pts].
    """
    if len(pts) < 2 * min_points_per_seg:
        return [pts]

    total_w = float(pts[:, 0].max() - pts[:, 0].min())
    min_seg_w = total_w * min_width_ratio

    # Compute smoothed velocity
    # Use a sliding-window average over 5-7 points to reduce noise
    win = min(7, len(pts) // 4)
    if win < 2:
        return [pts]

    dx = np.convolve(np.diff(pts[:, 0]), np.ones(win) / win, mode='same')
    dy = np.convolve(np.diff(pts[:, 1]), np.ones(win) / win, mode='same')
    # Pad to match pts length
    dx = np.concatenate([dx, [dx[-1]]])
    dy = np.concatenate([dy, [dy[-1]]])

    speed = np.hypot(dx, dy)
    max_speed = float(speed.max())
    if max_speed < 1e-6:
        return [pts]

    speed_norm = speed / max_speed

    # Find candidate split points: local speed minima below 20% of max speed
    # that occur after a significant rightward progression
    candidates = []
    n = len(pts)
    for i in range(min_points_per_seg, n - min_points_per_seg):
        # Speed local minimum
        local_min = (speed_norm[i] < 0.20 and
                     speed_norm[i] <= speed_norm[max(0, i-2)] and
                     speed_norm[i] <= speed_norm[min(n-1, i+2)])
        # x-progression: moved right by at least min_seg_w from last split
        if candidates:
            x_since_last = pts[i, 0] - pts[candidates[-1], 0]
        else:
            x_since_last = pts[i, 0] - pts[0, 0]

        if local_min and x_since_last >= min_seg_w:
            candidates.append(i)

    # Build sub-strokes from split candidates
    if not candidates:
        # Fallback: try to split by spatial x-windows based on estimated letter count
        # Heuristic: total width / typical letter width gives letter count
        total_h = float(pts[:, 1].max() - pts[:, 1].min())
        est_letter_w = max(total_h * 0.6, min_seg_w)
        est_count = max(2, int(round(total_w / est_letter_w)))
        est_count = min(est_count, 8)  # cap at 8 letters per gesture

        splits = []
        for i in range(1, est_count):
            x_target = pts[0, 0] + (total_w * i / est_count)
            # Find closest point to this x target
            idx = int(np.argmin(np.abs(pts[:, 0] - x_target)))
            if idx > min_points_per_seg and idx < n - min_points_per_seg:
                splits.append(idx)
        candidates = splits

    if not candidates:
        return [pts]

    # Build segments
    split_points = [0] + candidates + [n]
    segments = []
    for i in range(len(split_points) - 1):
        seg = pts[split_points[i]:split_points[i+1]]
        if len(seg) >= max(2, min_points_per_seg // 2):
            seg_w = float(seg[:, 0].max() - seg[:, 0].min()) if len(seg) > 1 else 0.0
            if seg_w >= min_seg_w * 0.5:  # allow slightly smaller segments
                segments.append(seg)

    return segments if len(segments) >= 2 else [pts]


# ---------------------------------------------------------------------------
# Public AirWritingRecognizer Class
# ---------------------------------------------------------------------------

class AirWritingRecognizer:
    """
    Genuine trajectory-based air-writing recognizer for AIR TEXT.
    Combines:
    1. Pretrained multi-feature Logistic Regression classifier on feature vectors:
       - N Chamfer distance embeddings against all canonical templates
       - 8-bin soft orientation histogram
       - 16-bin horizontal projection profile
       - 16-bin vertical projection profile
       - 1-dim aspect ratio
       - 1-dim loop topology indicator
    2. Personal Calibration with Exemplar Priority:
       - Persists user handwriting samples to calibration JSON
       - Prioritizes personal exemplars with custom boost
    3. Spatio-Temporal Stroke Segmentation:
       - Clusters multi-stroke characters (H, E, F, T, etc.)
       - Identifies word boundaries and spaces
       - Splits single continuous-stroke word writing (e.g. "Hello") into letters
    """

    def __init__(self, confidence_threshold: float = 0.45,
                 calibration_file: str = DEFAULT_CALIBRATION_FILE,
                 use_ml: bool = True):
        self.threshold = confidence_threshold
        self.calibration = CalibrationStore(calibration_file)
        self.use_ml = use_ml
        if self.use_ml and SKLEARN_AVAILABLE:
            self.clf, self.classes = get_trained_classifier()
        else:
            self.clf, self.classes = None, None

    def recognize_character(
        self,
        strokes: List[List[Tuple[float, float]]],
    ) -> Tuple[str, float, List[str], bool]:
        """
        Primary recognition method for single-character air writing ("Finish Character").
        Evaluates input against personal calibrated exemplars and canonical reference templates.

        Returns:
            (best_letter, confidence, top_candidates, is_uncertain)
        """
        if not strokes:
            return ("", 0.0, [], True)

        # Parse strokes, filtering out degenerate/noise ones
        parsed = []
        for s in strokes:
            if len(s) >= 2:
                s_arr = np.array(s, dtype=float)
                diffs = np.diff(s_arr, axis=0)
                path_l = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
                # Keep if path length is meaningful (> 6 px in screen coords, or > 0.05 in normalized)
                if path_l >= 0.05:
                    parsed.append([(float(x), float(y)) for x, y in s])

        if not parsed:
            # Fallback if strokes was list of single points
            if any(len(s) >= 1 for s in strokes):
                parsed = [[(float(x), float(y)) for x, y in s] for s in strokes if len(s) >= 1]
            else:
                return ("", 0.0, [], True)

        all_pts = np.vstack([np.array(s) for s in parsed])
        bbox = np.ptp(all_pts, axis=0)
        max_span = max(bbox[0], bbox[1])

        # A serif capital I is often three strokes: top bar, vertical stem,
        # bottom bar. The generic classifier can confuse this sparse shape
        # with an open curve, so recognize its defining geometry directly.
        width = max(float(bbox[0]), 1e-6)
        height = max(float(bbox[1]), 1e-6)
        if width / height <= 0.95:
            boxes = []
            for stroke in parsed:
                stroke_pts = np.asarray(stroke, dtype=float)
                lo = stroke_pts.min(axis=0)
                hi = stroke_pts.max(axis=0)
                boxes.append((lo, hi, hi - lo))
            line_dx = []
            line_dy = []
            line_mx = []
            line_my = []
            for stroke in parsed:
                stroke_pts = np.asarray(stroke, dtype=float)
                delta = np.diff(stroke_pts, axis=0)
                line_dx.extend(np.abs(delta[:, 0]))
                line_dy.extend(np.abs(delta[:, 1]))
                line_mx.extend((stroke_pts[:-1, 0] + stroke_pts[1:, 0]) * 0.5)
                line_my.extend((stroke_pts[:-1, 1] + stroke_pts[1:, 1]) * 0.5)
            line_dx = np.asarray(line_dx, dtype=float)
            line_dy = np.asarray(line_dy, dtype=float)
            line_mx = np.asarray(line_mx, dtype=float)
            line_my = np.asarray(line_my, dtype=float)
            line_len = np.hypot(line_dx, line_dy)
            x_min = float(all_pts[:, 0].min())
            y_min = float(all_pts[:, 1].min())
            horizontal = line_dx >= 1.5 * np.maximum(line_dy, 1e-6)
            vertical = line_dy >= 1.5 * np.maximum(line_dx, 1e-6)
            center_x = (line_mx >= x_min + 0.28 * width) & (line_mx <= x_min + 0.72 * width)
            top_y = line_my <= y_min + 0.30 * height
            bottom_y = line_my >= y_min + 0.70 * height
            has_trajectory_stem = float(np.sum(line_dy[vertical & center_x])) >= 0.45 * height
            has_trajectory_top = float(np.sum(line_dx[horizontal & top_y])) >= 0.35 * width
            has_trajectory_bottom = float(np.sum(line_dx[horizontal & bottom_y])) >= 0.35 * width
            has_center_stem = any(
                size[1] >= 0.55 * height and size[0] <= 0.30 * width
                and abs((lo[0] + hi[0]) * 0.5 - (all_pts[:, 0].min() + width * 0.5)) <= 0.25 * width
                for lo, hi, size in boxes
            ) or has_trajectory_stem
            has_top_bar = any(
                size[0] >= 0.35 * width and size[1] <= 0.24 * height
                and (lo[1] + hi[1]) * 0.5 <= y_min + 0.30 * height
                for lo, hi, size in boxes
            ) or has_trajectory_top
            has_bottom_bar = any(
                size[0] >= 0.35 * width and size[1] <= 0.24 * height
                and (lo[1] + hi[1]) * 0.5 >= y_min + 0.70 * height
                for lo, hi, size in boxes
            ) or has_trajectory_bottom
            if has_center_stem and has_top_bar and has_bottom_bar:
                return ("I", 0.90, ["I", "H", "T"], False)

        # Determine if coords are screen-pixels (large values) or normalized (small)
        is_screen_scale = float(np.max(np.abs(all_pts))) > 5.0

        if max_span < 1e-3:
            return ("", 0.0, [], True)
        if is_screen_scale and max_span < 8.0:
            # Tiny pixel-space jitter — reject as noise
            return ("", 0.0, [], True)
        if len(all_pts) < 2:
            return ("", 0.0, [], True)

        cand_feat = extract_features(parsed)
        if cand_feat is None:
            return ("", 0.0, [], True)

        # 1. Match against personal calibrated exemplars
        calib_bonuses: Dict[str, float] = {}
        for letter, feat_list in self.calibration.features.items():
            if not feat_list:
                continue
            costs = [compare_features(cand_feat, ref_f) for ref_f in feat_list]
            min_c = min(costs)
            if min_c < 0.40:
                bonus = max(0.0, (0.40 - min_c) / 0.40) * 0.45
                calib_bonuses[letter] = bonus

        # 2. Geometric Chamfer Costs against Canonical Templates
        # Use outer-bin clusters to handle natural human slant (+-15 degrees)
        # Left stems can spread into bins 0-3; right stems into bins 12-15
        left_stem_mass  = float(np.sum(cand_feat['h_proj'][:4]))
        right_stem_mass = float(np.sum(cand_feat['h_proj'][-4:]))
        center_void     = float(np.sum(cand_feat['h_proj'][4:12]))   # center should be relatively low for H
        has_two_stems = (left_stem_mass > 0.18 and right_stem_mass > 0.18 and cand_feat['aspect'] > 0.35)
        chamfer_costs: Dict[str, float] = {}
        for tmpl_key, (tmpl_feat, base_letter) in _CANONICAL_FEATURES.items():
            cost = compare_features(cand_feat, tmpl_feat)
            h_energy = cand_feat['ori'][0]
            diag_energy = cand_feat['ori'][2] + cand_feat['ori'][6]

            if base_letter == 'H':
                if cand_feat['has_mid_bar'] and h_energy > 0.05:
                    cost -= 0.12          # reward clear crossbar
                elif not cand_feat['has_mid_bar'] and h_energy < 0.04:
                    cost += 0.25          # penalize missing crossbar
                if has_two_stems:
                    cost -= 0.14          # reward two vertical stems (slant-tolerant check)
                if diag_energy > 0.45:
                    cost += 0.20
            elif base_letter == 'V':
                if cand_feat['has_mid_bar'] or h_energy > 0.18:
                    cost += 0.30
                if diag_energy > 0.35:
                    cost -= 0.06
            elif base_letter == 'M':
                if cand_feat['has_mid_bar'] or h_energy > 0.18:
                    cost += 0.30
                if diag_energy > 0.20 and cand_feat['ori'][4] > 0.20:
                    cost -= 0.05
            elif base_letter == 'S':
                if has_two_stems:
                    cost += 0.25

            if base_letter == 'I':
                # Only penalize I if it's very wide AND lacks a vertical component.
                # A real serif I has top/bottom bars → aspect can be up to 0.9. Do NOT penalize that.
                # Only penalize truly wide shapes (aspect > 0.90) that lack a vertical stem.
                v_proj_center = float(np.sum(cand_feat['v_proj'][4:12]))  # middle-band vertical projection
                h_energy = cand_feat['ori'][0]
                has_vertical_stem = (cand_feat['v_proj'].max() > 0.10 and h_energy < 0.65)
                if cand_feat['aspect'] > 0.90 and not has_vertical_stem:
                    cost += 0.40   # truly horizontal shape — unlikely to be I
                elif cand_feat['aspect'] < 0.20:
                    cost -= 0.10   # reward narrow vertical shapes for I
            elif base_letter != 'I' and cand_feat['aspect'] < 0.20:
                cost += 0.35

            if base_letter not in chamfer_costs or cost < chamfer_costs[base_letter]:
                chamfer_costs[base_letter] = cost

        # 3. Hybrid ML + Geometric Scoring
        if self.clf is not None:
            vec = extract_feature_vector(cand_feat)
            probs = self.clf.predict_proba([vec])[0]
            scores: Dict[str, float] = {}
            for i, cls_name in enumerate(self.classes):
                lr_prob = float(probs[i])
                c_cost = chamfer_costs.get(cls_name, 1.0)
                geom_sim = max(0.0, 1.0 - c_cost / 0.50)
                bonus = calib_bonuses.get(cls_name, 0.0)
                scores[cls_name] = 0.40 * lr_prob + 0.60 * geom_sim + bonus

            sorted_letters = sorted(scores, key=scores.get, reverse=True)
            top_candidates = sorted_letters[:3]
            best_letter = top_candidates[0]
            best_score = float(scores[best_letter])
            second_score = float(scores[sorted_letters[1]]) if len(sorted_letters) > 1 else 0.0
            margin = best_score - second_score
            confidence = min(1.0, max(0.0, best_score))

            is_uncertain = (confidence < self.threshold) or (margin < 0.08 and confidence < 0.65)

            # ── Diagnostic Logging ───────────────────────────────────────────
            all_pts_diag = np.vstack([np.array(s) for s in parsed])
            bbox_diag    = np.ptp(all_pts_diag, axis=0)
            print(f"[DIAG] strokes={len(parsed)} pts={len(all_pts_diag)} "
                  f"bbox=({bbox_diag[0]:.1f}x{bbox_diag[1]:.1f}) "
                  f"aspect={cand_feat['aspect']:.2f} "
                  f"has_mid_bar={cand_feat['has_mid_bar']} "
                  f"has_loop={cand_feat['has_loop']} "
                  f"left_stem={left_stem_mass:.3f} right_stem={right_stem_mass:.3f}")
            top5 = [(l, round(scores[l], 4)) for l in sorted_letters[:5]]
            print(f"[DIAG] Top-5: {top5}  → best={best_letter} conf={confidence:.3f} margin={margin:.3f} uncertain={is_uncertain}")
            # ────────────────────────────────────────────────────────────────

            return (best_letter, round(confidence, 3), top_candidates, is_uncertain)

        # 3. Fallback: Heuristic Cost Matching (when ML classifier unavailable)
        scores_cost: Dict[str, float] = {}
        for tmpl_key, (tmpl_feat, base_letter) in _CANONICAL_FEATURES.items():
            cost = compare_features(cand_feat, tmpl_feat)
            h_energy = cand_feat['ori'][0]
            diag_energy = cand_feat['ori'][2] + cand_feat['ori'][6]

            if base_letter == 'H':
                if cand_feat['has_mid_bar'] and h_energy > 0.10:
                    cost -= 0.08
                elif not cand_feat['has_mid_bar'] and h_energy < 0.06:
                    cost += 0.20
                if diag_energy > 0.45:
                    cost += 0.20
            elif base_letter == 'V':
                if cand_feat['has_mid_bar'] or h_energy > 0.18:
                    cost += 0.30
                if diag_energy > 0.35:
                    cost -= 0.06
            elif base_letter == 'M':
                if cand_feat['has_mid_bar'] or h_energy > 0.18:
                    cost += 0.30
                if diag_energy > 0.20 and cand_feat['ori'][4] > 0.20:
                    cost -= 0.05

            if base_letter == 'I':
                # Only penalize I if it's very wide AND lacks a vertical component.
                # A real serif I has top/bottom bars → aspect can be up to 0.9. Do NOT penalize that.
                # Only penalize truly wide shapes (aspect > 0.90) that lack a vertical stem.
                v_proj_center = float(np.sum(cand_feat['v_proj'][4:12]))  # middle-band vertical projection
                h_energy = cand_feat['ori'][0]
                has_vertical_stem = (cand_feat['v_proj'].max() > 0.10 and h_energy < 0.65)
                if cand_feat['aspect'] > 0.90 and not has_vertical_stem:
                    cost += 0.40   # truly horizontal shape — unlikely to be I
                elif cand_feat['aspect'] < 0.20:
                    cost -= 0.10   # reward narrow vertical shapes for I
            elif base_letter != 'I' and cand_feat['aspect'] < 0.20:
                cost += 0.35

            if base_letter in calib_bonuses:
                cost -= calib_bonuses[base_letter]

            if base_letter not in scores_cost or cost < scores_cost[base_letter]:
                scores_cost[base_letter] = cost

        if not scores_cost:
            return ("", 0.0, [], True)

        sorted_letters = sorted(scores_cost, key=scores_cost.get)
        best_letter = sorted_letters[0]
        best_cost = max(0.0, float(scores_cost[best_letter]))
        second_cost = max(0.0, float(scores_cost[sorted_letters[1]])) if len(sorted_letters) > 1 else best_cost + 0.50
        margin = (second_cost - best_cost) / max(best_cost, 1e-4)

        raw_conf = max(0.0, 1.0 - float((best_cost / 0.65) ** 1.3))
        confidence = min(1.0, max(0.0, raw_conf * (0.75 + 0.25 * min(margin, 1.0))))
        is_uncertain = (confidence < self.threshold) or (margin < 0.10 and confidence < 0.70)
        return (best_letter, round(float(confidence), 3), sorted_letters[:3], is_uncertain)

    def recognize_letter(
        self,
        strokes: List[np.ndarray],
    ) -> Tuple[str, float]:
        """Backward-compatible helper returning (letter, confidence)."""
        letter, conf, _, _ = self.recognize_character([[tuple(pt) for pt in s] for s in strokes])
        return (letter, conf)

    def cluster_strokes(
        self,
        strokes: Any,
    ) -> List[Tuple[List[np.ndarray], bool]]:
        """
        Cluster raw strokes into individual letters and detect word spaces.
        Supports both:
        - Timed stroke dicts: [{'pts': ..., 'start_time': ..., 'end_time': ...}]
        - Untimed raw stroke coordinate lists: [[(x, y), ...], ...]
        """
        if not strokes:
            return []

        timed_items = []
        curr_t = 0.0

        for s in strokes:
            if isinstance(s, dict):
                pts = np.array(s['pts'], dtype=float)
                if len(pts) < 2:
                    continue
                timed_items.append({
                    'pts': pts,
                    'start_time': float(s.get('start_time', curr_t)),
                    'end_time': float(s.get('end_time', curr_t + 0.3)),
                })
                curr_t = float(s.get('end_time', curr_t + 0.3)) + 0.15
            else:
                s_arr = np.array(s, dtype=float)
                if len(s_arr) < 2:
                    continue
                timed_items.append({
                    'pts': s_arr,
                    'start_time': curr_t,
                    'end_time': curr_t + 0.3,
                })
                curr_t += 0.45

        if not timed_items:
            return []
        if len(timed_items) == 1:
            # Single stroke: check if it's a continuous word (wide bounding box)
            # and try to segment it into sub-character components via local minima detection
            single_pts = timed_items[0]['pts']
            single_w   = float(single_pts[:, 0].max() - single_pts[:, 0].min())
            single_h   = float(single_pts[:, 1].max() - single_pts[:, 1].min())
            single_ar  = single_w / max(single_h, 1e-4)

            if single_ar > 1.8 and len(single_pts) >= 20:
                # Likely a continuous word — segment into sub-strokes by finding
                # local velocity minima (points of slowdown between letters)
                sub_strokes = _segment_continuous_stroke(single_pts,
                                                         min_points_per_seg=10,
                                                         min_width_ratio=0.15)
                if len(sub_strokes) > 1:
                    # Re-enter clustering with segmented sub-strokes
                    timed_segmented = []
                    t_now = timed_items[0]['start_time']
                    duration = timed_items[0]['end_time'] - timed_items[0]['start_time']
                    dt = duration / len(sub_strokes)
                    for seg in sub_strokes:
                        timed_segmented.append({
                            'pts': seg,
                            'start_time': t_now,
                            'end_time': t_now + dt,
                        })
                        t_now += dt
                    timed_items = timed_segmented
                else:
                    return [([timed_items[0]['pts']], False)]
            else:
                return [([timed_items[0]['pts']], False)]

        # Extract geometric bounding boxes
        items = []
        for it in timed_items:
            pts = it['pts']
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            items.append({
                'pts': pts,
                'start_time': it['start_time'],
                'end_time': it['end_time'],
                'min_x': float(mn[0]),
                'max_x': float(mx[0]),
                'min_y': float(mn[1]),
                'max_y': float(mx[1]),
                'w': float(mx[0] - mn[0]),
                'h': float(mx[1] - mn[1]),
                'cx': float((mn[0] + mx[0]) / 2.0),
                'cy': float((mn[1] + mx[1]) / 2.0),
            })

        # Reference character height: maximum stroke height across drawing
        all_vals = [float(np.max(np.abs(it['pts']))) for it in items]
        is_screen_scale = max(all_vals) > 5.0
        char_h = float(np.max([it['h'] for it in items]))
        if is_screen_scale:
            if char_h < 15.0:
                char_h = 20.0
        else:
            if char_h < 0.2:
                char_h = 0.5

        clusters: List[Tuple[List[np.ndarray], bool]] = []
        curr = [items[0]]
        curr_min_x = items[0]['min_x']
        curr_max_x = items[0]['max_x']
        curr_min_y = items[0]['min_y']
        curr_max_y = items[0]['max_y']
        has_space = False

        for it in items[1:]:
            dt = it['start_time'] - curr[-1]['end_time']
            gap_x = it['min_x'] - curr_max_x
            overlap_x = min(curr_max_x, it['max_x']) - max(curr_min_x, it['min_x'])
            center_dx = it['cx'] - (curr_min_x + curr_max_x) / 2.0
            potential_w = max(curr_max_x, it['max_x']) - min(curr_min_x, it['min_x'])
            y_overlap = min(curr_max_y, it['max_y']) - max(curr_min_y, it['min_y'])

            # Multi-stroke letter clustering condition
            is_overlapping = (overlap_x > 0.05 * char_h) or (it['min_x'] <= curr_max_x and it['max_x'] >= curr_min_x)
            is_parallel_stem = (potential_w <= 0.65 * char_h) and (y_overlap > 0.45 * char_h) and (dt < 0.35)
            is_same_letter = (is_overlapping or is_parallel_stem) and (potential_w <= 1.25 * char_h) and (dt < 0.65)

            if is_same_letter:
                curr.append(it)
                curr_min_x = min(curr_min_x, it['min_x'])
                curr_max_x = max(curr_max_x, it['max_x'])
                curr_min_y = min(curr_min_y, it['min_y'])
                curr_max_y = max(curr_max_y, it['max_y'])
            else:
                # Word-level spacing: requires significant physical gap or intentional pause
                is_space = ((gap_x > 1.35 * char_h) and (center_dx > 1.30 * char_h)) or (dt > 1.40)
                clusters.append(([s['pts'] for s in curr], has_space))
                curr = [it]
                curr_min_x = it['min_x']
                curr_max_x = it['max_x']
                curr_min_y = it['min_y']
                curr_max_y = it['max_y']
                has_space = is_space

        if curr:
            clusters.append(([s['pts'] for s in curr], has_space))

        return clusters

    def cluster_strokes_timed(
        self,
        strokes_with_time: List[Dict],
    ) -> List[Tuple[List[np.ndarray], bool]]:
        """Convenience alias for cluster_strokes with timestamped stroke dicts."""
        return self.cluster_strokes(strokes_with_time)

    def recognize(
        self,
        strokes: Any,
    ) -> Tuple[str, float]:
        """
        Recognize text from air-writing strokes.
        First clusters strokes along the horizontal axis:
        - If all strokes belong to one letter, evaluates directly.
        - If multiple letters or words are detected, evaluates each cluster sequentially.
        """
        if not strokes:
            return ("", 0.0)

        clusters = self.cluster_strokes(strokes)
        if not clusters:
            return ("", 0.0)

        # Single character case
        if len(clusters) == 1:
            cluster_strokes, _ = clusters[0]
            stroke_tuples = [[(float(x), float(y)) for x, y in s] for s in cluster_strokes]
            letter, conf, _, _ = self.recognize_character(stroke_tuples)
            return (letter, conf)

        # Multi-character word/sentence case
        recognized_parts = []
        confidences = []

        for cluster_strokes, has_space in clusters:
            if has_space and recognized_parts:
                recognized_parts.append(" ")

            stroke_tuples = [[(float(x), float(y)) for x, y in s] for s in cluster_strokes]
            letter, conf, _, _ = self.recognize_character(stroke_tuples)
            if letter:
                recognized_parts.append(letter)
                confidences.append(conf)

        if not recognized_parts:
            return ("", 0.0)

        raw_text = "".join(recognized_parts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return (raw_text, round(avg_conf, 3))
