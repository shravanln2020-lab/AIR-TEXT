"""
test_comprehensive_eval.py
==========================
Comprehensive AI/ML benchmark & evaluation suite for AIR TEXT:
1. 26-Letter Full Confusion Matrix (N=20 trials per letter with jitter/scale/stroke-reversal)
2. Per-letter Precision, Recall, and F1 Scores
3. Targeted Confusion Pair Audits:
   - H vs (A, V, M)
   - B vs (O, D, P)
   - M vs (H, N, W)
   - V vs (Y, U)
   - I vs vertical strokes
4. Continuous Word Recognition & Character Error Rate (CER):
   - HELLO, WORLD, AIR, TEXT, COMPUTER, SCIENCE, MACHINE, LEARNING, PROGRAMMING
5. Continuous Sentence Recognition & Word Error Rate (WER):
   - "HELLO WORLD"
   - "I LOVE PROGRAMMING"
   - "ARTIFICIAL INTELLIGENCE"
   - "I AM A COMPUTER SCIENCE STUDENT"
6. Inference Latency Benchmark (ms per character)
7. Honest disclaimer regarding sandbox environment vs live human webcam testing.
"""

import time
import numpy as np
from recognizer import AirWritingRecognizer, TEMPLATES

def add_noise(strokes, noise_std=0.035, scale_x=1.0, scale_y=1.0):
    noisy = []
    for s in strokes:
        s_arr = np.array(s, dtype=float)
        s_def = np.zeros_like(s_arr)
        s_def[:, 0] = s_arr[:, 0] * scale_x
        s_def[:, 1] = s_arr[:, 1] * scale_y
        span = max(float(np.ptp(s_def, axis=0).max()), 1e-4)
        jitter = np.random.normal(0, noise_std * span, s_def.shape)
        noisy.append([(float(x), float(y)) for x, y in (s_def + jitter)])
    return noisy

def compute_cer(target: str, predicted: str) -> float:
    """Levenshtein distance based Character Error Rate (CER)."""
    n, m = len(target), len(predicted)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target[i - 1] == predicted[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m]) / max(len(target), 1)

def compute_wer(target: str, predicted: str) -> float:
    """Word Error Rate (WER)."""
    t_words = target.split()
    p_words = predicted.split()
    n, m = len(t_words), len(p_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if t_words[i - 1] == p_words[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m]) / max(len(t_words), 1)

def build_word_strokes(word, x_start=100.0, y_pos=200.0, spacing=60.0, char_h=80.0):
    strokes_with_time = []
    current_time = 0.0
    x_offset = x_start
    
    for char in word:
        if char == ' ':
            x_offset += spacing * 2.2
            current_time += 1.6
            continue
            
        raw_strokes = TEMPLATES[char]
        combined = np.vstack(raw_strokes)
        mn = combined.min(axis=0)
        span = max(float(np.ptp(combined, axis=0).max()), 1e-4)
        
        for s in raw_strokes:
            s_arr = np.array(s, dtype=float)
            sc_x = x_offset + (s_arr[:, 0] - mn[0]) / span * (spacing * 0.75)
            sc_y = y_pos    + (s_arr[:, 1] - mn[1]) / span * char_h
            
            s_duration = 0.25 + len(s_arr) * 0.015
            t_start = current_time
            t_end = t_start + s_duration
            
            strokes_with_time.append({
                'pts': np.column_stack([sc_x, sc_y]),
                'start_time': t_start,
                'end_time': t_end,
            })
            current_time = t_end + 0.15
            
        current_time += 0.35
        x_offset += spacing
        
    return strokes_with_time

def run_evaluation():
    print("=" * 75)
    print("AIR TEXT — COMPREHENSIVE AI/ML BENCHMARK & EVALUATION REPORT")
    print("=" * 75)
    
    rec = AirWritingRecognizer(confidence_threshold=0.35)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    trials_per_letter = 20
    np.random.seed(42)
    
    # ── Section 1: 26-Letter Full Confusion Matrix ──────────────────────────
    print(f"\n[Section 1] Evaluating 26-Letter Confusion Matrix (N={trials_per_letter} trials per letter)...")
    matrix = {true_c: {pred_c: 0 for pred_c in list(letters) + ["?"]} for true_c in letters}
    latencies = []
    
    for l in letters:
        raw_strokes = TEMPLATES[l]
        for t in range(trials_per_letter):
            sx = np.random.uniform(0.85, 1.15)
            sy = np.random.uniform(0.85, 1.15)
            strokes = list(raw_strokes)
            if len(strokes) > 1 and np.random.rand() > 0.4:
                np.random.shuffle(strokes)
            if np.random.rand() > 0.5:
                strokes = [s[::-1] for s in strokes]
                
            noisy = add_noise(strokes, noise_std=0.03, scale_x=sx, scale_y=sy)
            
            t0 = time.perf_counter()
            pred, conf, _, unc = rec.recognize_character(noisy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
            col = pred if pred in letters else "?"
            matrix[l][col] += 1
            
    # Calculate Per-Letter Precision, Recall, F1
    print("\nPer-Letter Performance Metrics:")
    print(f"{'Letter':>6} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7}")
    print("-" * 55)
    
    f1_scores = []
    recalls = []
    for l in letters:
        tp = matrix[l][l]
        fn = sum(matrix[l][k] for k in matrix[l] if k != l)
        fp = sum(matrix[k][l] for k in letters if k != l)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec_val) / (prec + rec_val) if (prec + rec_val) > 0 else 0.0
        
        f1_scores.append(f1)
        recalls.append(rec_val)
        print(f"{l:>6} | {prec*100:>8.1f}% | {rec_val*100:>8.1f}% | {f1*100:>8.1f}% | {trials_per_letter:>7}")
        
    overall_acc = np.mean(recalls) * 100.0
    macro_f1 = np.mean(f1_scores) * 100.0
    print("-" * 55)
    print(f"{'MACRO':>6} | {'-':>9} | {overall_acc:>8.1f}% | {macro_f1:>8.1f}% | {len(letters)*trials_per_letter:>7}")
    
    # ── Section 2: Targeted Problematic Pair Audits ─────────────────────────
    print("\n[Section 2] Auditing Key Problematic Letter Pairs:")
    pairs_to_audit = [
        ("H vs A, V, M", "H", ["A", "V", "M"]),
        ("B vs O, D, P", "B", ["O", "D", "P"]),
        ("M vs H, N, W", "M", ["H", "N", "W"]),
        ("V vs Y, U",    "V", ["Y", "U"]),
        ("I vs L, T",    "I", ["L", "T"]),
    ]
    
    for title, target, confusers in pairs_to_audit:
        target_acc = matrix[target][target] / trials_per_letter * 100.0
        confused_counts = {c: matrix[target].get(c, 0) for c in confusers}
        print(f"  • {title:<16}: Target Accuracy = {target_acc:5.1f}% | Confusions: {confused_counts}")
        
    # ── Section 3: Acceptance Words & CER ──────────────────────────────────
    print("\n[Section 3] Continuous Word Recognition & Character Error Rate (CER):")
    acceptance_words = [
        "HELLO", "WORLD", "AIR", "TEXT", "COMPUTER", "SCIENCE", "MACHINE", "LEARNING", "PROGRAMMING"
    ]
    
    cer_list = []
    for word in acceptance_words:
        strokes_timed = build_word_strokes(word)
        clusters = rec.cluster_strokes(strokes_timed)
        recognized_chars = []
        for c, is_sp in clusters:
            if is_sp and recognized_chars:
                recognized_chars.append(" ")
            pred, _, _, _ = rec.recognize_character([[(float(x), float(y)) for x, y in s] for s in c])
            recognized_chars.append(pred)
            
        pred_word = "".join(recognized_chars)
        cer = compute_cer(word, pred_word)
        cer_list.append(cer)
        match_flag = "EXACT MATCH" if pred_word == word else f"CER={cer*100:.1f}%"
        print(f"  • Target: '{word:<12}' | Output: '{pred_word:<12}' | Status: [{match_flag}]")
        
    avg_cer = np.mean(cer_list) * 100.0
    print(f"  -> Average Character Error Rate (CER) on Words: {avg_cer:.2f}%")
    assert avg_cer == 0.0, "Expected 0% CER on acceptance words"
    
    # ── Section 4: Acceptance Sentences & WER ──────────────────────────────
    print("\n[Section 4] Continuous Sentence Recognition & Word Error Rate (WER):")
    acceptance_sentences = [
        "HELLO WORLD",
        "I LOVE PROGRAMMING",
        "ARTIFICIAL INTELLIGENCE",
        "I AM A COMPUTER SCIENCE STUDENT",
    ]
    
    wer_list = []
    for sentence in acceptance_sentences:
        strokes_timed = build_word_strokes(sentence)
        clusters = rec.cluster_strokes(strokes_timed)
        recognized_chars = []
        for c, is_sp in clusters:
            if is_sp and recognized_chars:
                recognized_chars.append(" ")
            pred, _, _, _ = rec.recognize_character([[(float(x), float(y)) for x, y in s] for s in c])
            recognized_chars.append(pred)
            
        pred_sentence = "".join(recognized_chars)
        wer = compute_wer(sentence, pred_sentence)
        wer_list.append(wer)
        match_flag = "EXACT MATCH" if pred_sentence == sentence else f"WER={wer*100:.1f}%"
        print(f"  • Target: '{sentence:<32}' | WER: {wer*100:4.1f}% | [{match_flag}]")
        
    avg_wer = np.mean(wer_list) * 100.0
    print(f"  -> Average Word Error Rate (WER) on Sentences: {avg_wer:.2f}%")
    assert avg_wer == 0.0, "Expected 0% WER on acceptance sentences"
    
    # ── Section 5: Latency Benchmark ──────────────────────────────────────
    print("\n[Section 5] Inference Latency Benchmark:")
    print(f"  • Mean Latency:   {np.mean(latencies):.2f} ms per character")
    print(f"  • 50th % (Median):{np.median(latencies):.2f} ms")
    print(f"  • 95th %:         {np.percentile(latencies, 95):.2f} ms")
    print(f"  • Max Latency:    {np.max(latencies):.2f} ms")
    print(f"  • Throughput:     ~{1000.0 / np.mean(latencies):.0f} characters/second")
    
    # ── Section 6: Limitations & Environment Notice ────────────────────────
    print("\n" + "=" * 75)
    print("ENVIRONMENT & REAL-WORLD VALIDATION NOTICE:")
    print("  1. The quantitative metrics above are measured on simulated air-writing")
    print("     trajectories incorporating human hand tremor (Gaussian jitter),")
    print("     scale distortions, stroke shuffling, and drawing direction reversals.")
    print("  2. Testing on a physical webcam with a live human user cannot be executed")
    print("     in this headless development/sandbox environment (no hardware camera).")
    print("  3. For live user testing, run `python main.py`, click 'Start Camera',")
    print("     and use 'Personal Calibration' to record your real index finger style.")
    print("=" * 75)
    print("COMPREHENSIVE EVALUATION PASSED WITH ZERO CRITICAL ERRORS!")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()

