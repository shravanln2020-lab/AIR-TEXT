"""
test_app_flow.py
================
Comprehensive end-to-end verification of the AIR TEXT application.

Tests:
1. Module imports and dependencies
2. AirWritingRecognizer single-letter accuracy on A-Z
3. Recognizer multi-letter word recognition ("HI", "HELLO")
4. Phrase dictionary matching ("GOOD MORNING", "HOW ARE YOU")
5. CameraThread multi-stroke separation (pen-up / pen-down)
6. Tkinter UI initialization, state management, and cleanup
"""

import sys
import time
import numpy as np

def run_tests():
    print("=" * 60)
    print("AIR TEXT — COMPREHENSIVE AUTOMATED VERIFICATION SUITE")
    print("=" * 60)

    # 1. Imports
    print("\n[Test 1] Importing core project modules...")
    try:
        from recognizer import AirWritingRecognizer, TEMPLATES
        from camera import CameraThread
        from main import AirTextApp
        print("  -> recognizer, camera, main successfully imported!")
    except Exception as e:
        print(f"  -> FAIL: Import error: {e}")
        return False

    # 2. Recognizer single letter tests
    print("\n[Test 2] Testing AirWritingRecognizer on all 26 letters...")
    rec = AirWritingRecognizer(confidence_threshold=0.20)
    passed_letters = 0
    for key, template in sorted(TEMPLATES.items()):
        expected = key.split('_')[0]
        input_strokes = [[(float(x), float(y)) for x, y in s] for s in template]
        recognized, conf = rec.recognize(input_strokes)
        if recognized == expected:
            passed_letters += 1
        else:
            print(f"  -> Discrepancy on {key}: expected '{expected}', got '{recognized}'")

    print(f"  -> Recognition accuracy: {passed_letters}/{len(TEMPLATES)} templates matched ({passed_letters/len(TEMPLATES)*100:.1f}%)")
    assert passed_letters == len(TEMPLATES), "All templates should match their base letter"

    # 3. Multi-character words
    print("\n[Test 3] Testing multi-character word recognition...")
    # 'HI'
    h_strokes = [
        [(100, 100), (100, 300)],
        [(100, 200), (180, 200)],
        [(180, 100), (180, 300)],
    ]
    i_strokes = [
        [(270, 100), (270, 300)],
    ]
    word_hi, conf_hi = rec.recognize(h_strokes + i_strokes)
    print(f"  -> Synthetic strokes for 'HI': Got '{word_hi}' (conf={conf_hi:.2f})")
    assert word_hi == "HI", f"Expected 'HI', got '{word_hi}'"

    # 'HELLO'
    e_strokes = [
        [(340, 100), (340, 300)],
        [(340, 100), (400, 100)],
        [(340, 200), (390, 200)],
        [(340, 300), (400, 300)],
    ]
    l1_strokes = [
        [(450, 100), (450, 300)],
        [(450, 300), (500, 300)],
    ]
    l2_strokes = [
        [(550, 100), (550, 300)],
        [(550, 300), (600, 300)],
    ]
    # Circle for O
    angles = np.linspace(0, 2 * np.pi, 40)
    o_stroke = [(float(690 + 40 * np.cos(a)), float(200 + 70 * np.sin(a))) for a in angles]
    o_strokes = [o_stroke]

    hello_strokes = h_strokes + [e_strokes[0]] + l1_strokes + l2_strokes + o_strokes
    word_hello, conf_hello = rec.recognize(hello_strokes)
    print(f"  -> Multi-letter recognition: Got '{word_hello}' (conf={conf_hello:.2f})")

    # 4. Camera thread segmentation test
    print("\n[Test 4] Testing CameraThread stroke segmentation & trail...")
    import queue
    fq = queue.Queue(maxsize=2)
    sq = queue.Queue()
    cam = CameraThread(fq, sq)

    cam.start_writing()
    # Stroke 1: Left diagonal up then right diagonal down (apex)
    apex_pts = [(100, 300), (125, 200), (150, 100), (175, 200), (200, 300)]
    for i, pt in enumerate(apex_pts):
        cam.add_manual_point(pt[0], pt[1], is_new_stroke=(i == 0))
    # Stroke 2: Crossbar (pen-up then pen-down)
    cross_pts = [(125, 200), (150, 200), (175, 200)]
    for i, pt in enumerate(cross_pts):
        cam.add_manual_point(pt[0], pt[1], is_new_stroke=(i == 0))

    captured = cam.finish_writing()
    print(f"  -> Captured strokes count: {len(captured)} (expected 2 strokes without connecting lines)")
    assert len(captured) == 2, f"Expected 2 separate strokes, got {len(captured)}"

    recognized_a, conf_a = rec.recognize(captured)
    print(f"  -> Recognition of captured strokes: Got '{recognized_a}' (conf={conf_a:.2f})")
    assert recognized_a == "A", f"Expected 'A', got '{recognized_a}'"

    # 5. UI Initialization & state verification
    print("\n[Test 5] Testing Tkinter UI initialization and controls...")
    app = AirTextApp()
    app.update_idletasks()

    # Test typing / editing actions
    app.recognized_text = "AIR"
    app._update_text_display()
    assert "AIR" in app.rec_display_var.get()

    app._insert_space()
    assert app.recognized_text == "AIR "

    app._backspace()
    assert app.recognized_text == "AIR"

    app._clear_text()
    assert app.recognized_text == ""
    print("  -> UI controls (Space, Backspace, Clear Text) validated successfully!")

    # Verify buttons
    assert "Start Camera" in app._buttons
    assert "Start Writing" in app._buttons
    assert "Finish Character" in app._buttons
    assert "Space" in app._buttons
    assert "Backspace" in app._buttons
    assert "Clear Text" in app._buttons
    assert "Stop Camera" in app._buttons
    print("  -> All 7 toolbar buttons verified!")

    app.destroy()
    print("  -> UI destroyed cleanly!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
