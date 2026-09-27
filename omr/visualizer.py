"""
Visualizer - Annotated grading overlays, bold bubble circling, and diagnostic markers.
"""

import cv2
import numpy as np


def annotate_graded_sheet(warped_img, results, summary, options=("A", "B", "C", "D")):
    """
    Produces a visually rich annotated image:
      - Bold vibrant circle around EVERY detected shaded bubble.
      - Solid center dot inside each shaded bubble.
      - Thin gray guide circles around all options to confirm sub-pixel centering.
      - Text label on the margin showing the detected choice (e.g. "Q1: B").
      - Header banner summarizing the detection rate (e.g. 30/30 shaded).
    """
    annotated = warped_img.copy()
    h, w = annotated.shape[:2]

    # Draw circles on every question bubble
    for q_num, data in results.items():
        coords = data["bubble_coords"]
        r = data["radius"]
        chosen = data["chosen"]
        chosen_idx = data["chosen_idx"]
        correct = data["correct_answer"]
        is_correct = data["is_correct"]

        # 1. Subtle gray guide circle on all option bubbles to verify concentric centering
        for opt_idx, (bx, by) in enumerate(coords):
            cv2.circle(annotated, (bx, by), r, (210, 210, 210), 1)

        # 2. Bold highlight on the detected shaded bubble
        if chosen_idx is not None:
            cx, cy = coords[chosen_idx]

            if is_correct is False and correct in options:
                # Wrong (when key is active): Red ring on chosen, Green on correct
                cv2.circle(annotated, (cx, cy), r + 4, (0, 0, 235), 3)
                corr_idx = options.index(correct)
                cor_x, cor_y = coords[corr_idx]
                cv2.circle(annotated, (cor_x, cor_y), r + 4, (0, 210, 0), 3)
            else:
                # Standard pure detection: Bold vibrant green circle + center dot
                cv2.circle(annotated, (cx, cy), r + 4, (0, 210, 0), 3)
                cv2.circle(annotated, (cx, cy), 3, (0, 210, 0), -1)

            # Draw text label in the right margin of the column
            label_x = coords[-1][0] + r + 15
            cv2.putText(
                annotated,
                f"{chosen}",
                (label_x, cy + 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (0, 160, 0),
                2
            )

    # 3. Add top Score / Detection Banner
    banner_h = 70
    banner = np.zeros((banner_h, w, 3), dtype="uint8")
    banner[:] = (30, 30, 30)

    if summary.get("score") is not None:
        main_title = f"SCORE: {summary['score']} / {summary['max_score']} " \
                     f"({(summary['score'] / float(summary['max_score'])) * 100:.1f}%)"
    else:
        main_title = f"DETECTED ANSWERS: {summary['answered']} / {summary['total_questions']} SHADED"

    stats_text = f"Total: {summary['total_questions']} | Shaded: {summary['answered']} | Blank: {summary['blank']}"

    cv2.putText(banner, main_title, (30, 44), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 120), 2)
    cv2.putText(banner, stats_text, (w - 550, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 1)

    annotated = np.vstack([banner, annotated])
    return annotated
