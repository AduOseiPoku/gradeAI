"""
Evaluator - Fill-ratio bubble sampling, multi-shade classification, and grading.
"""

import cv2
import numpy as np


def sample_fill_ratio(pencil_mask, cx, cy, radius):
    """
    Computes the fraction of filled (white) pixels inside a circular bubble mask.
    Returns a float between 0.0 (empty) and 1.0 (filled).
    """
    h, w = pencil_mask.shape[:2]
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    
    roi = pencil_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    roi_h, roi_w = roi.shape
    mask = np.zeros((roi_h, roi_w), dtype="uint8")
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), radius, 255, -1)

    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return 0.0

    filled_pixels = cv2.countNonZero(cv2.bitwise_and(roi, mask))
    return float(filled_pixels) / float(total_pixels)


def classify_bubble_group(fill_ratios, options=("A", "B", "C", "D"),
                          blank_threshold=0.14, ambiguity_gap=0.08):
    """
    Classifies a question's options based on absolute fill ratio and relative contrast:
      - BLANK: if highest ratio is below blank_threshold.
      - AMBIGUOUS: if top 2 options are both shaded and within ambiguity_gap.
      - Option (A/B/C/D): the single clear highest fill ratio.
    """
    sorted_indices = sorted(range(len(fill_ratios)), key=lambda k: fill_ratios[k], reverse=True)
    best_idx = sorted_indices[0]
    best_val = fill_ratios[best_idx]
    second_val = fill_ratios[sorted_indices[1]] if len(sorted_indices) > 1 else 0.0

    # Calculate average of the unshaded options for contrast validation
    other_vals = [fill_ratios[i] for i in range(len(fill_ratios)) if i != best_idx]
    avg_others = float(np.mean(other_vals)) if other_vals else 0.0

    # Shaded if:
    # 1. Above absolute minimum threshold (0.14), OR
    # 2. Strong contrast relative to blank bubbles in the same row (> 1.8x background)
    is_shaded = (best_val >= blank_threshold) or (best_val > 0.12 and best_val >= (avg_others * 1.8))

    if not is_shaded:
        return "BLANK", best_val, None
    elif len(sorted_indices) > 1 and (best_val - second_val) < ambiguity_gap and second_val >= blank_threshold:
        return "AMBIGUOUS", best_val, None
    else:
        return options[best_idx], best_val, best_idx


def evaluate_bubbles(pencil_mask, grid_config, answer_key=None,
                     options=("A", "B", "C", "D")):
    """
    Evaluates every question across all detected columns and rows.
    """
    h, w = pencil_mask.shape[:2]
    answer_rows = grid_config["answer_rows"]
    col_ranges = grid_config["col_ranges"]
    radius = grid_config["bubble_radius"]
    num_options = len(options)

    results = {}
    total_score = 0
    q_counter = 1

    for col_idx, (col_x1_ratio, col_x2_ratio) in enumerate(col_ranges):
        col_x1 = int(w * col_x1_ratio)
        col_x2 = int(w * col_x2_ratio)
        option_spacing = (col_x2 - col_x1) / float(num_options)

        for row_idx, cy in enumerate(answer_rows):
            q_num = q_counter
            q_counter += 1

            bubble_coords = []
            fill_ratios = []

            for opt_idx in range(num_options):
                cx = int(col_x1 + (opt_idx + 0.5) * option_spacing)
                ratio = sample_fill_ratio(pencil_mask, cx, cy, radius)
                bubble_coords.append((cx, cy))
                fill_ratios.append(round(ratio, 3))

            chosen, confidence, chosen_idx = classify_bubble_group(
                fill_ratios, options=options
            )

            is_correct = None
            if answer_key and q_num in answer_key:
                is_correct = (chosen == answer_key[q_num])
                if is_correct:
                    total_score += 1

            results[q_num] = {
                "question": q_num,
                "column": col_idx + 1,
                "row": row_idx + 1,
                "chosen": chosen,
                "confidence": confidence,
                "chosen_idx": chosen_idx,
                "fill_ratios": fill_ratios,
                "bubble_coords": bubble_coords,
                "radius": radius,
                "correct_answer": answer_key.get(q_num) if answer_key else None,
                "is_correct": is_correct,
            }

    summary = {
        "total_questions": len(results),
        "answered": sum(1 for r in results.values() if r["chosen"] not in ("BLANK", "AMBIGUOUS")),
        "blank": sum(1 for r in results.values() if r["chosen"] == "BLANK"),
        "ambiguous": sum(1 for r in results.values() if r["chosen"] == "AMBIGUOUS"),
        "score": total_score if answer_key else None,
        "max_score": len(answer_key) if answer_key else None,
    }

    return results, summary
