"""
Grid Detector - Dynamic discovery of timing marks, answer rows, and column blocks.
Supports arbitrary question counts (30, 40, 50, 180 questions).
"""

import cv2
import numpy as np


def detect_timing_marks(pencil_mask, x_strip_ratio=(0.0, 0.08)):
    """
    Finds the black tick marks down the left timing strip.
    Returns their sorted vertical center coordinates (y-positions).
    """
    h, w = pencil_mask.shape[:2]
    x1 = int(w * x_strip_ratio[0])
    x2 = int(w * x_strip_ratio[1])
    strip = pencil_mask[:, x1:x2]

    contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strip_area = strip.shape[0] * strip.shape[1]

    raw_marks = []
    for c in contours:
        area = cv2.contourArea(c)
        if strip_area * 0.00015 < area < strip_area * 0.025:
            x, y, mw, mh = cv2.boundingRect(c)
            if 0.4 < (mw / float(mh + 1e-5)) < 6.0:
                cy = y + mh // 2
                raw_marks.append(cy)

    raw_marks = sorted(raw_marks)

    # Merge nearby duplicate detections belonging to the same mark
    merged = []
    min_dist = h * 0.007
    for y in raw_marks:
        if not merged or (y - merged[-1]) > min_dist:
            merged.append(y)

    return merged


def partition_answer_rows(timing_marks, expected_rows=None):
    """
    Separates the timing marks for the register number (top) from the
    answer rows (bottom) by finding the largest vertical gap or separator.
    """
    if len(timing_marks) < 5:
        return timing_marks

    diffs = np.diff(timing_marks)
    median_diff = np.median(diffs)

    # Look for a gap significantly larger than the median spacing
    gap_indices = np.where(diffs > (median_diff * 1.8))[0]

    if len(gap_indices) > 0:
        split_idx = gap_indices[-1] + 1
        answer_rows = timing_marks[split_idx:]
    else:
        if expected_rows and len(timing_marks) > expected_rows:
            answer_rows = timing_marks[-expected_rows:]
        else:
            answer_rows = timing_marks

    return answer_rows


def infer_column_layout(num_rows, num_columns_override=None):
    """
    Deduces the number of columns and exact calibrated horizontal bounds.
    Standard patterns:
      - 15 rows -> 2 columns (30 questions total)
      - 25 rows -> 2 columns (50 questions) or 4 columns (100 questions)
      - 45 rows -> 4 columns (180 questions total)
    """
    if num_columns_override is not None:
        num_cols = num_columns_override
    elif num_rows <= 16:
        num_cols = 2
    elif num_rows <= 28:
        num_cols = 2
    elif num_rows >= 40:
        num_cols = 4
    else:
        num_cols = 2

    # Calibrated column ranges (col_x1, col_x2) where options A-D are spaced evenly inside.
    # For a 1600-width canvas:
    # 2 columns (30 questions):
    #   Col 1: A=505, B=570, C=635, D=700 -> Range: (0.296, 0.458)
    #   Col 2: A=933, B=998, C=1064, D=1130 -> Range: (0.563, 0.727)
    if num_cols == 2:
        col_ranges = [
            (0.296, 0.458),   # Column 1 (Q1-15)
            (0.563, 0.727),   # Column 2 (Q16-30)
        ]
    elif num_cols == 3:
        col_ranges = [
            (0.18, 0.38),
            (0.44, 0.64),
            (0.70, 0.90),
        ]
    elif num_cols == 4:
        # 180 questions (4 columns of 45 rows)
        col_ranges = [
            (0.10, 0.28),   # Col 1: Q1-45
            (0.33, 0.51),   # Col 2: Q46-90
            (0.56, 0.74),   # Col 3: Q91-135
            (0.78, 0.96),   # Col 4: Q136-180
        ]
    else:
        col_ranges = []
        width_per_col = 0.85 / num_cols
        for c in range(num_cols):
            x1 = 0.08 + c * width_per_col + 0.03
            x2 = x1 + width_per_col * 0.75
            col_ranges.append((round(x1, 3), round(x2, 3)))

    return num_cols, col_ranges


def detect_answer_grid(pencil_mask, num_columns_override=None, expected_rows=None):
    """
    Builds the complete dynamic answer grid:
    1. Extracts timing marks.
    2. Isolates the answer rows.
    3. Infers column count and horizontal bounds.
    4. Computes row spacing and optimal bubble radius.
    """
    h, w = pencil_mask.shape[:2]
    all_timing_marks = detect_timing_marks(pencil_mask)
    answer_rows = partition_answer_rows(all_timing_marks, expected_rows=expected_rows)

    if len(answer_rows) < 5:
        default_rows = expected_rows if expected_rows else 15
        start_y = int(h * 0.53)
        end_y = int(h * 0.86)
        step = (end_y - start_y) / float(default_rows)
        answer_rows = [int(start_y + (i + 0.5) * step) for i in range(default_rows)]

    num_rows = len(answer_rows)
    num_cols, col_ranges = infer_column_layout(num_rows, num_columns_override)

    if num_rows > 1:
        row_diffs = np.diff(answer_rows)
        median_spacing = float(np.median(row_diffs))
    else:
        median_spacing = 30.0

    bubble_radius = int(median_spacing * 0.32)
    bubble_radius = max(8, min(bubble_radius, 25))

    return {
        "answer_rows": answer_rows,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "col_ranges": col_ranges,
        "bubble_radius": bubble_radius,
        "total_questions": num_rows * num_cols,
    }
