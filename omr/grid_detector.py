"""
Grid Detector - Autonomous discovery of timing marks, answer rows, and column blocks.
Supports arbitrary question counts (30, 40, 50, 180 questions) using vertical density
projections and bottom anchor blocks to dynamically detect horizontal column bounds.
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


def discover_columns_from_density(pencil_mask, answer_rows, bubble_radius, expected_cols=None):
    """
    Autonomously discovers horizontal column bounds using vertical density projection
    across the answer rows. Gutters between columns have zero mark density, allowing
    the engine to segment distinct column blocks directly from the paper.

    Returns:
        (col_ranges, is_autonomous, debug_viz)
    """
    h, w = pencil_mask.shape[:2]
    y1 = max(0, answer_rows[0] - int(bubble_radius * 1.5))
    y2 = min(h, answer_rows[-1] + int(bubble_radius * 1.5))
    
    # Exclude left timing strip to avoid false column detection
    x_offset = int(w * 0.12)
    x_end = int(w * 0.98)
    band = pencil_mask[y1:y2, x_offset:x_end]

    # Compute 1D vertical density profile (sum of mark pixels down vertical columns)
    density = np.sum(band > 0, axis=0).astype(np.float32)

    if np.max(density) < 10:
        # Not enough marks to construct a reliable profile
        return None, False, None

    # Smooth the 1D profile with a Gaussian filter to bridge spacing between A, B, C, D
    # while preserving the wide gutters between separate column blocks
    kernel_size = 45
    smoothed = cv2.GaussianBlur(density.reshape(1, -1), (kernel_size, 1), 0).flatten()

    # Normalize to [0, 1]
    norm_density = smoothed / float(np.max(smoothed) + 1e-6)

    # Active column mask (where mark density is significant)
    thresh_val = 0.14
    is_active = (norm_density > thresh_val).astype(np.uint8)

    # Find contiguous active regions
    diff = np.diff(np.pad(is_active, (1, 1), 'constant'))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Bridge small inner gaps (< 2.5% of page width) within the same column block
    merged_blocks = []
    max_gap = int(w * 0.035)

    curr_start = None
    curr_end = None

    for s, e in zip(starts, ends):
        if curr_start is None:
            curr_start, curr_end = s, e
        elif s - curr_end <= max_gap:
            curr_end = e
        else:
            merged_blocks.append((curr_start, curr_end))
            curr_start, curr_end = s, e
    if curr_start is not None:
        merged_blocks.append((curr_start, curr_end))

    # Filter blocks by reasonable column width (e.g. at least 6% of page width)
    min_col_w = int(w * 0.06)
    valid_blocks = [(s, e) for s, e in merged_blocks if (e - s) >= min_col_w]

    # Verification: If expected_cols is known, ensure we found a plausible match
    if expected_cols and len(valid_blocks) != expected_cols:
        # If count does not match expected, fall back to calibrated defaults
        return None, False, None

    if len(valid_blocks) in (2, 3, 4):
        # Successfully discovered columns autonomously!
        col_ranges = []
        for s, e in valid_blocks:
            abs_x1 = x_offset + s
            abs_x2 = x_offset + e
            col_ranges.append((round(abs_x1 / float(w), 3), round(abs_x2 / float(w), 3)))

        # Create diagnostic visualization image
        vis_h = 120
        vis_img = np.zeros((vis_h, w, 3), dtype=np.uint8)
        vis_img[:] = (245, 245, 245)

        # Plot density curve
        for x_idx in range(len(norm_density) - 1):
            px1 = x_offset + x_idx
            px2 = x_offset + x_idx + 1
            py1 = int(vis_h - 15 - norm_density[x_idx] * (vis_h - 30))
            py2 = int(vis_h - 15 - norm_density[x_idx + 1] * (vis_h - 30))
            cv2.line(vis_img, (px1, py1), (px2, py2), (180, 80, 0), 1)

        # Draw discovered column bands
        for (r1, r2) in col_ranges:
            bx1, bx2 = int(r1 * w), int(r2 * w)
            cv2.rectangle(vis_img, (bx1, 5), (bx2, vis_h - 5), (0, 180, 0), 2)
            cv2.putText(vis_img, "Column Block", (bx1 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)

        return col_ranges, True, vis_img

    return None, False, None


def infer_column_layout(num_rows, num_columns_override=None):
    """
    Standard calibrated fallback when paper density is too sparse to auto-segment.
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

    if num_cols == 2:
        col_ranges = [(0.296, 0.458), (0.563, 0.727)]
    elif num_cols == 3:
        col_ranges = [(0.18, 0.38), (0.44, 0.64), (0.70, 0.90)]
    elif num_cols == 4:
        col_ranges = [(0.10, 0.28), (0.33, 0.51), (0.56, 0.74), (0.78, 0.96)]
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
    1. Extracts timing marks for row Y-coordinates.
    2. Isolates the answer rows.
    3. Autonomously discovers column blocks via vertical density projection.
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

    if num_rows > 1:
        row_diffs = np.diff(answer_rows)
        median_spacing = float(np.median(row_diffs))
    else:
        median_spacing = 30.0

    bubble_radius = int(median_spacing * 0.32)
    bubble_radius = max(8, min(bubble_radius, 25))

    # --- Autonomous Column Discovery ---
    # First, attempt to discover the column bands directly from paper density
    expected_cols = num_columns_override
    auto_ranges, is_autonomous, proj_vis = discover_columns_from_density(
        pencil_mask, answer_rows, bubble_radius, expected_cols=expected_cols
    )

    if is_autonomous and auto_ranges:
        col_ranges = auto_ranges
        num_cols = len(col_ranges)
        discovery_method = "Autonomous Paper Density Projection"
    else:
        num_cols, col_ranges = infer_column_layout(num_rows, num_columns_override)
        discovery_method = "Calibrated Layout Profile (Fallback)"
        proj_vis = None

    return {
        "answer_rows": answer_rows,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "col_ranges": col_ranges,
        "bubble_radius": bubble_radius,
        "total_questions": num_rows * num_cols,
        "discovery_method": discovery_method,
        "is_autonomous": is_autonomous,
        "projection_vis": proj_vis,
    }
