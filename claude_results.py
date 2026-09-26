r"""
OMR Prototype v2 - Timing-mark sheet layout
---------------------------------------------
Built for sheets like your "MCQ Answer Sheet" template:
    - A column of timing marks (tick marks) down the left edge instead
      of 4 corner squares
    - A register-number bubble grid (columns x digits 0-9) instead of
      a QR code
    - Answers laid out in TWO columns (e.g. Q1-15 left, Q16-30 right),
      4 options (A-D) per row

Usage:
    python omr_test_v2.py
    (file picker opens - select your sheet image)

    or:
    python omr_test_v2.py path\to\image.jpg

Output: numbered debug images + results.txt in "debug_output_v2"

IMPORTANT: the coordinate estimates in CONFIG below are PROPORTIONAL
GUESSES based on the sheet layout you shared, not measurements of
your actual printed sheet. Expect to tune these once you see the
debug images against a real photo - that's the whole point of this
script existing. Read the comments next to each CONFIG value.
"""

import cv2
import numpy as np
import os
import sys

DEBUG_DIR = "debug_output_v2"

# ---------------------------------------------------------------------------
# CONFIG - tune these against YOUR real sheet using the debug images.
# All positions are fractions (0.0-1.0) of the WARPED sheet's width/height,
# so they survive different photo resolutions.
# ---------------------------------------------------------------------------
CONFIG = {
    # Vertical strip (as a fraction of page width) where the left-edge
    # timing marks live. Widen this if 8_timing_marks.jpg misses marks,
    # narrow it if it's picking up other stuff (bubble grid, text).
    "timing_strip_x_range": (0.0, 0.06),

    # How many timing marks correspond to the register-number section
    # vs. the answer section. In your sheet: register number has 10 rows
    # (digits 0-9), and the answer section has 15 row-positions shared
    # by both columns (row for Q1 = row for Q16, etc).
    "num_register_rows": 10,
    "num_answer_row_positions": 15,

    # Register number grid: how many digit columns, and roughly where
    # the grid starts/ends horizontally (fraction of page width).
    # MEASURE this from your real sheet - current values are a rough
    # guess based on the image you shared.
    "register_num_digits": 9,
    "register_x_range": (0.08, 0.45),

    # Answer bubbles: each column (left block = Q1-15, right block =
    # Q16-30) has 4 options (A-D). x_range is where that column's own
    # 4 bubbles live, as a fraction of page width.
    "answer_options": 4,
    "left_block_x_range": (0.08, 0.42),
    "right_block_x_range": (0.55, 0.90),

    # Fill-ratio thresholds - same idea as v1, will need tuning against
    # real shaded bubbles vs. blank ones.
    "blank_threshold": 0.20,
    "ambiguous_gap": 0.10,
}

WARP_OUTPUT_SIZE = (850, 1100)  # (width, height) - matches typical A4 portrait ratio


def pick_image_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter not available. Pass the image path as a command-line argument:")
        print("    python omr_test_v2.py path\\to\\image.jpg")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select an answer sheet photo/scan",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)
    return file_path


def ensure_debug_dir():
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)


def save_debug(name, img):
    path = os.path.join(DEBUG_DIR, name)
    cv2.imwrite(path, img)
    print(f"  saved: {path}")


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: could not read image at {path}")
        sys.exit(1)
    return img


def adaptive_thresh(gray):
    """
    Local/adaptive threshold instead of a single global Otsu value -
    much more resilient to uneven lighting across a phone photo than
    the global threshold used in v1.
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )


def detect_page_corners(thresh_img, original_img):
    """
    This sheet has no corner fiducial squares, so instead we try to find
    the page's own outer boundary (the largest rectangular contour) and
    use that for perspective correction.

    NOTE: this only works well if the sheet is photographed with some
    contrast against its background (e.g. on a dark desk/table). If your
    sheet fills the entire photo frame edge-to-edge (like a flatbed scan
    or a tightly cropped screenshot), there IS no separate page boundary
    to detect - in that case this function will correctly return None,
    and main() falls back to treating the input image as already-flat.
    """
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = original_img.shape[0] * original_img.shape[1]
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # The page itself should be a large fraction of the frame, but if it's
    # the ENTIRE frame (no background visible), approxPolyDP often won't
    # cleanly return 4 points - that's our signal to fall back gracefully.
    if area < img_area * 0.5:
        return None

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        return None

    return approx.reshape(4, 2)


def order_corners(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def warp_perspective(original_img, corners, output_size=WARP_OUTPUT_SIZE):
    ordered = order_corners(corners)
    dst = np.array(
        [[0, 0], [output_size[0] - 1, 0],
         [output_size[0] - 1, output_size[1] - 1], [0, output_size[1] - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(original_img, matrix, output_size)


def detect_timing_marks(warped_gray, x_range):
    """
    Finds the tick marks in the left-edge timing-mark strip and returns
    their y-centers, sorted top to bottom. These y-positions become the
    row grid for BOTH the register-number section and the answer
    section - measured from the actual marks rather than guessed from
    even spacing, which is the key improvement over v1's placeholder grid.
    """
    h, w = warped_gray.shape[:2]
    x1, x2 = int(w * x_range[0]), int(w * x_range[1])
    strip = warped_gray[:, x1:x2]

    strip_thresh = cv2.adaptiveThreshold(
        strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    contours, _ = cv2.findContours(strip_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    strip_area = strip.shape[0] * strip.shape[1]
    marks = []
    for c in contours:
        area = cv2.contourArea(c)
        # Timing marks are small, roughly uniform blobs - filter obvious
        # noise (specks) and obvious non-marks (large blobs/text bleed).
        if strip_area * 0.0003 < area < strip_area * 0.02:
            x, y, mw, mh = cv2.boundingRect(c)
            cy = y + mh // 2
            marks.append(cy)

    marks = sorted(marks)

    # Merge marks that are suspiciously close together (duplicate
    # contours from the same physical tick) before returning.
    merged = []
    for m in marks:
        if not merged or (m - merged[-1]) > (warped_gray.shape[0] * 0.01):
            merged.append(m)
    return merged


def partition_timing_marks(marks, num_register_rows, num_answer_rows):
    """
    Splits the full list of detected timing-mark y-positions into the
    register-number rows (first N) and answer-row positions (next M).
    Returns (None, None) if the count doesn't match what's expected, so
    you know to go tune detect_timing_marks() rather than silently
    reading garbage positions.
    """
    expected = num_register_rows + num_answer_rows
    if len(marks) != expected:
        print(f"  WARNING: expected {expected} timing marks "
              f"({num_register_rows} register + {num_answer_rows} answer), "
              f"found {len(marks)}.")
        print("  Check 8_timing_marks.jpg - tune 'timing_strip_x_range' or the "
              "area filters in detect_timing_marks() to match what you see.")
        return None, None

    register_ys = marks[:num_register_rows]
    answer_ys = marks[num_register_rows:]
    return register_ys, answer_ys


def sample_fill_ratio(gray, cx, cy, radius):
    h, w = gray.shape[:2]
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return 1 - (np.mean(roi) / 255)


def decode_register_number(gray, register_ys, x_range, num_digits, debug_img):
    h, w = gray.shape[:2]
    x1, x2 = int(w * x_range[0]), int(w * x_range[1])
    col_spacing = (x2 - x1) / num_digits
    radius = int(col_spacing * 0.25)

    digits = []
    confidences = []

    for col in range(num_digits):
        cx = int(x1 + col_spacing * (col + 0.5))
        fill_ratios = []
        for row_idx, cy in enumerate(register_ys):
            ratio = sample_fill_ratio(gray, cx, cy, radius)
            fill_ratios.append(ratio)
            color = (0, 0, 255) if ratio > CONFIG["blank_threshold"] else (255, 0, 0)
            cv2.circle(debug_img, (cx, cy), radius, color, 1)

        sorted_ratios = sorted(fill_ratios, reverse=True)
        best_idx = int(np.argmax(fill_ratios))
        blank = sorted_ratios[0] < CONFIG["blank_threshold"]
        ambiguous = (sorted_ratios[0] - sorted_ratios[1]) < CONFIG["ambiguous_gap"]

        if blank:
            digits.append("_")
        elif ambiguous:
            digits.append("?")
        else:
            digits.append(str(best_idx))  # row 0 = digit "0", row 9 = digit "9"
        confidences.append(round(sorted_ratios[0], 3))

    return "".join(digits), confidences


def decode_answer_block(gray, row_ys, x_range, num_options, question_offset, debug_img):
    h, w = gray.shape[:2]
    x1, x2 = int(w * x_range[0]), int(w * x_range[1])
    col_spacing = (x2 - x1) / num_options
    radius = int(min(col_spacing, (row_ys[1] - row_ys[0] if len(row_ys) > 1 else 20)) * 0.3)

    results = []
    for i, cy in enumerate(row_ys):
        fill_ratios = []
        for opt in range(num_options):
            cx = int(x1 + col_spacing * (opt + 0.5))
            ratio = sample_fill_ratio(gray, cx, cy, radius)
            fill_ratios.append(ratio)
            color = (0, 0, 255) if ratio > CONFIG["blank_threshold"] else (255, 0, 0)
            cv2.circle(debug_img, (cx, cy), radius, color, 1)

        sorted_ratios = sorted(fill_ratios, reverse=True)
        best_idx = int(np.argmax(fill_ratios))
        blank = sorted_ratios[0] < CONFIG["blank_threshold"]
        ambiguous = (sorted_ratios[0] - sorted_ratios[1]) < CONFIG["ambiguous_gap"]

        answer = "BLANK" if blank else ("AMBIGUOUS" if ambiguous else chr(65 + best_idx))
        results.append({
            "question": question_offset + i + 1,
            "fill_ratios": [round(r, 3) for r in fill_ratios],
            "answer": answer,
        })
    return results


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Opening file picker - select your answer sheet image...")
        image_path = pick_image_file()

    print(f"\nProcessing: {image_path}\n")
    ensure_debug_dir()

    img = load_image(image_path)
    save_debug("1_original.jpg", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_debug("2_grayscale.jpg", gray)

    thresh = adaptive_thresh(gray)
    save_debug("3_adaptive_threshold.jpg", thresh)

    # --- Perspective correction (page-border based, with graceful fallback) ---
    corners = detect_page_corners(thresh, img)
    if corners is not None:
        corners_img = img.copy()
        for (x, y) in corners:
            cv2.circle(corners_img, (int(x), int(y)), 10, (0, 255, 0), 3)
        save_debug("4_page_corners.jpg", corners_img)

        warped = warp_perspective(img, corners)
        print("  Page corners found - applied perspective correction.")
    else:
        warped = cv2.resize(img, WARP_OUTPUT_SIZE)
        print("  No separate page boundary found - treating input as already-flat")
        print("  (expected if your sheet fills the whole photo/screenshot frame).")

    save_debug("5_warped_or_resized.jpg", warped)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    save_debug("6_warped_grayscale.jpg", warped_gray)

    warped_thresh = adaptive_thresh(warped_gray)
    save_debug("7_warped_threshold.jpg", warped_thresh)

    # --- Timing mark detection (replaces v1's corner-fiducial approach for row grid) ---
    marks = detect_timing_marks(warped_gray, CONFIG["timing_strip_x_range"])

    marks_debug = warped.copy()
    strip_x1 = int(warped.shape[1] * CONFIG["timing_strip_x_range"][0])
    strip_x2 = int(warped.shape[1] * CONFIG["timing_strip_x_range"][1])
    cv2.rectangle(marks_debug, (strip_x1, 0), (strip_x2, warped.shape[0]), (255, 0, 0), 2)
    for y in marks:
        cv2.line(marks_debug, (0, y), (warped.shape[1], y), (0, 0, 255), 1)
        cv2.circle(marks_debug, (strip_x1 + 10, y), 5, (0, 255, 0), -1)
    save_debug("8_timing_marks.jpg", marks_debug)
    print(f"  Found {len(marks)} timing marks.")

    register_ys, answer_ys = partition_timing_marks(
        marks, CONFIG["num_register_rows"], CONFIG["num_answer_row_positions"]
    )

    results_lines = []

    if register_ys is None:
        results_lines.append(
            "Could not partition timing marks - see warning above. "
            "Tune CONFIG['timing_strip_x_range'] and re-run."
        )
        print("\nStopping here - fix timing mark detection before bubble decoding.")
    else:
        decode_debug = warped.copy()

        register_number, reg_confidences = decode_register_number(
            warped_gray, register_ys, CONFIG["register_x_range"],
            CONFIG["register_num_digits"], decode_debug,
        )
        results_lines.append(f"Register Number: {register_number}")
        results_lines.append(f"  (per-digit confidence: {reg_confidences})")
        print(f"\nRegister Number: {register_number}")

        left_results = decode_answer_block(
            warped_gray, answer_ys, CONFIG["left_block_x_range"],
            CONFIG["answer_options"], question_offset=0, debug_img=decode_debug,
        )
        right_results = decode_answer_block(
            warped_gray, answer_ys, CONFIG["right_block_x_range"],
            CONFIG["answer_options"], question_offset=CONFIG["num_answer_row_positions"],
            debug_img=decode_debug,
        )

        results_lines.append("\nAnswers:")
        for l, r in zip(left_results, right_results):
            line = (f"Q{l['question']:2d}: {l['answer']:10s} ratios={l['fill_ratios']}   |   "
                    f"Q{r['question']:2d}: {r['answer']:10s} ratios={r['fill_ratios']}")
            print(line)
            results_lines.append(line)

        save_debug("9_decoded_bubbles.jpg", decode_debug)

    with open(os.path.join(DEBUG_DIR, "results.txt"), "w") as f:
        f.write("\n".join(results_lines))

    print(f"\nAll debug images saved to: {os.path.abspath(DEBUG_DIR)}")
    print("Start by checking 8_timing_marks.jpg - if the count is wrong, everything")
    print("downstream (register number + answers) will be misread. Tune CONFIG")
    print("at the top of this script based on what you see, then re-run.")


if __name__ == "__main__":
    main()