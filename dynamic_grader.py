"""
Dynamic Grader - Main CLI entrypoint for grading arbitrary OMR sheets.
Supports phone camera photos, blind pink ink dropout, dynamic row/column discovery,
and variable question counts (30, 40, 50, 180 questions).
"""

import argparse
import os
import sys
import cv2
import numpy as np

from omr.document_detector import detect_and_warp_document
from omr.preprocessor import preprocess_sheet
from omr.grid_detector import detect_answer_grid
from omr.evaluator import evaluate_bubbles
from omr.visualizer import annotate_graded_sheet

# Optional 30-Question Answer Key
SAMPLE_ANSWER_KEY_30 = {
    1: 'B',  2: 'D',  3: 'D',  4: 'B',  5: 'C',
    6: 'B',  7: 'C',  8: 'A',  9: 'D', 10: 'B',
    11: 'C', 12: 'D', 13: 'C', 14: 'D', 15: 'B',
    16: 'B', 17: 'C', 18: 'D', 19: 'A', 20: 'A',
    21: 'C', 22: 'D', 23: 'B', 24: 'A', 25: 'C',
    26: 'A', 27: 'C', 28: 'D', 29: 'B', 30: 'A'
}


def pick_image_gui():
    """Opens a native file picker dialog if available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select an Answer Sheet Photo / Scan",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception:
        return None


def grade_sheet(image_path, answer_key=None, output_path="graded_sheet_output.png",
                num_columns_override=None, save_debug=True, debug_dir="debug_output"):
    """
    Full end-to-end shaded answer detection pipeline:
      1. Load image and warp document boundary.
      2. Drop pink ink and create inverted pencil mask.
      3. Dynamically discover timing marks and answer columns.
      4. Sample bubbles by fill ratio and classify shaded marks.
      5. Circle every shaded answer with bold rings & export visual overlay.
    """
    if not os.path.exists(image_path):
        print(f"Error: Input file does not exist: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    if save_debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    print(f"\n[1/5] Processing sheet: {image_path}")
    print(f"      Original image dimensions: {img.shape[1]}x{img.shape[0]}")

    # Step 1: Document boundary detection & perspective flattening
    warped, was_warped = detect_and_warp_document(img, target_size=(1600, 2200))
    warp_status = "Perspective corrected" if was_warped else "Standardized/Resized directly"
    print(f"[2/5] Page alignment: {warp_status} -> 1600x2200")
    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "1_warped_sheet.jpg"), warped)

    # Step 2: Blind ink dropout (Red channel) & pencil mask
    red_channel, pencil_mask = preprocess_sheet(warped)
    print(f"[3/5] Preprocessing: Pink ink dropped via Red channel")
    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "2_red_channel.jpg"), red_channel)
        cv2.imwrite(os.path.join(debug_dir, "3_pencil_mask.jpg"), pencil_mask)

    # Step 3: Dynamic Answer Grid & Timing Marks
    grid_config = detect_answer_grid(
        pencil_mask,
        num_columns_override=num_columns_override,
    )
    print(f"[4/5] Grid discovery:")
    print(f"      - Detected answer rows: {grid_config['num_rows']}")
    print(f"      - Inferred columns: {grid_config['num_cols']}")
    print(f"      - Total questions: {grid_config['total_questions']}")
    print(f"      - Bubble radius: {grid_config['bubble_radius']} px")

    # Step 4: Bubble sampling & answer extraction
    results, summary = evaluate_bubbles(
        pencil_mask,
        grid_config,
        answer_key=answer_key,
    )

    # Step 5: Visual annotation & export
    annotated = annotate_graded_sheet(warped, results, summary)
    cv2.imwrite(output_path, annotated)
    print(f"[5/5] Success! Annotated output saved to: {output_path}")

    # Display results table
    print("\n" + "=" * 50)
    print(f"{'Q#':<5} | {'Detected':<10} | {'Confidence':<10} | {'Status'}")
    print("-" * 50)
    for q_num in sorted(results.keys()):
        data = results[q_num]
        status = ""
        if answer_key and q_num in answer_key:
            corr = answer_key[q_num]
            status = "✓" if data['is_correct'] else f"✗ (Key: {corr})"
        else:
            status = "Shaded" if data['chosen'] not in ("BLANK", "AMBIGUOUS") else data['chosen']
        print(f"Q{q_num:<4} | {data['chosen']:<10} | {data['confidence']:<10.2f} | {status}")

    print("=" * 50)
    if summary.get("score") is not None:
        pct = (summary['score'] / float(summary['max_score'])) * 100
        print(f"\nFINAL SCORE: {summary['score']} / {summary['max_score']} ({pct:.1f}%)")
    print(f"Total Detected: {summary['answered']} / {summary['total_questions']} Shaded | "
          f"Blank: {summary['blank']} | Ambiguous: {summary['ambiguous']}\n")

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Universal Dynamic OMR Grading Engine")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to answer sheet image")
    parser.add_argument("--output", "-o", type=str, default="graded_sheet_output.png", help="Output path")
    parser.add_argument("--cols", "-c", type=int, default=None, help="Force number of question columns")
    parser.add_argument("--key", action="store_true", help="Compare answers against the sample answer key")
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        candidates = [
            r"C:\Users\adupo\Downloads\scanable sheet.PNG",
            "debug_output_v2/1_original.jpg",
            "sample_sheet.png",
        ]
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break

    if not input_path:
        print("No image path specified. Opening file picker...")
        input_path = pick_image_gui()

    if not input_path:
        print("No image selected. Usage: python dynamic_grader.py --input path\\to\\sheet.jpg")
        sys.exit(1)

    key = SAMPLE_ANSWER_KEY_30 if args.key else None
    grade_sheet(input_path, answer_key=key, output_path=args.output, num_columns_override=args.cols)


if __name__ == "__main__":
    main()
