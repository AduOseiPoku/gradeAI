"""
Runner for the calibrated Dynamic OMR Engine.
Detects and circles all shaded answers on the sheet.
"""

import os
from dynamic_grader import grade_sheet

if __name__ == "__main__":
    image_path = r"C:\Users\adupo\Downloads\scanable sheet.PNG"
    if not os.path.exists(image_path):
        image_path = "debug_output_v2/1_original.jpg"

    grade_sheet(
        image_path=image_path,
        answer_key=None,  # Pure shaded answer detection mode
        output_path="graded_sheet_output.png",
        save_debug=True,
        debug_dir="debug_output"
    )