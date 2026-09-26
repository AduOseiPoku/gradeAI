"""
OMR Package - Universal, phone-camera resilient Optical Mark Recognition engine.
"""

from .document_detector import detect_and_warp_document
from .preprocessor import preprocess_sheet, drop_pink_ink
from .grid_detector import detect_answer_grid
from .evaluator import evaluate_bubbles
from .visualizer import annotate_graded_sheet

__all__ = [
    "detect_and_warp_document",
    "preprocess_sheet",
    "drop_pink_ink",
    "detect_answer_grid",
    "evaluate_bubbles",
    "annotate_graded_sheet",
]
