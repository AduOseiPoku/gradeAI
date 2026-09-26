"""
Document Detector - Phone photo boundary detection and perspective flattening.
"""

import cv2
import numpy as np


def order_quad_points(pts):
    """
    Orders 4 points as: [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff (y - x), bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_and_warp_document(image, target_size=(1600, 2200), min_area_ratio=0.45):
    """
    Finds the 4 corners of a paper sheet on a desk/background and applies perspective warp.
    If the image is already flat/cropped (no clear outer boundary > min_area_ratio),
    it safely resizes directly to target_size without distortion.

    Args:
        image: Original BGR image from camera or file.
        target_size: (width, height) output dimensions.
        min_area_ratio: Minimum fraction of frame area for paper contour.

    Returns:
        (warped_img, was_warped)
    """
    h, w = image.shape[:2]
    total_area = h * w

    # Work on a downscaled image for fast edge and contour extraction
    proc_h = 800
    scale = h / float(proc_h)
    proc_w = int(w / scale)
    small = cv2.resize(image, (proc_w, proc_h))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu or Canny with morphological closing to seal page boundary
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_quad = None
    for c in contours:
        area = cv2.contourArea(c) * (scale * scale)
        if area < total_area * min_area_ratio:
            # If largest contour is smaller than 45% of frame, don't warp inner boxes!
            break

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            paper_quad = approx.reshape(4, 2) * scale
            break

    if paper_quad is None:
        # Fallback: image already fills frame or is flat scan
        warped = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return warped, False

    ordered = order_quad_points(paper_quad)
    tw, th = target_size
    dst = np.array([
        [0, 0],
        [tw - 1, 0],
        [tw - 1, th - 1],
        [0, th - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, target_size, flags=cv2.INTER_CUBIC)
    return warped, True
