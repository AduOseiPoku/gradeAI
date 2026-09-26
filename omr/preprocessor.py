"""
Preprocessor - Blind color dropout (pink/red ink suppression) and shadow-tolerant thresholding.
"""

import cv2
import numpy as np


def drop_pink_ink(image_bgr):
    """
    Suppresses pink/magenta ink by utilizing the Red channel.
    Pink ink reflects red almost as brightly as white paper,
    rendering printed text, boxes, and instructions invisible,
    while black graphite/pen and black timing marks stay dark.
    """
    b, g, r = cv2.split(image_bgr)
    return r


def normalize_lighting(gray_img, kernel_size=45):
    """
    Corrects uneven shadows and lighting gradients common in phone photos
    using morphological background estimation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Dilate estimates the local background brightness
    bg = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel)
    # Absolute difference removes low-frequency lighting gradients
    diff = cv2.absdiff(bg, gray_img)
    normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized


def preprocess_sheet(image_bgr):
    """
    Full preprocessing pipeline:
    1. Red channel extraction for pink dropout.
    2. Adaptive thresholding to generate a clean binary mask.
       Pencil marks and timing marks are 255 (White), background is 0 (Black).
    """
    red_channel = drop_pink_ink(image_bgr)
    
    # Adaptive thresholding handles varying phone exposure across the page
    pencil_mask = cv2.adaptiveThreshold(
        red_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=15
    )

    # Remove very tiny dust/camera noise specks with a 2x2 opening
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    pencil_mask = cv2.morphologyEx(pencil_mask, cv2.MORPH_OPEN, clean_kernel)

    return red_channel, pencil_mask
