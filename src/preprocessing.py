"""
preprocessing.py — Image preprocessing utilities for skin lesion classification.

Includes CLAHE enhancement, lesion extraction, and standard transformations.
"""

import numpy as np
import cv2
from PIL import Image


def resize_image(image: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """Resize image to the target size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances contrast in dermoscopic images, making lesion boundaries
    and color variations more visible to the CNN.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l_channel)

    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def extract_lesion(image: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """
    Extract the lesion region from a dermoscopic image.

    Uses thresholding and morphological operations (opening + dilation)
    to create a mask isolating the lesion, then blacks out the background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def preprocess_pipeline(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline: resize → CLAHE → normalize.

    Args:
        image: Input RGB image as numpy array.
        target_size: Output dimensions (height, width).

    Returns:
        Preprocessed image as float32 array in [0, 1].
    """
    image = resize_image(image, target_size)
    image = apply_clahe(image)
    image = normalize_image(image)
    return image
