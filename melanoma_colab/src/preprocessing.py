import cv2
import numpy as np


def remove_hairs(image: np.ndarray) -> np.ndarray:
    """
    Remove hair artifacts from dermoscopic images using
    Black-Hat morphological transformation + Telea inpainting.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    return result


def resize_image(image: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize image to (size, size)."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
