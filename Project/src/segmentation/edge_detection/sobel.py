import cv2
import numpy as np


def apply_sobel(image):
    """Sobel magnitude cho cả ảnh xám và BGR."""
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)
    sobel = cv2.convertScaleAbs(magnitude)
    return sobel
