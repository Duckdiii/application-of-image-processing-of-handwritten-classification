"# Placeholder: erosion operation\n"
import cv2
import numpy as np


def apply_erosion(image, kernel_size=5, iterations=1):
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)
