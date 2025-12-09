"# Placeholder: median filter implementation\n"
import cv2


def apply_median_filter(image, kernel_size=5):

    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)
