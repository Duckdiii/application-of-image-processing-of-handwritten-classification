"# Placeholder: Gaussian blur implementation\n"
import cv2


def apply_gaussian_blur(image, kernel_size=5):

    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
