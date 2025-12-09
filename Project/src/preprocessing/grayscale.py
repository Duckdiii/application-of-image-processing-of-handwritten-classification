"# Placeholder: convert image to grayscale\n"
import cv2


def apply_grayscale(image):

    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
