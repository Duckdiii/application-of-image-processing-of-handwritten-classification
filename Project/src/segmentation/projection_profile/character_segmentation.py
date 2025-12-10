import cv2
import numpy as np

def segment_characters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    vertical_projection = np.sum(binary, axis=0)

    result = image.copy()

    threshold = np.max(vertical_projection) * 0.2
    chars = []
    in_char = False
    start = 0

    for i, val in enumerate(vertical_projection):
        if val > threshold and not in_char:
            start = i
            in_char = True
        elif val <= threshold and in_char:
            end = i
            chars.append((start, end))
            in_char = False

    for (x1, x2) in chars:
        cv2.rectangle(result, (x1, 0), (x2, result.shape[0]), (0, 0, 255), 2)

    return result
