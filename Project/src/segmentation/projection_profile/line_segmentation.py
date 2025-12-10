import cv2
import numpy as np

def segment_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    horizontal_projection = np.sum(binary, axis=1)

    result = image.copy()

    threshold = np.max(horizontal_projection) * 0.2
    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(horizontal_projection):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            end = i
            lines.append((start, end))
            in_line = False

    for (y1, y2) in lines:
        cv2.rectangle(result, (0, y1), (result.shape[1], y2), (0, 255, 0), 2)

    return result
