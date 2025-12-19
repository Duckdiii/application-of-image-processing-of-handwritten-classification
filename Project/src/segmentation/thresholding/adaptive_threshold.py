import cv2


def apply_adaptive_threshold(image, block_size=15, C=2):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Đảm bảo block_size là số lẻ và tối thiểu 3
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )

    return binary
