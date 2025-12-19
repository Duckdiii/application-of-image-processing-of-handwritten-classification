import cv2
import numpy as np


def _crop_to_content(binary_slice):
    coords = cv2.findNonZero(binary_slice)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def segment_characters(line_image, min_char_width=15, min_char_height=15, pad=2):

    if len(line_image.shape) == 2:
        gray = line_image
        base_color = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        base_color = line_image

    # Làm mượt nhẹ và threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Nối các nét đứt nhỏ theo chiều ngang
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)

    # Vertical Projection
    vertical_projection = np.sum(binary, axis=0)
    threshold = np.max(vertical_projection) * 0.1

    chars = []
    start = 0
    in_char = False

    for i, val in enumerate(vertical_projection):
        if val > threshold and not in_char:
            start = i
            in_char = True
        elif val <= threshold and in_char:
            end = i
            if (end - start) >= min_char_width:
                chars.append((start, end))
            in_char = False
    if in_char:
        chars.append((start, len(vertical_projection)))

    char_images = []
    visualization = base_color.copy()

    for (x1, x2) in chars:
        # Tìm bounding box thực tế trong slice để bỏ khoảng trắng dư
        slice_bin = binary[:, x1:x2]
        bbox = _crop_to_content(slice_bin)
        if bbox is None:
            continue
        bx, by, bw, bh = bbox

        x_start = x1 + bx
        x_end = x_start + bw
        y_start = by
        y_end = y_start + bh

        # Thêm padding và bảo vệ biên
        x_start = max(0, x_start - pad)
        x_end = min(binary.shape[1], x_end + pad)
        y_start = max(0, y_start - pad)
        y_end = min(binary.shape[0], y_end + pad)

        # Bỏ qua nếu quá nhỏ
        if (x_end - x_start) < min_char_width or (y_end - y_start) < min_char_height:
            continue

        roi = line_image[y_start:y_end, x_start:x_end]
        char_images.append(roi)

        # Vẽ khung hiển thị
        cv2.rectangle(visualization, (x_start, y_start), (x_end, y_end), (0, 0, 255), 1)

    return char_images, visualization
