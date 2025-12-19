import cv2
import numpy as np


def segment_lines(image):
    """Tách dòng bằng projection profile, có làm mượt và lọc nhiễu nhẹ."""
    # Hỗ trợ ảnh BGR hoặc ảnh xám
    if len(image.shape) == 2:
        gray = image
        vis_base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_base = image

    # Giảm nhiễu nhẹ trước khi threshold
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Projection theo trục ngang
    horizontal_projection = np.sum(binary, axis=1)
    threshold = np.max(horizontal_projection) * 0.1

    lines = []
    start = 0
    in_line = False

    for i, val in enumerate(horizontal_projection):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            end = i
            # Lọc nhiễu: chỉ giữ dòng cao hơn 8 pixel
            if (end - start) > 8:
                lines.append((start, end))
            in_line = False

    # Nốt cuối
    if in_line:
        lines.append((start, len(horizontal_projection)))

    # Cắt ảnh ra thành các ảnh con
    line_images = []
    visualization_image = vis_base.copy()

    for (y1, y2) in lines:
        roi = image[y1:y2, 0:image.shape[1]]
        line_images.append(roi)
        cv2.rectangle(visualization_image, (0, y1), (image.shape[1], y2), (0, 255, 0), 2)

    return line_images, visualization_image
