import cv2


def apply_canny(image, low=50, high=150):
    """Canny cho cả ảnh xám và BGR; làm mượt nhẹ trước khi detect."""
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges
