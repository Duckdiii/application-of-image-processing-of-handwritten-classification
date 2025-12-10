import cv2

def apply_canny(image, low=50, high=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, low, high)

    return edges
