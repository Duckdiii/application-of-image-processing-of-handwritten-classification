import cv2
import numpy as np

def get_deskew_angle(image):
    """
    Hàm tính toán góc nghiêng của văn bản trong ảnh.
    Phiên bản cải tiến để tránh lỗi xoay 90 độ.
    """
    # 1. Chuyển sang ảnh xám nếu cần
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Nhị phân hóa (Đảo ngược màu: Chữ trắng, nền đen)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 3. Lấy tọa độ các điểm chữ
    coords = np.column_stack(np.where(thresh > 0))
    
    # 4. Tìm hình chữ nhật bao quanh tối thiểu
    # rect là một tuple: ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(coords)
    (center), (w, h), angle = rect
    
    # --- LOGIC MỚI ĐỂ XỬ LÝ GÓC ---
    
    # OpenCV trả về góc trong khoảng [-90, 0).
    # Nếu w < h, nghĩa là OpenCV đang coi cạnh ngắn là 'width' và cạnh dài là 'height'.
    # Góc 'angle' lúc này là góc của cạnh ngắn với trục hoành.
    # Góc của cạnh dài (hướng văn bản) sẽ là angle + 90.
    if w < h:
        angle = angle + 90
        
    # Sau bước trên, 'angle' là góc của cạnh dài văn bản so với phương ngang.
    # Góc này có thể dương (nghiêng lên) hoặc âm (nghiêng xuống).
    # Chúng ta cần xoay ngược lại một góc '-angle' để đưa nó về 0 độ.
    rotation_angle = -angle

    # KHIÊN BẢO VỆ: Đảm bảo chỉ xoay các góc nhỏ.
    # Nếu góc xoay quá lớn (> 45 hoặc < -45), nghĩa là đã có sự nhầm lẫn
    # giữa ngang và dọc. Ta cộng/trừ 90 để đưa nó về phạm vi góc nhỏ.
    # Điều này ngăn chặn việc ảnh bị xoay đứng 90 độ.
    if rotation_angle > 45:
        rotation_angle -= 90
    elif rotation_angle < -45:
        rotation_angle += 90
        
    return rotation_angle

def rotate_image(image, angle):
    # (Hàm này giữ nguyên như cũ)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def deskew(image):
    # (Hàm này giữ nguyên như cũ, có thể bỏ print nếu không cần debug)
    angle = get_deskew_angle(image)
    
    # Chỉ xoay nếu góc nghiêng đáng kể (ví dụ > 0.1 độ)
    if abs(angle) < 0.1:
        return image

    return rotate_image(image, angle)