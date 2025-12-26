import cv2
import numpy as np

def word_segmentation(line_image):

    # Tiền xử lý: Nhị phân hóa và chuẩn bị ảnh visualization
    if len(line_image.shape) == 3:
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        vis_image = line_image.copy()
    else:
        gray = line_image
        vis_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR) # Để vẽ box màu
    
    # Giả định chữ đen nền trắng, dùng THRESH_BINARY_INV để chữ thành trắng, nền đen
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tìm Contours (Các nét dính liền)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lấy Bounding Box và lọc nhiễu
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 5: 
            bboxes.append((x, y, w, h))

    #Xử lý các trường hợp đặc biệt
    if not bboxes:
        return [], vis_image, [] # Không có gì để xử lý

    # Sắp xếp từ trái sang phải
    bboxes.sort(key=lambda b: b[0])

    # Nếu chỉ có 1 box -> đó là 1 từ
    if len(bboxes) == 1:
        x, y, w, h = bboxes[0]
        word_img = line_image[y:y+h, x:x+w]
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return [word_img], vis_image, [(x, y, x + w, y + h)]

    # Tính khoảng cách (gap) giữa các box liền kề
    gaps = []
    for i in range(len(bboxes) - 1):
        curr_box = bboxes[i]
        next_box = bboxes[i+1]
        gap = next_box[0] - (curr_box[0] + curr_box[2])
        gaps.append(gap)

    # Tìm ngưỡng (Threshold) để tách từ
    valid_gaps = [g for g in gaps if g > 0]
    if not valid_gaps:
        # Nếu không có khoảng cách dương, coi tất cả là 1 từ
        threshold = np.inf 
    else:
        # Heuristic: ngưỡng là 1.5 lần trung bình các khoảng cách dương
        threshold = np.mean(valid_gaps) * 1.5

    # Gom nhóm thành Từ (Words)
    words = []
    current_word_boxes = [bboxes[0]]

    for i in range(len(gaps)):
        gap = gaps[i]
        next_box = bboxes[i+1]

        if gap > threshold:
            words.append(current_word_boxes)
            current_word_boxes = [next_box]
        else:
            current_word_boxes.append(next_box)
    
    words.append(current_word_boxes)

    #Cắt ảnh (Crop) và tạo bounding box cho từng từ
    word_images = []
    word_boxes = []
    for group in words:
        x_min = min([b[0] for b in group])
        y_min = min([b[1] for b in group])
        x_max = max([b[0] + b[2] for b in group])
        y_max = max([b[1] + b[3] for b in group])
        
        # FIX: Thêm kiểm tra để đảm bảo bounding box và ảnh cắt ra hợp lệ
        if x_min < x_max and y_min < y_max:
            word_img = line_image[y_min:y_max, x_min:x_max]
            word_images.append(word_img)
            word_boxes.append((x_min, y_min, x_max, y_max))

            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return word_images, vis_image, word_boxes