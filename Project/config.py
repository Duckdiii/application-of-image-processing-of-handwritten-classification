import os

# Đường dẫn chính (điều chỉnh tại đây nếu thay đổi vị trí dự án/ dữ liệu)
BASE_DATA_DIR = r"D:\Data\Python\digital-image-processing"

# Thư mục lưu kết quả preprocessing (batch)
ARCHIVE_PREPROCESSING_DIR = os.path.join(BASE_DATA_DIR, "archive", "preprocessing")

# Thư mục mặc định khi mở Segmentation UI (dùng ảnh đã preprocessing)
DEFAULT_PREPROCESSING_DATASET = ARCHIVE_PREPROCESSING_DIR

# Thư mục lưu kết quả segmentation (ảnh đã xử lý, line/char crop)
ARCHIVE_SEGMENTATION_DIR = os.path.join(BASE_DATA_DIR, "archive", "segmentation")

# ============================ MODEL PATHS ============================
# Cập nhật đường dẫn đến file model của bạn tại đây.
# Ví dụ: "D:/models/yolov8_detection.pt"

# --- Model YOLOv8 cho Object Detection ---
YOLOV8_DETECTION_MODEL_PATH = "D:\\Data\\Python\\digital-image-processing\\result2\\runs\\result_train_det\\weights\\best.pt"

# --- Model YOLOv8 cho Classification ---
# Tải model classification của bạn và đặt đường dẫn ở đây
YOLOV8_CLASSIFICATION_MODEL_PATH = "D:\Data\\Python\\digital-image-processing\\model\\best.pt"
