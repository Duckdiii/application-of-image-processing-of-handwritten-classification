import cv2
from ultralytics import YOLO
import torch
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    DEVICE = 0
    print(f"YOLOv8 Classifier: GPU được phát hiện - {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print(" YOLOv8 Classifier: GPU không được phát hiện, dùng CPU.")

class YOLOv8Classifier:

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"File mô hình classification không tồn tại: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(DEVICE)
        
        # Lấy tên các lớp từ model
        self.class_names = self.model.names

    def predict(self, image):

        results = self.model(image, verbose=False) # Chạy model
        
        if not results:
            return None, None
            
        # results là một list, ta lấy phần tử đầu tiên
        result = results[0]
        
        # Lấy index của lớp có xác suất cao nhất
        top1_index = result.probs.top1
        # Lấy xác suất tương ứng
        top1_confidence = result.probs.top1conf.item()
        
        # Lấy tên lớp từ index
        class_name = self.class_names[top1_index]
        
        return class_name, top1_confidence

if __name__ == '__main__':
    cls_model_path = "path/to/your/classification_model.pt"
    image_to_test = "path/to/your/test_image.jpg"

    if not Path(cls_model_path).exists() or not Path(image_to_test).exists():
        print("Vui lòng đặt đường dẫn chính xác cho model và ảnh test.")
    else:
        try:
            # 1. Khởi tạo classifier
            classifier = YOLOv8Classifier(model_path=cls_model_path)
            
            # 2. Đọc ảnh
            img = cv2.imread(image_to_test)
            
            # 3. Dự đoán
            predicted_class, confidence = classifier.predict(img)
            
            # 4. In kết quả
            if predicted_class is not None:
                print(f"Dự đoán thành công!")
                print(f"   - Lớp: '{predicted_class}'")
                print(f"   - Độ tin cậy: {confidence:.2f}")
                
                # Hiển thị ảnh với kết quả
                cv2.putText(img, f"{predicted_class} ({confidence:.2f})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Kết quả dự đoán", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")
