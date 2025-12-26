import cv2
import numpy as np
from pathlib import Path
import torch
import warnings

# Tắt warnings
warnings.filterwarnings('ignore')

# Import YOLO với xử lý lỗi GIT
try:
    from ultralytics import YOLO
except ImportError as e:
    if 'GIT' in str(e):
        print("⚠ Cảnh báo: Git không được phát hiện, vẫn cố gắng tiếp tục...")
        # Cài git hoặc thử import lại
        import sys
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython", "-q"])
            from ultralytics import YOLO
        except:
            raise ImportError("Vui lòng cài Git hoặc chạy: pip install gitpython")
    else:
        raise

MODEL_PATH = "D:\\Data\\Python\\digital-image-processing\\result2\\runs\\result_train_det\\weights\\best.pt"
CONFIDENCE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.45

# Thiết lập device - Dùng GPU nếu có sẵn, nếu không thì dùng CPU
if torch.cuda.is_available():
    DEVICE = 0  # GPU device 0
    print(f"✓ GPU được phát hiện: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = 'cpu'
    print("GPU không được phát hiện, dùng CPU")


class YOLOv8Detector:

    def __init__(self, model_path=None, confidence=CONFIDENCE_THRESHOLD):
        if model_path is None:
            model_path = MODEL_PATH
            
        if model_path is None:
            raise ValueError("Vui lòng cung cấp đường dẫn đến mô hình YOLOv8!")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"File mô hình không tồn tại: {model_path}")
        
        self.model = YOLO(model_path)
        self.confidence = confidence

        # Try to get class name mapping from the model if available
        self.class_names = None
        try:
            # ultralytics YOLO object may store names under model.names or names
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            elif hasattr(self.model, 'names'):
                self.class_names = self.model.names
        except Exception:
            self.class_names = None
    
    def detect(self, image):

        results = self.model(image, conf=self.confidence, iou=IOU_THRESHOLD, device=DEVICE)
        return results
    
    def detect_with_visualization(self, image):

        results = self.detect(image)
        
        # Vẽ bounding boxes
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Vẽ bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Vẽ label và confidence (nếu có tên lớp thì dùng tên)
                label_text = str(int(cls))
                try:
                    if self.class_names is not None:
                        # class_names may be dict or list
                        if isinstance(self.class_names, dict):
                            label_text = f"{self.class_names.get(int(cls), int(cls))}: {conf:.2f}"
                        else:
                            label_text = f"{self.class_names[int(cls)]}: {conf:.2f}"
                    else:
                        label_text = f"{int(cls)}: {conf:.2f}"
                except Exception:
                    label_text = f"{int(cls)}: {conf:.2f}"

                cv2.putText(
                    annotated_image,
                    label_text,
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return annotated_image, results
    
    def get_detections_info(self, results):

        detections_info = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, classes):
                x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
                x_center = (x1f + x2f) / 2.0
                y_center = (y1f + y2f) / 2.0

                # Resolve label from class if available
                label = None
                try:
                    if self.class_names is not None:
                        if isinstance(self.class_names, dict):
                            label = str(self.class_names.get(int(cls), int(cls)))
                        else:
                            label = str(self.class_names[int(cls)])
                    else:
                        label = str(int(cls))
                except Exception:
                    label = str(int(cls))

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'label': label,
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'x_center': x_center,
                    'y_center': y_center,
                }
                detections_info.append(detection)
        
        return detections_info


# ===== Helper Functions =====

def load_yolov8_model(model_path):

    return YOLOv8Detector(model_path=model_path)


def detect_from_image(image_path, model_path):

    detector = YOLOv8Detector(model_path=model_path)
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    annotated_image, results = detector.detect_with_visualization(image)
    detections_info = detector.get_detections_info(results)
    
    return annotated_image, results, detections_info


if __name__ == "__main__":

    MODEL_PATH = "your_model_path.pt"
    
    try:
        detector = YOLOv8Detector(model_path=MODEL_PATH)
        print("Mô hình YOLOv8 đã tải thành công!")
    except Exception as e:
        print(f"Lỗi: {e}")
