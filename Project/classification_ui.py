import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import os
import sys
from pathlib import Path

# Thêm project root vào path để import các module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.detection.yolov8_model import YOLOv8Detector
from src.classification.yolov8_classifier import YOLOv8Classifier
from src.segmentation.deskew.deskew import deskew
from src.segmentation.thresholding.adaptive_threshold import apply_adaptive_threshold
from src.segmentation.projection_profile.line_segmentation import segment_lines
from src.segmentation.projection_profile.character_segmentation import word_segmentation


# ========== Global Model Instances ==========
yolo_detector = None
yolo_classifier = None

# ========== DEMO CNN MODEL ==========
def predict_with_cnn(image):
    """Giả lập dự đoán với model CNN."""
    return "A"

# ========== PREDICTION WRAPPERS ==========
def predict_with_yolo_detection(image):
    global yolo_detector
    model_path = config.YOLOV8_DETECTION_MODEL_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Không tìm thấy model detection tại: {model_path}\nVui lòng kiểm tra file config.py.")
    
    if yolo_detector is None or yolo_detector.model.model.pt_path != model_path:
        yolo_detector = YOLOv8Detector(model_path=model_path)
    
    annotated_image, results = yolo_detector.detect_with_visualization(image)
    detections_info = yolo_detector.get_detections_info(results)
    
    return annotated_image, detections_info

def predict_with_yolo_classification(image):
    global yolo_classifier
    model_path = config.YOLOV8_CLASSIFICATION_MODEL_PATH

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Không tìm thấy model classification tại: {model_path}\nVui lòng kiểm tra file config.py.")

    if yolo_classifier is None:
        yolo_classifier = YOLOv8Classifier(model_path=model_path)
        
    class_name, confidence = yolo_classifier.predict(image)
    return class_name, confidence


class ClassificationUI:
    def __init__(self, root, image_path: str):
        self.root = root
        self.root.title("Classification - Nhận Diện Ký Tự")
        self.root.geometry("1200x800")

        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.processed_cv_image = self.original_cv_image.copy()
        self.current_detections = None

        self.current_model_name = tk.StringVar(value="YOLOv8_Segment_And_Classify")

        self.setup_ui()
        self.update_original_image()

        folder_path = os.path.dirname(self.image_path)
        self.load_images_to_gallery(folder_path)

    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=1)

        # Frame ảnh gốc
        self.frame_original = tk.Frame(self.root, bg="#A9A9A9", bd=2, relief="sunken")
        self.frame_original.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        tk.Label(self.frame_original, text="Ảnh Gốc (Original)", bg="#A9A9A9", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_original_img = tk.Label(self.frame_original, bg="#A9A9A9")
        self.lbl_original_img.pack(expand=True)

        # Frame kết quả
        self.frame_result = tk.Frame(self.root, bg="#3A3A3A", bd=2, relief="sunken")
        self.frame_result.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        tk.Label(self.frame_result, text="KẾT QUẢ NHẬN DIỆN", bg="#3A3A3A", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        self.result_big_label = tk.Label(self.frame_result, text="?", bg="#3A3A3A", fg="#00FF00", font=("Arial", 20, "bold"), justify=tk.LEFT, wraplength=500)
        self.result_big_label.pack(expand=True, fill="both", padx=10, pady=10)
        self.model_name_label = tk.Label(self.frame_result, text="Model: ---", bg="#3A3A3A", fg="white", font=("Arial", 12))
        self.model_name_label.pack(pady=10)

        # Frame điều khiển
        self.frame_controls = tk.Frame(self.root, bg="#5DADE2", width=250)
        self.frame_controls.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
        self.frame_controls.pack_propagate(False)

        tk.Label(self.frame_controls, text="Chọn Model", bg="#5DADE2", font=("Arial", 14, "bold")).pack(pady=20)
        
        models = [
            ("YOLOv8 Segment & Classify", "YOLOv8_Segment_And_Classify"),
            ("YOLOv8 Detection", "YOLOv8_Detection"),
            ("CNN (Demo)", "CNN_Demo")
        ]
        for text, value in models:
            tk.Radiobutton(self.frame_controls, text=text, variable=self.current_model_name, value=value, bg="#5DADE2", anchor="w").pack(fill='x', padx=20)

        tk.Button(self.frame_controls, text="🤖 Dự đoán", command=self.predict_image, height=2, bg="#2ECC71", fg="white").pack(fill="x", padx=15, pady=(25, 10))
        tk.Button(self.frame_controls, text="🔄 Reset", command=self.reset_result, height=1, bg="#E74C3C", fg="white").pack(fill="x", padx=15, pady=(0, 10))

        # Gallery
        self.frame_gallery = tk.Frame(self.root, bg="#F4D03F")
        self.frame_gallery.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)
        self.canvas_gallery = tk.Canvas(self.frame_gallery, bg="#F4D03F")
        self.scrollbar_gallery = tk.Scrollbar(self.frame_gallery, orient="horizontal", command=self.canvas_gallery.xview)
        self.gallery_content = tk.Frame(self.canvas_gallery, bg="#F4D03F")
        self.gallery_content.bind("<Configure>", lambda e: self.canvas_gallery.configure(scrollregion=self.canvas_gallery.bbox("all")))
        self.canvas_gallery.create_window((0, 0), window=self.gallery_content, anchor="nw")
        self.canvas_gallery.configure(xscrollcommand=self.scrollbar_gallery.set)
        self.canvas_gallery.pack(side="top", fill="both", expand=True)
        self.scrollbar_gallery.pack(side="bottom", fill="x")

    def update_original_image(self):
        self.show_image_on_label(self.original_cv_image, self.lbl_original_img)

    def show_image_on_label(self, cv_image, label_widget):
        if cv_image is None: return
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        frame = label_widget.master
        fw, fh = frame.winfo_width(), frame.winfo_height()
        fw = 400 if fw < 100 else fw
        fh = 400 if fh < 100 else fh
        scale = min(fw / w, fh / h) * 0.9
        img = Image.fromarray(img_rgb).resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo)
        label_widget.image = photo

    def predict_image(self):
        self.reset_result()
        model_choice = self.current_model_name.get()
        
        try:
            if model_choice == "YOLOv8_Segment_And_Classify":
                self.model_name_label.config(text="Model: YOLOv8 Segment & Classify")
                self.result_big_label.config(text="Đang xử lý...", font=("Arial", 20, "bold"))
                self.root.update_idletasks()

                processed_img = deskew(self.original_cv_image)
                processed_img = apply_adaptive_threshold(processed_img)

                line_images, _, _ = segment_lines(processed_img)
                if not line_images:
                    self.result_big_label.config(text="[Không tìm thấy dòng nào]")
                    return

                all_recognized_words = []
                for line_img in line_images:
                    word_images, _, _ = word_segmentation(line_img)
                    
                    recognized_line = []
                    for word_img in word_images:
                        class_name, confidence = predict_with_yolo_classification(word_img)
                        if class_name is not None:
                            recognized_line.append(class_name)
                    
                    if recognized_line:
                        all_recognized_words.append(" ".join(recognized_line))

                final_text = "\n".join(all_recognized_words)
                if not final_text:
                    final_text = "[Không nhận diện được ký tự nào]"
                
                self.result_big_label.config(text=final_text)

            elif model_choice == "YOLOv8_Detection":
                annotated_image, detections = predict_with_yolo_detection(self.original_cv_image)
                self.processed_cv_image = annotated_image
                self.show_image_on_label(annotated_image, self.lbl_original_img)
                self.current_detections = detections
                
                if detections:
                    best_detection = max(detections, key=lambda d: d.get('confidence', 0.0))
                    label = best_detection.get('label', 'N/A')
                    self.result_big_label.config(text=str(label), font=("Arial", 80, "bold"))
                    self.model_name_label.config(text=f"Model: YOLOv8 Detect ({len(detections)} objects)")
                else:
                    self.result_big_label.config(text="0", font=("Arial", 80, "bold"))
                    self.model_name_label.config(text="Model: YOLOv8 Detect (No objects found)")

            elif model_choice == "CNN_Demo":
                predicted_label = predict_with_cnn(self.original_cv_image)
                self.result_big_label.config(text=str(predicted_label), font=("Arial", 80, "bold"))
                self.model_name_label.config(text="Model: CNN (Demo)")

        except Exception as e:
            messagebox.showerror("Lỗi Dự Đoán", str(e))
            self.reset_result()

    def reset_result(self):
        self.result_big_label.config(text="?", font=("Arial", 80, "bold"))
        self.model_name_label.config(text="Model: ---")
        self.current_detections = None
        self.processed_cv_image = self.original_cv_image.copy()
        self.update_original_image()

    def load_images_to_gallery(self, folder_path):
        for widget in self.gallery_content.winfo_children():
            widget.destroy()
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        try:
            files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
            for f in files:
                path = os.path.join(folder_path, f)
                img = Image.open(path)
                img.thumbnail((80, 80))
                photo = ImageTk.PhotoImage(img)
                btn = tk.Button(self.gallery_content, image=photo, command=lambda p=path: self.display_original_from_gallery(p))
                btn.image = photo
                btn.pack(side="left", padx=5, pady=5)
        except Exception as e:
            messagebox.showerror("Lỗi Gallery", f"Không thể đọc thư mục: {e}")

    def display_original_from_gallery(self, image_path):
        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.reset_result()
