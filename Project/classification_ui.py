import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import os

# ========== DEMO CNN MODEL ==========
# Sau này bạn thay bằng model thật
def predict_with_cnn(image):
    return "A"   # giả lập kết quả


class ClassificationUI:
    def __init__(self, root, image_path: str):
        self.root = root
        self.root.title("Classification - Nhận Diện CNN")
        self.root.geometry("1200x800")

        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)

        self.current_model_name = tk.StringVar(value="CNN Digit")

        self.setup_ui()
        self.update_original_image()

        folder_path = os.path.dirname(self.image_path)
        self.load_images_to_gallery(folder_path)

    # ============================ GIAO DIỆN CHÍNH ============================

    def setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=1)

        # ===== KHUNG ẢNH GỐC =====
        self.frame_original = tk.Frame(self.root, bg="#A9A9A9", bd=2, relief="sunken")
        self.frame_original.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        tk.Label(
            self.frame_original, text="Ảnh Gốc (Original)",
            bg="#A9A9A9", font=("Arial", 12, "bold")
        ).pack(pady=5)

        self.lbl_original_img = tk.Label(self.frame_original, bg="#A9A9A9")
        self.lbl_original_img.pack(expand=True)

        # ===== KHUNG HIỂN THỊ KẾT QUẢ CNN (THAY ẢNH SEGMENTATION) =====
        self.frame_result = tk.Frame(self.root, bg="#808080", bd=2, relief="sunken")
        self.frame_result.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        tk.Label(
            self.frame_result, text="KẾT QUẢ NHẬN DIỆN (CNN Output)",
            bg="#808080", fg="white", font=("Arial", 12, "bold")
        ).pack(pady=10)

        # Ô hiển thị kết quả LỚN
        self.result_big_label = tk.Label(
            self.frame_result,
            text="?",
            bg="#808080",
            fg="lime",
            font=("Arial", 80, "bold")
        )
        self.result_big_label.pack(expand=True)

        # Hiển thị tên model
        self.model_name_label = tk.Label(
            self.frame_result,
            text="Model: ---",
            bg="#808080",
            fg="white",
            font=("Arial", 12)
        )
        self.model_name_label.pack(pady=10)

        # ===== BẢNG CNN MODEL =====
        self.frame_controls = tk.Frame(self.root, bg="#5DADE2", width=250)
        self.frame_controls.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        self.frame_controls.pack_propagate(False)

        tk.Label(
            self.frame_controls, text="Bảng CNN Model",
            bg="#5DADE2", font=("Arial", 14, "bold")
        ).pack(pady=20)

        tk.Label(
            self.frame_controls, text="Chọn mô hình:",
            bg="#5DADE2"
        ).pack(pady=(10, 5))

        tk.Radiobutton(
            self.frame_controls, text="CNN Digit (MNIST)",
            variable=self.current_model_name,
            value="CNN Digit",
            bg="#5DADE2"
        ).pack(anchor="w", padx=30)

        tk.Radiobutton(
            self.frame_controls, text="CNN Character (A-Z)",
            variable=self.current_model_name,
            value="CNN Character",
            bg="#5DADE2"
        ).pack(anchor="w", padx=30)

        tk.Radiobutton(
            self.frame_controls, text="CNN Handwriting Custom",
            variable=self.current_model_name,
            value="CNN Custom",
            bg="#5DADE2"
        ).pack(anchor="w", padx=30)

        tk.Button(
            self.frame_controls, text="🤖 Dự đoán",
            command=self.predict_image,
            height=2, bg="#2ECC71", fg="white"
        ).pack(fill="x", padx=15, pady=(25, 10))

        tk.Button(
            self.frame_controls, text="🔄 Reset",
            command=self.reset_result,
            height=1, bg="#E74C3C", fg="white"
        ).pack(fill="x", padx=15, pady=(0, 10))

        tk.Button(
            self.frame_controls, text="💾 Lưu Kết Quả",
            command=self.save_result,
            height=1, bg="#3498DB", fg="white"
        ).pack(fill="x", padx=15)

        # ===== GALLERY ẢNH =====
        self.frame_gallery = tk.Frame(self.root, bg="#F4D03F", height=200)
        self.frame_gallery.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2
        )

        tk.Label(
            self.frame_gallery, text="Danh sách ảnh trong thư mục",
            bg="#F4D03F", font=("Arial", 10, "bold")
        ).pack(anchor="nw", padx=5, pady=2)

        self.canvas_gallery = tk.Canvas(self.frame_gallery, bg="#F4D03F")
        self.scrollbar_gallery = tk.Scrollbar(
            self.frame_gallery, orient="horizontal",
            command=self.canvas_gallery.xview
        )

        self.gallery_content = tk.Frame(self.canvas_gallery, bg="#F4D03F")

        self.gallery_content.bind(
            "<Configure>", lambda e: self.canvas_gallery.configure(
                scrollregion=self.canvas_gallery.bbox("all"))
        )

        self.canvas_gallery.create_window(
            (0, 0), window=self.gallery_content, anchor="nw"
        )
        self.canvas_gallery.configure(
            xscrollcommand=self.scrollbar_gallery.set
        )

        self.canvas_gallery.pack(side="top", fill="both", expand=True)
        self.scrollbar_gallery.pack(side="bottom", fill="x")

    # ============================ HIỂN THỊ ẢNH GỐC ============================

    def update_original_image(self):
        self.show_image_on_label(self.original_cv_image, self.lbl_original_img)

    def show_image_on_label(self, cv_image_bgr, label_widget):
        if cv_image_bgr is None:
            return

        img_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        parent_frame = label_widget.master
        frame_width = parent_frame.winfo_width()
        frame_height = parent_frame.winfo_height()

        if frame_width < 100:
            frame_width = 400
        if frame_height < 100:
            frame_height = 400

        scale = min(frame_width / w, frame_height / h) * 0.9
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = Image.fromarray(img_rgb)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        label_widget.config(image=photo)
        label_widget.image = photo

    # ============================ CNN PREDICT ============================

    def predict_image(self):
        try:
            predicted_label = predict_with_cnn(self.original_cv_image)
            model_name = self.current_model_name.get()

            self.result_big_label.config(text=str(predicted_label))
            self.model_name_label.config(text=f"Model: {model_name}")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi dự đoán: {e}")

    def reset_result(self):
        self.result_big_label.config(text="?")
        self.model_name_label.config(text="Model: ---")

    def save_result(self):
        text = self.result_big_label.cget("text")
        if text == "?":
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Lưu kết quả classification"
        )

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Kết quả dự đoán: {text}\n")
                f.write(f"Model: {self.current_model_name.get()}\n")

            messagebox.showinfo("Thành công", f"Đã lưu kết quả tại:\n{file_path}")

    # ============================ GALLERY ============================

    def load_images_to_gallery(self, folder_path):
        for widget in self.gallery_content.winfo_children():
            widget.destroy()

        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        try:
            files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            for f in files:
                path = os.path.join(folder_path, f)
                img = Image.open(path)
                img.thumbnail((80, 80))
                photo = ImageTk.PhotoImage(img)

                btn = tk.Button(
                    self.gallery_content, image=photo,
                    command=lambda p=path: self.display_original_from_gallery(p)
                )
                btn.image = photo
                btn.pack(side="left", padx=5, pady=15)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc thư mục: {e}")

    def display_original_from_gallery(self, image_path):
        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.update_original_image()
        self.reset_result()
