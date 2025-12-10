import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import os
from src.segmentation.thresholding.otsu import apply_otsu_threshold
from src.segmentation.thresholding.adaptive_threshold import apply_adaptive_threshold
from src.segmentation.edge_detection.canny import apply_canny
from src.segmentation.edge_detection.sobel import apply_sobel
from src.segmentation.projection_profile.line_segmentation import segment_lines
from src.segmentation.projection_profile.character_segmentation import segment_characters

#from segmentation import segment_characters


class SegmentationUI:
    def __init__(self, root, image_path: str):
        self.root = root
        self.root.title("Segmentation - Tách Ký Tự")
        self.root.geometry("1200x800")

        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.processed_cv_image = self.original_cv_image.copy()

        self.threshold_value = tk.IntVar(value=127)
        self.kernel_size = tk.IntVar(value=5)

        self.setup_ui()
        self.update_all_images()

        # Load gallery từ thư mục của ảnh hiện tại
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

        # ===== KHUNG ẢNH KẾT QUẢ =====
        self.frame_processed = tk.Frame(self.root, bg="#808080", bd=2, relief="sunken")
        self.frame_processed.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        tk.Label(
            self.frame_processed, text="Ảnh Segmentation (Result)",
            bg="#808080", fg="white", font=("Arial", 12, "bold")
        ).pack(pady=5)

        self.lbl_processed_img = tk.Label(self.frame_processed, bg="#808080")
        self.lbl_processed_img.pack(expand=True)

        # ===== BẢNG ĐIỀU KHIỂN =====
        self.frame_controls = tk.Frame(self.root, bg="#5DADE2", width=250)
        self.frame_controls.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        self.frame_controls.pack_propagate(False)

        tk.Label(
            self.frame_controls, text="Bảng Điều Khiển - Segmentation",
            bg="#5DADE2", font=("Arial", 14, "bold")
        ).pack(pady=20)

        tk.Button(
            self.frame_controls, text="💾 Lưu Ảnh Segmentation",
            command=self.save_processed_image,
            height=2, bg="#2ECC71", fg="white"
        ).pack(fill="x", padx=10, pady=(0, 5))

        tk.Button(
            self.frame_controls, text="🔄 Reset Ảnh",
            command=self.reset_image,
            height=1, bg="#E74C3C", fg="white"
        ).pack(fill="x", padx=10, pady=(0, 15))

        tk.Button(self.frame_controls, text="Otsu Threshold",
          command=self.apply_otsu).pack(fill="x", padx=20, pady=2)

        tk.Button(self.frame_controls, text="Adaptive Threshold",
          command=self.apply_adaptive).pack(fill="x", padx=20, pady=2)

        tk.Button(self.frame_controls, text="Canny Edge",
          command=self.apply_canny).pack(fill="x", padx=20, pady=2)

        tk.Button(self.frame_controls, text="Sobel Edge",
          command=self.apply_sobel).pack(fill="x", padx=20, pady=2)

        tk.Button(self.frame_controls, text="Tách Dòng (Projection)",
          command=self.apply_line_seg).pack(fill="x", padx=20, pady=2)

        tk.Button(self.frame_controls, text="Tách Ký Tự (Projection)",
          command=self.apply_char_seg).pack(fill="x", padx=20, pady=2)

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

    # ============================ HIỂN THỊ ẢNH ============================

    def update_all_images(self):
        self.show_image_on_label(self.original_cv_image, self.lbl_original_img)
        self.show_image_on_label(self.processed_cv_image, self.lbl_processed_img)

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

    # ============================ SEGMENTATION ============================
    def apply_otsu(self):
        self.processed_cv_image = apply_otsu_threshold(self.original_cv_image)
        self.update_all_images()

    def apply_adaptive(self):
        self.processed_cv_image = apply_adaptive_threshold(self.original_cv_image)
        self.update_all_images()

    def apply_canny(self):
        self.processed_cv_image = apply_canny(self.original_cv_image)
        self.update_all_images()

    def apply_sobel(self):
        self.processed_cv_image = apply_sobel(self.original_cv_image)
        self.update_all_images()

    def apply_line_seg(self):
        self.processed_cv_image = segment_lines(self.original_cv_image)
        self.update_all_images()

    def apply_char_seg(self):
        self.processed_cv_image = segment_characters(self.original_cv_image)
        self.update_all_images()
    def apply_segmentation(self):
        try:
            self.processed_cv_image = segment_characters(
                self.original_cv_image,
                threshold_value=int(self.threshold_value.get()),
                kernel_size=int(self.kernel_size.get())
            )
            self.update_all_images()
        except Exception as e:
            messagebox.showerror("Lỗi Segmentation", f"Lỗi: {e}")

    def reset_image(self):
        self.processed_cv_image = self.original_cv_image.copy()
        self.update_all_images()

    def save_processed_image(self):
        if self.processed_cv_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để lưu.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ],
            title="Lưu ảnh segmentation"
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_cv_image)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

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
                    self.gallery_content,
                    image=photo,
                    command=lambda p=path: self.display_original_from_gallery(p)
                )
                btn.image = photo
                btn.pack(side="left", padx=5, pady=15)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc thư mục: {e}")

    def display_original_from_gallery(self, image_path):
        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.processed_cv_image = self.original_cv_image.copy()
        self.update_all_images()
