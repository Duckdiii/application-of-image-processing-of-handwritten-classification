from segmentation_ui import SegmentationUI
from classification_ui import ClassificationUI
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import config

from src.preprocessing.morphology.grayscale import apply_grayscale

from src.preprocessing.filters.gaussian_blur import apply_gaussian_blur
from src.preprocessing.filters.median_filter import apply_median_filter
from src.preprocessing.morphology.dilation import apply_dilation
from src.preprocessing.morphology.erosion import apply_erosion


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ứng dụng Xử lý ảnh - Digital Image Processing")
        self.root.geometry("1200x800")

        # Biáº¿n lÆ°u trá»¯ áº£nh (OpenCV format: BGR hoáº·c Grayscale)
        self.original_cv_image = None
        self.original_cv_image_backup = None  # Luu anh goc de reset
        self.processed_cv_image = None  # DÃ¹ng Ä‘á»ƒ lÆ°u vÃ  reset áº£nh
        self.mode_input_image = None  # Luu anh lam dau vao cho mode hien tai
        self.current_folder_path = ""
        self.current_mode = None
        self.image_path = None

        self.setup_ui()

    def setup_ui(self):
        # --- C?U H?NH LAYOUT (GRID) ---
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=1)

        # --- 1. KHUNG ?NH G?C ---
        self.frame_original = tk.Frame(
            self.root, bg="#A9A9A9", bd=2, relief="sunken")
        self.frame_original.grid(
            row=0, column=0, sticky="nsew", padx=2, pady=2)
        tk.Label(self.frame_original, text="Ảnh Gốc (Original)",
                 bg="#A9A9A9", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_original_img = tk.Label(self.frame_original, bg="#A9A9A9")
        self.lbl_original_img.pack(expand=True)

        # --- 2. KHUNG ?NH X? L? ---
        self.frame_processed = tk.Frame(
            self.root, bg="#808080", bd=2, relief="sunken")
        self.frame_processed.grid(
            row=0, column=1, sticky="nsew", padx=2, pady=2)
        tk.Label(self.frame_processed, text="Ảnh đang Xử lý (Result)",
                 bg="#808080", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_processed_img = tk.Label(self.frame_processed, bg="#808080")
        self.lbl_processed_img.pack(expand=True)

        # --- 3. KHUNG CH?C N?NG (B?NG ?I?U KHI?N) ---
        self.frame_controls = tk.Frame(self.root, bg="#5DADE2", width=250)
        self.frame_controls.grid(
            row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
        self.frame_controls.pack_propagate(False)

        tk.Label(self.frame_controls, text="Bảng Điều Khiển",
                 bg="#5DADE2", font=("Arial", 14, "bold")).pack(pady=(12, 10))

        # N?t ch?n th? m?c
        tk.Button(self.frame_controls, text="Chọn Thư Mục Ảnh", command=self.load_folder,
                  height=2, bg="white").pack(fill="x", padx=10, pady=(0, 8))

        # N?t L?u v? Reset
        tk.Button(self.frame_controls, text="Lưu Ảnh Xử Lý", command=self.save_image_preprocessing,
                  height=2, bg="#2ECC71", fg="white").pack(fill="x", padx=10, pady=(4, 4))
        tk.Button(self.frame_controls, text="Reset Ảnh", command=self.reset_image,
                  height=1, bg="#E74C3C", fg="white").pack(fill="x", padx=10, pady=(4, 10))

        # C?c n?t ch?c n?ng x? l?
        tk.Label(self.frame_controls, text="Chọn thuật toán:",
                 bg="#5DADE2").pack(pady=(6, 4))
        tk.Button(self.frame_controls, text="Grayscale", command=lambda: self.set_mode(
            "gray")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Gaussian Blur", command=lambda: self.set_mode(
            "gaussian")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Median Filter", command=lambda: self.set_mode(
            "median")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Dilation (Gi?n)", command=lambda: self.set_mode(
            "dilation")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Erosion (Co)", command=lambda: self.set_mode(
            "erosion")).pack(fill="x", padx=20, pady=2)

        # Slider ?i?u ch?nh tham s? (Kernel size)
        tk.Label(self.frame_controls, text="Kích thước Kernel / Mức ??:",
                 bg="#5DADE2").pack(pady=(12, 4))
        self.slider_kernel = tk.Scale(self.frame_controls, from_=1, to=21,
                                      orient="horizontal", bg="#5DADE2", command=self.on_slider_change)
        self.slider_kernel.set(3)
        self.slider_kernel.pack(fill="x", padx=20, pady=(0, 6))
        tk.Button(self.frame_controls, text="Áp dụng & Gi? (Chain)",
                  command=self.apply_and_commit,
                  bg="#27AE60", fg="white").pack(fill="x", padx=20, pady=(6, 2))

        tk.Button(self.frame_controls, text="Xem thử (Test)",
                  command=self.test_current_mode,
                  bg="#F39C12", fg="white").pack(fill="x", padx=20, pady=(2, 8))
        #   N?t segmentation v? classification model
        tk.Button(self.frame_controls, text="Segmentation (Phân đoạn)",
                  command=self.open_segmentation_window).pack(fill="x", padx=20, pady=2)

        tk.Button(self.frame_controls, text="Classification (Phân loại)",
                  command=self.open_classification_window).pack(fill="x", padx=20, pady=6)
        # --- KHU V?C BATCH PREPROCESSING CHO TO?N B? DATASET ---
        batch_frame = tk.LabelFrame(
            self.frame_controls,
            text="Batch Preprocessing (toàn bộ thư mục)",
            bg="#5DADE2", fg="black", labelanchor="n"
        )
        batch_frame.pack(fill="x", padx=10, pady=(10, 8))

        # ===== BI?N CHECKBOX =====
        self.var_batch_gray = tk.BooleanVar(value=True)
        self.var_batch_gaussian = tk.BooleanVar(value=False)
        self.var_batch_median = tk.BooleanVar(value=False)
        self.var_batch_dilation = tk.BooleanVar(value=False)
        self.var_batch_erosion = tk.BooleanVar(value=False)

        # ===== GRAYSCALE =====
        row_gray = tk.Frame(batch_frame, bg="#5DADE2")
        row_gray.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_gray, text="Grayscale",
                       variable=self.var_batch_gray,
                       bg="#5DADE2").pack(side="left", anchor="w")

        # ===== GAUSSIAN BLUR =====
        row_g = tk.Frame(batch_frame, bg="#5DADE2")
        row_g.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_g, text="Gaussian Blur",
                       variable=self.var_batch_gaussian,
                       bg="#5DADE2").pack(side="left")
        tk.Label(row_g, text="k:", bg="#5DADE2").pack(
            side="left", padx=(10, 2))
        self.entry_k_gaussian = tk.Entry(row_g, width=4)
        self.entry_k_gaussian.insert(0, "3")
        self.entry_k_gaussian.pack(side="left")

        # ===== MEDIAN FILTER =====
        row_m = tk.Frame(batch_frame, bg="#5DADE2")
        row_m.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_m, text="Median Filter",
                       variable=self.var_batch_median,
                       bg="#5DADE2").pack(side="left")
        tk.Label(row_m, text="k:", bg="#5DADE2").pack(
            side="left", padx=(10, 2))
        self.entry_k_median = tk.Entry(row_m, width=4)
        self.entry_k_median.insert(0, "3")
        self.entry_k_median.pack(side="left")

        # ===== DILATION =====
        row_d = tk.Frame(batch_frame, bg="#5DADE2")
        row_d.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_d, text="Dilation (Giãn)",
                       variable=self.var_batch_dilation,
                       bg="#5DADE2").pack(side="left")
        tk.Label(row_d, text="k:", bg="#5DADE2").pack(
            side="left", padx=(10, 2))
        self.entry_k_dilation = tk.Entry(row_d, width=4)
        self.entry_k_dilation.insert(0, "3")
        self.entry_k_dilation.pack(side="left")

        # ===== EROSION =====
        row_e = tk.Frame(batch_frame, bg="#5DADE2")
        row_e.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_e, text="Erosion (Co)",
                       variable=self.var_batch_erosion,
                       bg="#5DADE2").pack(side="left")
        tk.Label(row_e, text="k:", bg="#5DADE2").pack(
            side="left", padx=(10, 2))
        self.entry_k_erosion = tk.Entry(row_e, width=4)
        self.entry_k_erosion.insert(0, "3")
        self.entry_k_erosion.pack(side="left")

        # N?t ch?y batch
        tk.Button(batch_frame, text="Xu ly toàn bộ thư mục",
                  command=self.batch_preprocess_folder,
                  bg="#1ABC9C", fg="white").pack(fill="x", padx=5, pady=(8, 5))
        # Progress bar
        self.batch_progress = ttk.Progressbar(
            batch_frame,
            orient="horizontal",
            length=200,
            mode="determinate"
        )
        self.batch_progress.pack(fill="x", padx=5, pady=(5, 2))

        self.lbl_batch_percent = tk.Label(
            batch_frame, text="0%", bg="#5DADE2"
        )
        self.lbl_batch_percent.pack(anchor="center")
        # --- 4. KHUNG GALLERY ?NH (BOTTOM) ---
        self.frame_gallery = tk.Frame(self.root, bg="#F4D03F", height=200)
        self.frame_gallery.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

        tk.Label(self.frame_gallery, text="Danh sách ảnh trong thư mục", bg="#F4D03F", font=(
            "Arial", 10, "bold")).pack(anchor="nw", padx=5, pady=2)

        # Canvas v? Scrollbar (?? hi?n th? d?ng th? vi?n ngang)
        self.canvas_gallery = tk.Canvas(self.frame_gallery, bg="#F4D03F")
        self.scrollbar_gallery = tk.Scrollbar(
            self.frame_gallery, orient="vertical", command=self.canvas_gallery.yview
        )

        # Gallery Content Frame
        self.gallery_content = tk.Frame(self.canvas_gallery, bg="#F4D03F")

        # C?n bind ?? c?p nh?t scrollregion khi n?i dung thay ??i
        self.gallery_content.bind("<Configure>", lambda e: self.canvas_gallery.configure(
            scrollregion=self.canvas_gallery.bbox("all")))

        # Th?m gallery_content v?o canvas
        # NOTE: Thay v? d?ng pack, ch?ng ta d?ng create_window ?? gallery_content c? th? cu?n ngang
        self.canvas_gallery.create_window(
            (0, 0), window=self.gallery_content, anchor="nw")
        self.canvas_gallery.configure(
            yscrollcommand=self.scrollbar_gallery.set)

        self.canvas_gallery.pack(side="left", fill="both", expand=True)
        self.scrollbar_gallery.pack(side="right", fill="y")

    def open_segmentation_window(self):
        if self.image_path is None:
            messagebox.showwarning(
                "Cảnh báo", "Vui lòng mở ảnh trước khi segmentation!")
            return
        seg_window = tk.Toplevel(self.root)
        SegmentationUI(seg_window, self.image_path)

    def open_classification_window(self):
        if self.image_path is None:
            messagebox.showwarning(
                "Cảnh báo", "Vui lòng mở ảnh trước khi classification!")
            return
        clf_window = tk.Toplevel(self.root)
        ClassificationUI(clf_window, self.image_path)

    def load_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.current_folder_path = folder_selected
            self.load_images_to_gallery()

    def load_images_to_gallery(self):
        for widget in self.gallery_content.winfo_children():
            widget.destroy()

        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        try:
            files = [
                f for f in os.listdir(self.current_folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            max_columns = 9
            row = 0
            col = 0

            for f in files:
                path = os.path.join(self.current_folder_path, f)
                img = Image.open(path)
                img.thumbnail((80, 80))
                photo = ImageTk.PhotoImage(img)

                btn = tk.Button(
                    self.gallery_content,
                    image=photo,
                    command=lambda p=path: self.display_original(p)
                )
                btn.image = photo

                btn.grid(row=row, column=col, padx=5, pady=5)

                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc thư mục: {e}")

    def open_image(self):
        from tkinter import filedialog
        import cv2

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not file_path:
            return

        # âœ… LÆ¯U ÄÆ¯á»œNG DáºªN áº¢NH (CÃI Báº N ÄANG THIáº¾U)
        self.image_path = file_path

        # Load áº£nh
        self.original_image = cv2.imread(file_path)
        self.processed_image = self.original_image.copy()

        # Hiá»ƒn thá»‹ áº£nh
        self.display_image(self.original_image)

    def display_original(self, image_path):
        try:
            self.original_cv_image = cv2.imread(image_path)
            self.original_cv_image_backup = self.original_cv_image.copy()
            self.original_cv_image = self.original_cv_image_backup.copy()
            self.processed_cv_image = self.original_cv_image_backup.copy()
            self.mode_input_image = self.processed_cv_image.copy()

            img_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)

            self.show_image_on_label(img_rgb, self.lbl_original_img)

            self.show_image_on_label(img_rgb, self.lbl_processed_img)

            self.current_mode = None

            self.image_path = image_path

        except Exception as e:
            messagebox.showerror("Lỗi hiển thị", f"Không thể tải ảnh: {e}")

    def reset_image(self):
        if self.original_cv_image_backup is not None:
            self.original_cv_image = self.original_cv_image_backup.copy()
            self.processed_cv_image = self.original_cv_image_backup.copy()
            self.mode_input_image = self.processed_cv_image.copy()

            img_rgb = cv2.cvtColor(self.processed_cv_image, cv2.COLOR_BGR2RGB)
            self.show_image_on_label(img_rgb, self.lbl_processed_img)
            self.current_mode = None
            messagebox.showinfo("Thông báo", "Đã reset ảnh thành công.")
        else:
            messagebox.showwarning("Cảnh báo", "Không có ảnh gốc để reset.")

    def save_image_preprocessing(self):
        """Lưu ảnh đã xử lý vào thư mục preprocessing."""
        if self.processed_cv_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh xử lý để lưu.")
            return

        save_dir = config.ARCHIVE_PREPROCESSING_DIR
        os.makedirs(save_dir, exist_ok=True)

        if self.image_path:
            base_name = os.path.basename(self.image_path)
            name, ext = os.path.splitext(base_name)
            filename = f"{name}_preprocessed.png"
        else:
            filename = "preprocessed.png"

        save_path = os.path.join(save_dir, filename)

        img_to_save = self.processed_cv_image
        if len(img_to_save.shape) == 3 and img_to_save.shape[2] == 3:
            pass

        cv2.imwrite(save_path, img_to_save)
        messagebox.showinfo("Thông báo", f"Đã lưu ảnh tại:\n{save_path}")

    def show_image_on_label(self, cv_image_rgb, label_widget):
        h, w, _ = cv_image_rgb.shape

        parent_frame = label_widget.master

        frame_width = parent_frame.winfo_width()
        frame_height = parent_frame.winfo_height()

        if frame_width < 100:
            frame_width = 400
        if frame_height < 100:
            frame_height = 400

        scale_w = frame_width / w
        scale_h = frame_height / h
        scale = min(scale_w, scale_h) * 0.90

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            return

        img = Image.fromarray(cv_image_rgb)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        label_widget.config(image=photo)
        label_widget.image = photo

    def set_mode(self, mode):
        self.current_mode = mode
        if self.processed_cv_image is not None:
            self.mode_input_image = self.processed_cv_image.copy()
        elif self.original_cv_image is not None:
            self.mode_input_image = self.original_cv_image.copy()
        if self.original_cv_image is not None:
            self.test_current_mode()

    def on_slider_change(self, val):
        if self.original_cv_image is not None and self.current_mode:
            self.test_current_mode()

    def process_image(self):
        self.test_current_mode()

    def _get_kernel_from_entry(self, entry_widget):
        """Parse kernel value from entry; ensure odd and positive, fallback to 3."""
        try:
            k = int(entry_widget.get().strip())
        except Exception:
            k = 3
        if k <= 0:
            k = 3
        if k % 2 == 0:
            k += 1
        return k

    def batch_preprocess_folder(self):
        if not self.current_folder_path:
            messagebox.showwarning(
                "Cảnh báo", "Vui lòng chọn thư mục ảnh trước!")
            return

        output_dir = config.ARCHIVE_PREPROCESSING_DIR
        os.makedirs(output_dir, exist_ok=True)

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(self.current_folder_path)
                 if f.lower().endswith(valid_exts)]

        if not files:
            messagebox.showwarning("Thông báo", "Thư mục không có ảnh hợp lệ.")
            return

        # ✅ Lấy KERNEL RIÊNG
        k_gaussian = self._get_kernel_from_entry(self.entry_k_gaussian)
        k_median = self._get_kernel_from_entry(self.entry_k_median)
        k_dilation = self._get_kernel_from_entry(self.entry_k_dilation)
        k_erosion = self._get_kernel_from_entry(self.entry_k_erosion)

        total = len(files)
        count = 0

        # ✅ RESET PROGRESS BAR
        self.batch_progress["value"] = 0
        self.batch_progress["maximum"] = total
        self.lbl_batch_percent.config(text="0%")
        self.root.update_idletasks()

        for i, fname in enumerate(files, start=1):
            in_path = os.path.join(self.current_folder_path, fname)
            img = cv2.imread(in_path)
            if img is None:
                continue

            # ===== PIPELINE =====
            if self.var_batch_gray.get():
                img = apply_grayscale(img)

            if self.var_batch_gaussian.get():
                img = apply_gaussian_blur(img, k_gaussian)

            if self.var_batch_median.get():
                img = apply_median_filter(img, k_median)

            if self.var_batch_dilation.get():
                img = apply_dilation(img, k_dilation)

            if self.var_batch_erosion.get():
                img = apply_erosion(img, k_erosion)

            name, _ = os.path.splitext(fname)
            out_path = os.path.join(output_dir, f"{name}_preprocessed.png")
            cv2.imwrite(out_path, img)
            count += 1

            # ✅ CẬP NHẬT PROGRESS
            self.batch_progress["value"] = i
            percent = int((i / total) * 100)
            self.lbl_batch_percent.config(text=f"{percent}%")
            self.root.update_idletasks()

        messagebox.showinfo(
            "Hoàn thành",
            f"Đã xử lý xong {count}/{total} ảnh.\nLưu tại:\n{output_dir}"
        )

    def _apply_current_mode(self, commit=True):
        if self.original_cv_image is None or self.current_mode is None:
            return

        k_size = int(self.slider_kernel.get()) | 1

        if self.mode_input_image is None:
            if self.processed_cv_image is not None:
                self.mode_input_image = self.processed_cv_image.copy()
            else:
                self.mode_input_image = self.original_cv_image

        base = self.mode_input_image if self.mode_input_image is not None else self.original_cv_image

        temp = self._apply_mode(base.copy(), k_size)

        if commit:
            self.processed_cv_image = temp

        self._show_processed(temp)

    def test_current_mode(self):
        self._apply_current_mode(commit=True)

    def apply_and_commit(self):
        self._apply_current_mode(commit=True)

    def _apply_mode(self, img, k_size):
        if self.current_mode == "gray":
            return apply_grayscale(img)

        elif self.current_mode == "gaussian":
            return apply_gaussian_blur(img, k_size)

        elif self.current_mode == "median":
            return apply_median_filter(img, k_size)

        elif self.current_mode == "dilation":
            return apply_dilation(img, k_size)

        elif self.current_mode == "erosion":
            return apply_erosion(img, k_size)

        return img

    def _show_processed(self, img):
        if len(img.shape) == 2:
            display_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.show_image_on_label(display_rgb, self.lbl_processed_img)
