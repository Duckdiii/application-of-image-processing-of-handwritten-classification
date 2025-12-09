import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np

# Import các hàm xử lý từ src (Backend)

from src.preprocessing.morphology.grayscale import apply_grayscale

from src.preprocessing.filters.gaussian_blur import apply_gaussian_blur
from src.preprocessing.filters.median_filter import apply_median_filter
from src.preprocessing.morphology.dilation import apply_dilation
from src.preprocessing.morphology.erosion import apply_erosion


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý ảnh - Digital Image Processing")
        self.root.geometry("1200x800")

        # Biến lưu trữ ảnh (OpenCV format: BGR hoặc Grayscale)
        self.original_cv_image = None
        self.processed_cv_image = None  # Dùng để lưu và reset ảnh
        self.current_folder_path = ""
        self.current_mode = None

        self.setup_ui()

    def setup_ui(self):
        # --- CẤU HÌNH LAYOUT (GRID) ---
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=1)

        # --- 1. KHUNG ẢNH GỐC ---
        self.frame_original = tk.Frame(
            self.root, bg="#A9A9A9", bd=2, relief="sunken")
        self.frame_original.grid(
            row=0, column=0, sticky="nsew", padx=2, pady=2)
        tk.Label(self.frame_original, text="Ảnh Gốc (Original)",
                 bg="#A9A9A9", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_original_img = tk.Label(self.frame_original, bg="#A9A9A9")
        self.lbl_original_img.pack(expand=True)

        # --- 2. KHUNG ẢNH XỬ LÝ ---
        self.frame_processed = tk.Frame(
            self.root, bg="#808080", bd=2, relief="sunken")
        self.frame_processed.grid(
            row=0, column=1, sticky="nsew", padx=2, pady=2)
        tk.Label(self.frame_processed, text="Ảnh Đang Xử Lý (Result)",
                 bg="#808080", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        self.lbl_processed_img = tk.Label(self.frame_processed, bg="#808080")
        self.lbl_processed_img.pack(expand=True)

        # --- 3. KHUNG CHỨC NĂNG (BẢNG ĐIỀU KHIỂN) ---
        self.frame_controls = tk.Frame(self.root, bg="#5DADE2", width=250)
        self.frame_controls.grid(
            row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
        self.frame_controls.pack_propagate(False)

        tk.Label(self.frame_controls, text="Bảng Điều Khiển",
                 bg="#5DADE2", font=("Arial", 14, "bold")).pack(pady=20)

        # Nút chọn thư mục
        tk.Button(self.frame_controls, text="📂 Chọn Thư Mục Ảnh", command=self.load_folder,
                  height=2, bg="white").pack(fill="x", padx=10, pady=(0, 10))

        # Nút Lưu và Reset (MỚI)
        tk.Button(self.frame_controls, text="💾 Lưu Ảnh Xử Lý", command=self.save_processed_image,
                  height=2, bg="#2ECC71", fg="white").pack(fill="x", padx=10, pady=(5, 5))
        tk.Button(self.frame_controls, text="🔄 Reset Ảnh", command=self.reset_image,
                  height=1, bg="#E74C3C", fg="white").pack(fill="x", padx=10, pady=(5, 10))

        # Các nút chức năng xử lý
        tk.Label(self.frame_controls, text="Chọn thuật toán:",
                 bg="#5DADE2").pack(pady=(10, 5))
        tk.Button(self.frame_controls, text="Grayscale", command=lambda: self.set_mode(
            "gray")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Gaussian Blur", command=lambda: self.set_mode(
            "gaussian")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Median Filter", command=lambda: self.set_mode(
            "median")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Dilation (Giãn)", command=lambda: self.set_mode(
            "dilation")).pack(fill="x", padx=20, pady=2)
        tk.Button(self.frame_controls, text="Erosion (Co)", command=lambda: self.set_mode(
            "erosion")).pack(fill="x", padx=20, pady=2)

        # Slider điều chỉnh tham số (Kernel size)
        tk.Label(self.frame_controls, text="Kích thước Kernel / Mức độ:",
                 bg="#5DADE2").pack(pady=(20, 5))
        self.slider_kernel = tk.Scale(self.frame_controls, from_=1, to=21,
                                      orient="horizontal", bg="#5DADE2", command=self.on_slider_change)
        self.slider_kernel.set(3)
        self.slider_kernel.pack(fill="x", padx=20)

        # --- 4. KHUNG GALLERY ẢNH (BOTTOM) ---
        self.frame_gallery = tk.Frame(self.root, bg="#F4D03F", height=200)
        self.frame_gallery.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

        tk.Label(self.frame_gallery, text="Danh sách ảnh trong thư mục", bg="#F4D03F", font=(
            "Arial", 10, "bold")).pack(anchor="nw", padx=5, pady=2)

        # Canvas và Scrollbar (Đã thiết kế để dùng tối đa không gian ngang)
        self.canvas_gallery = tk.Canvas(self.frame_gallery, bg="#F4D03F")
        self.scrollbar_gallery = tk.Scrollbar(
            self.frame_gallery, orient="horizontal", command=self.canvas_gallery.xview)

        # Gallery Content Frame
        self.gallery_content = tk.Frame(self.canvas_gallery, bg="#F4D03F")

        # Cần bind để cập nhật scrollregion khi nội dung thay đổi
        self.gallery_content.bind("<Configure>", lambda e: self.canvas_gallery.configure(
            scrollregion=self.canvas_gallery.bbox("all")))

        # Thêm gallery_content vào canvas
        # NOTE: Thay vì dùng pack, chúng ta dùng create_window để gallery_content có thể cuộn ngang
        self.canvas_gallery.create_window(
            (0, 0), window=self.gallery_content, anchor="nw")
        self.canvas_gallery.configure(
            xscrollcommand=self.scrollbar_gallery.set)

        self.canvas_gallery.pack(side="top", fill="both", expand=True)
        self.scrollbar_gallery.pack(side="bottom", fill="x")

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
            files = [f for f in os.listdir(self.current_folder_path) if os.path.splitext(f)[
                1].lower() in valid_extensions]

            for f in files:
                path = os.path.join(self.current_folder_path, f)
                # Tạo thumbnail
                img = Image.open(path)
                img.thumbnail((80, 80))
                photo = ImageTk.PhotoImage(img)

                # Tạo nút bấm chứa ảnh
                # Dùng pady lớn hơn (ví dụ 10-15) để ảnh được căn giữa trong chiều cao cố định (200px),
                # giúp "lấp đầy khoảng trống" một cách trực quan.
                btn = tk.Button(self.gallery_content, image=photo,
                                command=lambda p=path: self.display_original(p))
                btn.image = photo
                # Tăng pady để căn giữa ảnh nhỏ trong gallery
                btn.pack(side="left", padx=5, pady=15)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc thư mục: {e}")

    def display_original(self, image_path):
        """Đọc và hiển thị ảnh gốc, đồng thời reset ảnh xử lý."""
        try:
            # Đọc ảnh bằng OpenCV
            self.original_cv_image = cv2.imread(image_path)
            # Khởi tạo ảnh xử lý bằng cách sao chép ảnh gốc
            self.processed_cv_image = self.original_cv_image.copy()

            # Convert BGR (OpenCV) to RGB (Pillow) để hiển thị
            img_rgb = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)

            # Hiển thị ảnh gốc
            self.show_image_on_label(img_rgb, self.lbl_original_img)

            # Hiển thị ảnh xử lý LÀM ẢNH GỐC (Trạng thái chưa xử lý)
            self.show_image_on_label(img_rgb, self.lbl_processed_img)

            # THAY ĐỔI CHÍNH: Reset chế độ xử lý.
            # Loại bỏ dòng tự động gọi process_image() ở đây.
            self.current_mode = None

        except Exception as e:
            messagebox.showerror("Lỗi hiển thị", f"Không thể tải ảnh: {e}")

    def reset_image(self):
        """Đặt lại ảnh đang xử lý thành ảnh gốc."""
        if self.original_cv_image is not None:
            # Sao chép lại ảnh gốc để reset
            self.processed_cv_image = self.original_cv_image.copy()

            # Chuyển đổi để hiển thị
            img_rgb = cv2.cvtColor(self.processed_cv_image, cv2.COLOR_BGR2RGB)
            self.show_image_on_label(img_rgb, self.lbl_processed_img)
            self.current_mode = None
            messagebox.showinfo("Thông báo", "Đã reset ảnh thành công.")
        else:
            messagebox.showwarning("Cảnh báo", "Không có ảnh gốc để reset.")

    def save_processed_image(self):
        """Lưu ảnh đang xử lý."""
        if self.processed_cv_image is None or self.original_cv_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh xử lý để lưu.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                       ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Lưu ảnh đã xử lý"
        )

        if file_path:
            try:
                # Lưu ảnh trực tiếp bằng OpenCV (dạng BGR hoặc Grayscale)
                cv2.imwrite(file_path, self.processed_cv_image)
                messagebox.showinfo(
                    "Thành công", f"Đã lưu ảnh thành công tại:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

    def show_image_on_label(self, cv_image_rgb, label_widget):
        h, w, _ = cv_image_rgb.shape

        # --- SỬA LỖI Ở ĐÂY ---
        # Lấy kích thước của Frame cha (Parent) chứa cái Label đó
        # Frame cha được set grid sticky="nsew" nên kích thước nó sẽ ổn định
        parent_frame = label_widget.master

        frame_width = parent_frame.winfo_width()
        frame_height = parent_frame.winfo_height()

        # Fallback: Nếu chưa render xong (kích thước = 1) thì lấy mặc định 400
        if frame_width < 100:
            frame_width = 400
        if frame_height < 100:
            frame_height = 400

        scale_w = frame_width / w
        scale_h = frame_height / h
        # Giảm 10% để tạo lề thoáng hơn chút
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
        self.process_image()

    def on_slider_change(self, val):
        if self.original_cv_image is not None and self.current_mode:
            self.process_image()

    def process_image(self):
        if self.original_cv_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return

        k_size = int(self.slider_kernel.get())

        # Dùng biến tạm để xử lý trên bản sao của ảnh gốc
        temp_processed = self.original_cv_image.copy()

        try:
            # --- Thực hiện xử lý ảnh ---
            if self.current_mode == "gray":
                temp_processed = apply_grayscale(temp_processed)

            elif self.current_mode == "gaussian":
                temp_processed = apply_gaussian_blur(temp_processed, k_size)

            elif self.current_mode == "median":
                temp_processed = apply_median_filter(temp_processed, k_size)

            elif self.current_mode == "dilation":
                temp_processed = apply_dilation(temp_processed, k_size)

            elif self.current_mode == "erosion":
                temp_processed = apply_erosion(temp_processed, k_size)

            # Lưu kết quả CV (BGR/Gray) vào biến chính để có thể lưu file
            self.processed_cv_image = temp_processed

            # --- Chuyển đổi sang RGB cho hiển thị trên giao diện ---
            if len(self.processed_cv_image.shape) == 2:  # Nếu là Grayscale
                display_img_rgb = cv2.cvtColor(
                    self.processed_cv_image, cv2.COLOR_GRAY2RGB)
            else:  # Nếu là BGR (hoặc đã là RGB)
                display_img_rgb = cv2.cvtColor(
                    self.processed_cv_image, cv2.COLOR_BGR2RGB)

            self.show_image_on_label(display_img_rgb, self.lbl_processed_img)

        except Exception as e:
            messagebox.showerror("Lỗi xử lý", f"Lỗi: {e}")
