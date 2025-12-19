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
from src.segmentation.deskew.deskew import deskew

class SegmentationUI:
    def __init__(self, root, image_path: str):
        self.root = root
        self.root.title("Segmentation - Tách Ký Tự")
        self.root.geometry("1200x800")

        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.processed_cv_image = self.original_cv_image.copy()
        self.cropped_lines = []
        self.final_char_imgs = []

        self.setup_ui()
        self.update_all_images()

        # Mặc định hiển thị thư mục preprocessing; nếu không tồn tại thì dùng thư mục ảnh hiện tại
        default_dataset = r"D:\Data\Python\digital-image-processing\archive\preprocessing"
        folder_path = default_dataset if os.path.isdir(default_dataset) else os.path.dirname(self.image_path)
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
        # rowspan=2 để phủ cả khu vực gallery (kéo dài xuống dưới)
        self.frame_controls.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=2, pady=2)
        self.frame_controls.pack_propagate(False)

        tk.Label(
            self.frame_controls, text="Bảng Điều Khiển - Segmentation",
            bg="#5DADE2", font=("Arial", 14, "bold")
        ).pack(pady=(12, 10))

        tk.Button(
            self.frame_controls, text="Lưu ảnh Segmentation",
            command=self.save_processed_image,
            height=2, bg="#2ECC71", fg="white"
        ).pack(fill="x", padx=10, pady=(0, 6))
        tk.Button(
            self.frame_controls, text="Lưu tất cả kết quả",
            command=self.save_all_results,
            height=1, bg="#27AE60", fg="white"
        ).pack(fill="x", padx=10, pady=(0, 8))

        tk.Button(
            self.frame_controls, text="Reset ảnh",
            command=self.reset_image,
            height=1, bg="#E74C3C", fg="white"
        ).pack(fill="x", padx=10, pady=(0, 12))

        # Nhóm thuật toán
        btn_pad = dict(fill="x", padx=20, pady=2)
        tk.Label(self.frame_controls, text="Thuật toán:", bg="#5DADE2").pack(pady=(4, 4))
        tk.Label(self.frame_controls, text="Thresholding:", bg="#EC4A05").pack(pady=(4, 4))
        tk.Button(self.frame_controls, text="Otsu Threshold",
                  command=self.apply_otsu).pack(**btn_pad)
        tk.Button(self.frame_controls, text="Adaptive Threshold",
                  command=self.apply_adaptive).pack(**btn_pad)
        tk.Label(self.frame_controls, text="Edge Detection:", bg="#EC4A05").pack(pady=(8, 4))
        tk.Button(self.frame_controls, text="Canny Edge",
                  command=self.apply_canny).pack(**btn_pad)
        tk.Button(self.frame_controls, text="Sobel Edge",
                  command=self.apply_sobel).pack(**btn_pad)
        tk.Label(self.frame_controls, text="Deskew:", bg="#EC4A05").pack(pady=(8, 4))
        tk.Button(self.frame_controls, text="Deskew Ảnh",
                  command=self.apply_deskew).pack(**btn_pad)
        tk.Label(self.frame_controls, text="Projection Profile Segmentation:", bg="#EC4A05").pack(pady=(8, 4))
        tk.Button(self.frame_controls, text="Tách Dòng (Projection)",
                  command=self.apply_line_seg).pack(**btn_pad)
        tk.Button(self.frame_controls, text="Tách Ký Tự (Projection)",
                  command=self.apply_char_seg).pack(**btn_pad)
        tk.Button(self.frame_controls, text="Xem ký tự đã cắt",
                  command=self.show_characters_window).pack(**btn_pad)

        # ===== GALLERY ẢNH =====
        self.frame_gallery = tk.Frame(self.root, bg="#F4D03F", height=220)
        self.frame_gallery.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=2, pady=2
        )

        tk.Label(
            self.frame_gallery, text="Danh sách ảnh trong thư mục",
            bg="#F4D03F", font=("Arial", 10, "bold")
        ).pack(anchor="nw", padx=5, pady=2)

        self.canvas_gallery = tk.Canvas(self.frame_gallery, bg="#F4D03F", highlightthickness=0)
        self.scrollbar_gallery = tk.Scrollbar(
            self.frame_gallery, orient="vertical",
            command=self.canvas_gallery.yview
        )

        self.gallery_content = tk.Frame(self.canvas_gallery, bg="#F4D03F")
        self.gallery_content.bind(
            "<Configure>", lambda e: self.canvas_gallery.configure(
                scrollregion=self.canvas_gallery.bbox("all"))
        )

        self.canvas_gallery.create_window(
            (0, 0), window=self.gallery_content, anchor="nw"
        )
        self.canvas_gallery.configure(yscrollcommand=self.scrollbar_gallery.set)

        self.canvas_gallery.pack(side="left", fill="both", expand=True)
        self.scrollbar_gallery.pack(side="right", fill="y")

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
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        self.processed_cv_image = apply_otsu_threshold(base)
        self.update_all_images()

    def apply_adaptive(self):
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        self.processed_cv_image = apply_adaptive_threshold(base)
        self.update_all_images()

    def apply_canny(self):
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        self.processed_cv_image = apply_canny(base)
        self.update_all_images()

    def apply_sobel(self):
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        self.processed_cv_image = apply_sobel(base)
        self.update_all_images()

    def apply_line_seg(self):
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        lines, vis_img = segment_lines(base)
        self.cropped_lines = lines
        self.processed_cv_image = vis_img
        self.update_all_images()

    def apply_char_seg(self):
        self.final_char_imgs = []
        if not getattr(self, 'cropped_lines', None):
            messagebox.showwarning("Cảnh báo", "Chưa có dòng nào để tách ký tự. Hãy chạy Tách Dòng trước.")
            return

        print(f"Đang xử lý {len(self.cropped_lines)} dòng...")
        for line_img in self.cropped_lines:
            chars, _ = segment_characters(line_img)
            self.final_char_imgs.extend(chars)

        print(f"Tổng cộng đã tách được: {len(self.final_char_imgs)} ký tự.")
        messagebox.showinfo("Thành công", f"Đã tách được {len(self.final_char_imgs)} ký tự. Sẵn sàng nhận diện.")

    def apply_deskew(self):
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        self.processed_cv_image = deskew(base)
        self.update_all_images()

    def show_characters_window(self):
        if not getattr(self, "final_char_imgs", None) or len(self.final_char_imgs) == 0:
            messagebox.showwarning("Canh bao", "Chua co ky tu de xem. Hay chay Tach Ky Tu truoc.")
            return

        viewer = tk.Toplevel(self.root)
        viewer.title("Ky tu da cat")
        viewer.geometry("900x600")

        top_controls = tk.Frame(viewer)
        top_controls.pack(side="top", fill="x", padx=8, pady=6)
        tk.Label(top_controls, text="Kich thuoc thumbnail:").pack(side="left")
        thumb_size_var = tk.IntVar(value=90)
        thumb_slider = tk.Scale(
            top_controls, from_=40, to=200, orient="horizontal",
            variable=thumb_size_var, showvalue=True,
            command=lambda _: render_thumbnails()
        )
        thumb_slider.pack(side="left", fill="x", expand=True, padx=8)

        canvas = tk.Canvas(viewer, bg="#f5f5f5", highlightthickness=0)
        scrollbar = tk.Scrollbar(viewer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        grid_frame = tk.Frame(canvas, bg="#f5f5f5")
        canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        image_refs = []

        def render_thumbnails():
            for widget in grid_frame.winfo_children():
                widget.destroy()
            image_refs.clear()

            size = int(thumb_size_var.get())
            max_columns = 8
            row = 0
            col = 0

            for idx, char_img in enumerate(self.final_char_imgs):
                if len(char_img.shape) == 2:
                    rgb_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)

                pil_img = Image.fromarray(rgb_img)
                pil_img.thumbnail((size, size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)

                lbl = tk.Label(grid_frame, image=photo, bd=1, relief="solid", bg="white")
                lbl.image = photo
                lbl.grid(row=row, column=col, padx=6, pady=6)
                image_refs.append(photo)

                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

            grid_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        render_thumbnails()

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

    def save_all_results(self):
        """Lưu ảnh đã xử lý + các dòng/ký tự đã cắt vào thư mục segmentation."""
        save_root = r"D:\Data\Python\digital-image-processing\archive\segmentation"
        try:
            os.makedirs(save_root, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không tạo được thư mục đích:\n{e}")
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "seg_result"
        case_dir = os.path.join(save_root, base_name)
        lines_dir = os.path.join(case_dir, "lines")
        chars_dir = os.path.join(case_dir, "chars")

        try:
            os.makedirs(case_dir, exist_ok=True)
            os.makedirs(lines_dir, exist_ok=True)
            os.makedirs(chars_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không tạo được thư mục con:\n{e}")
            return

        saved_processed = False
        if self.processed_cv_image is not None:
            processed_path = os.path.join(case_dir, f"{base_name}_processed.png")
            cv2.imwrite(processed_path, self.processed_cv_image)
            saved_processed = True

        saved_lines = 0
        if getattr(self, "cropped_lines", None):
            for idx, ln in enumerate(self.cropped_lines, start=1):
                line_path = os.path.join(lines_dir, f"line_{idx:03d}.png")
                cv2.imwrite(line_path, ln)
                saved_lines += 1

        saved_chars = 0
        if getattr(self, "final_char_imgs", None):
            for idx, ch in enumerate(self.final_char_imgs, start=1):
                char_path = os.path.join(chars_dir, f"char_{idx:04d}.png")
                cv2.imwrite(char_path, ch)
                saved_chars += 1

        msg_parts = []
        if saved_processed:
            msg_parts.append("Ảnh kết quả")
        if saved_lines:
            msg_parts.append(f"{saved_lines} dòng")
        if saved_chars:
            msg_parts.append(f"{saved_chars} ký tự")
        if not msg_parts:
            msg_parts.append("Chưa có dữ liệu để lưu")

        messagebox.showinfo(
            "Đã lưu",
            f"{', '.join(msg_parts)}\nThư mục: {case_dir}"
        )

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

            max_columns = 6
            row = 0
            col = 0

            for f in files:
                path = os.path.join(folder_path, f)
                img = Image.open(path)
                img.thumbnail((90, 90))
                photo = ImageTk.PhotoImage(img)

                btn = tk.Button(
                    self.gallery_content,
                    image=photo,
                    command=lambda p=path: self.display_original_from_gallery(p),
                    relief="flat",
                )
                btn.image = photo
                btn.grid(row=row, column=col, padx=8, pady=8)

                col += 1
                if col >= max_columns:
                    col = 0
                    row += 1

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc thư mục: {e}")

    def display_original_from_gallery(self, image_path):
        self.image_path = image_path
        self.original_cv_image = cv2.imread(image_path)
        self.processed_cv_image = self.original_cv_image.copy()
        self.update_all_images()
