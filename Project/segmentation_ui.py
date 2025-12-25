import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import cv2
import os
import config
import numpy as np

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
        self.line_spans = []
        self.final_char_imgs = []
        self.char_boxes = []
        self.char_base_shape = None

        self.setup_ui()
        self.update_all_images()

        # Mặc định hiển thị thư mục preprocessing; nếu không tồn tại thì dùng thư mục ảnh hiện tại
        default_dataset = config.DEFAULT_PREPROCESSING_DATASET
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
        # ===== BATCH SEGMENTATION =====
        batch_frame = tk.LabelFrame(
            self.frame_controls,
            text="Batch Segmentation (dataset)",
            bg="#5DADE2",
            fg="black",
            labelanchor="n",
        )
        batch_frame.pack(fill="x", padx=10, pady=(10, 8))

        self.var_batch_deskew = tk.BooleanVar(value=True)
        self.var_batch_otsu = tk.BooleanVar(value=False)
        self.var_batch_adaptive = tk.BooleanVar(value=True)
        self.var_batch_canny = tk.BooleanVar(value=False)
        self.var_batch_sobel = tk.BooleanVar(value=False)
        self.var_batch_line = tk.BooleanVar(value=True)
        self.var_batch_char = tk.BooleanVar(value=True)

        row_pre = tk.Frame(batch_frame, bg="#5DADE2")
        row_pre.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_pre, text="Deskew", variable=self.var_batch_deskew, bg="#5DADE2").pack(side="left")
        tk.Checkbutton(row_pre, text="Otsu", variable=self.var_batch_otsu, bg="#5DADE2").pack(side="left", padx=(8, 0))

        row_adapt = tk.Frame(batch_frame, bg="#5DADE2")
        row_adapt.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_adapt, text="Adaptive", variable=self.var_batch_adaptive, bg="#5DADE2").pack(side="left")
        tk.Label(row_adapt, text="block:", bg="#5DADE2").pack(side="left", padx=(6, 2))
        self.entry_batch_block = tk.Entry(row_adapt, width=4)
        self.entry_batch_block.insert(0, "15")
        self.entry_batch_block.pack(side="left")
        tk.Label(row_adapt, text="C:", bg="#5DADE2").pack(side="left", padx=(6, 2))
        self.entry_batch_c = tk.Entry(row_adapt, width=3)
        self.entry_batch_c.insert(0, "2")
        self.entry_batch_c.pack(side="left")

        row_edge = tk.Frame(batch_frame, bg="#5DADE2")
        row_edge.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_edge, text="Canny", variable=self.var_batch_canny, bg="#5DADE2").pack(side="left")
        tk.Checkbutton(row_edge, text="Sobel", variable=self.var_batch_sobel, bg="#5DADE2").pack(side="left", padx=(8, 0))

        row_seg = tk.Frame(batch_frame, bg="#5DADE2")
        row_seg.pack(fill="x", padx=5, pady=1)
        tk.Checkbutton(row_seg, text="Line Seg", variable=self.var_batch_line, bg="#5DADE2").pack(side="left")
        tk.Checkbutton(row_seg, text="Char Seg", variable=self.var_batch_char, bg="#5DADE2").pack(side="left", padx=(8, 0))

        tk.Button(
            batch_frame,
            text="Run batch segmentation",
            command=self.batch_segment_dataset,
            bg="#1ABC9C",
            fg="white",
        ).pack(fill="x", padx=5, pady=(8, 4))

        self.batch_seg_progress = ttk.Progressbar(
            batch_frame,
            orient="horizontal",
            length=200,
            mode="determinate",
        )
        self.batch_seg_progress.pack(fill="x", padx=5, pady=(4, 2))
        self.lbl_batch_seg_status = tk.Label(batch_frame, text="0/0", bg="#5DADE2")
        self.lbl_batch_seg_status.pack(anchor="center")

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
        lines, vis_img, line_spans = segment_lines(base)
        self.cropped_lines = lines
        self.line_spans = line_spans
        self.processed_cv_image = vis_img
        self.update_all_images()
    def apply_char_seg(self):
        self.final_char_imgs = []
        self.char_boxes = []
        base = self.processed_cv_image if self.processed_cv_image is not None else self.original_cv_image
        self.char_base_shape = base.shape[:2] if base is not None else None
        if not getattr(self, 'cropped_lines', None):
            messagebox.showwarning("C?nh bAo", "Chua co dong nao de tach ky tu. Hay chay Tach Dong truoc.")
            return

        print(f"Dang xu ly {len(self.cropped_lines)} dong...")
        for line_img, line_span in zip(self.cropped_lines, self.line_spans):
            chars, _, boxes = segment_characters(line_img)
            self.final_char_imgs.extend(chars)
            y_offset = line_span[0]
            for (x1, y1, x2, y2) in boxes:
                self.char_boxes.append((x1, y1 + y_offset, x2, y2 + y_offset))

        print(f"Tong cong da tach duoc: {len(self.final_char_imgs)} ky tu.")
        messagebox.showinfo("Thanh cong", f"Da tach duoc {len(self.final_char_imgs)} ky tu. San sang nhan dien.")

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
        save_root = config.ARCHIVE_SEGMENTATION_DIR
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
            sheet = self._build_char_sheet(self.final_char_imgs)
            if sheet is not None:
                sheet_path = os.path.join(case_dir, "chars_sheet.png")
                cv2.imwrite(sheet_path, sheet)

        if self.char_boxes and self.char_base_shape:
            labels_dir = os.path.join(case_dir, "char_labels")
            self._write_yolo_labels_per_char(labels_dir, self.char_boxes, self.char_base_shape)

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

    def _build_char_sheet(self, chars, cell_size=64, cols=16, padding=4):
        if not chars:
            return None

        rows = (len(chars) + cols - 1) // cols
        sheet_h = rows * cell_size + (rows + 1) * padding
        sheet_w = cols * cell_size + (cols + 1) * padding
        sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

        max_inner = max(1, cell_size - (padding * 2))

        for idx, char_img in enumerate(chars):
            if char_img is None:
                continue
            if len(char_img.shape) == 2:
                gray = char_img
            else:
                gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape[:2]
            if h == 0 or w == 0:
                continue

            scale = min(max_inner / w, max_inner / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            row = idx // cols
            col = idx % cols
            x0 = padding + col * (cell_size + padding) + (cell_size - new_w) // 2
            y0 = padding + row * (cell_size + padding) + (cell_size - new_h) // 2
            sheet[y0:y0 + new_h, x0:x0 + new_w] = resized

        return sheet

    def _write_yolo_labels_per_char(self, output_dir, boxes, image_shape, name_prefix="char"):
        if not boxes or image_shape is None:
            return 0

        height, width = image_shape[:2]
        if width <= 0 or height <= 0:
            return 0

        os.makedirs(output_dir, exist_ok=True)
        saved = 0

        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height

            char_name = f"{name_prefix}_{idx:04d}.png"
            line = f"{char_name} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            label_path = os.path.join(output_dir, f"{name_prefix}_{idx:04d}.txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(line)
            saved += 1

        return saved

    def _get_int_from_entry(self, entry_widget, default):
        try:
            value = int(entry_widget.get().strip())
        except Exception:
            value = default
        return value

    def batch_segment_dataset(self):
        dataset_dir = getattr(self, "dataset_folder", None) or config.DEFAULT_PREPROCESSING_DATASET
        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showwarning("Canh bao", "Khong tim thay thu muc dataset de chay batch.")
            return

        ops_selected = any([
            self.var_batch_deskew.get(),
            self.var_batch_otsu.get(),
            self.var_batch_adaptive.get(),
            self.var_batch_canny.get(),
            self.var_batch_sobel.get(),
            self.var_batch_line.get(),
            self.var_batch_char.get(),
        ])
        if not ops_selected:
            messagebox.showwarning("Canh bao", "Hay chon it nhat 1 ky thuat de chay batch.")
            return

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [
            f for f in os.listdir(dataset_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        if not files:
            messagebox.showwarning("Canh bao", "Thu muc dataset khong co anh hop le.")
            return

        block_size = self._get_int_from_entry(self.entry_batch_block, 15)
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        c_value = self._get_int_from_entry(self.entry_batch_c, 2)

        save_root = config.ARCHIVE_SEGMENTATION_DIR
        os.makedirs(save_root, exist_ok=True)

        total = len(files)
        self.batch_seg_progress["value"] = 0
        self.batch_seg_progress["maximum"] = total
        self.lbl_batch_seg_status.config(text=f"0/{total}")
        self.root.update_idletasks()

        saved = 0
        for idx, fname in enumerate(files, start=1):
            in_path = os.path.join(dataset_dir, fname)
            img = cv2.imread(in_path)
            if img is None:
                continue

            processed = img

            if self.var_batch_deskew.get():
                processed = deskew(processed)

            if self.var_batch_adaptive.get():
                processed = apply_adaptive_threshold(processed, block_size, c_value)
            elif self.var_batch_otsu.get():
                processed = apply_otsu_threshold(processed)

            if self.var_batch_canny.get():
                processed = apply_canny(processed)
            elif self.var_batch_sobel.get():
                processed = apply_sobel(processed)
            processed_to_save = processed
            lines = []
            line_spans = []
            chars = []
            char_boxes = []

            if self.var_batch_line.get():
                lines, vis_img, line_spans = segment_lines(processed)
                processed_to_save = vis_img

            if self.var_batch_char.get():
                if self.var_batch_line.get():
                    for line_img, line_span in zip(lines, line_spans):
                        chs, _, boxes = segment_characters(line_img)
                        chars.extend(chs)
                        y_offset = line_span[0]
                        for (x1, y1, x2, y2) in boxes:
                            char_boxes.append((x1, y1 + y_offset, x2, y2 + y_offset))
                else:
                    chs, vis_char, boxes = segment_characters(processed)
                    chars.extend(chs)
                    char_boxes.extend(boxes)
                    processed_to_save = vis_char

            base_name = os.path.splitext(fname)[0]
            case_dir = os.path.join(save_root, base_name)
            lines_dir = os.path.join(case_dir, "lines")
            chars_dir = os.path.join(case_dir, "chars")
            os.makedirs(case_dir, exist_ok=True)
            os.makedirs(lines_dir, exist_ok=True)
            os.makedirs(chars_dir, exist_ok=True)

            cv2.imwrite(os.path.join(case_dir, f"{base_name}_processed.png"), processed_to_save)

            for line_idx, line_img in enumerate(lines, start=1):
                line_path = os.path.join(lines_dir, f"line_{line_idx:03d}.png")
                cv2.imwrite(line_path, line_img)
            if chars:
                for char_idx, char_img in enumerate(chars, start=1):
                    char_path = os.path.join(chars_dir, f"char_{char_idx:04d}.png")
                    cv2.imwrite(char_path, char_img)
                sheet = self._build_char_sheet(chars)
                if sheet is not None:
                    sheet_path = os.path.join(case_dir, "chars_sheet.png")
                    cv2.imwrite(sheet_path, sheet)

            if char_boxes:
                labels_dir = os.path.join(case_dir, "char_labels")
                self._write_yolo_labels_per_char(labels_dir, char_boxes, processed.shape)

            saved += 1
            self.batch_seg_progress["value"] = idx
            self.lbl_batch_seg_status.config(text=f"{idx}/{total}")
            self.root.update_idletasks()

        messagebox.showinfo(
            "Hoan thanh",
            f"Da xu ly {saved}/{total} anh.\nLuu tai: {save_root}"
        )

    # ============================ GALLERY ============================
    def load_images_to_gallery(self, folder_path):
        for widget in self.gallery_content.winfo_children():
            widget.destroy()

        self.dataset_folder = folder_path

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






