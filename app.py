import customtkinter as ctk         # Tạo giao diện người dùng (GUI) hiện đại, hỗ trợ Dark mode, theme, nút đẹp
import cv2                          # xử lý hình ảnh và video
from PIL import Image, ImageTk      # Pillow - chuyển đổi, hiển thị, xử lý ảnh
from ultralytics import YOLO        # Ultralytics YOLO - tải và chạy mô hình học sâu nhận diện đối tượng (ở đây là biển báo giao thông)
from tkinter import filedialog      # Tkinter gốc của Python - tạo hộp thoại chọn file hoặc lưu file
import threading                    # cho phép chạy các tác vụ nặng (như đọc camera, nhận diện YOLO) song song với GUI, tránh treo ứng dụng
import os                           # thao tác với hệ thống file và thư mục
import datetime                     # xử lý thời gian, ngày tháng
import torch                        # PyTorch, framework học sâu được YOLO sử dụng - kiểm tra thiết bị xử lý (CPU/GPU), và hỗ trợ YOLO chạy nhanh hơn

# ------------------ GIAO DIỆN CHÍNH ------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TrafficSignApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Sign Detection System")
        self.geometry("1100x750")

        # Tải mô hình YOLO
        self.model = YOLO("yolo11n_traffic_150_quyen.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Giao diện
        self.create_header()
        self.create_main_area()
        self.create_result_area()

        # Biến trạng thái
        self.cap = None
        self.running = False
        self.current_frame = None  # Lưu frame hiện tại

    # ------------------ GIAO DIỆN ------------------
    def create_header(self):
        header = ctk.CTkLabel(self, text="TRAFFIC SIGN DETECTION SYSTEM",
                              font=ctk.CTkFont(size=26, weight="bold"))
        header.pack(pady=15)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=5)

        self.btn_camera = ctk.CTkButton(button_frame, text="📷 Nhận diện bằng Camera",
                                        command=self.start_camera_mode)
        self.btn_camera.grid(row=0, column=0, padx=15)

        self.btn_image = ctk.CTkButton(button_frame, text="🖼️ Nhận diện bằng Ảnh",
                                       command=self.start_image_mode)
        self.btn_image.grid(row=0, column=1, padx=15)

    def create_main_area(self):
        self.display_frame = ctk.CTkFrame(self, width=900, height=500)
        self.display_frame.pack(pady=10)
        self.display_frame.pack_propagate(False)

        self.display_label = ctk.CTkLabel(self.display_frame, text="Chưa có nội dung hiển thị",
                                          font=ctk.CTkFont(size=18))
        self.display_label.pack(expand=True)

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=5)

    def create_result_area(self):
        self.result_frame = ctk.CTkFrame(self, width=900, height=180)
        self.result_frame.pack(pady=10)
        self.result_frame.pack_propagate(False)

        self.result_label = ctk.CTkLabel(self.result_frame, text="Kết quả nhận diện sẽ hiển thị tại đây",
                                         font=ctk.CTkFont(size=16))
        self.result_label.pack(expand=True)

    # ------------------ CAMERA MODE ------------------
    def start_camera_mode(self):
        self.stop_camera()
        self.clear_controls()
        self.running = True
        self.cap = cv2.VideoCapture(1)

        # Nút điều khiển camera
        self.btn_capture = ctk.CTkButton(self.control_frame, text="📸 Chụp ảnh", command=self.capture_image)
        self.btn_capture.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(self.control_frame, text="🛑 Tắt camera", command=self.stop_camera)
        self.btn_stop.pack(side="left", padx=10)

        threading.Thread(target=self.update_camera, daemon=True).start()

    def update_camera(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Nhận diện YOLO
            results = self.model(frame, verbose=False, device=self.device)
            annotated = results[0].plot()
            self.current_frame = annotated.copy()  # Lưu frame đã annotate

            # Chuyển đổi để hiển thị
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.display_label.configure(image=imgtk, text="")
            self.display_label.image = imgtk

            # Cập nhật class nhận diện
            detected_classes = set()
            for r in results:
                for c in r.boxes.cls:
                    detected_classes.add(self.model.names[int(c)])

            if detected_classes:
                self.result_label.configure(
                    text="📋 Phát hiện: " + ", ".join(detected_classes))
            else:
                self.result_label.configure(text="⚠️ Không phát hiện biển báo nào")

    def capture_image(self):
        if self.current_frame is not None:
            os.makedirs("captured", exist_ok=True)
            filename = f"captured_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join("captured", filename)
            cv2.imwrite(path, self.current_frame)
            self.result_label.configure(text=f"📸 Ảnh đã lưu tại: {path}")

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    # ------------------ IMAGE MODE ------------------
    def start_image_mode(self):
        self.stop_camera()
        self.clear_controls()
        self.display_label.configure(image=None, text="🖼️ Nhấn 'Chọn ảnh' để nhận diện")

        self.btn_choose = ctk.CTkButton(self.control_frame, text="📂 Chọn ảnh", command=self.load_image)
        self.btn_choose.pack(side="left", padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        img = cv2.imread(path)

        # Nhận diện bằng YOLO
        results = self.model(img, verbose=False, device=self.device)
        annotated = results[0].plot()
        self.current_frame = annotated.copy()  # Lưu ảnh đã annotate

        # Hiển thị ảnh
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_rgb).resize((900, 500))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.display_label.configure(image=imgtk, text="")
        self.display_label.image = imgtk

        # Hiển thị class nhận diện
        detected_classes = set()
        for r in results:
            for c in r.boxes.cls:
                detected_classes.add(self.model.names[int(c)])

        if detected_classes:
            self.result_label.configure(text=f"✅ Phát hiện: {', '.join(detected_classes)}")
        else:
            self.result_label.configure(text="⚠️ Không phát hiện biển báo nào")

        # Nút phụ
        self.clear_controls()
        self.btn_save = ctk.CTkButton(self.control_frame, text="💾 Lưu ảnh có khung", command=self.save_detected_image)
        self.btn_save.pack(side="left", padx=10)

        self.btn_new = ctk.CTkButton(self.control_frame, text="➕ Ảnh khác", command=self.load_image)
        self.btn_new.pack(side="left", padx=10)

    def save_detected_image(self):
        if self.current_frame is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, self.current_frame)
            self.result_label.configure(text=f"💾 Ảnh đã lưu: {save_path}")

    # ------------------ TIỆN ÍCH ------------------
    def clear_controls(self):
        for w in self.control_frame.winfo_children():
            w.destroy()

    def on_closing(self):
        self.stop_camera()
        self.destroy()


# ------------------ CHẠY CHƯƠNG TRÌNH ------------------
if __name__ == "__main__":
    app = TrafficSignApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
