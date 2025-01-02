import cv2
import os
import sys
import warnings
import logging
import queue
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import set_random_seed
import numpy as np
import cvlib
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import threading

# Tắt các thông báo của TF_CPP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_random_seed(42)  # Đảm bảo kết quả dự đoán của mô hình là nhất quán

# Tắt các cảnh báo từ TensorFlow, Keras và cả các cảnh báo chung khác
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module='tensorflow')

# Tắt log từ thư viện absl (thư viện phụ được sử dụng bởi TensorFlow)
logging.getLogger('absl').setLevel(logging.ERROR)

MODEL_PATH = r"E:\DO_AN\CODE\Save_Model"
ICON_PATH = r"E:\DO_AN\CODE\Logo_HaUI.png"

# Tải mô hình
gender_model = load_model(os.path.join(MODEL_PATH, "Gender1.h5"), compile=False)
emotion_model = load_model(os.path.join(MODEL_PATH, "Emotion1.h5"), compile=False)

gender_labels = ['Male', 'Female']  # Nhãn giới tính
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']  # Nhãn cảm xúc

# Tạo cửa sổ tkinter
root = tk.Tk()
root.geometry('1200x700')
root.resizable(False, False)
root.configure(padx=10, pady=10)
root.title('Ứng dụng phân tích khuôn mặt - Dự đoán giới tính & cảm xúc')
root.configure(bg="#EAEDED")
root.iconphoto(True, PhotoImage(file=ICON_PATH))

is_running = False
frame_queue = queue.Queue(maxsize=10)  # Giới hạn số lượng khung hình trong hàng đợi để tránh lỗi bộ nhớ


# Hàm khởi động camera
def use_camera():
    global is_running
    is_running = True
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    exit_button.config(state="normal")

    # Tạo luồng xử lý camera
    worker_thread = threading.Thread(target=camera_worker, daemon=True)
    if not hasattr(threading.current_thread(), "_workers"):
        threading.current_thread()._workers = []
    threading.current_thread()._workers.append(worker_thread)
    worker_thread.start()


# Hàm thoát chương trình
def quit_program():
    answer = messagebox.askyesno("Quit", "Bạn có chắc muốn thoát không?")
    if answer:
        global is_running
        is_running = False
        if hasattr(threading.current_thread(), "_workers"):
            for thread in threading.current_thread()._workers:
                thread.join()
        root.destroy()


# Hàm để dừng camera
def cancel_feed():
    global is_running
    is_running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")

    if hasattr(threading.current_thread(), "_workers"):
        for thread in threading.current_thread()._workers:
            thread.join()


# Luồng xử lý camera
def camera_worker():
    global is_running
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Sử dụng DirectShow backend cho các hệ thống Windows

    while is_running:
        ret, frame = capture.read()

        # Phát hiện khuôn mặt
        faces, confidences = cvlib.detect_face(frame)

        for face, confidence in zip(faces, confidences):
            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]

            # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)  # Màu BGR

            # Cắt phần vùng khuôn mặt đã phát hiện
            face_crop = np.copy(frame[startY:endY, startX:endX])

            # Bỏ qua những khuôn mặt có kích thước quá nhỏ
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                continue

            # Tiền xử lý khuôn mặt cho dự đoán giới tính
            face_crop = cv2.resize(face_crop, (150, 150), interpolation=cv2.INTER_AREA)
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # Dự đoán giới tính và cảm xúc
            conf_model_gender = gender_model.predict(face_crop)[0]
            idx_model_gender = np.argmax(conf_model_gender)
            label_model_gender = gender_labels[idx_model_gender]

            conf_model_emotion = emotion_model.predict(face_crop)[0]
            idx_model_emotion = np.argmax(conf_model_emotion)
            label_model_emotion = emotion_labels[idx_model_emotion]

            # Gắn nhãn giới tính và cảm xúc lên ảnh
            label = "{},{}".format(label_model_gender, label_model_emotion)
            label = "{}: {}".format(label_model_gender, label_model_emotion)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            break  # Chỉ xử lý khuôn mặt đầu tiên được phát hiện

        # Chuyển đổi ảnh từ định dạng BGR của OpenCV sang định dạng ảnh PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((640, 480),
                             Image.Resampling.BICUBIC)  # Sử dụng BICUBIC để tăng chất lượng khi thu nhỏ hoặc phóng to ảnh

        # Đưa khung hình vào hàng đợi để luồng giao diện xử lý
        if not frame_queue.full():
            frame_queue.put(image)

        # Lắng nghe phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            continue
    capture.release()
    cv2.destroyAllWindows()


# Hàm xử lý hàng đợi và hiển thị khung hình mới nhất lên giao diện
def process_queue():
    if not frame_queue.empty():
        image = frame_queue.get()

        # Cập nhật khung hình trên giao diện
        imgtk = ImageTk.PhotoImage(image=image)
        image_label.config(image=imgtk)
        image_label.image = imgtk

    # Lên lịch kiểm tra khung hình tiếp theo
    root.after(10, process_queue)


theme_var = tk.StringVar(value="Default")
main_frame = tk.Frame(root, bg='#FFFFFF', highlightbackground="#3483eb", highlightthickness=2)
main_frame = tk.Frame(root, bg='#B3E5FC')

main_frame.pack(side=tk.LEFT, padx=10, pady=10)
main_frame.pack_propagate(False)
main_frame.configure(width=1050, height=620)

# Tiêu đề 1
label_title = tk.Label(main_frame, text='Analyze Gender & Emotion',
                       font=("Verdana", 22, "bold"),
                       fg="#1b4f72",
                       bg='#FFFFFF')
# Tiêu đề 2
label_title2 = tk.Label(main_frame, text='Powered by AI',
                        font=("Verdana", 14, "italic"),
                        fg="#2e8b57",
                        bg='#FFFFFF')
# Tiêu đề 3
label_title3 = tk.Label(main_frame, text='Developed by Nguyen Minh Duc',
                        font=("Verdana", 12, "italic"),
                        fg="#2e8b57",
                        bg='#FFFFFF')

# Khung hiển thị camera (luồng video trực tiếp)
image_label = tk.Label(main_frame, bg='#F8F9F9', relief="ridge", borderwidth=3)
image_label.place(x=150, y=100, width=750, height=450)

# Nút "BẮT ĐẦU"
start_button = tk.Button(main_frame,
                         text="START",
                         font=('Verdana', 12, 'bold'),
                         fg='#FFFFFF',
                         relief="solid",
                         bd=1,
                         bg='#4caf50',
                         command=use_camera, cursor="hand2")
start_button.place(x=300, y=570, width=80, height=35)

# Nút "DỪNG"
stop_button = tk.Button(main_frame, text="STOP",
                        font=('Verdana', 12, 'bold'),
                        fg='#FFFFFF',
                        relief="solid",
                        bd=1,
                        bg='#f1c40f',
                        command=cancel_feed,
                        state="disabled", cursor="hand2")
stop_button.place(x=500, y=570, width=80, height=35)

# Nút "THOÁT"
exit_button = tk.Button(main_frame, text="EXIT",
                        font=('Verdana', 12, 'bold'),
                        fg='#FFFFFF',
                        relief="solid",
                        bd=1,
                        bg='#d32f2f',
                        command=quit_program,
                        state="normal", cursor="hand2")
exit_button.place(x=700, y=570, width=80, height=35)

label_title.pack()
label_title2.pack()
label_title3.pack()

# Menu lựa chọn chủ đề cho giao diện
theme_menu = tk.Menubutton(root, text="Theme", font=("Verdana", 10), relief="groove", bg="#e1e2e1", padx=10)
theme_menu.menu = tk.Menu(theme_menu, tearoff=0)
theme_menu["menu"] = theme_menu.menu
theme_menu.menu.add_radiobutton(label="Default", variable=theme_var, value="Default",
                                command=lambda: apply_theme("default"))
theme_menu.menu.add_radiobutton(label="Dark", variable=theme_var, value="Dark", command=lambda: apply_theme("dark"))
theme_menu.menu.add_radiobutton(label="Light", variable=theme_var, value="Light", command=lambda: apply_theme("light"))
theme_menu.pack(side=tk.TOP, anchor="ne", padx=20, pady=10)

# Bắt đầu xử lý hàng đợi để cập nhật giao diện người dùng
root.after(25, process_queue)


# Hàm áp dụng chủ đề vào giao diện
def apply_theme(theme):
    if theme == "dark":
        # Chủ đề tối
        root.configure(bg="black")
        main_frame.configure(bg="#212121")
        label_title.configure(bg="#212121", fg="white")
        label_title2.configure(bg="#212121", fg="white")
        label_title3.configure(bg="#212121", fg="white")
        start_button.configure(bg="#37474F", fg="white")
        stop_button.configure(bg="#37474F", fg="white")
        exit_button.configure(bg="#FF5722", fg="white")
    elif theme == "light":
        # Chủ đề sáng
        root.configure(bg="#EAEDED")
        main_frame.configure(bg="#B3E5FC")
        label_title.configure(bg="#B3E5FC", fg="darkblue")
        label_title2.configure(bg="#B3E5FC", fg="darkgreen")
        label_title3.configure(bg="#B3E5FC", fg="darkgreen")
        start_button.configure(bg="#0288D1", fg="white")
        stop_button.configure(bg="#00897B", fg="white")
        exit_button.configure(bg="#FF7043", fg="black")
    else:
        # Chủ đề mặc định
        root.configure(bg="#EAEDED")
        main_frame.configure(bg="#faf79b")
        label_title.configure(bg="#faf79b", fg="red")
        label_title2.configure(bg="#faf79b", fg="blue")
        label_title3.configure(bg="#faf79b", fg="blue")
        start_button.configure(bg="blue", fg="white")
        stop_button.configure(bg="blue", fg="white")
        exit_button.configure(bg="blue", fg="white")


# Vòng lặp xử lý của giao diện chính
root.mainloop()
