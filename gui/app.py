# Nhập các thư viện cần thiết
import sys
import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Thêm đường dẫn vào sys.path để truy cập các tiện ích (ví dụ: module preprocess)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
# Nhập hàm preprocess_image từ module preprocess
from preprocess import preprocess_image

# Danh sách các nhãn cảm xúc tương ứng với đầu ra của mô hình
emotion_labels = ['Ang', 'Dis', 'Fea', 'Hap', 'Neu', 'Sad', 'Sur']

# Biến toàn cục để lưu trữ mô hình hiện tại
current_model = None

# Hàm tải mô hình mặc định khi khởi động
def load_default_model():
    global current_model
    # Xây dựng đường dẫn đến tệp mô hình mặc định
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_model.keras'))
    try:
        # Tải mô hình bằng Keras
        current_model = load_model(default_path)
        # Cập nhật nhãn giao diện để hiển thị tên mô hình đã tải
        model_label.config(text=f"Mô hình: {os.path.basename(default_path)}")
    except Exception as e:
        # Hiển thị thông báo lỗi nếu tải mô hình thất bại và thoát
        messagebox.showerror("Lỗi", f"Không thể tải mô hình mặc định:\n{str(e)}")
        sys.exit(1)

# Hàm cho phép người dùng chọn và tải mô hình tùy chỉnh
def select_model_and_load():
    global current_model
    # Mở hộp thoại để chọn tệp mô hình (.h5 hoặc .keras)
    file_path = filedialog.askopenfilename(
        title="Chọn Tệp Mô hình (.h5 hoặc .keras)",
        filetypes=[("Mô hình", "*.keras *.h5")]
    )
    # Thoát nếu không có tệp nào được chọn
    if not file_path:
        return
    try:
        # Tải mô hình đã chọn
        model = load_model(file_path)
        current_model = model
        # Cập nhật nhãn giao diện với tên mô hình mới
        model_label.config(text=f"Mô hình: {os.path.basename(file_path)}")
        # Hiển thị thông báo thành công
        messagebox.showinfo("Tải Mô hình", "Mô hình đã được tải thành công.")
    except Exception as e:
        # Hiển thị thông báo lỗi nếu tải mô hình thất bại
        messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n{str(e)}")

# Hàm dự đoán cảm xúc từ các ảnh được tải lên
def predict_emotion():
    # Vô hiệu hóa nút tải ảnh để tránh nhấn nhiều lần
    button.config(state="disabled")
    # Cập nhật nhãn trạng thái để báo đang xử lý
    stats_label.config(text="Đang xử lý... Vui lòng đợi.")

    # Hàm nội bộ để xử lý ảnh trong một luồng riêng
    def process_images():
        # Mở hộp thoại để chọn một hoặc nhiều tệp ảnh
        image_paths = filedialog.askopenfilenames(
            title="Chọn Ảnh",
            filetypes=[("Tệp ảnh", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        # Nếu không có ảnh nào được chọn, đặt lại giao diện và thoát
        if not image_paths:
            stats_label.config(text="Phân bố cảm xúc: Không có ảnh nào được chọn")
            button.config(state="normal")
            return

        # Xóa các kết quả trước đó khỏi Treeview
        for row in tree.get_children():
            tree.delete(row)

        # Khởi tạo từ điển để đếm số lần xuất hiện của từng cảm xúc
        emotion_counts = {label: 0 for label in emotion_labels}
        total_images = 0
        # Danh sách để lưu xác suất nhằm tính trung bình
        all_probabilities = []

        # Xử lý từng ảnh được chọn
        for image_path in image_paths:
            try:
                # Lấy tên tệp ảnh
                image_name = os.path.basename(image_path)
                # Tiền xử lý ảnh về kích thước yêu cầu (224x224)
                img = preprocess_image(image_path, img_size=224)
                # Dự đoán bằng mô hình hiện tại
                prediction = current_model.predict(img, verbose=0)[0]
                # Lấy chỉ số của xác suất cao nhất
                emotion_index = np.argmax(prediction)
                # Ánh xạ chỉ số sang nhãn cảm xúc tương ứng
                emotion = emotion_labels[emotion_index]
                # Chuyển xác suất thành phần trăm
                probabilities = prediction * 100
                # Định dạng xác suất thành chuỗi để hiển thị
                prob_str = ", ".join([f"{label}: {prob:.2f}%" for label, prob in zip(emotion_labels, probabilities)])
                # Lưu xác suất để tính trung bình
                all_probabilities.append(probabilities)
                # Thêm kết quả vào Treeview (cập nhật giao diện phải thực hiện trong luồng chính)
                window.after(0, lambda name=image_name, emo=emotion, prob=prob_str:
                             tree.insert("", "end", values=(name, emo, prob)))
                # Tăng số đếm cho cảm xúc được dự đoán
                emotion_counts[emotion] += 1
                total_images += 1
            except Exception as e:
                # Nếu xảy ra lỗi, hiển thị lỗi trong Treeview
                window.after(0, lambda name=image_name, err=str(e):
                             tree.insert("", "end", values=(name, f"Lỗi: {err}", "")))

        # Nếu có ít nhất một ảnh được xử lý, tính toán và hiển thị thống kê
        if total_images > 0:
            # Tính phần trăm phân bố của các cảm xúc
            emotion_percentages = {label: (count / total_images) * 100 for label, count in emotion_counts.items()}
            # Định dạng phân bố thành chuỗi
            dist_str = "Phân bố cảm xúc: " + ", ".join(
                [f"{label}: {percent:.2f}%" for label, percent in emotion_percentages.items()])
            # Tính xác suất trung bình của tất cả ảnh
            avg_probabilities = np.mean(all_probabilities, axis=0)
            # Định dạng xác suất trung bình thành chuỗi
            avg_str = "Xác suất trung bình: " + ", ".join(
                [f"{label}: {prob:.2f}%" for label, prob in zip(emotion_labels, avg_probabilities)])
            # Kết hợp phân bố và xác suất trung bình để hiển thị
            stats_str = f"{dist_str}\n{avg_str}"
            # Cập nhật nhãn thống kê trong luồng chính
            window.after(0, lambda: stats_label.config(text=stats_str))
        else:
            # Nếu không có ảnh hợp lệ nào được xử lý, cập nhật nhãn trạng thái
            window.after(0, lambda: stats_label.config(text="Phân bố cảm xúc: Không có ảnh hợp lệ nào được xử lý"))
        # Kích hoạt lại nút tải ảnh
        window.after(0, lambda: button.config(state="normal"))

    # Chạy xử lý ảnh trong một luồng riêng để giữ giao diện phản hồi
    thread = threading.Thread(target=process_images)
    thread.start()

# Thiết lập cửa sổ giao diện chính
window = tk.Tk()
window.title("Nhận diện cảm xúc")
window.geometry("800x600")  # Đặt kích thước cửa sổ

# Tạo khung để chứa các thành phần giao diện
frame = tk.Frame(window)
frame.pack(pady=10)

# Nhãn để hiển thị tên của mô hình hiện tại
model_label = tk.Label(frame, text="Mô hình: Đang tải...", fg="blue")
model_label.pack(pady=5)

# Nút để chọn mô hình tùy chỉnh
select_model_btn = tk.Button(frame, text="Chọn Mô hình", command=select_model_and_load)
select_model_btn.pack(pady=5)

# Nút để tải ảnh lên nhằm dự đoán cảm xúc
button = tk.Button(frame, text="Tải Ảnh Lên", command=predict_emotion)
button.pack(pady=5)

# Nhãn để hiển thị phân bố cảm xúc và xác suất trung bình
stats_label = tk.Label(frame, text="Phân bố cảm xúc:", wraplength=750, justify="left")
stats_label.pack(pady=5)

# Tạo Treeview để hiển thị kết quả dự đoán
columns = ("Tên Ảnh", "Cảm Xúc Dự Đoán", "Xác Suất (%)")
tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
# Thiết lập tiêu đề cho các cột
for col in columns:
    tree.heading(col, text=col)
# Cấu hình chiều rộng các cột
tree.column("Tên Ảnh", width=150)
tree.column("Cảm Xúc Dự Đoán", width=150)
tree.column("Xác Suất (%)", width=500)
tree.pack(pady=5)

# Thêm thanh cuộn dọc cho Treeview
scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Tải mô hình mặc định khi ứng dụng khởi động
load_default_model()

# Bắt đầu vòng lặp sự kiện chính của giao diện
window.mainloop()