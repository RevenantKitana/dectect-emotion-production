# Nhập thư viện tkinter để tạo giao diện người dùng đồ họa
import tkinter as tk

# Tạo cửa sổ chính của ứng dụng
window = tk.Tk()
# Đặt tiêu đề cho cửa sổ
window.title("Nhận diện cảm xúc")

# Thêm nhãn và nút vào giao diện
# Tạo một nhãn với nội dung "Emotion Recognition App" và đặt nó vào cửa sổ
label = tk.Label(window, text="Ứng dụng Nhận diện Cảm xúc")
label.pack()  # Sắp xếp nhãn tự động trong cửa sổ

# Tạo một nút với nội dung "Start", hiện chưa gắn hàm xử lý (command=None)
button = tk.Button(window, text="Bắt đầu", command=None)
button.pack()  # Sắp xếp nút tự động trong cửa sổ

# Bắt đầu vòng lặp sự kiện chính để chạy giao diện
window.mainloop()