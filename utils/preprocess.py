# Nhập các thư viện cần thiết
import cv2  # Thư viện OpenCV để đọc và xử lý ảnh
import numpy as np  # Thư viện NumPy để xử lý mảng và tính toán số học

# Hàm để tiền xử lý ảnh trước khi đưa vào mô hình
def preprocess_image(image_path, img_size=224):
    """
    Tiền xử lý ảnh: đọc, chuyển màu, thay đổi kích thước, chuẩn hóa.
    
    Args:
        image_path (str): Đường dẫn đến tệp ảnh.
        img_size (int): Kích thước mục tiêu (mặc định 224x224).
    
    Returns:
        numpy.ndarray: Mảng ảnh đã tiền xử lý, sẵn sàng cho mô hình.
    """
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)
    # Chuyển đổi không gian màu từ BGR (mặc định của OpenCV) sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Thay đổi kích thước ảnh về (img_size x img_size), ví dụ: 224x224
    img = cv2.resize(img, (img_size, img_size))  # Sử dụng img_size cho cả chiều rộng và cao
    # Thêm chiều batch để phù hợp với định dạng đầu vào của mô hình (1, img_size, img_size, 3)
    img = np.expand_dims(img, axis=0)
    # Chuẩn hóa giá trị pixel về khoảng [0, 1] bằng cách chia cho 255
    img = img / 255.0
    return img  # Trả về ảnh đã tiền xử lý

def apply_median(frame, k=3):
    return cv2.medianBlur(frame, k)

def apply_clahe_rgb(frame_rgb):
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

def apply_canny(gray):
    return cv2.Canny(gray, 80, 160)

def preprocess_face(face_rgb):
    median = apply_median(face_rgb, 3)

    ycrcb = cv2.cvtColor(median, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    merged = cv2.merge([y_eq, cr, cb])
    clahe_rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

    clahe_gray = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2GRAY)
    edges = apply_canny(clahe_gray)

    resized = cv2.resize(clahe_rgb, (224, 224))
    model_input = resized.astype("float32") / 255.0

    return median, clahe_rgb, edges, model_input
