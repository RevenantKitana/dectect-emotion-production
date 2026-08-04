# Nhập các thư viện cần thiết
import os  # Thư viện để thao tác với hệ thống tệp
import matplotlib.pyplot as plt  # Thư viện để vẽ biểu đồ
import seaborn as sns  # Thư viện để tạo biểu đồ trực quan (như heatmap)
from sklearn.metrics import classification_report, confusion_matrix  # Hàm để tạo báo cáo phân loại và ma trận nhầm lẫn
import numpy as np  # Thư viện NumPy để xử lý mảng và tính toán số học

# Tạo thư mục 'results' để lưu các kết quả (biểu đồ, báo cáo) nếu chưa tồn tại
os.makedirs("results", exist_ok=True)

# Hàm để vẽ biểu đồ lịch sử huấn luyện (accuracy và loss)
def plot_training_history(history):
    # Lấy dữ liệu lịch sử huấn luyện từ đối tượng history
    acc = history.history.get('accuracy', [])  # Độ chính xác trên tập huấn luyện
    val_acc = history.history.get('val_accuracy', [])  # Độ chính xác trên tập kiểm tra
    loss = history.history.get('loss', [])  # Mất mát trên tập huấn luyện
    val_loss = history.history.get('val_loss', [])  # Mất mát trên tập kiểm tra

    # Tìm giá trị tốt nhất của độ chính xác và mất mát trên tập kiểm tra
    best_acc = max(val_acc)  # Độ chính xác cao nhất
    best_epoch = val_acc.index(best_acc) + 1  # Epoch đạt độ chính xác cao nhất
    best_loss = min(val_loss)  # Mất mát thấp nhất

    # In thông tin về độ chính xác và mất mát tốt nhất
    print(f"✅ Độ chính xác tốt nhất trên tập kiểm tra: {best_acc:.4f} tại Epoch {best_epoch}")
    print(f"✅ Mất mát thấp nhất trên tập kiểm tra: {best_loss:.4f}")

    # Vẽ biểu đồ Accuracy
    plt.figure(figsize=(8, 5))  # Tạo khung biểu đồ với kích thước 8x5
    plt.plot(acc, label='Độ chính xác huấn luyện', marker='o')  # Vẽ đường độ chính xác huấn luyện
    plt.plot(val_acc, label='Độ chính xác kiểm tra', marker='o')  # Vẽ đường độ chính xác kiểm tra
    plt.xlabel('Epoch')  # Đặt nhãn trục X
    plt.ylabel('Độ chính xác')  # Đặt nhãn trục Y
    plt.title('Độ chính xác huấn luyện vs kiểm tra')  # Đặt tiêu đề biểu đồ
    plt.legend()  # Hiển thị chú thích
    plt.grid(True)  # Thêm lưới cho biểu đồ
    plt.tight_layout()  # Tối ưu bố cục
    plt.savefig('results/accuracy_plot.png', dpi=300)  # Lưu biểu đồ vào thư mục results
    plt.close()  # Đóng khung biểu đồ để tiết kiệm bộ nhớ

    # Vẽ biểu đồ Loss
    plt.figure(figsize=(8, 5))  # Tạo khung biểu đồ với kích thước 8x5
    plt.plot(loss, label='Mất mát huấn luyện', marker='o')  # Vẽ đường mất mát huấn luyện
    plt.plot(val_loss, label='Mất mát kiểm tra', marker='o')  # Vẽ đường mất mát kiểm tra
    plt.xlabel('Epoch')  # Đặt nhãn trục X
    plt.ylabel('Mất mát')  # Đặt nhãn trục Y
    plt.title('Mất mát huấn luyện vs kiểm tra')  # Đặt tiêu đề biểu đồ
    plt.legend()  # Hiển thị chú thích
    plt.grid(True)  # Thêm lưới cho biểu đồ
    plt.tight_layout()  # Tối ưu bố cục
    plt.savefig('results/loss_plot.png', dpi=300)  # Lưu biểu đồ vào thư mục results
    plt.close()  # Đóng khung biểu đồ để tiết kiệm bộ nhớ

# Hàm để đánh giá mô hình trên tập kiểm tra
def evaluate_model(model, validation_dataset):
    # Khởi tạo danh sách để lưu nhãn thực tế và nhãn dự đoán
    y_true = []  # Nhãn thực tế
    y_pred = []  # Nhãn dự đoán

    # Dự đoán từng lô dữ liệu trong tập kiểm tra
    for images, labels in validation_dataset:
        preds = model.predict(images, verbose=0)  # Dự đoán với mô hình, không hiển thị tiến trình
        y_true.extend(np.argmax(labels.numpy(), axis=1))  # Lấy nhãn thực tế (chuyển từ one-hot sang số)
        y_pred.extend(np.argmax(preds, axis=1))  # Lấy nhãn dự đoán (lớp có xác suất cao nhất)

    # Chuyển danh sách thành mảng NumPy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Lấy danh sách tên lớp từ tập dữ liệu, nếu không có thì dùng số thứ tự
    class_labels = getattr(validation_dataset, 'class_names', [str(i) for i in range(np.max(y_true) + 1)])

    # Tạo báo cáo phân loại (precision, recall, f1-score, v.v.)
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print("📊 Báo cáo phân loại:")
    print(report)

    # Lưu báo cáo phân loại vào tệp văn bản
    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Tạo ma trận nhầm lẫn (confusion matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))  # Tạo khung biểu đồ với kích thước 8x6
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)  # Vẽ heatmap với số liệu
    plt.xlabel('Nhãn dự đoán')  # Đặt nhãn trục X
    plt.ylabel('Nhãn thực tế')  # Đặt nhãn trục Y
    plt.title('Ma trận nhầm lẫn')  # Đặt tiêu đề biểu đồ
    plt.tight_layout()  # Tối ưu bố cục
    plt.savefig("results/confusion_matrix.png", dpi=300)  # Lưu ma trận nhầm lẫn vào thư mục results
    plt.close()  # Đóng khung biểu đồ để tiết kiệm bộ nhớ

    # Phân tích lớp bị nhầm lẫn nhiều nhất
    if cm.shape[0] > 1:  # Kiểm tra nếu có hơn một lớp
        error_matrix = cm.copy()  # Sao chép ma trận nhầm lẫn
        np.fill_diagonal(error_matrix, 0)  # Xóa các giá trị trên đường chéo (dự đoán đúng)
        max_confused_class = np.argmax(np.sum(error_matrix, axis=1))  # Lớp bị nhầm nhiều nhất
        confused_with = np.argmax(error_matrix[max_confused_class])  # Lớp bị nhầm với
        # In thông báo về lớp bị nhầm lẫn nhiều nhất
        print(f"⚠️ Lớp '{class_labels[max_confused_class]}' thường bị nhầm với '{class_labels[confused_with]}' "
              f"({error_matrix[max_confused_class][confused_with]} lần)")