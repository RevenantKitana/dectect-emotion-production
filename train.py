# Nhập các thư viện cần thiết
import tensorflow as tf  # Thư viện TensorFlow để xây dựng và huấn luyện mô hình
from tensorflow.keras import layers, models  # Các module để tạo layer và mô hình
from tensorflow.keras.applications import MobileNetV2  # Mô hình MobileNetV2 được huấn luyện trước
from tensorflow.keras.callbacks import ModelCheckpoint  # Callback để lưu mô hình tốt nhất
from tensorflow.keras.optimizers import AdamW  # Bộ tối ưu AdamW với weight decay
from utils.visualize import plot_training_history, evaluate_model  # Hàm để vẽ biểu đồ và đánh giá mô hình

# ==== CẤU HÌNH ====
# Vô hiệu hóa GPU (chạy trên CPU)
tf.config.set_visible_devices([], 'GPU')
# Thiết lập số luồng xử lý song song trong TensorFlow để tối ưu hiệu suất
tf.config.threading.set_intra_op_parallelism_threads(12)  # Số luồng trong một thao tác
tf.config.threading.set_inter_op_parallelism_threads(4)   # Số luồng giữa các thao tác

# ==== THAM SỐ ====
# Định nghĩa các tham số chính cho mô hình và dữ liệu
IMG_SIZE = 224        # Kích thước ảnh đầu vào (224x224)
BATCH_SIZE = 64       # Kích thước lô dữ liệu mỗi lần huấn luyện
NUM_CLASSES = 7       # Số lớp cảm xúc (7 cảm xúc)
DROPOUT = 0.1         # Tỷ lệ dropout để tránh overfitting
AUTOTUNE = tf.data.AUTOTUNE  # Tự động tối ưu hiệu suất xử lý dữ liệu
LR1, LR2 = 2e-3, 3e-5  # Tỷ lệ học (learning rate) cho giai đoạn 1 và 2
EPOCHS1, EPOCHS2 = 15, 45  # Số epoch huấn luyện cho giai đoạn 1 và 2

# ==== AUGMENTATION & NORMALIZATION ====
# Tạo pipeline tăng cường dữ liệu (data augmentation) cho tập huấn luyện
data_aug = tf.keras.Sequential([
    layers.RandomZoom(0.1),  # Phóng to ngẫu nhiên ảnh với tỷ lệ tối đa 10%
])
# Chuẩn hóa giá trị pixel về khoảng [0, 1]
normalize = layers.Rescaling(1./255)

# ==== LOAD DATASET ====
# Hàm để chuẩn bị tập dữ liệu từ thư mục
def prepare_ds(path, training=True):
    """Tải và tiền xử lý tập dữ liệu ảnh"""
    # Tải dữ liệu ảnh từ thư mục với cấu hình cụ thể
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, 
        image_size=(IMG_SIZE, IMG_SIZE),  # Thay đổi kích thước ảnh
        batch_size=BATCH_SIZE,            # Kích thước lô
        label_mode='categorical',         # Nhãn dạng one-hot encoding
        shuffle=training                  # Xáo trộn dữ liệu nếu là tập huấn luyện
    )
    # Áp dụng tăng cường dữ liệu nếu là tập huấn luyện
    if training:
        ds = ds.map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # Chuẩn hóa dữ liệu và tối ưu hiệu suất
    return ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE).prefetch(1)

# Tải và chuẩn bị tập dữ liệu huấn luyện (train) và kiểm tra (test)fffffg
train_ds = prepare_ds("dataset/train", training=True).shuffle(100)  # Xáo trộn với buffer 100
val_ds = prepare_ds("dataset/test", training=False)  # Không tăng cường dữ liệu cho tập kiểm tra

# ==== MODEL ====
# Tải mô hình MobileNetV2 được huấn luyện trước trên ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# Đóng băng các tầng của MobileNetV2 để không huấn luyện lại trong giai đoạn 1
base_model.trainable = False

# Xây dựng mô hình tùy chỉnh
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # Đầu vào ảnh RGB
x = base_model(inputs, training=False)                # Trích xuất đặc trưng bằng MobileNetV2
x = layers.GlobalAveragePooling2D()(x)               # Gộp đặc trưng thành vector
# Thêm các tầng fully connected
for units in [256, 128]:
    x = layers.Dense(units, activation='relu')(x)    # Tầng dense với ReLU
    x = layers.Dropout(DROPOUT)(x)                   # Dropout để giảm overfitting
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)  # Đầu ra với softmax cho phân loại
# Tạo mô hình hoàn chỉnh
model = models.Model(inputs, outputs)

# ==== GIAI ĐOẠN 1 (HUẤN LUYỆN PHẦN MỀM) ====
# Biên dịch mô hình với tối ưu AdamW và hàm mất mát categorical crossentropy
model.compile(optimizer=AdamW(learning_rate=LR1, weight_decay=1e-5),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Huấn luyện mô hình giai đoạn 1 (chỉ huấn luyện các tầng mới thêm)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1)

# ==== GIAI ĐOẠN 2 (FINE-TUNING) ====
# Mở khóa huấn luyện cho toàn bộ MobileNetV2
base_model.trainable = True
# Đóng băng các tầng đầu tiên, chỉ huấn luyện 30 tầng cuối
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Biên dịch lại mô hình với tỷ lệ học nhỏ hơn
model.compile(optimizer=AdamW(learning_rate=LR2, weight_decay=1e-5),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callback để lưu mô hình tốt nhất dựa trên val_loss
checkpoint_callback = ModelCheckpoint(
    "models/best_model.keras",  # Đường dẫn lưu mô hình
    monitor='val_loss',         # Theo dõi mất mát trên tập kiểm tra
    save_best_only=True         # Chỉ lưu mô hình có val_loss thấp nhất
)

# Huấn luyện mô hình giai đoạn 2 (fine-tuning) với callback
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2,
                    callbacks=[checkpoint_callback])

# ==== LƯU & ĐÁNH GIÁ ====
# Lưu mô hình hoàn chỉnh sau khi huấn luyện
model.save("models/final_model.keras")

# Vẽ biểu đồ lịch sử huấn luyện (loss, accuracy)
plot_training_history(history)
# Đánh giá mô hình trên tập kiểm tra
evaluate_model(model, val_ds)