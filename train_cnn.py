import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from utils.visualize import plot_training_history, evaluate_model

# ==== CẤU HÌNH ====
tf.config.set_visible_devices([], 'GPU')  # chỉ CPU
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(4)

# ==== THAM SỐ ====
IMG_SIZE = 224
BATCH_SIZE = 16      # giảm để tiết kiệm RAM
NUM_CLASSES = 7
DROPOUT = 0.1
AUTOTUNE = tf.data.AUTOTUNE
LR1, LR2 = 2e-3, 3e-5
EPOCHS1, EPOCHS2 = 15, 45

# ==== AUGMENTATION & NORMALIZATION ====
data_aug = tf.keras.Sequential([
    layers.RandomZoom(0.1),  # augmentation nhẹ
])
normalize = layers.Rescaling(1./255)

# ==== LOAD DATASET ====
def prepare_ds(path, training=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=training
    )
    if training:
        ds = ds.shuffle(64)  # giảm shuffle buffer
        ds = ds.map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(1)  # prefetch 1 batch, tiết kiệm RAM

train_ds = prepare_ds("dataset/train", training=True)
val_ds = prepare_ds("dataset/test", training=False)

# ==== CNN TỰ BUILD ====
def build_cnn(input_shape=(224, 224, 3), num_classes=7, dropout=DROPOUT):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(dropout)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(dropout)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(dropout)(x)

    # Block 4
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)  # giảm dense từ 512 → 256
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_cnn()

# ==== GIAI ĐOẠN 1 ====
model.compile(optimizer=AdamW(learning_rate=LR1, weight_decay=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1)

# ==== GIAI ĐOẠN 2 (FINE-TUNING) ====
# Fine-tune toàn bộ mạng CNN tự build
model.compile(optimizer=AdamW(learning_rate=LR2, weight_decay=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    "models/best_model_cnn_optimized.keras",
    monitor='val_loss',
    save_best_only=True
)

history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS2, callbacks=[checkpoint_callback])

# ==== LƯU & ĐÁNH GIÁ ====
model.save("models/final_model_cnn_optimized.keras")
plot_training_history(history)
evaluate_model(model, val_ds)
