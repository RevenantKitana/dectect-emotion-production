import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Tải MobileNetV2 với weights từ ImageNet và bỏ phần top classifier
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# In cấu trúc các layers của MobileNetV2
base_model.summary()
