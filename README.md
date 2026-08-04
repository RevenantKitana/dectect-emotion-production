# Emotion Recognition using MobileNetV2

## 📋 Mô tả dự án

Hệ thống nhận diện cảm xúc khuôn mặt thời gian thực sử dụng Deep Learning với kiến trúc MobileNetV2. Dự án được phát triển nhằm 

Hệ thống phân loại 7 loại cảm xúc cơ bản theo mô hình Ekman:
1. **Angry** (Tức giận)
2. **Disgust** (Ghê tởm)
3. **Fear** (Sợ hãi)
4. **Happy** (Vui vẻ)
5. **Neutral** (Trung lập)
6. **Sad** (Buồn bã)
7. **Surprise** (Ngạc nhiên)

## 🛠️ Công nghệ & Kiến trúc

### Deep Learning Framework
- **TensorFlow/Keras**: Framework chính cho việc xây dựng và huấn luyện mô hình
- **MobileNetV2**: Kiến trúc CNN được tối ưu hóa cho inference nhanh
  - Sử dụng kỹ thuật Transfer Learning từ ImageNet weights
  - Fine-tuning các lớp cuối để phù hợp với bài toán classification 7 classes
  - Áp dụng Data Augmentation để tăng cường dữ liệu huấn luyện

### Computer Vision
- **OpenCV**: Xử lý ảnh và video real-time
  - Haar Cascade Classifier cho face detection
  - Các kỹ thuật tiền xử lý: CLAHE, Edge Detection, Median Filter
- **Pillow (PIL)**: Xử lý và manipulate hình ảnh

### GUI Development
- **Tkinter/CustomTkinter**: Giao diện desktop application
  - Theme tùy chỉnh với modern UI/UX
  - Real-time visualization
  - Upload và preview images

### Data Processing & Visualization
- **NumPy & Pandas**: Xử lý dữ liệu và matrix operations
- **Scikit-learn**: Metrics evaluation và preprocessing
- **Matplotlib & Seaborn**: Visualization cho báo cáo và phân tích

## ✨ Tính năng chính

### 1. Real-time Emotion Detection
- Nhận diện cảm xúc từ webcam với độ trễ thấp (<50ms)
- Hiển thị confidence score cho từng emotion
- Hỗ trợ multi-face detection trong một frame

### 2. GUI Application
- Upload và phân tích ảnh tĩnh
- Hiển thị kết quả với confidence bar chart
- Lưu kết quả phân tích dưới dạng file

### 3. Advanced Preprocessing
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Cải thiện độ tương phản
- **Edge Detection**: Phát hiện đường viền khuôn mặt
- **Median Filter**: Giảm noise trong ảnh
- Auto face alignment và normalization

### 4. Model Training & Evaluation
- Custom training pipeline với callbacks
- Early stopping và model checkpointing
- Comprehensive evaluation với:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC curves và AUC scores
  - Training history visualization

### 5. Face Storage System
- Tự động lưu detected faces theo emotion categories
- Organized folder structure cho data collection
- Support cho việc tạo custom dataset

## 📊 Kết quả đạt được

### Performance Metrics
- **Test Accuracy**: ~70-75%
- **Inference Speed**: >20 FPS (real-time)
- **Model Size**: ~160MB (optimized for deployment)
- **Latency**: <50ms per frame

### Class-wise Performance
| Emotion   | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Happy     | ~85%      | ~82%   | ~83%     |
| Surprise  | ~75%      | ~78%   | ~76%     |
| Neutral   | ~70%      | ~72%   | ~71%     |
| Sad       | ~65%      | ~68%   | ~66%     |
| Angry     | ~68%      | ~65%   | ~66%     |
| Fear      | ~60%      | ~58%   | ~59%     |
| Disgust   | ~55%      | ~52%   | ~53%     |

*Note: Happy và Surprise có accuracy cao nhất do đặc trưng facial features rõ ràng*

## 📁 Cấu trúc dự án

```
emotion_recognition_usemobilenetv2/
├── dataset/                    # Dataset gốc
│   ├── train/                 # Training data
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── test/                  # Test data (cùng cấu trúc)
│
├── models/                     # Trained models
│   ├── best_model.keras       # Best checkpoint
│   ├── final_model.keras      # Final trained model
│   └── emotion_model.keras    # Current model
│
├── gui/                        # GUI application
│   ├── app.py                 # Main GUI application
│   └── ui_design.py           # UI components
│
├── real_time_detection/        # Real-time detection modules
│   ├── detect_emotion.py      # Standard detection
│   ├── detect_emotion_lite.py # Lightweight version
│   └── detect_emotion_SJ.py   # Advanced detection
│
├── utils/                      # Utility functions
│   ├── preprocess.py          # Preprocessing functions
│   └── visualize.py           # Visualization tools
│
├── results/                    # Training results & reports
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── accuracy_plot.png
│   └── loss_plot.png
│
├── saved_faces/                # Detected faces storage
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happiness/
│   ├── Neutral/
│   ├── Sadness/
│   └── Surprise/
│
├── train.py                    # Main training script
├── train_cnn.py               # CNN training with optimization
├── mobilenetV2_details.py     # Model architecture details
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```


## 🚀 Setup & Installation

### Yêu cầu hệ thống
- **Python**: 3.10 (khuyến nghị)
- **RAM**: Tối thiểu 8GB
- **GPU**: Optional (CUDA-compatible cho training nhanh hơn)
- **Webcam**: Cần thiết cho real-time detection

### Bước 1: Clone Repository
```bash
git clone https://github.com/RevenantKitana/emotion_recognition_usemobilenetv2.git
cd emotion_recognition_usemobilenetv2
```

### Bước 2: Tạo Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Download Pre-trained Model (Optional)
Nếu không muốn train từ đầu, có thể download pre-trained model từ releases.

## 💻 Hướng dẫn sử dụng

### 1. Training Model

#### Training cơ bản
```bash
python train.py
```

#### Training với CNN optimization
```bash
python train_cnn.py
```

**Training Options:**
- Số epochs: Mặc định 50 (có thể điều chỉnh trong code)
- Batch size: 32
- Learning rate: 0.0001 với Adam optimizer
- Data augmentation: Rotation, flip, zoom, brightness

**Output:**
- Model được lưu tại `models/`
- Training history plots tại `results/`
- Confusion matrix và classification report

### 2. Real-time Emotion Detection

#### Standard version
```bash
python real_time_detection/detect_emotion.py
```

#### Lightweight version (faster)
```bash
python real_time_detection/detect_emotion_lite.py
```

#### Advanced version với preprocessing
```bash
python real_time_detection/detect_emotion_SJ.py
```

**Controls:**
- `q`: Quit application
- `s`: Save detected face
- `c`: Clear saved faces
- `p`: Pause/Resume detection

### 3. GUI Application

```bash
python gui/app.py
```

**Features:**
- Upload ảnh từ máy tính
- Real-time detection từ webcam
- Hiển thị confidence scores
- Export results
- Theme switching (light/dark)

### 4. Custom Training với Dataset riêng

```python
from utils.preprocess import load_dataset
from train import create_model

# Load custom dataset
train_data = load_dataset('path/to/train')
test_data = load_dataset('path/to/test')

# Train model
model = create_model()
history = model.fit(train_data, validation_data=test_data)
```

## 📈 Kết quả Training

### Training History
- Training được thực hiện trên dataset ~28,000 images
- Validation split: 20%
- Training time: ~2-3 hours (GPU) / ~8-10 hours (CPU)

### Loss & Accuracy Curves
Kết quả được lưu tại `results/`:
- `accuracy_plot.png`: Training/validation accuracy
- `loss_plot.png`: Training/validation loss
- `confusion_matrix.png`: Confusion matrix
- `classification_report.txt`: Detailed metrics

## 🔧 Troubleshooting

### Common Issues

**1. OpenCV Camera Error**
```bash
# Windows
# Cài đặt lại OpenCV
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

**2. TensorFlow GPU Issues**
```bash
# Kiểm tra CUDA compatibility
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**3. Memory Error during Training**
- Giảm batch size trong code
- Close các ứng dụng khác
- Sử dụng data generator thay vì load toàn bộ vào RAM

**4. Low Accuracy**
- Tăng số epochs
- Điều chỉnh learning rate
- Thêm data augmentation
- Fine-tune thêm các lớp của MobileNetV2

## 📚 Dataset Information

### Dataset Structure
```
dataset/
├── train/          # ~22,000+ images
└── test/           # ~6,000+ images
    ├── angry/      # ~3,000 images
    ├── disgust/    # ~500 images
    ├── fear/       # ~2,000 images
    ├── happy/      # ~8,000 images (largest)
    ├── neutral/    # ~5,000 images
    ├── sad/        # ~4,000 images
    └── surprised/  # ~3,000 images
```

### Data Preprocessing
1. **Face Detection**: Haar Cascade
2. **Resize**: 224x224 pixels (MobileNetV2 input size)
3. **Normalization**: Pixel values scaled to [0,1]
4. **Augmentation**: 
   - Random rotation (±15°)
   - Horizontal flip
   - Zoom (±10%)
   - Brightness adjustment

### Dataset Sources
- FER2013
- CK+ (Extended Cohn-Kanade)
- JAFFE (Japanese Female Facial Expression)
- Custom collected data

## 🎯 Use Cases

### 1. Healthcare
- Monitoring bệnh nhân tâm thần
- Phát hiện depression và anxiety
- Đánh giá hiệu quả điều trị

### 2. Education
- Phân tích engagement của học sinh
- Adaptive learning systems
- Online education monitoring

### 3. Business
- Customer satisfaction analysis
- Employee wellness monitoring
- Market research và feedback analysis

### 4. Security
- Suspicious behavior detection
- Access control systems
- Interview analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Fork repository
# Clone your fork
git clone https://github.com/YOUR_USERNAME/emotion_recognition_usemobilenetv2.git

# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Open Pull Request
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors & Contact

- **GitHub**: [@RevenantKitana](https://github.com/RevenantKitana)
- **Email**: nqk6829@gmail.com

## 🙏 Acknowledgments

- MobileNetV2 paper: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- FER2013 dataset creators
- TensorFlow và OpenCV communities
- All contributors who helped improve this project

## 📝 Citation

If you use this project in your research, please cite:
```bibtex
@misc{emotion_recognition_mobilenetv2,
  author = {RevenantKitana},
  title = {Emotion Recognition using MobileNetV2},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/RevenantKitana/emotion_recognition_usemobilenetv2}
}
```

---
⭐ Star this repository if you find it helpful!

