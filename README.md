## 🎙️ Project: Audio Deepfake Detection (Phát hiện giọng nói giả mạo)

Mục tiêu

- Project này xây dựng hệ thống phát hiện audio deepfake — phân loại nhị phân (Real / Fake) cho giọng nói, sử dụng dataset ASVspoof 2019 LA (Logical Access).

---

Dataset

- ASVspoof 2019 LA: bộ dữ liệu chuẩn quốc tế cho bài toán phát hiện giọng nói giả mạo, gồm các file audio .flac chia thành 3 tập: train, dev, eval.

---

Pipeline tổng thể

- Audio .flac → Protocol txt → Metadata CSV → Log-Mel Spectrogram (.npy) → tf.data.Dataset → Model (CNN / MobileNetV3) → Train → Evaluate → Export (TFLite/ONNX)

---

Kiến trúc Models (2 model)

- CNN Baseline: cnn_baseline.py - Light CNN tự xây (4 conv blocks, Global Average Pooling, Dropout)
- MobileNetV3-Small: mobilenetv3_small.py - MobileNetV3-Small (transfer learning / fine-tuning)

---

Cấu trúc source code

- src/data/ - Tạo metadata CSV, load dataset - make_metadata.py, dataset_loader.py
- src/features/ - Trích xuất Log-Mel Spectrogram - logmel.py, mobilenetv3_precompute.py
- src/models/ - Định nghĩa kiến trúc model - cnn_baseline.py, mobilenetv3_small.py
- src/training/ - Huấn luyện model - train_cnn.py, train_mobilenetv3.py, callbacks.py
- src/evaluation/ - Đánh giá model (Accuracy, F1, ROC-AUC, EER, Confusion Matrix...) - evaluate.py, metrics.py
- src/deployment/ - Export model ra TFLite / ONNX để deploy - export_tflite.py, export_onnx.py

---

Các chỉ số đánh giá
Accuracy, Precision, Recall, F1-score, ROC-AUC, EER (Equal Error Rate), Confusion Matrix, và tìm best threshold theo F1 / Youden's J.

---

Tech Stack

- TensorFlow / Keras (training & model)
- librosa (xử lý audio, trích xuất Log-Mel)
- Python 3.10
- Export sang TFLite (quantization) và ONNX để triển khai
