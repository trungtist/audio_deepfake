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

# Hướng Dẫn Quy Trình (Pipeline) Huấn Luyện & Đánh Giá Mô Hình

Dự án: **Audio Deepfake Detection**

Tài liệu này chứa hướng dẫn chi tiết các bước để chạy toàn bộ dự án từ tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện mô hình đến xuất định dạng phục vụ triển khai (Deployment).

---

## 🛠️ Chuẩn Bị Môi Trường

Trước khi thực hiện bất kỳ lệnh nào bên dưới, hãy đảm bảo rằng bạn đã kích hoạt môi trường ảo (Virtual Environment):

```powershell
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1

# Trên Windows CMD:
.venv\Scripts\activate.bat
```

---

## 🔄 PIPELINE A: Mô hình CNN V2 (CNN Baseline)

_(Dòng mô hình Custom CNN truyền thống)_

Sử dụng cấu hình:

- `configs/cnn_v2.yaml` (Cho phiên bản v2 nâng cấp - Mặc định)
- `configs/cnn_baseline.yaml` (Cho mô hình Baseline cũ)

> [!NOTE]
> Tất cả các câu lệnh dưới đây sử dụng cấu hình mặc định `configs/cnn_v2.yaml`. Bạn có thể đổi sang `configs/cnn_baseline.yaml` nếu muốn kiểm tra mô hình Baseline cũ.

### Bước 1: Tạo Dữ Liệu Metadata (make_metadata.py)

Đọc protocol của bộ dataset ASVspoof2019 LA và tạo danh sách metadata dạng `.csv`.

- **Chạy đầy đủ (all splits):**
  ```powershell
  python -m src.data.make_metadata --config configs/cnn_v2.yaml --split all
  ```
- **Chạy debug nhanh (số mẫu nhỏ):**
  ```powershell
  python -m src.data.make_metadata --config configs/cnn_v2.yaml --split all --debug
  ```
- **Đầu ra (Outputs):**
  Các file CSV được tạo trong thư mục `data/metadata/` bao gồm: `train.csv`, `dev.csv`, `eval.csv`.

### Bước 2: Trích Xuất Đặc Trưng Log-Mel Spectrogram (logmel.py)

Trích xuất đặc trưng âm thanh từ file `.flac` sang định dạng `.npy`.

- **Chạy đầy đủ:**
  ```powershell
  python -m src.features.logmel --config configs/cnn_v2.yaml --split all
  ```
- **Chạy debug nhanh (100 mẫu):**
  ```powershell
  python -m src.features.logmel --config configs/cnn_v2.yaml --split train --max-samples 100
  ```
- **Kiểm tra nhanh kích thước file `.npy` sau khi xuất:**
  ```powershell
  python -c "import numpy as np, glob; f=glob.glob('data/processed/logmel_64_4s/train/*.npy')[0]; x=np.load(f); print(f); print(x.shape); print(x.dtype)"
  ```
- **Đầu ra (Outputs):**
  Các tệp tin đặc trưng đặc thù dạng `.npy` tại thư mục `data/processed/logmel_64_4s/{train,dev,eval}/`.

### Bước 3: Kiểm Tra Bộ Đọc Dữ Liệu (dataset_loader.py)

Kiểm tra xem dữ liệu `.npy` có được nạp và chuyển sang định dạng `tf.data.Dataset` đúng cách không.

- **Kiểm tra loader trên tập dev:**
  ```powershell
  python -m src.data.dataset_loader --config configs/cnn_v2.yaml --split dev --max-samples 32
  ```
- **Xác thực toàn bộ các file `.npy` đã tạo:**
  ```powershell
  python -m src.data.dataset_loader --config configs/cnn_v2.yaml --split train --validate-files
  ```

### Bước 4: Kiểm Tra Kiến Trúc Mô Hình (cnn_baseline.py)

Xây dựng mô hình ảo, in tóm tắt tham số (model summary) và kiểm tra quá trình biên dịch (compile).

- **Kiểm tra và In Summary:**
  ```powershell
  python -m src.models.cnn_baseline --config configs/cnn_v2.yaml --summary --compile
  ```

### Bước 5: Huấn Luyện Mô Hình (train_cnn.py)

Tiến hành train mô hình CNN trên tập dữ liệu đã chuẩn bị.

- **Chạy huấn luyện đầy đủ:**
  ```powershell
  python -m src.training.train_cnn --config configs/cnn_v2.yaml
  ```
- **Chạy train debug nhanh:**
  ```powershell
  python -m src.training.train_cnn --config configs/cnn_v2.yaml --debug
  ```
- **Đầu ra (Outputs):**
  - Mô hình tốt nhất lưu tại: `outputs/checkpoints/cnn_v2/best_cnn_v2.keras`
  - Mô hình hoàn tất epoch cuối: `outputs/saved_models/cnn_v2/cnn_v2_final.keras`
  - Nhật ký huấn luyện: `outputs/logs/cnn_v2/training_log.csv`

### Bước 6: Đánh Giá Mô Hình Keras (evaluate.py)

1. **Chạy đánh giá trên tập `dev` để tìm ngưỡng phân loại (threshold) tối ưu:**
   ```powershell
   python -m src.evaluation.evaluate --config configs/cnn_v2.yaml --split dev
   ```
2. **Dùng ngưỡng tối ưu vừa tìm được ở tập dev (ví dụ: `0.956`) chạy đánh giá cuối trên tập `eval`:**
   ```powershell
   python -m src.evaluation.evaluate --config configs/cnn_v2.yaml --split eval --threshold 0.956 --output-dir outputs/results/cnn_v2/evaluate_eval_thr_0956
   ```

### Bước 7: Xuất Mô Hình Sang Định Dạng TFLite (export_tflite.py)

Chuyển đổi file `.keras` sang định dạng gọn nhẹ dùng cho deploy di động hoặc phần cứng nhúng.

- **Xuất mô hình:**
  ```powershell
  python -m src.deployment.export_tflite --config configs/cnn_v2.yaml --formats fp32 fp16 dynamic
  ```
- **Đầu ra (Outputs):**
  Các file mô hình TFLite tại thư mục: `outputs/tflite/cnn_v2/`.

### Bước 8: Đánh Giá Mô Hình TFLite (evaluate_tflite.py)

Kiểm tra độ chính xác và tốc độ của mô hình TFLite đã xuất trên tập `eval`.

- **Đánh giá phiên bản FP16:**
  ```powershell
  python -m src.deployment.evaluate_tflite --config configs/cnn_v2.yaml --split eval --format fp16 --threshold 0.956
  ```
- **Đánh giá phiên bản FP32:**
  ```powershell
  python -m src.deployment.evaluate_tflite --config configs/cnn_v2.yaml --split eval --format fp32 --threshold 0.956
  ```

---

## 🔄 PIPELINE B: Mô hình MobileNetV3-Small

_(Dòng mô hình tối ưu sử dụng thêm bước Precompute)_

Sử dụng cấu hình: `configs/mobilenetv3_small.yaml`

### Bước 1: Tạo Dữ Liệu Metadata (make_metadata.py)

- **Chạy đầy đủ:**
  ```powershell
  python -m src.data.make_metadata --config configs/mobilenetv3_small.yaml --split all
  ```

### Bước 2: Trích Xuất Đặc Trưng Log-Mel Spectrogram (logmel.py)

- **Chạy đầy đủ:**
  ```powershell
  python -m src.features.logmel --config configs/mobilenetv3_small.yaml --split all
  ```

### Bước 3: Tiền Tính Toán Đặc Trưng Cho MobileNetV3 (mobilenetv3_precompute.py)

Đây là bước đặc thù của MobileNetV3 giúp tạo trước các đặc trưng tương thích, tối ưu tốc độ huấn luyện.

- **Chạy precompute đầy đủ:**
  ```powershell
  python -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split all
  ```
- **Chạy debug nhanh:**
  ```powershell
  python -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split train --max-samples 100
  ```
- **Chỉ kiểm tra validate các file đặc trưng đã tính toán:**
  ```powershell
  python -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split dev --validate-only
  ```

### Bước 4: Kiểm Tra Bộ Đọc Dữ Liệu (dataset_loader.py)

- **Kiểm tra bộ loader:**
  ```powershell
  python -m src.data.dataset_loader --config configs/mobilenetv3_small.yaml --split dev --max-samples 32
  ```

### Bước 5: Kiểm Tra Kiến Trúc Mô Hình (mobilenetv3_small.py)

- **Kiểm tra và In Summary:**
  ```powershell
  python -m src.models.mobilenetv3_small --config configs/mobilenetv3_small.yaml --summary --compile --forward-test
  ```

### Bước 6: Huấn Luyện Mô Hình (train_mobilenetv3.py)

- **Huấn luyện đầy đủ:**
  ```powershell
  python -m src.training.train_mobilenetv3 --config configs/mobilenetv3_small.yaml
  ```
- **Huấn luyện debug nhanh:**
  ```powershell
  python -m src.training.train_mobilenetv3 --config configs/mobilenetv3_small.yaml --debug
  ```
- **Đầu ra (Outputs):**
  - Checkpoint tốt nhất: `outputs/checkpoints/mobilenetv3_small/best_mobilenetv3_small.keras`
  - Log huấn luyện: `outputs/logs/mobilenetv3_small/training_log.csv`

### Bước 7: Đánh Giá Mô Hình Keras (evaluate.py)

- **Tìm ngưỡng tối ưu trên tập dev:**
  ```powershell
  python -m src.evaluation.evaluate --config configs/mobilenetv3_small.yaml --split dev
  ```
- **Đánh giá tập eval với ngưỡng tối ưu (ví dụ: `0.606`):**
  ```powershell
  python -m src.evaluation.evaluate --config configs/mobilenetv3_small.yaml --split eval --threshold 0.606 --output-dir outputs/results/mobilenetv3_small/evaluate_eval_thr_0606
  ```

### Bước 8: Xuất Mô Hình Sang Định Dạng TFLite (export_tflite.py)

- **Xuất mô hình:**
  ```powershell
  python -m src.deployment.export_tflite --config configs/mobilenetv3_small.yaml --formats fp32 fp16 dynamic
  ```
- **Đầu ra (Outputs):**
  Các file mô hình TFLite tại thư mục: `outputs/tflite/mobilenetv3_small/`.

### Bước 9: Đánh Giá Mô Hình TFLite (evaluate_tflite.py)

- **Đánh giá trên phiên bản FP16 với ngưỡng tối ưu:**
  ```powershell
  python -m src.deployment.evaluate_tflite --config configs/mobilenetv3_small.yaml --split eval --format fp16 --threshold 0.606
  ```
- **Đánh giá trên phiên bản FP32:**
  ```powershell
  python -m src.deployment.evaluate_tflite --config configs/mobilenetv3_small.yaml --split eval --format fp32 --threshold 0.606
  ```

---

## 🤖 Đánh Giá Hiệu Năng Trên Thiết Bị Android (Android Benchmarking)

Bạn có thể đo độ trễ (latency) và bộ nhớ tiêu thụ khi chạy mô hình `.tflite` trực tiếp trên thiết bị Android bằng công cụ benchmark trong thư mục `android_bench/`.

### Điều kiện tiên quyết:

- Điện thoại Android đã bật **Developer Options (Tùy chọn nhà phát triển)** và kích hoạt **USB Debugging**.
- Máy tính của bạn đã được cấu hình bộ công cụ **ADB (Android Debug Bridge)** và kết nối thành công với thiết bị (kiểm tra bằng lệnh `adb devices` trong terminal).

### Các bước thực hiện:

#### Bước 1: Đẩy công cụ benchmark và mô hình TFLite lên thiết bị

Đẩy tệp thực thi benchmark cùng mô hình `.tflite` (đã xuất ở các bước trước) vào thư mục tạm `/data/local/tmp/` trên Android qua cổng ADB:

```powershell
# Đẩy tệp thực thi benchmark
adb push android_bench/android_aarch64_benchmark_model /data/local/tmp/

# Đẩy mô hình TFLite (ví dụ model fp16 của CNN V2)
adb push outputs/tflite/cnn_v2/cnn_v2_fp16.tflite /data/local/tmp/
```

#### Bước 2: Cấp quyền thực thi cho tệp tin benchmark

Cấp quyền chạy chương trình cho tệp nhị phân vừa đẩy lên:

```powershell
adb shell chmod +x /data/local/tmp/android_aarch64_benchmark_model
```

#### Bước 3: Chạy lệnh đo đạc hiệu năng (Benchmark)

Bạn có thể chọn chạy trên CPU thông thường hoặc kích hoạt các bộ tăng tốc phần cứng phần cứng (GPU/NNAPI):

- **Chạy benchmark trên CPU (sử dụng 4 luồng xử lý):**

  ```powershell
  adb shell /data/local/tmp/android_aarch64_benchmark_model --graph=/data/local/tmp/cnn_v2_fp16.tflite --num_threads=4
  ```

- **Chạy tăng tốc bằng GPU Delegate (nếu GPU của thiết bị hỗ trợ):**

  ```powershell
  adb shell /data/local/tmp/android_aarch64_benchmark_model --graph=/data/local/tmp/cnn_v2_fp16.tflite --use_gpu=true
  ```

- **Chạy tăng tốc bằng NNAPI Delegate (sử dụng NPU/DSP của thiết bị):**
  ```powershell
  adb shell /data/local/tmp/android_aarch64_benchmark_model --graph=/data/local/tmp/cnn_v2_fp16.tflite --use_nnapi=true
  ```

#### Bước 4: Đọc kết quả

Kết quả benchmark hiển thị trực tiếp trên màn hình terminal. Hãy quan sát phần **`Inference timings in us`** (độ trễ tính bằng micro-giây):

- `Inference (avg)`: Thời gian suy luận trung bình cho mỗi mẫu âm thanh (chia giá trị này cho `1000` để đổi sang mili-giây - `ms`).

---

## 💡 Mẹo và Khắc Phục Sự Cố

1. **Lỗi `ModuleNotFoundError: No module named 'src'`**:
   - Luôn chạy lệnh ở thư mục gốc của dự án (`e:\MASTER\DeAn\audio_deepfake`).
   - Sử dụng cú pháp chạy dạng package module (`python -m src...` thay vì `python src/...`).
2. **Thay đổi cấu hình mô hình**:
   - Không nên thay đổi trực tiếp cấu hình trong quá trình đang chạy. Hãy tạo một file cấu hình copy mới trong thư mục `configs/` rồi truyền tên file qua tham số `--config`.
