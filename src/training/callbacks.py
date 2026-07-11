"""
callbacks.py

Nhiệm vụ:
- Tạo các Keras callbacks dùng trong quá trình train CNN baseline.
- Các callback được đọc từ configs/cnn_baseline.yaml.

Các callback chính:
1. ModelCheckpoint:
   Lưu model tốt nhất theo val_roc_auc.

2. EarlyStopping:
   Dừng train sớm nếu validation metric không cải thiện.

3. ReduceLROnPlateau:
   Giảm learning rate nếu validation loss không giảm.

4. CSVLogger:
   Lưu lịch sử train ra file CSV.

Cách chạy test từ thư mục gốc project:

    py -m src.training.callbacks --config configs/cnn_baseline.yaml

Lưu ý:
- File này không train model.
- File này chỉ kiểm tra việc tạo callback và tạo thư mục output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import tensorflow as tf
import yaml


# ============================================================
# Config utilities
# ============================================================

def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config rỗng: {config_path}")

    return config


def infer_project_root(config_path: str | Path) -> Path:
    """
    Suy ra thư mục gốc project.

    Ví dụ:
        audio_deepfake/configs/cnn_baseline.yaml

    Project root:
        audio_deepfake/
    """
    config_path = Path(config_path).resolve()

    if config_path.parent.name == "configs":
        return config_path.parent.parent

    return Path.cwd().resolve()


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """Chuyển path tương đối thành absolute path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def ensure_parent_dir(file_path: str | Path) -> None:
    """Tạo thư mục cha cho file nếu chưa tồn tại."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(dir_path: str | Path) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


# ============================================================
# Callback builders
# ============================================================

def build_model_checkpoint_callback(
    callback_config: Dict[str, Any],
    project_root: Path,
) -> Optional[tf.keras.callbacks.ModelCheckpoint]:
    """
    Tạo ModelCheckpoint callback.

    Chức năng:
        Lưu model tốt nhất trong quá trình train.

    Với config hiện tại:
        monitor = val_roc_auc
        mode = max

    Nghĩa là:
        Nếu val_roc_auc tốt hơn epoch trước thì lưu model.
    """
    if not callback_config.get("enable", True):
        return None

    filepath = resolve_path(project_root, callback_config["filepath"])
    ensure_parent_dir(filepath)

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(filepath),
        monitor=str(callback_config.get("monitor", "val_loss")),
        mode=str(callback_config.get("mode", "min")),
        save_best_only=bool(callback_config.get("save_best_only", True)),
        save_weights_only=bool(callback_config.get("save_weights_only", False)),
        verbose=int(callback_config.get("verbose", 1)),
    )

    return callback


def build_early_stopping_callback(
    callback_config: Dict[str, Any],
) -> Optional[tf.keras.callbacks.EarlyStopping]:
    """
    Tạo EarlyStopping callback.

    Chức năng:
        Dừng train nếu validation metric không cải thiện sau nhiều epoch.

    Ví dụ:
        patience = 5

    Nghĩa là:
        Nếu 5 epoch liên tiếp val_roc_auc không tăng,
        quá trình train sẽ dừng sớm.
    """
    if not callback_config.get("enable", True):
        return None

    callback = tf.keras.callbacks.EarlyStopping(
        monitor=str(callback_config.get("monitor", "val_loss")),
        mode=str(callback_config.get("mode", "min")),
        patience=int(callback_config.get("patience", 5)),
        restore_best_weights=bool(callback_config.get("restore_best_weights", True)),
        verbose=int(callback_config.get("verbose", 1)),
    )

    return callback


def build_reduce_lr_on_plateau_callback(
    callback_config: Dict[str, Any],
) -> Optional[tf.keras.callbacks.ReduceLROnPlateau]:
    """
    Tạo ReduceLROnPlateau callback.

    Chức năng:
        Giảm learning rate khi validation loss không còn cải thiện.

    Ví dụ:
        learning_rate = 0.001
        factor = 0.5

    Khi val_loss không giảm:
        0.001 → 0.0005 → 0.00025 → ...
    """
    if not callback_config.get("enable", True):
        return None

    callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=str(callback_config.get("monitor", "val_loss")),
        mode=str(callback_config.get("mode", "min")),
        factor=float(callback_config.get("factor", 0.5)),
        patience=int(callback_config.get("patience", 2)),
        min_lr=float(callback_config.get("min_lr", 1e-6)),
        verbose=int(callback_config.get("verbose", 1)),
    )

    return callback


def build_csv_logger_callback(
    callback_config: Dict[str, Any],
    project_root: Path,
) -> Optional[tf.keras.callbacks.CSVLogger]:
    """
    Tạo CSVLogger callback.

    Chức năng:
        Ghi lịch sử huấn luyện ra file CSV.

    File CSV thường có các cột:
        epoch, accuracy, loss, roc_auc, val_accuracy, val_loss, val_roc_auc

    File này rất hữu ích để:
        - Vẽ training curve.
        - Đưa bảng kết quả vào báo cáo.
        - Kiểm tra model có overfitting không.
    """
    if not callback_config.get("enable", True):
        return None

    filename = resolve_path(project_root, callback_config["filename"])
    ensure_parent_dir(filename)

    callback = tf.keras.callbacks.CSVLogger(
        filename=str(filename),
        separator=",",
        append=bool(callback_config.get("append", False)),
    )

    return callback


class EpochTimeLogger(tf.keras.callbacks.Callback):
    """
    Callback tự định nghĩa để in thời gian train từng epoch.

    Callback này không bắt buộc, nhưng hữu ích khi train trên laptop.
    """

    def on_epoch_begin(self, epoch, logs=None):
        import time
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        import time
        elapsed = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1} time: {elapsed:.2f} seconds")


def create_callbacks_from_config(
    config: Dict[str, Any],
    project_root: Path,
    include_time_logger: bool = True,
) -> List[tf.keras.callbacks.Callback]:
    """
    Tạo danh sách callbacks từ config.

    Hàm này sẽ được train_cnn.py gọi.

    Args:
        config: dictionary đọc từ cnn_baseline.yaml.
        project_root: thư mục gốc project.
        include_time_logger: có thêm callback in thời gian từng epoch không.

    Returns:
        List[tf.keras.callbacks.Callback]
    """
    callbacks_config = config.get("callbacks", {})

    callbacks: List[tf.keras.callbacks.Callback] = []

    # 1. ModelCheckpoint
    model_checkpoint_config = callbacks_config.get("model_checkpoint", {})
    model_checkpoint = build_model_checkpoint_callback(
        callback_config=model_checkpoint_config,
        project_root=project_root,
    )

    if model_checkpoint is not None:
        callbacks.append(model_checkpoint)

    # 2. EarlyStopping
    early_stopping_config = callbacks_config.get("early_stopping", {})
    early_stopping = build_early_stopping_callback(
        callback_config=early_stopping_config,
    )

    if early_stopping is not None:
        callbacks.append(early_stopping)

    # 3. ReduceLROnPlateau
    reduce_lr_config = callbacks_config.get("reduce_lr_on_plateau", {})
    reduce_lr = build_reduce_lr_on_plateau_callback(
        callback_config=reduce_lr_config,
    )

    if reduce_lr is not None:
        callbacks.append(reduce_lr)

    # 4. CSVLogger
    csv_logger_config = callbacks_config.get("csv_logger", {})
    csv_logger = build_csv_logger_callback(
        callback_config=csv_logger_config,
        project_root=project_root,
    )

    if csv_logger is not None:
        callbacks.append(csv_logger)

    # 5. Optional time logger
    if include_time_logger:
        callbacks.append(EpochTimeLogger())

    return callbacks


# ============================================================
# Output directory utilities
# ============================================================

def prepare_output_directories(
    config: Dict[str, Any],
    project_root: Path,
) -> None:
    """
    Tạo trước các thư mục output cần thiết.

    Các thư mục này gồm:
        outputs/checkpoints/cnn_baseline
        outputs/saved_models/cnn_baseline
        outputs/tflite/cnn_baseline
        outputs/logs/cnn_baseline
        outputs/results/cnn_baseline
    """
    paths_config = config.get("paths", {})

    output_dir_keys = [
        "output_dir",
        "checkpoint_dir",
        "saved_model_dir",
        "tflite_dir",
        "log_dir",
        "result_dir",
    ]

    for key in output_dir_keys:
        if key not in paths_config:
            continue

        dir_path = resolve_path(project_root, paths_config[key])
        ensure_dir(dir_path)


def print_callbacks_summary(callbacks: List[tf.keras.callbacks.Callback]) -> None:
    """In danh sách callbacks đã tạo."""
    print("\n" + "=" * 70)
    print("CALLBACKS SUMMARY")
    print("=" * 70)

    if not callbacks:
        print("Không có callback nào được tạo.")
        return

    for idx, callback in enumerate(callbacks, start=1):
        print(f"{idx}. {callback.__class__.__name__}")

        if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
            print(f"   monitor        : {callback.monitor}")
            print(f"   filepath       : {callback.filepath}")
            print(f"   save_best_only : {callback.save_best_only}")

        elif isinstance(callback, tf.keras.callbacks.EarlyStopping):
            print(f"   monitor        : {callback.monitor}")
            print(f"   patience       : {callback.patience}")
            print(f"   restore_best   : {callback.restore_best_weights}")

        elif isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau):
            print(f"   monitor        : {callback.monitor}")
            print(f"   factor         : {callback.factor}")
            print(f"   patience       : {callback.patience}")
            print(f"   min_lr         : {callback.min_lr}")

        elif isinstance(callback, tf.keras.callbacks.CSVLogger):
            print(f"   filename       : {callback.filename}")


# ============================================================
# CLI test
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and inspect Keras callbacks from config."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_baseline.yaml",
        help="Đường dẫn tới config YAML.",
    )

    parser.add_argument(
        "--no-time-logger",
        action="store_true",
        help="Không thêm EpochTimeLogger.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    project_root = infer_project_root(config_path)

    print("Project root:", project_root)
    print("Config file :", config_path)

    prepare_output_directories(
        config=config,
        project_root=project_root,
    )

    callbacks = create_callbacks_from_config(
        config=config,
        project_root=project_root,
        include_time_logger=not args.no_time_logger,
    )

    print_callbacks_summary(callbacks)

    print("\nCallbacks OK.")


if __name__ == "__main__":
    main()