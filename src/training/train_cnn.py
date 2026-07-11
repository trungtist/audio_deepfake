"""
train_cnn.py

Nhiệm vụ:
- Huấn luyện mô hình CNN baseline cho bài toán phát hiện Audio Deepfake.
- Đọc config từ configs/cnn_baseline.yaml.
- Load train/dev dataset thông qua dataset_loader.py.
- Build CNN từ cnn_baseline.py.
- Tạo callbacks từ callbacks.py.
- Train model bằng model.fit().
- Lưu best checkpoint, final model, training history và thông tin thực nghiệm.

Cách chạy test nhanh:

    py -m src.training.train_cnn --config configs/cnn_baseline.yaml --debug --epochs 2

Chạy train full:

    py -m src.training.train_cnn --config configs/cnn_baseline.yaml

Chạy với số mẫu tùy chọn:

    py -m src.training.train_cnn --config configs/cnn_baseline.yaml --max-train-samples 3000 --max-dev-samples 1000 --epochs 3

Lưu ý:
- File này mới là bước train model thật.
- Trước khi chạy full, nên chạy debug trước.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.data.dataset_loader import (
    load_yaml_config,
    infer_project_root,
    get_metadata_path_by_split,
    prepare_metadata_dataframe,
    make_tf_dataset,
    get_input_shape,
    get_train_dtype,
    get_dataloader_config,
    parse_autotune,
    compute_class_weight_from_dataframe,
    print_metadata_summary,
    inspect_one_batch,
)

from src.models.cnn_baseline import (
    build_cnn_from_config,
    compile_cnn_model,
    print_model_info,
    get_model_parameter_count,
    estimate_fp32_model_size_mb,
)

from src.training.callbacks import (
    create_callbacks_from_config,
    prepare_output_directories,
    resolve_path,
)


# ============================================================
# Reproducibility utilities
# ============================================================

def set_global_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả ổn định hơn."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def configure_mixed_precision(config: Dict[str, Any]) -> None:
    """
    Bật/tắt mixed precision theo config.

    Với laptop CPU, nên để false.
    Nếu sau này dùng GPU có Tensor Core, có thể cân nhắc bật.
    """
    training_config = config.get("training", {})
    mixed_precision_config = training_config.get("mixed_precision", {})
    enable = bool(mixed_precision_config.get("enable", False))

    if enable:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: ENABLED")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("Mixed precision: DISABLED")


# ============================================================
# Path utilities
# ============================================================

def get_result_dir(config: Dict[str, Any], project_root: Path) -> Path:
    """Lấy thư mục result từ config."""
    result_dir = config["paths"].get("result_dir", "outputs/results/cnn_baseline")
    result_dir = resolve_path(project_root, result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_saved_model_dir(config: Dict[str, Any], project_root: Path) -> Path:
    """Lấy thư mục saved model từ config."""
    saved_model_dir = config["paths"].get(
        "saved_model_dir",
        "outputs/saved_models/cnn_baseline",
    )
    saved_model_dir = resolve_path(project_root, saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    return saved_model_dir


def get_checkpoint_path(config: Dict[str, Any], project_root: Path) -> Path:
    """Lấy đường dẫn best checkpoint từ callback config."""
    checkpoint_config = config.get("callbacks", {}).get("model_checkpoint", {})
    filepath = checkpoint_config.get(
        "filepath",
        "outputs/checkpoints/cnn_baseline/best_cnn_baseline.keras",
    )
    return resolve_path(project_root, filepath)


# ============================================================
# Data sampling utilities
# ============================================================

def balanced_sample_dataframe(
    df: pd.DataFrame,
    max_samples: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Lấy mẫu cân bằng theo class.

    Dùng cho debug train.

    Lý do:
    - ASVspoof train.csv thường sắp xếp bonafide trước.
    - Nếu lấy head(1000) thì có thể toàn class 0.
    - Khi debug train, cần có cả class 0 và class 1.

    Nếu max_samples=None hoặc max_samples >= len(df), trả về df ban đầu.
    """
    if max_samples is None:
        return df.reset_index(drop=True)

    max_samples = int(max_samples)

    if max_samples >= len(df):
        return df.reset_index(drop=True)

    labels = sorted(df["label"].astype(int).unique().tolist())

    if len(labels) < 2:
        return df.head(max_samples).reset_index(drop=True)

    per_class = max_samples // len(labels)

    sampled_parts = []

    for label in labels:
        df_label = df[df["label"].astype(int) == label]
        n = min(per_class, len(df_label))

        sampled = df_label.sample(
            n=n,
            random_state=seed,
            replace=False,
        )

        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, axis=0)

    # Nếu max_samples là số lẻ hoặc một class không đủ mẫu, fill thêm từ phần còn lại
    remaining = max_samples - len(sampled_df)

    if remaining > 0:
        remaining_df = df.drop(index=sampled_df.index)

        if len(remaining_df) > 0:
            extra = remaining_df.sample(
                n=min(remaining, len(remaining_df)),
                random_state=seed,
                replace=False,
            )
            sampled_df = pd.concat([sampled_df, extra], axis=0)

    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return sampled_df


def stratified_sample_dataframe(
    df: pd.DataFrame,
    max_samples: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Lấy mẫu giữ tương đối tỷ lệ class.

    Dùng cho dev/eval debug.
    """
    if max_samples is None:
        return df.reset_index(drop=True)

    max_samples = int(max_samples)

    if max_samples >= len(df):
        return df.reset_index(drop=True)

    frac = max_samples / len(df)

    sampled_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, int(round(len(x) * frac))),
            random_state=seed,
            replace=False,
        ))
        .reset_index(drop=True)
    )

    if len(sampled_df) > max_samples:
        sampled_df = sampled_df.sample(
            n=max_samples,
            random_state=seed,
            replace=False,
        )

    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return sampled_df


def create_dataset_for_training(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    max_samples: Optional[int] = None,
    balanced_train_sample: bool = False,
) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    """
    Tạo dataset cho train/dev.

    Điểm khác so với test dataloader:
    - Với train debug, có thể lấy mẫu cân bằng để tránh toàn một class.
    - Với dev, giữ tỷ lệ class tương đối.
    """
    metadata_path = get_metadata_path_by_split(
        config=config,
        project_root=project_root,
        split=split,
    )

    df = prepare_metadata_dataframe(
        metadata_path=metadata_path,
        max_samples=None,
        require_audio_exists=True,
        require_feature_exists=True,
    )

    seed = int(config.get("project", {}).get("seed", 42))

    if max_samples is not None:
        if split == "train" and balanced_train_sample:
            df = balanced_sample_dataframe(
                df=df,
                max_samples=max_samples,
                seed=seed,
            )
        else:
            df = stratified_sample_dataframe(
                df=df,
                max_samples=max_samples,
                seed=seed,
            )

    input_shape = get_input_shape(config)
    train_dtype = get_train_dtype(config)

    dataloader_config = get_dataloader_config(config)

    batch_size = int(dataloader_config.get("batch_size", 16))
    shuffle_buffer_size = int(dataloader_config.get("shuffle_buffer_size", 4096))

    num_parallel_calls = parse_autotune(
        dataloader_config.get("num_parallel_calls", "AUTOTUNE")
    )

    prefetch = parse_autotune(
        dataloader_config.get("prefetch", "AUTOTUNE")
    )

    shuffle = split == "train"

    ds = make_tf_dataset(
        df=df,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        train_dtype=train_dtype,
        num_parallel_calls=num_parallel_calls,
        prefetch=prefetch,
        drop_remainder=False,
    )

    return ds, df


def get_debug_sample_limits(config: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Lấy số mẫu debug từ config.debug."""
    debug_config = config.get("debug", {})

    max_train_samples = debug_config.get("max_train_samples", 1000)
    max_dev_samples = debug_config.get("max_dev_samples", 500)

    return int(max_train_samples), int(max_dev_samples)


# ============================================================
# Training artifact utilities
# ============================================================

def convert_history_to_jsonable(history: tf.keras.callbacks.History) -> Dict[str, Any]:
    """Chuyển Keras history sang kiểu có thể lưu JSON."""
    output = {}

    for key, values in history.history.items():
        output[key] = [float(v) for v in values]

    return output


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """Lưu dict ra JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_yaml_snapshot(config: Dict[str, Any], output_path: Path) -> None:
    """Lưu bản copy config dùng cho experiment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    """Lưu model.summary() ra file txt."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    model.summary(print_fn=lambda line: lines.append(line))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_training_summary(
    config: Dict[str, Any],
    project_root: Path,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    class_weight: Optional[Dict[int, float]],
    training_time_seconds: float,
    final_model_path: Path,
) -> None:
    """Lưu thông tin tổng hợp của lần train."""
    result_dir = get_result_dir(config, project_root)
    checkpoint_path = get_checkpoint_path(config, project_root)

    counts = get_model_parameter_count(model)
    fp32_size_mb = estimate_fp32_model_size_mb(model)

    history_json = convert_history_to_jsonable(history)

    summary = {
        "project": config.get("project", {}),
        "model_name": model.name,
        "input_shape": list(model.input_shape[1:]),
        "output_shape": list(model.output_shape[1:]),
        "total_params": counts["total_params"],
        "trainable_params": counts["trainable_params"],
        "non_trainable_params": counts["non_trainable_params"],
        "estimated_fp32_size_mb": fp32_size_mb,
        "train_samples": int(len(train_df)),
        "dev_samples": int(len(dev_df)),
        "train_label_counts": {
            str(k): int(v)
            for k, v in train_df["label"].value_counts().sort_index().to_dict().items()
        },
        "dev_label_counts": {
            str(k): int(v)
            for k, v in dev_df["label"].value_counts().sort_index().to_dict().items()
        },
        "class_weight": None if class_weight is None else {
            str(k): float(v) for k, v in class_weight.items()
        },
        "training_time_seconds": float(training_time_seconds),
        "training_time_minutes": float(training_time_seconds / 60.0),
        "best_checkpoint_path": str(checkpoint_path),
        "final_model_path": str(final_model_path),
        "history": history_json,
    }

    save_json(
        data=summary,
        output_path=result_dir / "training_summary.json",
    )

    save_json(
        data=history_json,
        output_path=result_dir / "training_history.json",
    )

    save_yaml_snapshot(
        config=config,
        output_path=result_dir / "config_snapshot.yaml",
    )

    save_model_summary(
        model=model,
        output_path=result_dir / "model_summary.txt",
    )


# ============================================================
# Class weight
# ============================================================

def get_class_weight_for_training(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    no_class_weight: bool = False,
) -> Optional[Dict[int, float]]:
    """Tính class weight nếu config cho phép."""
    if no_class_weight:
        print("Class weight: DISABLED by CLI")
        return None

    class_weight_config = config.get("training", {}).get("class_weight", {})
    enable = bool(class_weight_config.get("enable", True))

    if not enable:
        print("Class weight: DISABLED by config")
        return None

    unique_labels = sorted(train_df["label"].astype(int).unique().tolist())

    if len(unique_labels) < 2:
        raise ValueError(
            f"Train data chỉ có class {unique_labels}. "
            "Không thể train binary classifier đúng nghĩa. "
            "Nếu đang debug, hãy tăng max_train_samples hoặc dùng balanced sampling."
        )

    class_weight = compute_class_weight_from_dataframe(train_df)

    print("\n" + "=" * 70)
    print("CLASS WEIGHT")
    print("=" * 70)
    print(class_weight)

    return class_weight


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN baseline for Audio Deepfake Detection."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_baseline.yaml",
        help="Đường dẫn tới config YAML.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Train nhanh với số mẫu nhỏ trong config.debug.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu train. Ghi đè config.debug nếu được truyền.",
    )

    parser.add_argument(
        "--max-dev-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu dev. Ghi đè config.debug nếu được truyền.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Số epoch train. Ghi đè training.epochs trong config.",
    )

    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Tắt class_weight khi train.",
    )

    parser.add_argument(
        "--no-time-logger",
        action="store_true",
        help="Không dùng EpochTimeLogger callback.",
    )

    parser.add_argument(
        "--inspect-batch",
        action="store_true",
        help="In thử một batch train/dev trước khi train.",
    )

    return parser.parse_args()


# ============================================================
# Main training
# ============================================================

def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    project_root = infer_project_root(config_path)

    seed = int(config.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    print("Project root:", project_root)
    print("Config file :", config_path)
    print("Seed        :", seed)

    configure_mixed_precision(config)

    prepare_output_directories(
        config=config,
        project_root=project_root,
    )

    # --------------------------------------------------------
    # Resolve sample limits
    # --------------------------------------------------------

    max_train_samples = args.max_train_samples
    max_dev_samples = args.max_dev_samples

    if args.debug:
        debug_train, debug_dev = get_debug_sample_limits(config)

        if max_train_samples is None:
            max_train_samples = debug_train

        if max_dev_samples is None:
            max_dev_samples = debug_dev

        print("\nDEBUG MODE ENABLED")
        print(f"max_train_samples: {max_train_samples}")
        print(f"max_dev_samples  : {max_dev_samples}")

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\nCreating train dataset...")
    train_ds, train_df = create_dataset_for_training(
        config=config,
        project_root=project_root,
        split="train",
        max_samples=max_train_samples,
        balanced_train_sample=True,
    )

    print("\nCreating dev dataset...")
    dev_ds, dev_df = create_dataset_for_training(
        config=config,
        project_root=project_root,
        split="dev",
        max_samples=max_dev_samples,
        balanced_train_sample=False,
    )

    print_metadata_summary(train_df, name="train")
    print_metadata_summary(dev_df, name="dev")

    if args.inspect_batch:
        print("\nInspecting train batch...")
        inspect_one_batch(train_ds)

        print("\nInspecting dev batch...")
        inspect_one_batch(dev_ds)

    # --------------------------------------------------------
    # Build and compile model
    # --------------------------------------------------------

    print("\nBuilding CNN baseline model...")
    model = build_cnn_from_config(config)
    model = compile_cnn_model(model, config)

    print_model_info(model)

    # --------------------------------------------------------
    # Class weight
    # --------------------------------------------------------

    class_weight = get_class_weight_for_training(
        config=config,
        train_df=train_df,
        no_class_weight=args.no_class_weight,
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = create_callbacks_from_config(
        config=config,
        project_root=project_root,
        include_time_logger=not args.no_time_logger,
    )

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------

    epochs = args.epochs

    if epochs is None:
        epochs = int(config.get("training", {}).get("epochs", 30))

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Epochs       : {epochs}")
    print(f"Train samples: {len(train_df)}")
    print(f"Dev samples  : {len(dev_df)}")
    print(f"Batch size   : {config.get('dataloader', {}).get('batch_size', 16)}")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    training_time_seconds = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Training time: {training_time_seconds / 60.0:.2f} minutes")

    # --------------------------------------------------------
    # Save final model and training artifacts
    # --------------------------------------------------------

    saved_model_dir = get_saved_model_dir(config, project_root)
    final_model_path = saved_model_dir / "cnn_baseline_final.keras"

    model.save(final_model_path)

    save_training_summary(
        config=config,
        project_root=project_root,
        train_df=train_df,
        dev_df=dev_df,
        model=model,
        history=history,
        class_weight=class_weight,
        training_time_seconds=training_time_seconds,
        final_model_path=final_model_path,
    )

    checkpoint_path = get_checkpoint_path(config, project_root)
    result_dir = get_result_dir(config, project_root)

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"Best checkpoint : {checkpoint_path}")
    print(f"Final model     : {final_model_path}")
    print(f"Training log    : {resolve_path(project_root, config['callbacks']['csv_logger']['filename'])}")
    print(f"Training summary: {result_dir / 'training_summary.json'}")
    print(f"Model summary   : {result_dir / 'model_summary.txt'}")

    print("\nTrain CNN baseline OK.")


if __name__ == "__main__":
    main()