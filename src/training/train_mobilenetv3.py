"""
train_mobilenetv3.py

Nhiệm vụ:
- Train MobileNetV3-Small cho bài toán Audio Deepfake Detection.
- Input: precomputed MobileNetV3 feature [128, 128, 3].
- Dùng metadata_mobilenetv3/*.csv.
- Dùng chung dataset_loader.py.
- Dùng model từ src/models/mobilenetv3_small.py.
- Lưu checkpoint, final model, training log, model summary, training summary.

Quy ước label:
    0 = bonafide / real
    1 = spoof / fake

Chạy debug:

    python -m src.training.train_mobilenetv3 --config configs/mobilenetv3_small.yaml --debug --epochs 3 --inspect-batch

Chạy full:

    python -m src.training.train_mobilenetv3 --config configs/mobilenetv3_small.yaml

Nếu muốn test khi chưa tải được ImageNet weights:

    python -m src.training.train_mobilenetv3 --config configs/mobilenetv3_small.yaml --debug --epochs 1 --allow-random-fallback

Fine-tune:

    python -m src.training.train_mobilenetv3 --config configs/mobilenetv3_small.yaml --fine-tune

Lưu ý:
- Khi train thật theo mục tiêu pretrained ImageNet, không nên dùng --allow-random-fallback.
"""

from __future__ import annotations

import argparse
import copy
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
    resolve_path,
    get_metadata_path_by_split,
    prepare_metadata_dataframe,
    print_metadata_summary,
    make_tf_dataset,
    get_input_shape,
    get_train_dtype,
    get_dataloader_config,
    parse_autotune,
    compute_class_weight_from_dataframe,
    inspect_one_batch,
)

from src.models.mobilenetv3_small import (
    build_model_from_config,
    compile_model,
    print_model_info,
    print_backbone_trainable_summary,
    set_backbone_trainable,
)


# ============================================================
# Seed utilities
# ============================================================

def set_global_seed(seed: int) -> None:
    """Cố định seed để kết quả dễ tái lập hơn."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Sampling utilities
# ============================================================

def balanced_sample_dataframe(
    df: pd.DataFrame,
    max_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Lấy mẫu cân bằng 2 class cho train debug.

    Ví dụ:
        max_samples = 1000
        → lấy 500 bonafide + 500 spoof nếu đủ dữ liệu.

    Dùng cho train debug để tránh tình trạng lấy head() toàn bonafide.
    """
    max_samples = int(max_samples)

    if max_samples >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_0 = df[df["label"].astype(int) == 0]
    df_1 = df[df["label"].astype(int) == 1]

    if len(df_0) == 0 or len(df_1) == 0:
        return df.sample(
            n=min(max_samples, len(df)),
            random_state=seed,
            replace=False,
        ).reset_index(drop=True)

    n0 = max_samples // 2
    n1 = max_samples - n0

    n0 = min(n0, len(df_0))
    n1 = min(n1, len(df_1))

    sampled_0 = df_0.sample(n=n0, random_state=seed, replace=False)
    sampled_1 = df_1.sample(n=n1, random_state=seed, replace=False)

    sampled_df = pd.concat([sampled_0, sampled_1], axis=0)

    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return sampled_df


def stratified_sample_dataframe(
    df: pd.DataFrame,
    max_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Lấy mẫu giữ tương đối tỷ lệ class.

    Dùng cho dev/eval debug để phân phối gần với dataset gốc.
    """
    max_samples = int(max_samples)

    if max_samples >= len(df):
        return df.reset_index(drop=True)

    frac = max_samples / len(df)
    sampled_parts = []

    for label_value in sorted(df["label"].astype(int).unique().tolist()):
        df_label = df[df["label"].astype(int) == label_value]

        n_samples = max(1, int(round(len(df_label) * frac)))
        n_samples = min(n_samples, len(df_label))

        sampled = df_label.sample(
            n=n_samples,
            random_state=seed,
            replace=False,
        )

        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, axis=0)

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


# ============================================================
# Dataset utilities
# ============================================================

def create_dataset_from_dataframe(
    df: pd.DataFrame,
    config: Dict[str, Any],
    split: str,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    Tạo tf.data.Dataset từ DataFrame đã chuẩn bị.

    Dùng chung make_tf_dataset() trong dataset_loader.py.
    """
    input_shape = get_input_shape(config)
    train_dtype = get_train_dtype(config)

    dataloader_config = get_dataloader_config(config)

    batch_size = int(dataloader_config.get("batch_size", 8))
    shuffle_buffer_size = int(dataloader_config.get("shuffle_buffer_size", 4096))

    num_parallel_calls = parse_autotune(
        dataloader_config.get("num_parallel_calls", "AUTOTUNE")
    )

    prefetch = parse_autotune(
        dataloader_config.get("prefetch", "AUTOTUNE")
    )

    drop_remainder = bool(dataloader_config.get("drop_remainder", False))

    ds = make_tf_dataset(
        df=df,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        train_dtype=train_dtype,
        num_parallel_calls=num_parallel_calls,
        prefetch=prefetch,
        drop_remainder=drop_remainder,
    )

    return ds


def create_train_dev_datasets(
    config: Dict[str, Any],
    project_root: Path,
    debug: bool = False,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame]:
    """
    Tạo train_ds và dev_ds cho MobileNetV3.

    Khác với create_train_dev_datasets_from_config() mặc định:
    - train debug dùng balanced sample để có đủ 2 class.
    - dev debug dùng stratified sample để phản ánh phân phối thật.
    """
    seed = int(config.get("project", {}).get("seed", 42))

    debug_config = config.get("debug", {})

    if debug:
        if max_train_samples is None:
            max_train_samples = debug_config.get("max_train_samples", 1000)

        if max_dev_samples is None:
            max_dev_samples = debug_config.get("max_dev_samples", 500)

    train_metadata_path = get_metadata_path_by_split(
        config=config,
        project_root=project_root,
        split="train",
    )

    dev_metadata_path = get_metadata_path_by_split(
        config=config,
        project_root=project_root,
        split="dev",
    )

    train_df = prepare_metadata_dataframe(
        metadata_path=train_metadata_path,
        max_samples=None,
        require_audio_exists=True,
        require_feature_exists=True,
    )

    dev_df = prepare_metadata_dataframe(
        metadata_path=dev_metadata_path,
        max_samples=None,
        require_audio_exists=True,
        require_feature_exists=True,
    )

    if max_train_samples is not None:
        train_df = balanced_sample_dataframe(
            df=train_df,
            max_samples=int(max_train_samples),
            seed=seed,
        )

    if max_dev_samples is not None:
        dev_df = stratified_sample_dataframe(
            df=dev_df,
            max_samples=int(max_dev_samples),
            seed=seed,
        )

    train_ds = create_dataset_from_dataframe(
        df=train_df,
        config=config,
        split="train",
        shuffle=True,
    )

    dev_ds = create_dataset_from_dataframe(
        df=dev_df,
        config=config,
        split="dev",
        shuffle=False,
    )

    return train_ds, dev_ds, train_df, dev_df


# ============================================================
# Callback utilities
# ============================================================

class EpochTimeLogger(tf.keras.callbacks.Callback):
    """Callback in thời gian mỗi epoch."""

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        elapsed = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1} time: {elapsed:.2f} seconds")


def create_callbacks_from_config(
    config: Dict[str, Any],
    project_root: Path,
) -> list[tf.keras.callbacks.Callback]:
    """Tạo callbacks từ config."""
    callbacks_config = config.get("callbacks", {})

    callbacks: list[tf.keras.callbacks.Callback] = []

    # ModelCheckpoint
    checkpoint_config = callbacks_config.get("model_checkpoint", {})
    if checkpoint_config.get("enable", True):
        filepath = resolve_path(
            project_root,
            checkpoint_config.get(
                "filepath",
                "outputs/checkpoints/mobilenetv3_small/best_mobilenetv3_small.keras",
            ),
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(filepath),
                monitor=checkpoint_config.get("monitor", "val_roc_auc"),
                mode=checkpoint_config.get("mode", "max"),
                save_best_only=bool(checkpoint_config.get("save_best_only", True)),
                save_weights_only=bool(checkpoint_config.get("save_weights_only", False)),
                verbose=int(checkpoint_config.get("verbose", 1)),
            )
        )

    # EarlyStopping
    early_config = callbacks_config.get("early_stopping", {})
    if early_config.get("enable", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_config.get("monitor", "val_roc_auc"),
                mode=early_config.get("mode", "max"),
                patience=int(early_config.get("patience", 5)),
                restore_best_weights=bool(early_config.get("restore_best_weights", True)),
                verbose=int(early_config.get("verbose", 1)),
            )
        )

    # ReduceLROnPlateau
    reduce_lr_config = callbacks_config.get("reduce_lr_on_plateau", {})
    if reduce_lr_config.get("enable", True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_lr_config.get("monitor", "val_loss"),
                mode=reduce_lr_config.get("mode", "min"),
                factor=float(reduce_lr_config.get("factor", 0.5)),
                patience=int(reduce_lr_config.get("patience", 2)),
                min_lr=float(reduce_lr_config.get("min_lr", 1e-6)),
                verbose=int(reduce_lr_config.get("verbose", 1)),
            )
        )

    # CSVLogger
    csv_logger_config = callbacks_config.get("csv_logger", {})
    if csv_logger_config.get("enable", True):
        filename = resolve_path(
            project_root,
            csv_logger_config.get(
                "filename",
                "outputs/logs/mobilenetv3_small/training_log.csv",
            ),
        )
        filename.parent.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=str(filename),
                append=bool(csv_logger_config.get("append", False)),
            )
        )

    callbacks.append(EpochTimeLogger())

    return callbacks


# ============================================================
# Output utilities
# ============================================================

def get_path_from_config(
    config: Dict[str, Any],
    project_root: Path,
    key: str,
    default_value: str,
) -> Path:
    """Lấy path từ config.paths."""
    paths_config = config.get("paths", {})
    return resolve_path(project_root, paths_config.get(key, default_value))


def save_model_summary_txt(
    model: tf.keras.Model,
    output_path: str | Path,
) -> None:
    """Lưu model.summary() ra file txt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    model.summary(print_fn=lambda x: lines.append(x))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json(
    data: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Lưu dict ra JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def save_config_snapshot(
    config: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Lưu snapshot config dùng trong lần train."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def history_to_dict(history: tf.keras.callbacks.History) -> Dict[str, Any]:
    """Chuyển History object sang dict JSON-serializable."""
    output = {}

    for key, values in history.history.items():
        output[key] = [float(v) for v in values]

    return output


def save_training_outputs(
    model: tf.keras.Model,
    config: Dict[str, Any],
    project_root: Path,
    history: tf.keras.callbacks.History,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    training_time_seconds: float,
    debug: bool,
) -> Dict[str, str]:
    """Lưu final model, history, summary, model summary, config snapshot."""
    saved_model_dir = get_path_from_config(
        config,
        project_root,
        key="saved_model_dir",
        default_value="outputs/saved_models/mobilenetv3_small",
    )

    result_dir = get_path_from_config(
        config,
        project_root,
        key="result_dir",
        default_value="outputs/results/mobilenetv3_small",
    )

    checkpoint_dir = get_path_from_config(
        config,
        project_root,
        key="checkpoint_dir",
        default_value="outputs/checkpoints/mobilenetv3_small",
    )

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = saved_model_dir / "mobilenetv3_small_final.keras"
    model_summary_path = result_dir / "model_summary.txt"
    training_history_path = result_dir / "training_history.json"
    training_summary_path = result_dir / "training_summary.json"
    config_snapshot_path = result_dir / "config_snapshot.yaml"

    model.save(final_model_path)

    save_model_summary_txt(
        model=model,
        output_path=model_summary_path,
    )

    history_dict = history_to_dict(history)
    save_json(
        data=history_dict,
        output_path=training_history_path,
    )

    checkpoint_path = Path(
        config.get("callbacks", {})
        .get("model_checkpoint", {})
        .get("filepath", "outputs/checkpoints/mobilenetv3_small/best_mobilenetv3_small.keras")
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    counts = {
        "total_params": int(model.count_params()),
        "trainable_params": int(
            np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        ),
        "non_trainable_params": int(
            np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        ),
    }

    summary = {
        "project": config.get("project", {}),
        "model": {
            "name": model.name,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            **counts,
        },
        "data": {
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
        },
        "training": {
            "debug": bool(debug),
            "training_time_seconds": float(training_time_seconds),
            "training_time_minutes": float(training_time_seconds / 60.0),
            "history_keys": list(history_dict.keys()),
        },
        "files": {
            "best_checkpoint": str(checkpoint_path),
            "final_model": str(final_model_path),
            "training_history": str(training_history_path),
            "training_summary": str(training_summary_path),
            "model_summary": str(model_summary_path),
            "config_snapshot": str(config_snapshot_path),
        },
    }

    save_json(
        data=summary,
        output_path=training_summary_path,
    )

    save_config_snapshot(
        config=config,
        output_path=config_snapshot_path,
    )

    return {
        "best_checkpoint": str(checkpoint_path),
        "final_model": str(final_model_path),
        "training_history": str(training_history_path),
        "training_summary": str(training_summary_path),
        "model_summary": str(model_summary_path),
        "config_snapshot": str(config_snapshot_path),
    }


# ============================================================
# Training stage utilities
# ============================================================

def set_optimizer_learning_rate(
    config: Dict[str, Any],
    learning_rate: float,
) -> Dict[str, Any]:
    """Copy config và override learning rate."""
    new_config = copy.deepcopy(config)

    if "training" not in new_config:
        new_config["training"] = {}

    if "optimizer" not in new_config["training"]:
        new_config["training"]["optimizer"] = {}

    new_config["training"]["optimizer"]["learning_rate"] = float(learning_rate)

    return new_config


def get_epochs_from_config(
    config: Dict[str, Any],
    args_epochs: Optional[int],
) -> int:
    """Lấy epochs từ args hoặc config."""
    if args_epochs is not None:
        return int(args_epochs)

    training_config = config.get("training", {})
    stages_config = training_config.get("stages", {})
    stage_1_config = stages_config.get("stage_1", {})

    if stage_1_config.get("enable", False):
        return int(stage_1_config.get("epochs", training_config.get("epochs", 20)))

    return int(training_config.get("epochs", 20))


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small for Audio Deepfake Detection."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mobilenetv3_small.yaml",
        help="Đường dẫn config MobileNetV3 YAML.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Train debug với số mẫu nhỏ trong config.debug.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override số epoch.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override số mẫu train.",
    )

    parser.add_argument(
        "--max-dev-samples",
        type=int,
        default=None,
        help="Override số mẫu dev.",
    )

    parser.add_argument(
        "--inspect-batch",
        action="store_true",
        help="Inspect một batch train/dev trước khi train.",
    )

    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Không dùng class_weight.",
    )

    parser.add_argument(
        "--allow-random-fallback",
        action="store_true",
        help="Nếu không load được ImageNet weights thì fallback sang weights=None. Chỉ nên dùng để test.",
    )

    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Bật stage 2 fine-tune nếu config có stage_2.",
    )

    return parser.parse_args()


# ============================================================
# Main
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
    print("Debug mode  :", args.debug)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nCreating MobileNetV3 train/dev datasets...")

    train_ds, dev_ds, train_df, dev_df = create_train_dev_datasets(
        config=config,
        project_root=project_root,
        debug=args.debug,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
    )

    print_metadata_summary(train_df, name="mobilenetv3 train")
    print_metadata_summary(dev_df, name="mobilenetv3 dev")

    if args.inspect_batch:
        print("\nInspecting train batch...")
        inspect_one_batch(train_ds)

        print("\nInspecting dev batch...")
        inspect_one_batch(dev_ds)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nBuilding MobileNetV3-Small model...")

    model = build_model_from_config(
        config=config,
        allow_random_fallback=args.allow_random_fallback,
    )

    model = compile_model(
        model=model,
        config=config,
    )

    print_model_info(model)
    print_backbone_trainable_summary(model)

    # --------------------------------------------------------
    # Class weight
    # --------------------------------------------------------

    use_class_weight = bool(
        config.get("training", {})
        .get("class_weight", {})
        .get("enable", True)
    )

    if args.no_class_weight:
        use_class_weight = False

    if use_class_weight:
        class_weight = compute_class_weight_from_dataframe(train_df)
    else:
        class_weight = None

    print("\n" + "=" * 70)
    print("CLASS WEIGHT")
    print("=" * 70)
    print(class_weight if class_weight is not None else "Disabled")

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = create_callbacks_from_config(
        config=config,
        project_root=project_root,
    )

    # --------------------------------------------------------
    # Train stage 1
    # --------------------------------------------------------

    epochs = get_epochs_from_config(
        config=config,
        args_epochs=args.epochs,
    )

    dataloader_config = get_dataloader_config(config)
    batch_size = int(dataloader_config.get("batch_size", 8))

    print("\n" + "=" * 70)
    print("TRAINING START - MOBILENETV3 STAGE 1")
    print("=" * 70)
    print(f"Epochs       : {epochs}")
    print(f"Train samples: {len(train_df)}")
    print(f"Dev samples  : {len(dev_df)}")
    print(f"Batch size   : {batch_size}")
    print("Backbone     : frozen")

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # --------------------------------------------------------
    # Optional fine-tune stage 2
    # --------------------------------------------------------

    stage_2_history = None
    training_config = config.get("training", {})
    stages_config = training_config.get("stages", {})
    stage_2_config = stages_config.get("stage_2", {})

    if args.fine_tune and stage_2_config.get("enable", False):
        print("\n" + "=" * 70)
        print("TRAINING START - MOBILENETV3 STAGE 2 FINE-TUNE")
        print("=" * 70)

        fine_tune_from_layer = stage_2_config.get("fine_tune_from_layer", -30)
        stage_2_lr = float(stage_2_config.get("learning_rate", 3e-5))
        stage_2_epochs = int(stage_2_config.get("epochs", 10))

        model = set_backbone_trainable(
            model=model,
            trainable=True,
            fine_tune_from_layer=int(fine_tune_from_layer),
        )

        stage_2_config_runtime = set_optimizer_learning_rate(
            config=config,
            learning_rate=stage_2_lr,
        )

        model = compile_model(
            model=model,
            config=stage_2_config_runtime,
        )

        print_backbone_trainable_summary(model)

        stage_2_history = model.fit(
            train_ds,
            validation_data=dev_ds,
            epochs=stage_2_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        # Gộp history stage 2 vào history chính
        for key, values in stage_2_history.history.items():
            if key in history.history:
                history.history[key].extend(values)
            else:
                history.history[key] = values

    training_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"Training time: {training_time / 60.0:.2f} minutes")

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    output_files = save_training_outputs(
        model=model,
        config=config,
        project_root=project_root,
        history=history,
        train_df=train_df,
        dev_df=dev_df,
        training_time_seconds=training_time,
        debug=args.debug,
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"Best checkpoint : {output_files['best_checkpoint']}")
    print(f"Final model     : {output_files['final_model']}")
    print(f"Training history: {output_files['training_history']}")
    print(f"Training summary: {output_files['training_summary']}")
    print(f"Model summary   : {output_files['model_summary']}")
    print(f"Config snapshot : {output_files['config_snapshot']}")

    print("\nTrain MobileNetV3-Small OK.")


if __name__ == "__main__":
    main()