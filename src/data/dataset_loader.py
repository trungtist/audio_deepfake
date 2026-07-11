"""
dataset_loader.py

Nhiệm vụ:
- Đọc metadata CSV: train.csv / dev.csv / eval.csv.
- Load Log-Mel Spectrogram .npy từ feature_path.
- Chuyển shape từ [64, 251] sang [64, 251, 1].
- Tạo tf.data.Dataset cho quá trình train/evaluate.

Cách chạy test từ thư mục gốc project:

    python -m src.data.dataset_loader --config configs/cnn_baseline.yaml --split train --max-samples 32

Test dev:

    python -m src.data.dataset_loader --config configs/cnn_baseline.yaml --split dev --max-samples 32

Kiểm tra file feature có tồn tại không:

    python -m src.data.dataset_loader --config configs/cnn_baseline.yaml --split train --max-samples 100 --validate-files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
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
        audio-deepfake-detection/configs/cnn_baseline.yaml

    Project root:
        audio-deepfake-detection/
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


def get_input_shape(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Lấy input shape từ config."""
    input_shape = config["feature"].get("input_shape", [64, 251, 1])

    if len(input_shape) != 3:
        raise ValueError(f"feature.input_shape phải có 3 chiều, got: {input_shape}")

    return tuple(int(v) for v in input_shape)


def get_train_dtype(config: Dict[str, Any]) -> str:
    """Lấy dtype dùng khi train."""
    return str(config["feature"].get("train_dtype", "float32"))


def get_metadata_path_by_split(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
) -> Path:
    """Lấy metadata path theo split."""
    paths_config = config["paths"]

    if split == "train":
        return resolve_path(project_root, paths_config["train_metadata"])

    if split == "dev":
        return resolve_path(project_root, paths_config["dev_metadata"])

    if split == "eval":
        return resolve_path(project_root, paths_config["eval_metadata"])

    raise ValueError(f"Split không hợp lệ: {split}")


def get_debug_max_samples(config: Dict[str, Any], split: str) -> Optional[int]:
    """Lấy số mẫu debug theo config."""
    debug_config = config.get("debug", {})

    if split == "train":
        return debug_config.get("max_train_samples")

    if split == "dev":
        return debug_config.get("max_dev_samples")

    if split == "eval":
        return debug_config.get("max_eval_samples")

    return None


# ============================================================
# Metadata utilities
# ============================================================

def read_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Đọc metadata CSV và kiểm tra cột bắt buộc."""
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata file: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = [
        "split",
        "utt_id",
        "label_text",
        "label",
        "audio_path",
        "feature_path",
        "audio_exists",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Metadata thiếu cột: {missing_columns}. "
            f"File: {metadata_path}"
        )

    return df


def prepare_metadata_dataframe(
    metadata_path: str | Path,
    max_samples: Optional[int] = None,
    require_audio_exists: bool = True,
    require_feature_exists: bool = True,
) -> pd.DataFrame:
    """
    Chuẩn bị dataframe metadata trước khi tạo dataset.

    Args:
        metadata_path: đường dẫn train.csv/dev.csv/eval.csv.
        max_samples: giới hạn số mẫu để debug.
        require_audio_exists: bỏ dòng audio_exists != 1.
        require_feature_exists: chỉ giữ dòng có feature_path tồn tại.

    Returns:
        pd.DataFrame đã lọc.
    """
    df = read_metadata(metadata_path)

    original_count = len(df)

    if require_audio_exists and "audio_exists" in df.columns:
        df = df[df["audio_exists"] == 1].copy()

    # Đảm bảo label là int và chỉ gồm 0/1
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    if require_feature_exists:
        exists_mask = df["feature_path"].apply(lambda p: Path(str(p)).exists())
        df = df[exists_mask].copy()

    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"Metadata sau khi lọc không còn dòng nào. "
            f"Original count: {original_count}, file: {metadata_path}"
        )

    return df


def print_metadata_summary(df: pd.DataFrame, name: str = "metadata") -> None:
    """In thống kê metadata."""
    print("\n" + "=" * 70)
    print(f"METADATA SUMMARY: {name}")
    print("=" * 70)

    print(f"Total samples: {len(df)}")

    label_counts = df["label"].value_counts().sort_index().to_dict()
    label_text_counts = df["label_text"].value_counts().to_dict()

    print(f"Label counts      : {label_counts}")
    print(f"Label text counts : {label_text_counts}")

    if "split" in df.columns:
        print(f"Split values      : {df['split'].unique().tolist()}")

    print("\nExample rows:")
    print(df[["utt_id", "label_text", "label", "feature_path"]].head(3).to_string(index=False))


def validate_feature_files(
    df: pd.DataFrame,
    input_shape: Tuple[int, int, int],
    max_check: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Kiểm tra file .npy có load được và đúng shape không.

    Dùng sau khi extract Log-Mel, đặc biệt nếu từng dừng extract giữa chừng.
    """
    expected_2d_shape = input_shape[:2]

    if max_check is not None:
        df_check = df.head(int(max_check)).copy()
    else:
        df_check = df

    total = len(df_check)
    bad_files = []

    for _, row in df_check.iterrows():
        feature_path = Path(str(row["feature_path"]))

        try:
            if not feature_path.exists():
                bad_files.append(
                    {
                        "feature_path": str(feature_path),
                        "error": "file_not_found",
                    }
                )
                continue

            x = np.load(feature_path)

            if x.shape == expected_2d_shape:
                continue

            if x.shape == input_shape:
                continue

            bad_files.append(
                {
                    "feature_path": str(feature_path),
                    "error": f"bad_shape_{x.shape}",
                }
            )

        except Exception as e:
            bad_files.append(
                {
                    "feature_path": str(feature_path),
                    "error": repr(e),
                }
            )

    stats = {
        "total_checked": total,
        "bad_count": len(bad_files),
        "bad_files": bad_files,
    }

    return stats


def print_validation_stats(stats: Dict[str, Any]) -> None:
    """In kết quả validate feature files."""
    print("\n" + "=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    print(f"Total checked : {stats['total_checked']}")
    print(f"Bad files     : {stats['bad_count']}")

    if stats["bad_files"]:
        print("\nMột số file lỗi:")
        for item in stats["bad_files"][:10]:
            print(f"  - {item['feature_path']}")
            print(f"    Error: {item['error']}")


# ============================================================
# TensorFlow dataset utilities
# ============================================================

def _load_npy_feature_py(
    feature_path: bytes,
    input_shape: Tuple[int, int, int],
    train_dtype: str,
) -> np.ndarray:
    """
    Hàm Python load .npy.

    Hàm này sẽ được gọi bên trong tf.numpy_function.
    """
    path = feature_path.decode("utf-8")
    x = np.load(path)

    # Nếu feature đang là [64, 251] thì thêm channel dim.
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    if x.shape != input_shape:
        raise ValueError(
            f"Feature shape không đúng. "
            f"Expected: {input_shape}, Got: {x.shape}, File: {path}"
        )

    if train_dtype == "float32":
        x = x.astype(np.float32)
    elif train_dtype == "float16":
        x = x.astype(np.float16)
    else:
        raise ValueError(f"train_dtype không hỗ trợ: {train_dtype}")

    return x


def make_tf_dataset(
    df: pd.DataFrame,
    input_shape: Tuple[int, int, int],
    batch_size: int = 16,
    shuffle: bool = False,
    shuffle_buffer_size: int = 4096,
    train_dtype: str = "float32",
    num_parallel_calls: Any = tf.data.AUTOTUNE,
    prefetch: Any = tf.data.AUTOTUNE,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    """
    Tạo tf.data.Dataset từ metadata dataframe.

    Output mỗi batch:
        X: [batch_size, 64, 251, 1]
        y: [batch_size]
    """
    feature_paths = df["feature_path"].astype(str).values
    labels = df["label"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((feature_paths, labels))

    if shuffle:
        buffer_size = min(int(shuffle_buffer_size), len(df))
        ds = ds.shuffle(
            buffer_size=buffer_size,
            seed=42,
            reshuffle_each_iteration=True,
        )

    def _load_feature_tf(path: tf.Tensor, label: tf.Tensor):
        x = tf.numpy_function(
            func=lambda p: _load_npy_feature_py(
                feature_path=p,
                input_shape=input_shape,
                train_dtype=train_dtype,
            ),
            inp=[path],
            Tout=tf.float32 if train_dtype == "float32" else tf.float16,
        )

        x.set_shape(input_shape)

        label = tf.cast(label, tf.float32)
        label.set_shape(())

        return x, label

    ds = ds.map(_load_feature_tf, num_parallel_calls=num_parallel_calls)

    ds = ds.batch(
        batch_size,
        drop_remainder=drop_remainder,
    )

    ds = ds.prefetch(prefetch)

    return ds


def get_dataloader_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy dataloader config."""
    return config.get("dataloader", {})


def parse_autotune(value: Any) -> Any:
    """Parse AUTOTUNE trong YAML."""
    if isinstance(value, str) and value.upper() == "AUTOTUNE":
        return tf.data.AUTOTUNE

    return value


def create_dataset_from_config(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    max_samples: Optional[int] = None,
    shuffle: Optional[bool] = None,
    require_feature_exists: bool = True,
) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    """
    Tạo tf.data.Dataset theo config và split.

    Hàm này sẽ được train_cnn.py sử dụng.
    """
    metadata_path = get_metadata_path_by_split(
        config=config,
        project_root=project_root,
        split=split,
    )

    df = prepare_metadata_dataframe(
        metadata_path=metadata_path,
        max_samples=max_samples,
        require_audio_exists=True,
        require_feature_exists=require_feature_exists,
    )

    input_shape = get_input_shape(config)
    train_dtype = get_train_dtype(config)

    dataloader_config = get_dataloader_config(config)

    batch_size = int(dataloader_config.get("batch_size", 16))
    shuffle_buffer_size = int(dataloader_config.get("shuffle_buffer_size", 4096))

    if shuffle is None:
        # Train thì shuffle, dev/eval thì không.
        shuffle = split == "train"

    num_parallel_calls = parse_autotune(
        dataloader_config.get("num_parallel_calls", "AUTOTUNE")
    )

    prefetch = parse_autotune(
        dataloader_config.get("prefetch", "AUTOTUNE")
    )

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


def create_train_dev_datasets_from_config(
    config: Dict[str, Any],
    project_root: Path,
    debug: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame]:
    """
    Tạo train_ds và dev_ds cho train_cnn.py.
    """
    train_max_samples = None
    dev_max_samples = None

    if debug:
        train_max_samples = get_debug_max_samples(config, "train")
        dev_max_samples = get_debug_max_samples(config, "dev")

    train_ds, train_df = create_dataset_from_config(
        config=config,
        project_root=project_root,
        split="train",
        max_samples=train_max_samples,
        shuffle=True,
        require_feature_exists=True,
    )

    dev_ds, dev_df = create_dataset_from_config(
        config=config,
        project_root=project_root,
        split="dev",
        max_samples=dev_max_samples,
        shuffle=False,
        require_feature_exists=True,
    )

    return train_ds, dev_ds, train_df, dev_df


# ============================================================
# Class weight
# ============================================================

def compute_class_weight_from_dataframe(df: pd.DataFrame) -> Dict[int, float]:
    """
    Tính class weight thủ công cho binary classification.

    Công thức:
        weight_c = total_samples / (num_classes * count_c)

    Với dataset mất cân bằng, class bonafide ít hơn spoof nên weight của
    bonafide thường lớn hơn.
    """
    labels = df["label"].astype(int).values

    classes = np.array([0, 1])
    total = len(labels)
    num_classes = len(classes)

    class_weight = {}

    for c in classes:
        count = int(np.sum(labels == c))

        if count == 0:
            raise ValueError(f"Không có sample nào cho class {c}")

        class_weight[int(c)] = float(total / (num_classes * count))

    return class_weight


# ============================================================
# Debug / test utilities
# ============================================================

def inspect_one_batch(ds: tf.data.Dataset) -> None:
    """In thông tin một batch để kiểm tra dataset."""
    for x_batch, y_batch in ds.take(1):
        x_np = x_batch.numpy()
        y_np = y_batch.numpy()

        print("\n" + "=" * 70)
        print("BATCH INSPECTION")
        print("=" * 70)

        print(f"X batch shape : {x_np.shape}")
        print(f"y batch shape : {y_np.shape}")
        print(f"X dtype       : {x_np.dtype}")
        print(f"y dtype       : {y_np.dtype}")

        print(f"X min         : {float(np.min(x_np)):.6f}")
        print(f"X max         : {float(np.max(x_np)):.6f}")
        print(f"X mean        : {float(np.mean(x_np)):.6f}")
        print(f"X std         : {float(np.std(x_np)):.6f}")
        print(f"Has NaN       : {bool(np.isnan(x_np).any())}")

        unique, counts = np.unique(y_np.astype(int), return_counts=True)
        label_counts = dict(zip(unique.tolist(), counts.tolist()))

        print(f"Labels        : {y_np}")
        print(f"Label counts  : {label_counts}")

        return

    raise ValueError("Dataset rỗng, không lấy được batch.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and inspect tf.data.Dataset from Log-Mel features."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_baseline.yaml",
        help="Đường dẫn tới config YAML.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "eval"],
        help="Split cần load.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu để test nhanh.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dùng số mẫu debug từ config.debug.",
    )

    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Kiểm tra file .npy có tồn tại và đúng shape không.",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Tắt shuffle khi inspect train dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    project_root = infer_project_root(config_path)

    print("Project root:", project_root)
    print("Config file :", config_path)

    split = args.split
    max_samples = args.max_samples

    if args.debug:
        max_samples = get_debug_max_samples(config, split)

    metadata_path = get_metadata_path_by_split(
        config=config,
        project_root=project_root,
        split=split,
    )

    df = prepare_metadata_dataframe(
        metadata_path=metadata_path,
        max_samples=max_samples,
        require_audio_exists=True,
        require_feature_exists=True,
    )

    print_metadata_summary(df, name=f"{split}.csv")

    input_shape = get_input_shape(config)

    if args.validate_files:
        stats = validate_feature_files(
            df=df,
            input_shape=input_shape,
            max_check=max_samples,
        )
        print_validation_stats(stats)

        if stats["bad_count"] > 0:
            raise ValueError("Có file feature lỗi. Hãy kiểm tra trước khi train.")

    dataloader_config = get_dataloader_config(config)
    batch_size = int(dataloader_config.get("batch_size", 16))
    shuffle_buffer_size = int(dataloader_config.get("shuffle_buffer_size", 4096))
    train_dtype = get_train_dtype(config)

    shuffle = split == "train" and not args.no_shuffle

    ds = make_tf_dataset(
        df=df,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        train_dtype=train_dtype,
        num_parallel_calls=tf.data.AUTOTUNE,
        prefetch=tf.data.AUTOTUNE,
    )

    inspect_one_batch(ds)

    if split == "train":
        unique_labels = sorted(df["label"].astype(int).unique().tolist())

        print("\n" + "=" * 70)
        print("CLASS WEIGHT")
        print("=" * 70)

        if len(unique_labels) < 2:
            print(
                "Không tính class_weight vì tập đang kiểm tra chỉ có "
                f"class {unique_labels}. "
                "Điều này thường xảy ra khi dùng --max-samples quá nhỏ."
            )
        else:
            class_weight = compute_class_weight_from_dataframe(df)
            print(class_weight)

    print("\nDataset loader OK.")

if __name__ == "__main__":
    main()