"""
mobilenetv3_precompute.py

Nhiệm vụ:
- Đọc Log-Mel feature đã extract cho CNN: [64, 251] hoặc [64, 251, 1].
- Chuyển thành feature dạng ảnh cho MobileNetV3-Small: [128, 128, 3].
- Lưu feature mới vào data/processed/mobilenetv3_128x128x3/.
- Tạo metadata phụ data/metadata_mobilenetv3/*.csv bằng cách copy metadata gốc
  và thay cột feature_path sang feature MobileNetV3.

Quan trọng:
- File này KHÔNG đọc lại protocol ASVspoof.
- File này KHÔNG tạo metadata gốc.
- Metadata gốc vẫn do src/data/make_metadata.py tạo.
- File này chỉ tạo feature_path mới cho MobileNetV3.

Cách chạy từ thư mục gốc project:

    py -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split train --max-samples 32

Chạy full train:

    py -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split train

Chạy full dev:

    py -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split dev

Chạy full eval:

    py -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split eval

Chạy cả train/dev/eval:

    py -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split all

Ghi đè feature đã tồn tại:

    py -m src.features.mobilenetv3_precompute --config configs/mobilenetv3_small.yaml --split train --force
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm


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

    Nếu config nằm trong configs/, project root là thư mục cha của configs/.
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


# ============================================================
# Path utilities
# ============================================================

def get_source_metadata_path(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
) -> Path:
    """Lấy metadata gốc theo split."""
    paths = config["paths"]

    if split == "train":
        return resolve_path(project_root, paths["source_train_metadata"])

    if split == "dev":
        return resolve_path(project_root, paths["source_dev_metadata"])

    if split == "eval":
        return resolve_path(project_root, paths["source_eval_metadata"])

    raise ValueError(f"Split không hợp lệ: {split}")


def get_output_metadata_path(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
) -> Path:
    """Lấy metadata MobileNetV3 output theo split."""
    paths = config["paths"]

    if split == "train":
        return resolve_path(project_root, paths["train_metadata"])

    if split == "dev":
        return resolve_path(project_root, paths["dev_metadata"])

    if split == "eval":
        return resolve_path(project_root, paths["eval_metadata"])

    raise ValueError(f"Split không hợp lệ: {split}")


def get_output_feature_dir(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
) -> Path:
    """Lấy thư mục lưu feature MobileNetV3 theo split."""
    paths = config["paths"]

    if split == "train":
        return resolve_path(project_root, paths["train_feature_dir"])

    if split == "dev":
        return resolve_path(project_root, paths["dev_feature_dir"])

    if split == "eval":
        return resolve_path(project_root, paths["eval_feature_dir"])

    raise ValueError(f"Split không hợp lệ: {split}")


def get_debug_max_samples(
    config: Dict[str, Any],
    split: str,
) -> Optional[int]:
    """Lấy max_samples debug cho precompute."""
    debug_config = config.get("debug", {}).get("precompute", {})

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

def read_source_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Đọc metadata gốc."""
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy source metadata: {metadata_path}")

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

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Source metadata thiếu cột: {missing}. File: {metadata_path}"
        )

    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    df = df.reset_index(drop=True)

    return df


def print_metadata_summary(df: pd.DataFrame, name: str) -> None:
    """In thống kê metadata."""
    print("\n" + "=" * 70)
    print(f"SOURCE METADATA SUMMARY: {name}")
    print("=" * 70)

    print(f"Total samples: {len(df)}")
    print(f"Label counts : {df['label'].value_counts().sort_index().to_dict()}")
    print(f"Label text   : {df['label_text'].value_counts().to_dict()}")

    if "split" in df.columns:
        print(f"Split values : {df['split'].unique().tolist()}")

    print("\nExample rows:")
    print(
        df[["utt_id", "label_text", "label", "feature_path"]]
        .head(3)
        .to_string(index=False)
    )


# ============================================================
# Feature conversion utilities
# ============================================================

def get_precompute_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy block precompute."""
    return config.get("precompute", {})


def get_output_shape(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Lấy output shape MobileNetV3."""
    precompute_config = get_precompute_config(config)

    output_shape = precompute_config.get(
        "output_shape",
        config.get("feature", {}).get("input_shape", [128, 128, 3]),
    )

    if len(output_shape) != 3:
        raise ValueError(f"output_shape phải có 3 chiều, got: {output_shape}")

    return tuple(int(v) for v in output_shape)


def get_storage_dtype(config: Dict[str, Any]) -> str:
    """Lấy dtype lưu feature."""
    precompute_config = get_precompute_config(config)
    return str(precompute_config.get("storage_dtype", "float16"))


def load_logmel_feature(feature_path: str | Path) -> np.ndarray:
    """
    Load Log-Mel feature gốc.

    Input hợp lệ:
        [64, 251]
        [64, 251, 1]

    Output:
        np.ndarray 2D [64, 251]
    """
    feature_path = Path(feature_path)

    if not feature_path.exists():
        raise FileNotFoundError(f"Không tìm thấy source feature: {feature_path}")

    x = np.load(feature_path)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)

    if x.ndim != 2:
        raise ValueError(
            f"Source Log-Mel phải là 2D hoặc 3D channel=1. "
            f"Got shape {x.shape}, file: {feature_path}"
        )

    return x.astype(np.float32)


def normalize_minmax_per_sample(
    x: np.ndarray,
    eps: float = 1e-6,
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> np.ndarray:
    """
    Normalize từng sample về khoảng [output_min, output_max].

    Log-Mel sau CNN pipeline thường đã z-score.
    MobileNetV3 pretrained ImageNet cần input image-like, nên ta scale lại.
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    denom = x_max - x_min

    if denom < eps:
        x_norm = np.zeros_like(x, dtype=np.float32)
    else:
        x_norm = (x - x_min) / (denom + eps)

    x_norm = x_norm * (output_max - output_min) + output_min
    return x_norm.astype(np.float32)


def resize_feature(
    x_2d: np.ndarray,
    target_height: int,
    target_width: int,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resize feature 2D sang [target_height, target_width].

    Dùng tf.image.resize để không cần thêm thư viện ngoài.
    """
    x = np.expand_dims(x_2d, axis=-1)       # [H, W, 1]
    x = np.expand_dims(x, axis=0)           # [1, H, W, 1]

    if method == "bilinear":
        resize_method = tf.image.ResizeMethod.BILINEAR
    elif method == "nearest":
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif method == "bicubic":
        resize_method = tf.image.ResizeMethod.BICUBIC
    else:
        raise ValueError(f"Resize method không hỗ trợ: {method}")

    x_resized = tf.image.resize(
        x,
        size=[target_height, target_width],
        method=resize_method,
        preserve_aspect_ratio=False,
        antialias=True,
    )

    x_resized = x_resized.numpy()[0, :, :, 0]  # [target_height, target_width]
    return x_resized.astype(np.float32)


def convert_to_three_channels(
    x_2d: np.ndarray,
    channels: int = 3,
    method: str = "repeat",
) -> np.ndarray:
    """
    Convert feature 2D sang 3 channels.

    method = repeat:
        [H, W] -> [H, W, 3]
    """
    if channels != 3:
        raise ValueError("MobileNetV3 pretrained ImageNet cần 3 channels.")

    if method != "repeat":
        raise ValueError(f"Channel method chưa hỗ trợ: {method}")

    x = np.expand_dims(x_2d, axis=-1)      # [H, W, 1]
    x = np.repeat(x, repeats=channels, axis=-1)

    return x.astype(np.float32)


def cast_storage_dtype(x: np.ndarray, storage_dtype: str) -> np.ndarray:
    """Cast dtype trước khi lưu."""
    if storage_dtype == "float16":
        return x.astype(np.float16)

    if storage_dtype == "float32":
        return x.astype(np.float32)

    if storage_dtype == "uint8":
        x = np.clip(x, 0.0, 255.0)
        return x.astype(np.uint8)

    raise ValueError(f"storage_dtype không hỗ trợ: {storage_dtype}")


def convert_logmel_to_mobilenetv3_feature(
    source_feature_path: str | Path,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Convert một file Log-Mel sang MobileNetV3 feature.

    Output shape mặc định:
        [128, 128, 3]
    """
    precompute_config = get_precompute_config(config)
    output_shape = get_output_shape(config)

    target_height, target_width, target_channels = output_shape

    normalization_config = precompute_config.get("normalization", {})
    resize_config = precompute_config.get("resize", {})
    channel_config = precompute_config.get("channel", {})
    scale_config = precompute_config.get("scale", {})

    eps = float(normalization_config.get("eps", 1e-6))
    norm_min = float(normalization_config.get("output_min", 0.0))
    norm_max = float(normalization_config.get("output_max", 1.0))

    resize_method = str(resize_config.get("method", "bilinear"))

    channel_method = str(channel_config.get("method", "repeat"))
    channels = int(channel_config.get("channels", target_channels))

    scale_min = float(scale_config.get("output_min", 0.0))
    scale_max = float(scale_config.get("output_max", 255.0))

    storage_dtype = get_storage_dtype(config)

    # 1. Load source Log-Mel
    x = load_logmel_feature(source_feature_path)

    # 2. Normalize per sample to [0, 1]
    x = normalize_minmax_per_sample(
        x,
        eps=eps,
        output_min=norm_min,
        output_max=norm_max,
    )

    # 3. Resize to [128, 128]
    x = resize_feature(
        x_2d=x,
        target_height=target_height,
        target_width=target_width,
        method=resize_method,
    )

    # 4. Convert to [128, 128, 3]
    x = convert_to_three_channels(
        x_2d=x,
        channels=channels,
        method=channel_method,
    )

    # 5. Scale to [0, 255]
    x = x * (scale_max - scale_min) + scale_min

    if x.shape != output_shape:
        raise ValueError(
            f"Output feature shape sai. Expected {output_shape}, got {x.shape}. "
            f"Source: {source_feature_path}"
        )

    # 6. Cast storage dtype
    x = cast_storage_dtype(x, storage_dtype=storage_dtype)

    return x


# ============================================================
# Precompute split
# ============================================================

def build_output_feature_path(
    output_feature_dir: Path,
    utt_id: str,
) -> Path:
    """Tạo path output feature từ utt_id."""
    return output_feature_dir / f"{utt_id}.npy"


def precompute_split(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    max_samples: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Precompute MobileNetV3 features cho một split.

    Args:
        config: mobilenetv3_small.yaml đã load.
        project_root: root project.
        split: train/dev/eval.
        max_samples: giới hạn số mẫu để debug.
        force: ghi đè feature đã tồn tại.

    Returns:
        dict thống kê.
    """
    source_metadata_path = get_source_metadata_path(
        config=config,
        project_root=project_root,
        split=split,
    )

    output_metadata_path = get_output_metadata_path(
        config=config,
        project_root=project_root,
        split=split,
    )

    output_feature_dir = get_output_feature_dir(
        config=config,
        project_root=project_root,
        split=split,
    )

    output_feature_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_source_metadata(source_metadata_path)

    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    df = df.reset_index(drop=True)

    print_metadata_summary(df, name=f"{split} source")

    precompute_config = get_precompute_config(config)

    if not force:
        force = bool(precompute_config.get("overwrite", False))

    output_shape = get_output_shape(config)
    storage_dtype = get_storage_dtype(config)

    output_rows = []
    total = len(df)
    created_count = 0
    skipped_existing_count = 0
    failed_count = 0
    missing_source_count = 0
    bad_items = []

    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"PRECOMPUTE MOBILENETV3 FEATURES: {split}")
    print("=" * 70)
    print(f"Source metadata : {source_metadata_path}")
    print(f"Output metadata : {output_metadata_path}")
    print(f"Output dir      : {output_feature_dir}")
    print(f"Output shape    : {output_shape}")
    print(f"Storage dtype   : {storage_dtype}")
    print(f"Force overwrite : {force}")
    print(f"Total rows      : {total}")

    for _, row in tqdm(df.iterrows(), total=total, desc=f"Precompute {split}"):
        row_dict = row.to_dict()

        utt_id = str(row_dict["utt_id"])
        source_feature_path = Path(str(row_dict["feature_path"]))
        output_feature_path = build_output_feature_path(
            output_feature_dir=output_feature_dir,
            utt_id=utt_id,
        )

        # Metadata mới sẽ trỏ sang feature MobileNetV3
        row_dict["source_feature_path"] = str(source_feature_path)
        row_dict["feature_path"] = str(output_feature_path)

        try:
            if not source_feature_path.exists():
                missing_source_count += 1
                failed_count += 1

                row_dict["feature_exists"] = 0
                row_dict["precompute_status"] = "missing_source_feature"

                bad_items.append(
                    {
                        "utt_id": utt_id,
                        "source_feature_path": str(source_feature_path),
                        "error": "missing_source_feature",
                    }
                )

                output_rows.append(row_dict)
                continue

            if output_feature_path.exists() and not force:
                skipped_existing_count += 1

                row_dict["feature_exists"] = 1
                row_dict["precompute_status"] = "skipped_existing"

                output_rows.append(row_dict)
                continue

            x_mobile = convert_logmel_to_mobilenetv3_feature(
                source_feature_path=source_feature_path,
                config=config,
            )

            output_feature_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(output_feature_path, x_mobile)

            created_count += 1

            row_dict["feature_exists"] = 1
            row_dict["precompute_status"] = "created"

            output_rows.append(row_dict)

        except Exception as e:
            failed_count += 1

            row_dict["feature_exists"] = 0
            row_dict["precompute_status"] = "failed"

            bad_items.append(
                {
                    "utt_id": utt_id,
                    "source_feature_path": str(source_feature_path),
                    "output_feature_path": str(output_feature_path),
                    "error": repr(e),
                }
            )

            output_rows.append(row_dict)

    output_df = pd.DataFrame(output_rows)

    output_df.to_csv(
        output_metadata_path,
        index=False,
        encoding="utf-8",
    )

    elapsed = time.time() - start_time

    stats = {
        "split": split,
        "total_rows": total,
        "created_count": created_count,
        "skipped_existing_count": skipped_existing_count,
        "failed_count": failed_count,
        "missing_source_count": missing_source_count,
        "bad_items": bad_items[:20],
        "output_metadata_path": str(output_metadata_path),
        "output_feature_dir": str(output_feature_dir),
        "output_shape": list(output_shape),
        "storage_dtype": storage_dtype,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
    }

    print("\n" + "=" * 70)
    print(f"PRECOMPUTE SUMMARY: {split}")
    print("=" * 70)
    print(f"Total rows        : {total}")
    print(f"Created           : {created_count}")
    print(f"Skipped existing  : {skipped_existing_count}")
    print(f"Failed            : {failed_count}")
    print(f"Missing source    : {missing_source_count}")
    print(f"Output metadata   : {output_metadata_path}")
    print(f"Elapsed           : {elapsed / 60.0:.2f} minutes")

    if bad_items:
        print("\nMột số lỗi đầu tiên:")
        for item in bad_items[:10]:
            print(f"- {item}")

    return stats


# ============================================================
# Validation helper
# ============================================================

def validate_precomputed_metadata(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    max_check: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Kiểm tra nhanh metadata MobileNetV3 và feature đã tạo.
    """
    metadata_path = get_output_metadata_path(
        config=config,
        project_root=project_root,
        split=split,
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy output metadata: {metadata_path}")

    df = pd.read_csv(metadata_path)

    output_shape = get_output_shape(config)

    if max_check is not None:
        df_check = df.head(int(max_check)).copy()
    else:
        df_check = df

    bad_files = []

    for _, row in tqdm(df_check.iterrows(), total=len(df_check), desc=f"Validate {split}"):
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

            if x.shape != output_shape:
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
        "split": split,
        "metadata_path": str(metadata_path),
        "total_checked": len(df_check),
        "bad_count": len(bad_files),
        "bad_files": bad_files[:20],
    }

    print("\n" + "=" * 70)
    print(f"VALIDATE PRECOMPUTED FEATURES: {split}")
    print("=" * 70)
    print(f"Metadata     : {metadata_path}")
    print(f"Checked      : {stats['total_checked']}")
    print(f"Bad files    : {stats['bad_count']}")

    if bad_files:
        print("\nMột số file lỗi:")
        for item in bad_files[:10]:
            print(f"- {item}")

    return stats


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute MobileNetV3 input features from Log-Mel features."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mobilenetv3_small.yaml",
        help="Đường dẫn config MobileNetV3 YAML.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "eval", "all"],
        help="Split cần precompute.",
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
        help="Dùng số mẫu debug từ config.debug.precompute.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè feature MobileNetV3 đã tồn tại.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Chỉ validate feature đã precompute, không tạo mới.",
    )

    parser.add_argument(
        "--max-check",
        type=int,
        default=None,
        help="Giới hạn số file validate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    project_root = infer_project_root(config_path)

    print("Project root:", project_root)
    print("Config file :", config_path)

    if args.split == "all":
        splits = ["train", "dev", "eval"]
    else:
        splits = [args.split]

    all_stats = []

    for split in splits:
        max_samples = args.max_samples

        if args.debug:
            max_samples = get_debug_max_samples(config, split)

        if args.validate_only:
            stats = validate_precomputed_metadata(
                config=config,
                project_root=project_root,
                split=split,
                max_check=args.max_check,
            )
        else:
            stats = precompute_split(
                config=config,
                project_root=project_root,
                split=split,
                max_samples=max_samples,
                force=args.force,
            )

            validate_precomputed_metadata(
                config=config,
                project_root=project_root,
                split=split,
                max_check=args.max_check,
            )

        all_stats.append(stats)

    print("\n" + "=" * 70)
    print("MOBILENETV3 PRECOMPUTE FINISHED")
    print("=" * 70)

    for stats in all_stats:
        print(
            f"Split: {stats['split']} | "
            f"Total: {stats.get('total_rows', stats.get('total_checked'))} | "
            f"Failed/Bad: {stats.get('failed_count', stats.get('bad_count'))}"
        )


if __name__ == "__main__":
    main()