"""
logmel.py

Nhiệm vụ:
- Trích xuất Log-Mel Spectrogram từ audio.
- Đọc metadata train/dev/eval đã tạo bởi make_metadata.py.
- Lưu đặc trưng Log-Mel thành file .npy.

Cách chạy từ thư mục gốc project:

    python -m src.features.logmel --config configs/cnn_baseline.yaml --split train

Chạy debug 100 file:

    python -m src.features.logmel --config configs/cnn_baseline.yaml --split train --max-samples 100

Chạy toàn bộ train/dev/eval:

    python -m src.features.logmel --config configs/cnn_baseline.yaml --split all

Chạy lại và ghi đè feature cũ:

    python -m src.features.logmel --config configs/cnn_baseline.yaml --split train --force
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config rỗng: {config_path}")

    return config


def infer_project_root(config_path: Path) -> Path:
    """
    Suy ra thư mục gốc project.

    Trường hợp phổ biến:
        audio-deepfake-detection/configs/cnn_baseline.yaml

    Project root:
        audio-deepfake-detection/
    """
    config_path = config_path.resolve()

    if config_path.parent.name == "configs":
        return config_path.parent.parent

    return Path.cwd().resolve()


def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """Chuyển path tương đối trong config thành absolute path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def get_audio_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy cấu hình audio."""
    audio_config = config.get("audio", {})

    required_keys = [
        "sample_rate",
        "duration",
        "target_samples",
        "mono",
        "pad_mode",
        "trim_mode",
    ]

    for key in required_keys:
        if key not in audio_config:
            raise KeyError(f"Thiếu audio.{key} trong config")

    return audio_config


def get_feature_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy cấu hình feature Log-Mel."""
    feature_config = config.get("feature", {})

    required_keys = [
        "n_mels",
        "n_fft",
        "hop_length",
        "win_length",
        "center",
        "fmin",
        "fmax",
        "power",
        "log_ref",
        "top_db",
        "expected_frames",
        "input_shape",
        "normalization",
        "storage_dtype",
        "train_dtype",
    ]

    for key in required_keys:
        if key not in feature_config:
            raise KeyError(f"Thiếu feature.{key} trong config")

    return feature_config


def pad_or_trim_audio(
    y: np.ndarray,
    target_samples: int,
    pad_mode: str = "constant",
    trim_mode: str = "start",
) -> np.ndarray:
    """
    Pad hoặc trim audio về cùng độ dài.

    Args:
        y: Waveform 1D.
        target_samples: Số sample mục tiêu.
        pad_mode: Kiểu padding, thường dùng "constant".
        trim_mode:
            - "start": lấy từ đầu audio.
            - "center": lấy đoạn giữa audio.

    Returns:
        Waveform 1D có độ dài target_samples.
    """
    current_samples = len(y)

    if current_samples == target_samples:
        return y

    if current_samples < target_samples:
        pad_length = target_samples - current_samples

        if pad_mode == "constant":
            y = np.pad(y, (0, pad_length), mode="constant")
        else:
            y = np.pad(y, (0, pad_length), mode=pad_mode)

        return y

    # current_samples > target_samples
    if trim_mode == "start":
        return y[:target_samples]

    if trim_mode == "center":
        start = (current_samples - target_samples) // 2
        end = start + target_samples
        return y[start:end]

    raise ValueError(f"trim_mode không hợp lệ: {trim_mode}")


def fix_time_frames(
    feature: np.ndarray,
    expected_frames: int,
) -> np.ndarray:
    """
    Đảm bảo số frame thời gian đúng với expected_frames.

    Với config hiện tại:
        sample_rate = 16000
        duration = 4.0
        target_samples = 64000
        hop_length = 256
        center = true

    Shape kỳ vọng:
        64 x 251

    Hàm này giúp tránh lỗi shape nếu librosa/STFT sinh lệch 1-2 frame.
    """
    n_mels, current_frames = feature.shape

    if current_frames == expected_frames:
        return feature

    if current_frames < expected_frames:
        pad_width = expected_frames - current_frames
        feature = np.pad(
            feature,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=0.0,
        )
        return feature

    # current_frames > expected_frames
    feature = feature[:, :expected_frames]
    return feature


def normalize_feature(
    feature: np.ndarray,
    normalization_config: Dict[str, Any],
) -> np.ndarray:
    """
    Chuẩn hóa đặc trưng Log-Mel.

    Hiện tại dùng sample_zscore:
        x = (x - mean) / (std + eps)

    Cách này chuẩn hóa từng audio riêng biệt,
    giúp giảm ảnh hưởng khác biệt về âm lượng.
    """
    enable = bool(normalization_config.get("enable", True))

    if not enable:
        return feature

    method = normalization_config.get("method", "sample_zscore")
    eps = float(normalization_config.get("eps", 1e-6))

    if method == "sample_zscore":
        mean = np.mean(feature)
        std = np.std(feature)
        feature = (feature - mean) / (std + eps)
        return feature

    if method == "minmax":
        min_value = np.min(feature)
        max_value = np.max(feature)
        feature = (feature - min_value) / (max_value - min_value + eps)
        return feature

    raise ValueError(f"Normalization method không hỗ trợ: {method}")


def extract_logmel_from_audio(
    audio_path: str | Path,
    audio_config: Dict[str, Any],
    feature_config: Dict[str, Any],
) -> np.ndarray:
    """
    Trích xuất Log-Mel Spectrogram từ một file audio.

    Returns:
        np.ndarray shape = [n_mels, expected_frames]
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy audio: {audio_path}")

    sample_rate = int(audio_config["sample_rate"])
    target_samples = int(audio_config["target_samples"])
    mono = bool(audio_config.get("mono", True))
    pad_mode = str(audio_config.get("pad_mode", "constant"))
    trim_mode = str(audio_config.get("trim_mode", "start"))

    n_mels = int(feature_config["n_mels"])
    n_fft = int(feature_config["n_fft"])
    hop_length = int(feature_config["hop_length"])
    win_length = int(feature_config["win_length"])
    center = bool(feature_config["center"])
    fmin = float(feature_config["fmin"])
    fmax = float(feature_config["fmax"])
    power = float(feature_config["power"])
    top_db = feature_config.get("top_db", 80.0)
    expected_frames = int(feature_config["expected_frames"])

    # Đọc audio và resample về sample_rate trong config.
    y, _ = librosa.load(
        path=audio_path,
        sr=sample_rate,
        mono=mono,
    )

    # Đảm bảo audio 1D float32.
    y = y.astype(np.float32)

    # Pad/trim về đúng 4 giây.
    y = pad_or_trim_audio(
        y=y,
        target_samples=target_samples,
        pad_mode=pad_mode,
        trim_mode=trim_mode,
    )

    # Tạo Mel-Spectrogram.
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=center,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    # Chuyển sang Log-Mel Spectrogram dạng dB.
    log_ref = feature_config.get("log_ref", "max")

    if log_ref == "max":
        logmel = librosa.power_to_db(
            mel,
            ref=np.max,
            top_db=top_db,
        )
    elif log_ref == "one":
        logmel = librosa.power_to_db(
            mel,
            ref=1.0,
            top_db=top_db,
        )
    else:
        raise ValueError(f"log_ref không hỗ trợ: {log_ref}")

    logmel = logmel.astype(np.float32)

    # Đảm bảo số frame đúng expected_frames.
    logmel = fix_time_frames(
        feature=logmel,
        expected_frames=expected_frames,
    )

    # Normalize.
    logmel = normalize_feature(
        feature=logmel,
        normalization_config=feature_config["normalization"],
    )

    logmel = logmel.astype(np.float32)

    expected_shape = (n_mels, expected_frames)

    if logmel.shape != expected_shape:
        raise ValueError(
            f"Shape Log-Mel không đúng. "
            f"Expected: {expected_shape}, Got: {logmel.shape}, File: {audio_path}"
        )

    return logmel


def save_feature(
    feature: np.ndarray,
    feature_path: str | Path,
    storage_dtype: str = "float16",
) -> None:
    """Lưu feature ra file .npy."""
    feature_path = Path(feature_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    if storage_dtype == "float16":
        feature = feature.astype(np.float16)
    elif storage_dtype == "float32":
        feature = feature.astype(np.float32)
    else:
        raise ValueError(f"storage_dtype không hỗ trợ: {storage_dtype}")

    np.save(feature_path, feature)


def load_feature_for_training(
    feature_path: str | Path,
    train_dtype: str = "float32",
    add_channel_dim: bool = True,
) -> np.ndarray:
    """
    Load feature .npy để train.

    Output mặc định:
        [n_mels, frames, 1]
    """
    feature_path = Path(feature_path)

    if not feature_path.exists():
        raise FileNotFoundError(f"Không tìm thấy feature: {feature_path}")

    feature = np.load(feature_path)

    if train_dtype == "float32":
        feature = feature.astype(np.float32)
    elif train_dtype == "float16":
        feature = feature.astype(np.float16)
    else:
        raise ValueError(f"train_dtype không hỗ trợ: {train_dtype}")

    if add_channel_dim:
        feature = np.expand_dims(feature, axis=-1)

    return feature


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    """Đọc metadata CSV."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_path}")

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


def process_metadata_file(
    metadata_path: Path,
    audio_config: Dict[str, Any],
    feature_config: Dict[str, Any],
    max_samples: Optional[int] = None,
    force: bool = False,
    strict_audio: bool = True,
) -> Dict[str, Any]:
    """
    Extract Log-Mel cho toàn bộ dòng trong một metadata CSV.

    Args:
        metadata_path: train.csv/dev.csv/eval.csv.
        audio_config: config audio.
        feature_config: config feature.
        max_samples: giới hạn số mẫu để debug.
        force: nếu True thì ghi đè feature đã tồn tại.
        strict_audio: nếu True, audio thiếu sẽ tính lỗi.

    Returns:
        Dict thống kê.
    """
    df = read_metadata(metadata_path)

    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    storage_dtype = str(feature_config.get("storage_dtype", "float16"))

    total = len(df)
    saved_count = 0
    skipped_count = 0
    error_count = 0
    missing_audio_count = 0

    error_rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=total,
        desc=f"Extracting {metadata_path.name}",
    ):
        utt_id = str(row["utt_id"])
        audio_path = Path(str(row["audio_path"]))
        feature_path = Path(str(row["feature_path"]))

        try:
            if feature_path.exists() and not force:
                skipped_count += 1
                continue

            if not audio_path.exists():
                missing_audio_count += 1

                message = f"Audio không tồn tại: {audio_path}"

                if strict_audio:
                    raise FileNotFoundError(message)

                error_rows.append(
                    {
                        "utt_id": utt_id,
                        "audio_path": str(audio_path),
                        "feature_path": str(feature_path),
                        "error": message,
                    }
                )
                continue

            feature = extract_logmel_from_audio(
                audio_path=audio_path,
                audio_config=audio_config,
                feature_config=feature_config,
            )

            save_feature(
                feature=feature,
                feature_path=feature_path,
                storage_dtype=storage_dtype,
            )

            saved_count += 1

        except Exception as e:
            error_count += 1

            error_rows.append(
                {
                    "utt_id": utt_id,
                    "audio_path": str(audio_path),
                    "feature_path": str(feature_path),
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )

    stats = {
        "metadata_path": str(metadata_path),
        "total_rows": total,
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "missing_audio_count": missing_audio_count,
        "force": force,
        "max_samples": max_samples,
        "error_rows": error_rows,
    }

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """In thống kê extract feature."""
    print("\n" + "=" * 70)
    print(f"Metadata: {stats['metadata_path']}")
    print("=" * 70)

    print(f"Total rows          : {stats['total_rows']}")
    print(f"Saved features      : {stats['saved_count']}")
    print(f"Skipped existing    : {stats['skipped_count']}")
    print(f"Errors              : {stats['error_count']}")
    print(f"Missing audio       : {stats['missing_audio_count']}")
    print(f"Force overwrite     : {stats['force']}")

    if stats["max_samples"] is not None:
        print(f"Max samples         : {stats['max_samples']}")

    if stats["error_rows"]:
        print("\nMột số lỗi đầu tiên:")
        for item in stats["error_rows"][:5]:
            print(f"  - utt_id={item['utt_id']}")
            print(f"    audio={item['audio_path']}")
            print(f"    error={item['error']}")


def get_debug_max_samples(
    config: Dict[str, Any],
    split: str,
) -> Optional[int]:
    """Lấy số mẫu debug theo split từ config."""
    debug_config = config.get("debug", {})

    if split == "train":
        return debug_config.get("max_train_samples")

    if split == "dev":
        return debug_config.get("max_dev_samples")

    if split == "eval":
        return debug_config.get("max_eval_samples")

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Log-Mel Spectrogram features from metadata CSV."
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
        choices=["train", "dev", "eval", "all"],
        help="Split cần extract feature.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu để debug nhanh.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dùng max sample trong config.debug.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè feature .npy nếu đã tồn tại.",
    )

    parser.add_argument(
        "--no-strict-audio",
        action="store_true",
        help="Không dừng hoặc tính lỗi nghiêm trọng khi thiếu audio.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    project_root = infer_project_root(config_path)

    audio_config = get_audio_config(config)
    feature_config = get_feature_config(config)

    print("Project root:", project_root)
    print("Config file :", config_path)

    print("\nAudio config:")
    print(f"  sample_rate    : {audio_config['sample_rate']}")
    print(f"  duration       : {audio_config['duration']}")
    print(f"  target_samples : {audio_config['target_samples']}")

    print("\nFeature config:")
    print(f"  n_mels         : {feature_config['n_mels']}")
    print(f"  n_fft          : {feature_config['n_fft']}")
    print(f"  hop_length     : {feature_config['hop_length']}")
    print(f"  expected_frames: {feature_config['expected_frames']}")
    print(f"  input_shape    : {feature_config['input_shape']}")
    print(f"  storage_dtype  : {feature_config['storage_dtype']}")

    splits = ["train", "dev", "eval"] if args.split == "all" else [args.split]

    strict_audio = not args.no_strict_audio

    all_stats = []

    for split in splits:
        metadata_path = get_metadata_path_by_split(
            config=config,
            project_root=project_root,
            split=split,
        )

        max_samples = args.max_samples

        if args.debug:
            max_samples = get_debug_max_samples(config, split)

        if max_samples is not None:
            max_samples = int(max_samples)

        stats = process_metadata_file(
            metadata_path=metadata_path,
            audio_config=audio_config,
            feature_config=feature_config,
            max_samples=max_samples,
            force=args.force,
            strict_audio=strict_audio,
        )

        print_stats(stats)
        all_stats.append(stats)

    total_saved = sum(item["saved_count"] for item in all_stats)
    total_skipped = sum(item["skipped_count"] for item in all_stats)
    total_errors = sum(item["error_count"] for item in all_stats)

    print("\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)
    print(f"Total saved   : {total_saved}")
    print(f"Total skipped : {total_skipped}")
    print(f"Total errors  : {total_errors}")

    if total_errors == 0:
        print("\nHoàn tất extract Log-Mel không có lỗi.")
    else:
        print("\nHoàn tất nhưng có lỗi. Hãy kiểm tra log in ra phía trên.")


if __name__ == "__main__":
    main()