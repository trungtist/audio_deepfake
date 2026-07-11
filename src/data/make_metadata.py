"""
make_metadata.py

Nhiệm vụ:
- Đọc protocol của ASVspoof2019 LA.
- Tạo metadata train/dev/eval dạng CSV.
- Mỗi dòng metadata gồm:
  split, speaker_id, utt_id, system_id, attack_key, label_text, label,
  audio_path, feature_path, audio_exists

Cách chạy từ thư mục gốc project:

    python -m src.data.make_metadata --config configs/cnn_baseline.yaml

Chạy debug ít mẫu:

    python -m src.data.make_metadata --config configs/cnn_baseline.yaml --debug
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


MetadataRow = Dict[str, Any]


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load file YAML config."""
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

    Khi đó project root là:
        audio-deepfake-detection/
    """
    config_path = config_path.resolve()

    if config_path.parent.name == "configs":
        return config_path.parent.parent

    return Path.cwd().resolve()


def resolve_path(project_root: Path, path_value: str) -> Path:
    """Chuyển path trong config thành absolute path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def get_label_mapping(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Lấy mapping label từ config.

    Ví dụ:
        bonafide -> 0
        spoof    -> 1
    """
    labels_config = config.get("labels", {})

    real_name = labels_config.get("real", {}).get("name", "bonafide")
    real_value = labels_config.get("real", {}).get("value", 0)

    fake_name = labels_config.get("fake", {}).get("name", "spoof")
    fake_value = labels_config.get("fake", {}).get("value", 1)

    return {
        real_name: int(real_value),
        fake_name: int(fake_value),
    }


def read_asvspoof_protocol(
    protocol_path: Path,
    audio_dir: Path,
    feature_dir: Path,
    split: str,
    label_mapping: Dict[str, int],
    max_samples: int | None = None,
    strict_audio: bool = False,
) -> Tuple[List[MetadataRow], Dict[str, Any]]:
    """
    Đọc protocol ASVspoof2019 LA và tạo danh sách metadata.

    Format protocol thường là:
        speaker_id utt_id system_id attack_key label

    Ví dụ:
        LA_0079 LA_T_1138215 - - bonafide
        LA_0079 LA_T_1271820 A01 spoof spoof

    Trong đó:
        label = bonafide hoặc spoof
    """
    if not protocol_path.exists():
        raise FileNotFoundError(f"Không tìm thấy protocol file: {protocol_path}")

    feature_dir.mkdir(parents=True, exist_ok=True)

    rows: List[MetadataRow] = []

    total_lines = 0
    skipped_invalid_format = 0
    skipped_unknown_label = 0
    skipped_missing_audio = 0

    label_counter: Counter[str] = Counter()
    audio_missing_examples: List[str] = []

    with open(protocol_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total_lines += 1
            parts = line.split()

            if len(parts) < 5:
                skipped_invalid_format += 1
                continue

            speaker_id = parts[0]
            utt_id = parts[1]
            system_id = parts[2]
            attack_key = parts[3]
            label_text = parts[-1]

            if label_text not in label_mapping:
                skipped_unknown_label += 1
                continue

            label = label_mapping[label_text]

            audio_path = audio_dir / f"{utt_id}.flac"
            feature_path = feature_dir / f"{utt_id}.npy"

            audio_exists = audio_path.exists()

            if not audio_exists:
                skipped_missing_audio += 1

                if len(audio_missing_examples) < 5:
                    audio_missing_examples.append(str(audio_path))

                if strict_audio:
                    continue

            rows.append(
                {
                    "split": split,
                    "speaker_id": speaker_id,
                    "utt_id": utt_id,
                    "system_id": system_id,
                    "attack_key": attack_key,
                    "label_text": label_text,
                    "label": label,
                    "audio_path": audio_path.as_posix(),
                    "feature_path": feature_path.as_posix(),
                    "audio_exists": int(audio_exists),
                }
            )

            label_counter[label_text] += 1

            if max_samples is not None and len(rows) >= max_samples:
                break

    stats = {
        "split": split,
        "protocol_path": protocol_path.as_posix(),
        "audio_dir": audio_dir.as_posix(),
        "feature_dir": feature_dir.as_posix(),
        "total_lines_read": total_lines,
        "total_rows_created": len(rows),
        "label_distribution": dict(label_counter),
        "skipped_invalid_format": skipped_invalid_format,
        "skipped_unknown_label": skipped_unknown_label,
        "missing_audio_count": skipped_missing_audio,
        "missing_audio_examples": audio_missing_examples,
        "strict_audio": strict_audio,
        "max_samples": max_samples,
    }

    return rows, stats


def write_metadata_csv(rows: List[MetadataRow], output_path: Path) -> None:
    """Ghi metadata ra file CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"Không có dữ liệu để ghi: {output_path}")

    fieldnames = [
        "split",
        "speaker_id",
        "utt_id",
        "system_id",
        "attack_key",
        "label_text",
        "label",
        "audio_path",
        "feature_path",
        "audio_exists",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(stats: Dict[str, Any]) -> None:
    """In thống kê sau khi tạo metadata."""
    print("\n" + "=" * 70)
    print(f"Split: {stats['split']}")
    print("=" * 70)

    print(f"Protocol: {stats['protocol_path']}")
    print(f"Audio dir: {stats['audio_dir']}")
    print(f"Feature dir: {stats['feature_dir']}")

    print(f"Total lines read     : {stats['total_lines_read']}")
    print(f"Total rows created   : {stats['total_rows_created']}")
    print(f"Label distribution   : {stats['label_distribution']}")

    print(f"Invalid format skipped: {stats['skipped_invalid_format']}")
    print(f"Unknown label skipped : {stats['skipped_unknown_label']}")
    print(f"Missing audio count   : {stats['missing_audio_count']}")

    if stats["max_samples"] is not None:
        print(f"Debug max samples     : {stats['max_samples']}")

    if stats["missing_audio_examples"]:
        print("\nMột số audio bị thiếu:")
        for path in stats["missing_audio_examples"]:
            print(f"  - {path}")


def build_split_metadata(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    label_mapping: Dict[str, int],
    debug: bool = False,
    strict_audio: bool = False,
) -> Tuple[List[MetadataRow], Dict[str, Any], Path]:
    """Tạo metadata cho một split: train/dev/eval."""
    paths_config = config["paths"]
    debug_config = config.get("debug", {})

    if split == "train":
        protocol_path = resolve_path(project_root, paths_config["train_protocol"])
        audio_dir = resolve_path(project_root, paths_config["train_audio_dir"])
        feature_dir = resolve_path(project_root, paths_config["train_feature_dir"])
        metadata_path = resolve_path(project_root, paths_config["train_metadata"])
        max_samples = debug_config.get("max_train_samples") if debug else None

    elif split == "dev":
        protocol_path = resolve_path(project_root, paths_config["dev_protocol"])
        audio_dir = resolve_path(project_root, paths_config["dev_audio_dir"])
        feature_dir = resolve_path(project_root, paths_config["dev_feature_dir"])
        metadata_path = resolve_path(project_root, paths_config["dev_metadata"])
        max_samples = debug_config.get("max_dev_samples") if debug else None

    elif split == "eval":
        protocol_path = resolve_path(project_root, paths_config["eval_protocol"])
        audio_dir = resolve_path(project_root, paths_config["eval_audio_dir"])
        feature_dir = resolve_path(project_root, paths_config["eval_feature_dir"])
        metadata_path = resolve_path(project_root, paths_config["eval_metadata"])
        max_samples = debug_config.get("max_eval_samples") if debug else None

    else:
        raise ValueError(f"Split không hợp lệ: {split}")

    if max_samples is not None:
        max_samples = int(max_samples)

    rows, stats = read_asvspoof_protocol(
        protocol_path=protocol_path,
        audio_dir=audio_dir,
        feature_dir=feature_dir,
        split=split,
        label_mapping=label_mapping,
        max_samples=max_samples,
        strict_audio=strict_audio,
    )

    return rows, stats, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metadata CSV files for ASVspoof2019 LA."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_baseline.yaml",
        help="Đường dẫn tới file config YAML.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "dev", "eval", "all"],
        help="Split cần tạo metadata.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Chỉ tạo metadata với số lượng mẫu nhỏ theo config.debug.",
    )

    parser.add_argument(
        "--strict-audio",
        action="store_true",
        help="Nếu bật, các dòng có audio không tồn tại sẽ bị bỏ qua.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)

    project_root = infer_project_root(config_path)
    label_mapping = get_label_mapping(config)

    print("Project root:", project_root)
    print("Config file :", config_path)
    print("Label mapping:", label_mapping)

    if args.debug:
        print("\nĐang chạy DEBUG mode: metadata sẽ bị giới hạn số mẫu.")
    else:
        print("\nĐang chạy FULL mode: tạo metadata đầy đủ.")

    splits = ["train", "dev", "eval"] if args.split == "all" else [args.split]

    for split in splits:
        rows, stats, metadata_path = build_split_metadata(
            config=config,
            project_root=project_root,
            split=split,
            label_mapping=label_mapping,
            debug=args.debug,
            strict_audio=args.strict_audio,
        )

        write_metadata_csv(rows, metadata_path)
        print_stats(stats)

        print(f"\nĐã lưu metadata: {metadata_path}")
        print(f"Số dòng: {len(rows)}")

    print("\nHoàn tất tạo metadata.")


if __name__ == "__main__":
    main()