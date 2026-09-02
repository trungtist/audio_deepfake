"""
Export trained Keras models to TensorFlow Lite.

Examples:
    python -m src.deployment.export_tflite --config configs/cnn_v2.yaml --model-path outputs/checkpoints/cnn_v2/best_cnn_v2.keras --formats fp32 fp16 dynamic

    python -m src.deployment.export_tflite --config configs/mobilenetv3_small.yaml --model-path outputs/checkpoints/mobilenetv3_small/best_mobilenetv3_small.keras --formats fp32 fp16 dynamic
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import tensorflow as tf
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Đọc và parse file YAML thành dictionary.
    
    Args:
        path (str | Path): Đường dẫn tới file YAML.
        
    Returns:
        Dict[str, Any]: Dictionary chứa dữ liệu từ file config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_get(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """
    Lấy giá trị từ dictionary lồng nhau (nested dictionary) một cách an toàn.
    
    Args:
        data (Dict[str, Any]): Dictionary chứa dữ liệu gốc.
        keys (Iterable[str]): Danh sách các key theo thứ tự lồng nhau.
        default (Any, optional): Giá trị trả về mặc định nếu không tìm thấy key. Mặc định là None.
        
    Returns:
        Any: Giá trị tìm được hoặc giá trị mặc định.
    """
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def slugify(value: str) -> str:
    """
    Chuyển đổi chuỗi thành định dạng an toàn cho tên file (chỉ chứa chữ cái thường, số, dấu gạch dưới).
    
    Args:
        value (str): Chuỗi đầu vào.
        
    Returns:
        str: Chuỗi sau khi đã được chuẩn hóa.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "model"


def infer_model_id(cfg: Dict[str, Any], config_path: Path) -> str:
    """
    Tự động suy luận ID/tên của mô hình dựa trên nội dung file config YAML.
    
    Args:
        cfg (Dict[str, Any]): Nội dung file config đã được parse.
        config_path (Path): Đường dẫn tới file config để dùng làm tên dự phòng (fallback).
        
    Returns:
        str: ID của mô hình (đã được chuẩn hóa dạng slug).
    """
    candidates = [
        deep_get(cfg, ["project", "model_name"]),
        deep_get(cfg, ["project", "name"]),
        deep_get(cfg, ["model", "id"]),
        deep_get(cfg, ["model", "model_name"]),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return slugify(item)
    return slugify(config_path.stem)


def infer_checkpoint_path(cfg: Dict[str, Any], model_id: str) -> Optional[Path]:
    """
    Tự động tìm kiếm đường dẫn đến file checkpoint (.keras) tốt nhất dựa trên config.
    
    Args:
        cfg (Dict[str, Any]): Nội dung file config.
        model_id (str): ID của mô hình để tìm kiếm trong cấu trúc thư mục mặc định.
        
    Returns:
        Optional[Path]: Đường dẫn đến file checkpoint nếu tìm thấy, ngược lại trả về None.
    """
    direct_candidates = [
        deep_get(cfg, ["training", "checkpoint_path"]),
        deep_get(cfg, ["training", "best_model_path"]),
        deep_get(cfg, ["checkpoint", "path"]),
        deep_get(cfg, ["checkpoint", "best_model_path"]),
        deep_get(cfg, ["paths", "best_model_path"]),
        deep_get(cfg, ["output", "best_model_path"]),
        deep_get(cfg, ["outputs", "best_model_path"]),
    ]
    for item in direct_candidates:
        if isinstance(item, str) and item.strip():
            p = Path(item)
            if p.exists():
                return p

    checkpoint_dir_candidates = [
        deep_get(cfg, ["paths", "checkpoint_dir"]),
        deep_get(cfg, ["output", "checkpoint_dir"]),
        deep_get(cfg, ["outputs", "checkpoint_dir"]),
        deep_get(cfg, ["checkpoint", "dir"]),
        deep_get(cfg, ["checkpoint", "checkpoint_dir"]),
        deep_get(cfg, ["project", "checkpoint_dir"]),
        "outputs/checkpoints/" + model_id,
    ]
    filenames = [
        f"best_{model_id}.keras",
        f"{model_id}_best.keras",
        "best_model.keras",
        "best.keras",
    ]
    for dir_item in checkpoint_dir_candidates:
        if not isinstance(dir_item, str) or not dir_item.strip():
            continue
        d = Path(dir_item)
        if not d.exists() or not d.is_dir():
            continue
        for name in filenames:
            p = d / name
            if p.exists():
                return p
        keras_files = sorted(d.glob("*.keras"))
        best_files = [p for p in keras_files if "best" in p.name.lower()]
        if best_files:
            return best_files[0]
        if len(keras_files) == 1:
            return keras_files[0]
    return None


def get_file_size_mb(path: str | Path) -> float:
    """Tính toán kích thước của file theo đơn vị Megabytes (MB)."""
    return Path(path).stat().st_size / (1024 * 1024)


def load_keras_model(model_path: Path) -> tf.keras.Model:
    """
    Load mô hình Keras từ file. Tránh biên dịch lại (compile=False) vì chỉ phục vụ mục đích export.
    
    Args:
        model_path (Path): Đường dẫn tới file mô hình .keras.
        
    Returns:
        tf.keras.Model: Đối tượng mô hình TensorFlow Keras.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def make_converter(model: tf.keras.Model, export_type: str, allow_select_tf_ops: bool = False) -> tf.lite.TFLiteConverter:
    """
    Khởi tạo TFLiteConverter từ mô hình Keras và cấu hình các tùy chọn lượng tử hóa (quantization).
    
    Args:
        model (tf.keras.Model): Mô hình Keras đầu vào cần chuyển đổi.
        export_type (str): Định dạng lượng tử hóa mong muốn ('fp32', 'fp16', 'dynamic').
        allow_select_tf_ops (bool): Cờ cho phép dùng các Ops gốc của TensorFlow nếu TFLite built-in bị thiếu.
        
    Returns:
        tf.lite.TFLiteConverter: Đối tượng converter đã được thiết lập các thông số.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if export_type == "fp32":
        pass
    elif export_type == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif export_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        raise ValueError(f"Unsupported export type: {export_type}")

    if allow_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

    return converter


def export_one(model: tf.keras.Model, output_path: Path, export_type: str, allow_select_tf_ops: bool = False) -> Dict[str, Any]:
    """
    Thực hiện tiến trình chuyển đổi mô hình cho một định dạng duy nhất và lưu file .tflite ra đĩa.
    
    Args:
        model (tf.keras.Model): Mô hình Keras đã nạp sẵn.
        output_path (Path): Đường dẫn đích để lưu file TFLite.
        export_type (str): Định dạng lượng tử hóa.
        allow_select_tf_ops (bool): Cho phép chọn TF ops thay thế.
        
    Returns:
        Dict[str, Any]: Từ điển chứa thông tin tóm tắt về loại xuất, đường dẫn lưu và dung lượng file MB.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = make_converter(model, export_type, allow_select_tf_ops)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return {
        "type": export_type,
        "path": str(output_path),
        "size_mb": round(get_file_size_mb(output_path), 4),
    }


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh (arguments) do người dùng truyền vào khi chạy file."""
    parser = argparse.ArgumentParser(description="Export trained Keras model to TensorFlow Lite.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model-path", default=None, help="Path to trained .keras checkpoint.")
    parser.add_argument("--output-dir", default=None, help="Output directory for .tflite files.")
    parser.add_argument("--formats", nargs="+", default=["fp32", "fp16", "dynamic"], choices=["fp32", "fp16", "dynamic"])
    parser.add_argument("--allow-select-tf-ops", action="store_true", help="Use only if conversion fails with builtin ops.")
    return parser.parse_args()


def main() -> None:
    """
    Hàm thực thi chính:
    1. Đọc argument & config.
    2. Xác định mô hình (.keras) đích.
    3. Trích xuất mô hình và thông tin cấu hình mạng.
    4. Vòng lặp chuyển đổi cho từng định dạng mong muốn.
    5. Lưu tổng kết dưới dạng file JSON manifest.
    """
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    model_id = infer_model_id(cfg, config_path)

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = infer_checkpoint_path(cfg, model_id)
        if model_path is None:
            raise FileNotFoundError("Could not infer checkpoint path. Please pass --model-path explicitly.")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "tflite" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TFLITE EXPORT")
    print("=" * 80)
    print(f"Config     : {config_path}")
    print(f"Model ID   : {model_id}")
    print(f"Model path : {model_path}")
    print(f"Output dir : {output_dir}")
    print(f"Formats    : {args.formats}")

    model = load_keras_model(model_path)

    print("-" * 80)
    print("Loaded Keras model")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Params      : {model.count_params():,}")
    print(f"Keras size  : {get_file_size_mb(model_path):.4f} MB")
    print("-" * 80)

    results: List[Dict[str, Any]] = []
    for fmt in args.formats:
        out_path = output_dir / f"{model_id}_{fmt}.tflite"
        print(f"Exporting {fmt} -> {out_path}")
        result = export_one(
            model=model,
            output_path=out_path,
            export_type=fmt,
            allow_select_tf_ops=args.allow_select_tf_ops,
        )
        results.append(result)
        print(f"Saved {fmt}: {result['size_mb']:.4f} MB")

    manifest = {
        "config": str(config_path),
        "model_id": model_id,
        "keras_model_path": str(model_path),
        "keras_size_mb": round(get_file_size_mb(model_path), 4),
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "params": int(model.count_params()),
        "exports": results,
    }

    manifest_path = output_dir / f"{model_id}_tflite_export_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"Keras model: {manifest['keras_size_mb']:.4f} MB")
    for item in results:
        print(f"{item['type']:>8}: {item['size_mb']:.4f} MB | {item['path']}")
    print(f"Summary JSON: {manifest_path}")


if __name__ == "__main__":
    main()
