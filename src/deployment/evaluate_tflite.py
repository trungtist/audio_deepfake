"""
Evaluate a TensorFlow Lite model on ASVspoof2019 LA feature files.

This script evaluates already-exported .tflite models using the metadata CSV files
created in the project pipeline. It supports both:
  - CNN v2 input:            (64, 251, 1)
  - MobileNetV3-Small input: (128, 128, 3)

Examples
--------
CNN v2, eval split, FP16 TFLite:
    python -m src.deployment.evaluate_tflite --config configs/cnn_v2.yaml --split eval --format fp16

MobileNetV3-Small, eval split, FP16 TFLite:
    python -m src.deployment.evaluate_tflite --config configs/mobilenetv3_small.yaml --split eval --format fp16

Override threshold explicitly:
    python -m src.deployment.evaluate_tflite --config configs/cnn_v2.yaml --split eval --format fp16 --threshold 0.869

Evaluate a custom .tflite path:
    python -m src.deployment.evaluate_tflite --config configs/cnn_v2.yaml --split eval --tflite-path outputs/tflite/cnn_v2/cnn_v2_fp32.tflite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.evaluate import plot_confusion_matrix, plot_roc_curve


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_get(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "model"


def infer_model_id(cfg: Dict[str, Any], config_path: Path) -> str:
    candidates = [
        deep_get(cfg, ["project", "model_name"]),
        deep_get(cfg, ["model", "model_name"]),
        deep_get(cfg, ["model", "id"]),
        config_path.stem,
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return slugify(item)
    return "model"


def infer_metadata_path(cfg: Dict[str, Any], split: str) -> Path:
    key = f"{split}_metadata"
    path = deep_get(cfg, ["paths", key])
    if isinstance(path, str) and path.strip():
        return Path(path)

    # Conservative fallback.
    return Path("data") / "metadata" / f"{split}.csv"


def infer_result_dir(cfg: Dict[str, Any], model_id: str) -> Path:
    candidates = [
        deep_get(cfg, ["paths", "result_dir"]),
        deep_get(cfg, ["output", "result_dir"]),
        deep_get(cfg, ["outputs", "result_dir"]),
    ]
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return Path(item) / "tflite"
    return Path("outputs") / "results" / model_id / "tflite"


def infer_tflite_path(cfg: Dict[str, Any], model_id: str, fmt: str) -> Path:
    # Prefer explicit config output_paths when available.
    config_path = deep_get(cfg, ["deployment", "tflite", "output_paths", fmt])
    if isinstance(config_path, str) and config_path.strip():
        p = Path(config_path)
        if p.exists():
            return p

    output_dir_candidates = [
        deep_get(cfg, ["deployment", "tflite", "output_dir"]),
        deep_get(cfg, ["paths", "tflite_dir"]),
        str(Path("outputs") / "tflite" / model_id),
    ]

    candidate_names = [
        f"{model_id}_{fmt}.tflite",
        f"{model_id}_{fmt.lower()}.tflite",
    ]
    if fmt == "dynamic":
        candidate_names.extend([
            f"{model_id}_dynamic_range.tflite",
            f"{model_id}_dynamic.tflite",
        ])

    for item in output_dir_candidates:
        if not isinstance(item, str) or not item.strip():
            continue
        d = Path(item)
        for name in candidate_names:
            p = d / name
            if p.exists():
                return p

    # Return expected path even if missing, so error message is concrete.
    return Path("outputs") / "tflite" / model_id / f"{model_id}_{fmt}.tflite"


def infer_threshold(cfg: Dict[str, Any]) -> float:
    candidates = [
        deep_get(cfg, ["deployment", "inference_threshold"]),
        deep_get(cfg, ["deployment", "threshold"]),
        deep_get(cfg, ["evaluation", "threshold"]),
        deep_get(cfg, ["evaluation", "inference_threshold"]),
        deep_get(cfg, ["training", "best_threshold"]),
        deep_get(cfg, ["labels", "threshold_default"]),
        deep_get(cfg, ["training", "threshold"]),
    ]
    for item in candidates:
        if item is None:
            continue
        try:
            return float(item)
        except (TypeError, ValueError):
            continue
    return 0.5


# -----------------------------------------------------------------------------
# Metadata and feature loading
# -----------------------------------------------------------------------------


LABEL_COL_CANDIDATES = ["label", "target", "class", "y", "is_spoof", "cm_label"]
FEATURE_COL_CANDIDATES = [
    "feature_path",
    "npy_path",
    "logmel_path",
    "mobilenet_feature_path",
    "feature_file",
    "path",
]
UTT_ID_CANDIDATES = ["utterance_id", "utt_id", "file_id", "filename", "audio_id", "trial_id"]


def find_first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    return None


def normalize_label(value: Any) -> int:
    """Map bonafide/real to 0 and spoof/fake to 1."""
    if pd.isna(value):
        raise ValueError("Label is NaN")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)

    s = str(value).strip().lower()
    if s in {"0", "0.0", "bonafide", "bona-fide", "real", "genuine", "human", "authentic"}:
        return 0
    if s in {"1", "1.0", "spoof", "fake", "deepfake", "synthetic", "attack"}:
        return 1
    raise ValueError(f"Unsupported label value: {value!r}")


def resolve_path(path_value: Any, project_root: Path) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return project_root / p


def infer_feature_path_from_row(row: pd.Series, cfg: Dict[str, Any], split: str, project_root: Path) -> Path:
    """Fallback if metadata does not contain feature_path."""
    utt_col = None
    for col in UTT_ID_CANDIDATES:
        if col in row.index:
            utt_col = col
            break
    if utt_col is None:
        # Case-insensitive fallback.
        lower_map = {str(c).lower(): c for c in row.index}
        for col in UTT_ID_CANDIDATES:
            if col.lower() in lower_map:
                utt_col = lower_map[col.lower()]
                break
    if utt_col is None:
        raise KeyError(
            "Cannot infer feature path: metadata has no feature_path column and no utterance_id-like column."
        )

    utt_id = str(row[utt_col]).strip()
    if utt_id.endswith(".flac") or utt_id.endswith(".wav"):
        utt_id = Path(utt_id).stem

    feature_dir = deep_get(cfg, ["paths", f"{split}_feature_dir"])
    if not isinstance(feature_dir, str) or not feature_dir.strip():
        raise KeyError(f"Cannot infer feature path: paths.{split}_feature_dir is missing in config.")

    return resolve_path(Path(feature_dir) / f"{utt_id}.npy", project_root)


@dataclass
class EvalRows:
    feature_paths: List[Path]
    labels: np.ndarray
    utterance_ids: List[str]


def load_eval_rows(metadata_path: Path, cfg: Dict[str, Any], split: str, project_root: Path, max_samples: Optional[int]) -> EvalRows:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples).copy()

    label_col = find_first_column(df, LABEL_COL_CANDIDATES)
    if label_col is None:
        raise KeyError(f"Cannot find label column in metadata. Columns: {list(df.columns)}")

    feature_col = find_first_column(df, FEATURE_COL_CANDIDATES)
    utt_col = find_first_column(df, UTT_ID_CANDIDATES)

    feature_paths: List[Path] = []
    labels: List[int] = []
    utterance_ids: List[str] = []

    for _, row in df.iterrows():
        labels.append(normalize_label(row[label_col]))

        if feature_col is not None:
            p = resolve_path(row[feature_col], project_root)
            # If a generic path column points to raw audio, infer .npy instead.
            if p.suffix.lower() not in {".npy", ".npz"}:
                p = infer_feature_path_from_row(row, cfg, split, project_root)
        else:
            p = infer_feature_path_from_row(row, cfg, split, project_root)
        feature_paths.append(p)

        if utt_col is not None:
            utterance_ids.append(str(row[utt_col]))
        else:
            utterance_ids.append(p.stem)

    return EvalRows(
        feature_paths=feature_paths,
        labels=np.asarray(labels, dtype=np.int32),
        utterance_ids=utterance_ids,
    )


def prepare_feature(x: np.ndarray, expected_input_shape: Sequence[int]) -> np.ndarray:
    """Prepare one feature sample without batch dimension."""
    x = np.asarray(x)

    # Convert NPZ if needed.
    if isinstance(x, np.lib.npyio.NpzFile):
        if "arr_0" in x:
            x = x["arr_0"]
        else:
            first_key = list(x.keys())[0]
            x = x[first_key]

    # CNN features can be [64, 251], model expects [64, 251, 1].
    if len(expected_input_shape) == 3 and expected_input_shape[-1] == 1 and x.ndim == 2:
        x = np.expand_dims(x, axis=-1)

    # Occasionally a saved feature may include an extra batch dimension.
    if x.ndim == len(expected_input_shape) + 1 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)

    if tuple(x.shape) != tuple(expected_input_shape):
        raise ValueError(f"Feature shape mismatch. Expected {tuple(expected_input_shape)}, got {x.shape}")

    return x.astype(np.float32, copy=False)


def iter_feature_batches(
    feature_paths: Sequence[Path],
    input_shape: Sequence[int],
    batch_size: int,
) -> Iterable[Tuple[np.ndarray, List[Path]]]:
    batch: List[np.ndarray] = []
    paths: List[Path] = []

    for p in feature_paths:
        if not p.exists():
            raise FileNotFoundError(f"Feature file not found: {p}")
        loaded = np.load(p, allow_pickle=False)
        x = prepare_feature(loaded, input_shape)
        batch.append(x)
        paths.append(p)

        if len(batch) == batch_size:
            yield np.stack(batch, axis=0).astype(np.float32, copy=False), paths
            batch = []
            paths = []

    if batch:
        yield np.stack(batch, axis=0).astype(np.float32, copy=False), paths


# -----------------------------------------------------------------------------
# TFLite inference
# -----------------------------------------------------------------------------


class TFLiteRunner:
    def __init__(self, tflite_path: Path, num_threads: int = 1):
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
        self.tflite_path = tflite_path
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1:
            raise ValueError(f"Expected 1 input tensor, got {len(self.input_details)}")
        if len(self.output_details) != 1:
            raise ValueError(f"Expected 1 output tensor, got {len(self.output_details)}")

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.input_dtype = self.input_details[0]["dtype"]
        self.output_dtype = self.output_details[0]["dtype"]
        self.shape_signature = self.input_details[0].get("shape_signature", self.input_details[0]["shape"])

        # Allocate once for batch size 1 to read static info.
        self.interpreter.allocate_tensors()

    @property
    def input_shape_no_batch(self) -> Tuple[int, ...]:
        sig = np.asarray(self.shape_signature).tolist()
        shape = self.input_details[0]["shape"].tolist()
        ref = sig if len(sig) > 0 else shape
        no_batch = ref[1:]
        if any(int(v) <= 0 for v in no_batch):
            no_batch = shape[1:]
        return tuple(int(v) for v in no_batch)

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        batch_shape = list(x.shape)
        # Resize supports dynamic batch size. strict=False is more permissive across TF versions.
        self.interpreter.resize_tensor_input(self.input_index, batch_shape, strict=False)
        self.interpreter.allocate_tensors()

        if self.input_dtype != np.float32:
            # Most FP32/FP16 TFLite models still accept float32 input. This branch is here
            # for future int8/dynamic models if the input tensor changes dtype.
            x = x.astype(self.input_dtype, copy=False)
        else:
            x = x.astype(np.float32, copy=False)

        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)
        y = np.asarray(y).reshape(-1)
        return y.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Compute EER for positive class score where higher score means spoof."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    # Use threshold candidates from actual scores for exactness; cap if huge for speed.
    unique_scores = np.unique(y_score)
    if unique_scores.size > 5000:
        thresholds = np.linspace(float(np.min(y_score)), float(np.max(y_score)), 5001)
    else:
        thresholds = unique_scores

    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (y_score >= thr).astype(np.int32)
        val = f1_score(y_true, pred, zero_division=0)
        if val > best_f1:
            best_f1 = float(val)
            best_thr = float(thr)
    return best_thr, best_f1


def best_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thresholds[idx]), float(j[idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    eer, eer_thr = compute_eer(y_true, y_score)
    f1_thr, f1_best = best_f1_threshold(y_true, y_score)
    youden_thr, youden_j = best_youden_threshold(y_true, y_score)

    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    return {
        "threshold_used": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "best_f1": float(f1_best),
        "best_f1_threshold": float(f1_thr),
        "best_youden_threshold": float(youden_thr),
        "best_youden_j": float(youden_j),
        "confusion_matrix": cm.astype(int).tolist(),
        "pred_counts": {
            "bonafide_0": int(np.sum(y_pred == 0)),
            "spoof_1": int(np.sum(y_pred == 1)),
        },
        "label_counts": {
            "bonafide_0": int(np.sum(y_true == 0)),
            "spoof_1": int(np.sum(y_true == 1)),
        },
        "score_stats": {
            "min": float(np.min(y_score)),
            "max": float(np.max(y_score)),
            "mean": float(np.mean(y_score)),
            "std": float(np.std(y_score)),
        },
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_scores_csv(path: Path, utterance_ids: Sequence[str], y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "label", "score_spoof", "score_bonafide", "pred"])
        for utt, label, score, pred in zip(utterance_ids, y_true, y_score, y_pred):
            writer.writerow([utt, int(label), float(score), float(1.0 - score), int(pred)])


def print_summary(metrics: Dict[str, Any]) -> None:
    print("-" * 80)
    print("METRICS")
    print("-" * 80)
    print(f"Threshold : {metrics['threshold_used']:.6f}")
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")
    print(f"F1-score  : {metrics['f1']:.6f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.6f}")
    print(f"EER       : {metrics['eer']:.6f} | threshold={metrics['eer_threshold']:.6f}")
    print(f"Best F1   : {metrics['best_f1']:.6f} | threshold={metrics['best_f1_threshold']:.6f}")
    print(f"Best Youden threshold: {metrics['best_youden_threshold']:.6f}")
    print("Confusion matrix [[bonafide->bonafide, bonafide->spoof], [spoof->bonafide, spoof->spoof]]:")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"Label counts: {metrics['label_counts']}")
    print(f"Pred counts : {metrics['pred_counts']}")
    print(f"Score stats : {metrics['score_stats']}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TensorFlow Lite audio deepfake model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", default="eval", choices=["train", "dev", "eval"], help="Dataset split to evaluate.")
    parser.add_argument("--format", default="fp16", choices=["fp32", "fp16", "dynamic"], help="TFLite export format to evaluate.")
    parser.add_argument("--tflite-path", default=None, help="Explicit .tflite path. Overrides --format inference.")
    parser.add_argument("--metadata", default=None, help="Explicit metadata CSV path. Overrides config path.")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold. Defaults to config deployment.inference_threshold or 0.5.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for TFLite inference.")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of CPU threads for TFLite interpreter.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick testing.")
    parser.add_argument("--output-dir", default=None, help="Output directory for result JSON and score CSV.")
    parser.add_argument("--no-save-scores", action="store_true", help="Do not save per-utterance score CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    project_root = Path.cwd()
    cfg = load_yaml(config_path)
    model_id = infer_model_id(cfg, config_path)

    metadata_path = Path(args.metadata) if args.metadata else infer_metadata_path(cfg, args.split)
    tflite_path = Path(args.tflite_path) if args.tflite_path else infer_tflite_path(cfg, model_id, args.format)
    threshold = float(args.threshold) if args.threshold is not None else infer_threshold(cfg)

    output_dir = Path(args.output_dir) if args.output_dir else infer_result_dir(cfg, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TFLITE EVALUATION")
    print("=" * 80)
    print(f"Config      : {config_path}")
    print(f"Model ID    : {model_id}")
    print(f"Split       : {args.split}")
    print(f"Metadata    : {metadata_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"TFLite size : {file_size_mb(tflite_path):.4f} MB")
    print(f"Threshold   : {threshold}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Num threads : {args.num_threads}")

    runner = TFLiteRunner(tflite_path=tflite_path, num_threads=args.num_threads)
    input_shape = runner.input_shape_no_batch
    print(f"Input shape : {input_shape}")
    print(f"Input dtype : {runner.input_dtype}")
    print(f"Output dtype: {runner.output_dtype}")

    rows = load_eval_rows(
        metadata_path=metadata_path,
        cfg=cfg,
        split=args.split,
        project_root=project_root,
        max_samples=args.max_samples,
    )
    n_samples = len(rows.labels)
    print(f"Samples     : {n_samples}")
    print("-" * 80)

    scores: List[np.ndarray] = []
    total_infer_time = 0.0
    batch_count = 0
    processed = 0

    start_total = time.perf_counter()
    for x_batch, _paths in iter_feature_batches(rows.feature_paths, input_shape, args.batch_size):
        start_infer = time.perf_counter()
        y_batch = runner.predict_batch(x_batch)
        total_infer_time += time.perf_counter() - start_infer

        scores.append(y_batch)
        processed += len(y_batch)
        batch_count += 1

        if batch_count % 50 == 0 or processed == n_samples:
            print(f"Processed {processed}/{n_samples} samples")

    total_time = time.perf_counter() - start_total
    y_score = np.concatenate(scores, axis=0).astype(np.float32)
    y_true = rows.labels.astype(np.int32)

    metrics = compute_metrics(y_true=y_true, y_score=y_score, threshold=threshold)
    metrics["tflite_size_mb"] = round(file_size_mb(tflite_path), 4)
    metrics["num_samples"] = int(n_samples)
    metrics["total_time_sec_including_io"] = float(total_time)
    metrics["total_inference_time_sec"] = float(total_infer_time)
    metrics["avg_inference_ms_per_sample"] = float(total_infer_time / max(n_samples, 1) * 1000.0)
    metrics["avg_total_ms_per_sample_including_io"] = float(total_time / max(n_samples, 1) * 1000.0)
    metrics["batch_size"] = int(args.batch_size)
    metrics["num_threads"] = int(args.num_threads)
    metrics["tflite_path"] = str(tflite_path)
    metrics["metadata_path"] = str(metadata_path)
    metrics["config_path"] = str(config_path)
    metrics["split"] = args.split
    metrics["format"] = args.format

    y_pred = (y_score >= threshold).astype(np.int32)

    result_name = f"{model_id}_{args.format}_{args.split}_tflite_metrics.json"
    result_path = output_dir / result_name
    result_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_save_scores:
        score_path = output_dir / f"{model_id}_{args.format}_{args.split}_tflite_scores.csv"
        save_scores_csv(score_path, rows.utterance_ids, y_true, y_score, y_pred)
    else:
        score_path = None

    cm_path = output_dir / f"{model_id}_{args.format}_{args.split}_tflite_confusion_matrix.png"
    plot_confusion_matrix(
        confusion_matrix_values=metrics["confusion_matrix"],
        output_path=cm_path,
        title=f"Confusion Matrix ({model_id} TFLite {args.format})"
    )

    roc_path = output_dir / f"{model_id}_{args.format}_{args.split}_tflite_roc_curve.png"
    plot_roc_curve(
        y_true=y_true,
        y_score=y_score,
        output_path=roc_path,
        roc_auc=metrics.get("roc_auc"),
        title=f"ROC Curve ({model_id} TFLite {args.format})"
    )

    print_summary(metrics)
    print("-" * 80)
    print(f"Inference time only      : {metrics['avg_inference_ms_per_sample']:.4f} ms/sample")
    print(f"Including I/O + loading  : {metrics['avg_total_ms_per_sample_including_io']:.4f} ms/sample")
    print(f"Result JSON              : {result_path}")
    if score_path is not None:
        print(f"Score CSV                : {score_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
