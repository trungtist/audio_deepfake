"""
evaluate.py

Nhiệm vụ:
- Load model CNN đã train.
- Load dev/eval dataset từ metadata + Log-Mel .npy.
- Predict xác suất fake/spoof.
- Tính các metric:
    Accuracy
    Precision
    Recall
    F1-score
    ROC-AUC
    EER
    Confusion Matrix
    Classification Report
- Lưu kết quả đánh giá ra outputs/results/cnn_baseline/.

Cách chạy evaluate nhanh trên dev 500 mẫu:

    py -m src.evaluation.evaluate --config configs/cnn_baseline.yaml --split dev --max-samples 500

Evaluate toàn bộ dev:

    py -m src.evaluation.evaluate --config configs/cnn_baseline.yaml --split dev

Evaluate toàn bộ eval:

    py -m src.evaluation.evaluate --config configs/cnn_baseline.yaml --split eval

Chỉ định model path thủ công:

    py -m src.evaluation.evaluate --config configs/cnn_baseline.yaml --split dev --model-path outputs/checkpoints/cnn_baseline/best_cnn_baseline.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
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
    print_metadata_summary,
)

from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_roc_data,
    print_metrics_summary,
    save_metrics_json,
    score_to_label,
)


# ============================================================
# Path utilities
# ============================================================

def resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """Chuyển path tương đối thành absolute path."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def get_default_model_path(config: Dict[str, Any], project_root: Path) -> Path:
    """
    Lấy model checkpoint mặc định từ config.

    Mặc định:
        outputs/checkpoints/cnn_baseline/best_cnn_baseline.keras
    """
    checkpoint_config = config.get("callbacks", {}).get("model_checkpoint", {})

    model_path = checkpoint_config.get(
        "filepath",
        "outputs/checkpoints/cnn_baseline/best_cnn_baseline.keras",
    )

    return resolve_path(project_root, model_path)


def get_result_root(config: Dict[str, Any], project_root: Path) -> Path:
    """
    Lấy thư mục kết quả gốc.

    Mặc định:
        outputs/results/cnn_baseline
    """
    result_dir = config.get("paths", {}).get(
        "result_dir",
        "outputs/results/cnn_baseline",
    )

    result_dir = resolve_path(project_root, result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def prepare_evaluation_output_dir(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Tạo thư mục lưu kết quả evaluate.

    Nếu không truyền output_dir:
        outputs/results/cnn_baseline/evaluate_dev
        outputs/results/cnn_baseline/evaluate_eval
    """
    if output_dir is not None:
        output_path = resolve_path(project_root, output_dir)
    else:
        result_root = get_result_root(config, project_root)
        output_path = result_root / f"evaluate_{split}"

    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


# ============================================================
# Data utilities
# ============================================================

def stratified_sample_dataframe(
    df: pd.DataFrame,
    max_samples: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Lấy mẫu giữ tương đối tỷ lệ class.

    Dùng khi evaluate debug để tập nhỏ vẫn có cả bonafide và spoof.
    """
    if max_samples is None:
        return df.reset_index(drop=True)

    max_samples = int(max_samples)

    if max_samples >= len(df):
        return df.reset_index(drop=True)

    label_values = sorted(df["label"].astype(int).unique().tolist())

    if len(label_values) < 2:
        return df.head(max_samples).reset_index(drop=True)

    frac = max_samples / len(df)
    sampled_parts = []

    for label_value in label_values:
        df_label = df[df["label"].astype(int) == label_value]

        n_samples = max(1, int(round(len(df_label) * frac)))
        n_samples = min(n_samples, len(df_label))

        sampled = df_label.sample(
            n=n_samples,
            random_state=seed,
            replace=False,
        )

        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, axis=0).reset_index(drop=True)

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


def create_eval_dataset_from_config(
    config: Dict[str, Any],
    project_root: Path,
    split: str,
    max_samples: Optional[int] = None,
) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    """
    Tạo dataset cho evaluate.

    Khác với train:
    - Không shuffle.
    - Không dùng class_weight.
    - Có thể lấy max_samples để debug nhanh.
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

    ds = make_tf_dataset(
        df=df,
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
        shuffle_buffer_size=shuffle_buffer_size,
        train_dtype=train_dtype,
        num_parallel_calls=num_parallel_calls,
        prefetch=prefetch,
        drop_remainder=False,
    )

    return ds, df


# ============================================================
# Model and prediction
# ============================================================

def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    """Load Keras model đã train."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}\n"
            "Hãy train trước bằng train_cnn.py hoặc truyền --model-path đúng."
        )

    print(f"\nLoading model: {model_path}")

    # compile=False vì evaluate chỉ cần inference/predict.
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    return model


def predict_dataset(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict toàn bộ dataset.

    Returns:
        y_true:
            Nhãn thật 0/1.

        y_score:
            Xác suất model dự đoán class 1, tức spoof/fake.
    """
    y_true_list = []
    y_score_list = []

    for x_batch, y_batch in ds:
        y_score_batch = model.predict(
            x_batch,
            verbose=0,
        ).reshape(-1)

        y_true_list.append(y_batch.numpy().reshape(-1))
        y_score_list.append(y_score_batch)

    y_true = np.concatenate(y_true_list, axis=0).astype(int)
    y_score = np.concatenate(y_score_list, axis=0).astype(float)

    return y_true, y_score


# ============================================================
# Save outputs
# ============================================================

def save_predictions_csv(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_path: str | Path,
) -> None:
    """
    Lưu prediction chi tiết từng audio ra CSV.

    Các cột chính:
        utt_id
        label_text
        label
        score_fake
        pred_label
        pred_label_text
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = score_to_label(
        y_score=y_score,
        threshold=threshold,
    )

    output_df = df.copy().reset_index(drop=True)

    if len(output_df) != len(y_true):
        raise ValueError(
            f"Số dòng df khác số prediction. "
            f"len(df)={len(output_df)}, len(y_true)={len(y_true)}"
        )

    output_df["y_true"] = y_true.astype(int)
    output_df["score_fake"] = y_score.astype(float)
    output_df["pred_label"] = y_pred.astype(int)
    output_df["pred_label_text"] = np.where(
        output_df["pred_label"] == 1,
        "spoof",
        "bonafide",
    )
    output_df["correct"] = (output_df["y_true"] == output_df["pred_label"]).astype(int)

    columns_first = [
        "split",
        "utt_id",
        "label_text",
        "label",
        "y_true",
        "score_fake",
        "pred_label",
        "pred_label_text",
        "correct",
    ]

    remaining_columns = [
        col for col in output_df.columns
        if col not in columns_first
    ]

    output_df = output_df[columns_first + remaining_columns]

    output_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def save_evaluation_summary(
    config: Dict[str, Any],
    split: str,
    model_path: Path,
    output_dir: Path,
    metrics: Dict[str, Any],
    predictions_path: Path,
    confusion_matrix_path: Path,
    roc_curve_path: Optional[Path],
) -> None:
    """Lưu summary evaluate."""
    summary = {
        "project": config.get("project", {}),
        "split": split,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "eer": metrics.get("eer"),
            "eer_threshold": metrics.get("eer_threshold"),
            "threshold": metrics.get("threshold"),
            "num_samples": metrics.get("num_samples"),
            "label_counts": metrics.get("label_counts"),
            "prediction_counts": metrics.get("prediction_counts"),
        },
        "files": {
            "predictions_csv": str(predictions_path),
            "confusion_matrix_png": str(confusion_matrix_path),
            "roc_curve_png": None if roc_curve_path is None else str(roc_curve_path),
            "metrics_json": str(output_dir / f"{split}_metrics.json"),
        },
    }

    summary_path = output_dir / f"{split}_evaluation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# Plot utilities
# ============================================================

def plot_confusion_matrix(
    confusion_matrix_values: Any,
    output_path: str | Path,
    class_names: Tuple[str, str] = ("bonafide", "spoof"),
    title: str = "Confusion Matrix",
) -> None:
    """Vẽ và lưu confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(confusion_matrix_values)

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Set3")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontweight="bold"
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: str | Path,
    roc_auc: Optional[float] = None,
    title: str = "ROC Curve",
) -> Optional[Path]:
    """Vẽ và lưu ROC curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roc_data = compute_roc_data(
        y_true=y_true,
        y_score=y_score,
    )

    if not roc_data["fpr"]:
        return None

    fpr = np.asarray(roc_data["fpr"])
    tpr = np.asarray(roc_data["tpr"])

    fig, ax = plt.subplots(figsize=(5, 4))

    label = "ROC curve"

    if roc_auc is not None:
        label = f"ROC curve, AUC = {roc_auc:.4f}"

    ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN baseline model."
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
        default="dev",
        choices=["dev", "eval", "train"],
        help="Split cần đánh giá.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Đường dẫn model .keras. Nếu không truyền, dùng best checkpoint từ config.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số mẫu evaluate để test nhanh.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold phân loại. Nếu không truyền, lấy từ config.evaluation.threshold hoặc 0.5.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục lưu kết quả evaluate. Nếu không truyền, tự tạo theo split.",
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

    split = args.split

    print("Project root:", project_root)
    print("Config file :", config_path)
    print("Split       :", split)

    # --------------------------------------------------------
    # Resolve model path
    # --------------------------------------------------------

    if args.model_path is not None:
        model_path = resolve_path(project_root, args.model_path)
    else:
        model_path = get_default_model_path(config, project_root)

    # --------------------------------------------------------
    # Resolve threshold
    # --------------------------------------------------------

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(
            config.get("evaluation", {}).get("threshold", 0.5)
        )

    print("Threshold   :", threshold)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = prepare_evaluation_output_dir(
        config=config,
        project_root=project_root,
        split=split,
        output_dir=args.output_dir,
    )

    print("Output dir  :", output_dir)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nCreating evaluation dataset...")

    ds, df = create_eval_dataset_from_config(
        config=config,
        project_root=project_root,
        split=split,
        max_samples=args.max_samples,
    )

    print_metadata_summary(
        df=df,
        name=f"{split}",
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_trained_model(model_path)

    print("\nModel loaded.")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nPredicting...")

    y_true, y_score = predict_dataset(
        model=model,
        ds=ds,
    )

    print("Prediction done.")
    print(f"y_true shape : {y_true.shape}")
    print(f"y_score shape: {y_score.shape}")
    print(f"score min    : {float(np.min(y_score)):.6f}")
    print(f"score max    : {float(np.max(y_score)):.6f}")
    print(f"score mean   : {float(np.mean(y_score)):.6f}")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
        class_names=("bonafide", "spoof"),
    )

    print_metrics_summary(metrics)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    metrics_path = output_dir / f"{split}_metrics.json"
    predictions_path = output_dir / f"{split}_predictions.csv"
    confusion_matrix_path = output_dir / f"{split}_confusion_matrix.png"
    roc_curve_path = output_dir / f"{split}_roc_curve.png"

    save_metrics_json(
        metrics=metrics,
        output_path=metrics_path,
    )

    save_predictions_csv(
        df=df,
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
        output_path=predictions_path,
    )

    plot_confusion_matrix(
        confusion_matrix_values=metrics["confusion_matrix"],
        output_path=confusion_matrix_path,
        class_names=("bonafide", "spoof"),
        title=f"Confusion Matrix - {split}",
    )

    saved_roc_path = plot_roc_curve(
        y_true=y_true,
        y_score=y_score,
        output_path=roc_curve_path,
        roc_auc=metrics.get("roc_auc"),
        title=f"ROC Curve - {split}",
    )

    save_evaluation_summary(
        config=config,
        split=split,
        model_path=model_path,
        output_dir=output_dir,
        metrics=metrics,
        predictions_path=predictions_path,
        confusion_matrix_path=confusion_matrix_path,
        roc_curve_path=saved_roc_path,
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"Metrics          : {metrics_path}")
    print(f"Predictions      : {predictions_path}")
    print(f"Confusion matrix : {confusion_matrix_path}")

    if saved_roc_path is not None:
        print(f"ROC curve        : {saved_roc_path}")
    else:
        print("ROC curve        : Not saved, y_true không đủ 2 class")

    print(f"Summary          : {output_dir / f'{split}_evaluation_summary.json'}")

    print("\nEvaluate OK.")


if __name__ == "__main__":
    main()