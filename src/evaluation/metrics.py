"""
metrics.py

Nhiệm vụ:
- Tính các metric đánh giá mô hình phát hiện Audio Deepfake.
- Input chính:
    y_true  : nhãn thật, gồm 0/1
    y_score : xác suất dự đoán fake/spoof từ model, giá trị trong [0, 1]
- Output:
    accuracy
    precision
    recall
    f1_score
    roc_auc
    eer
    confusion_matrix
    classification_report
    best thresholds

Quy ước label:
    0 = bonafide / real
    1 = spoof / fake

Cách chạy test nhanh:

    py -m src.evaluation.metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Basic utilities
# ============================================================

def to_1d_numpy(array: Any, name: str = "array") -> np.ndarray:
    """
    Chuyển input về numpy array 1 chiều.

    Args:
        array: list, numpy array hoặc tensor-like.
        name: tên biến để báo lỗi rõ hơn.

    Returns:
        np.ndarray 1D.
    """
    arr = np.asarray(array)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if len(arr) == 0:
        raise ValueError(f"{name} rỗng.")

    return arr


def validate_binary_inputs(
    y_true: Any,
    y_score: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kiểm tra và chuẩn hóa y_true, y_score.

    y_true:
        Nhãn thật, chỉ gồm 0/1.

    y_score:
        Xác suất fake/spoof, giá trị lý tưởng trong [0, 1].
    """
    y_true = to_1d_numpy(y_true, name="y_true").astype(int)
    y_score = to_1d_numpy(y_score, name="y_score").astype(float)

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true và y_score phải có cùng số phần tử. "
            f"Got len(y_true)={len(y_true)}, len(y_score)={len(y_score)}"
        )

    unique_labels = sorted(np.unique(y_true).tolist())

    invalid_labels = [label for label in unique_labels if label not in [0, 1]]

    if invalid_labels:
        raise ValueError(
            f"y_true chỉ được chứa label 0/1. "
            f"Invalid labels: {invalid_labels}"
        )

    if np.isnan(y_score).any():
        raise ValueError("y_score có chứa NaN.")

    if np.isinf(y_score).any():
        raise ValueError("y_score có chứa Inf.")

    return y_true, y_score


def score_to_label(
    y_score: Any,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Chuyển xác suất fake thành label dự đoán.

    Nếu:
        y_score >= threshold -> 1, fake/spoof
        y_score < threshold  -> 0, real/bonafide
    """
    y_score = to_1d_numpy(y_score, name="y_score").astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return y_pred


def has_two_classes(y_true: np.ndarray) -> bool:
    """Kiểm tra y_true có đủ cả class 0 và class 1 không."""
    return len(np.unique(y_true)) == 2


# ============================================================
# EER and threshold utilities
# ============================================================

def compute_eer(
    y_true: Any,
    y_score: Any,
) -> Dict[str, Optional[float]]:
    """
    Tính Equal Error Rate.

    EER là điểm mà:
        FPR gần bằng FNR

    Trong bài toán anti-spoofing:
        FPR: tỷ lệ real bị dự đoán nhầm thành fake
        FNR: tỷ lệ fake bị dự đoán nhầm thành real

    EER càng thấp càng tốt.

    Returns:
        {
            "eer": float,
            "eer_threshold": float,
            "fpr_at_eer": float,
            "fnr_at_eer": float
        }
    """
    y_true, y_score = validate_binary_inputs(y_true, y_score)

    if not has_two_classes(y_true):
        return {
            "eer": None,
            "eer_threshold": None,
            "fpr_at_eer": None,
            "fnr_at_eer": None,
        }

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fpr - fnr)))

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "fpr_at_eer": float(fpr[idx]),
        "fnr_at_eer": float(fnr[idx]),
    }


def find_best_threshold_by_f1(
    y_true: Any,
    y_score: Any,
    num_thresholds: int = 1001,
) -> Dict[str, float]:
    """
    Tìm threshold tốt nhất theo F1-score.

    Duyệt threshold từ 0 đến 1.
    """
    y_true, y_score = validate_binary_inputs(y_true, y_score)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = score_to_label(y_score, threshold=float(threshold))
        current_f1 = f1_score(y_true, y_pred, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)

    return {
        "best_threshold_f1": best_threshold,
        "best_f1": best_f1,
    }


def find_best_threshold_by_youden(
    y_true: Any,
    y_score: Any,
) -> Dict[str, Optional[float]]:
    """
    Tìm threshold tốt nhất theo Youden's J statistic.

    Công thức:
        J = TPR - FPR

    Threshold tốt nhất là threshold có J lớn nhất.
    """
    y_true, y_score = validate_binary_inputs(y_true, y_score)

    if not has_two_classes(y_true):
        return {
            "best_threshold_youden": None,
            "best_youden_j": None,
        }

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=1,
    )

    youden_j = tpr - fpr
    idx = int(np.argmax(youden_j))

    return {
        "best_threshold_youden": float(thresholds[idx]),
        "best_youden_j": float(youden_j[idx]),
    }


def compute_roc_data(
    y_true: Any,
    y_score: Any,
) -> Dict[str, Any]:
    """
    Tính dữ liệu ROC curve.

    Dữ liệu này sẽ được evaluate.py dùng để vẽ ROC curve.
    """
    y_true, y_score = validate_binary_inputs(y_true, y_score)

    if not has_two_classes(y_true):
        return {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
        }

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=1,
    )

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


# ============================================================
# Main metric computation
# ============================================================

def compute_binary_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
    class_names: Tuple[str, str] = ("bonafide", "spoof"),
) -> Dict[str, Any]:
    """
    Tính toàn bộ metric cho binary classification.

    Args:
        y_true:
            Nhãn thật 0/1.

        y_score:
            Xác suất dự đoán class 1, tức spoof/fake.

        threshold:
            Ngưỡng phân loại. Mặc định 0.5.

        class_names:
            Tên class theo thứ tự label 0, label 1.

    Returns:
        Dict chứa toàn bộ metric.
    """
    y_true, y_score = validate_binary_inputs(y_true, y_score)
    y_pred = score_to_label(y_score, threshold=threshold)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(class_names),
        output_dict=False,
        zero_division=0,
    )

    if has_two_classes(y_true):
        roc_auc = float(roc_auc_score(y_true, y_score))
    else:
        roc_auc = None

    eer_result = compute_eer(y_true, y_score)
    best_f1_result = find_best_threshold_by_f1(y_true, y_score)
    best_youden_result = find_best_threshold_by_youden(y_true, y_score)

    label_counts = {
        str(k): int(v)
        for k, v in zip(*np.unique(y_true, return_counts=True))
    }

    pred_counts = {
        str(k): int(v)
        for k, v in zip(*np.unique(y_pred, return_counts=True))
    }

    result = {
        "threshold": float(threshold),
        "num_samples": int(len(y_true)),
        "label_counts": label_counts,
        "prediction_counts": pred_counts,

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,

        "eer": eer_result["eer"],
        "eer_threshold": eer_result["eer_threshold"],
        "fpr_at_eer": eer_result["fpr_at_eer"],
        "fnr_at_eer": eer_result["fnr_at_eer"],

        "best_threshold_f1": best_f1_result["best_threshold_f1"],
        "best_f1": best_f1_result["best_f1"],

        "best_threshold_youden": best_youden_result["best_threshold_youden"],
        "best_youden_j": best_youden_result["best_youden_j"],

        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }

    return result


# ============================================================
# Display and save utilities
# ============================================================

def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """In metric summary ra terminal."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    print(f"Samples       : {metrics['num_samples']}")
    print(f"Threshold     : {metrics['threshold']}")
    print(f"Label counts  : {metrics['label_counts']}")
    print(f"Pred counts   : {metrics['prediction_counts']}")

    print("\nMain metrics:")
    print(f"Accuracy      : {metrics['accuracy']:.6f}")
    print(f"Precision     : {metrics['precision']:.6f}")
    print(f"Recall        : {metrics['recall']:.6f}")
    print(f"F1-score      : {metrics['f1_score']:.6f}")

    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC       : {metrics['roc_auc']:.6f}")
    else:
        print("ROC-AUC       : None, y_true không đủ 2 class")

    if metrics["eer"] is not None:
        print(f"EER           : {metrics['eer']:.6f}")
        print(f"EER threshold : {metrics['eer_threshold']:.6f}")
    else:
        print("EER           : None, y_true không đủ 2 class")

    print("\nBest thresholds:")
    print(f"Best F1 threshold     : {metrics['best_threshold_f1']:.6f}")
    print(f"Best F1               : {metrics['best_f1']:.6f}")

    if metrics["best_threshold_youden"] is not None:
        print(f"Best Youden threshold : {metrics['best_threshold_youden']:.6f}")
        print(f"Best Youden J         : {metrics['best_youden_j']:.6f}")
    else:
        print("Best Youden threshold : None")

    print("\nConfusion matrix:")
    print("Rows = true label, Columns = predicted label")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification report:")
    print(metrics["classification_report_text"])


def save_metrics_json(
    metrics: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Lưu metrics ra file JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_metrics_json(
    input_path: str | Path,
) -> Dict[str, Any]:
    """Đọc metrics JSON."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy metrics file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CLI test
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test binary metrics for Audio Deepfake Detection."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold dùng để chuyển score thành label.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Dữ liệu giả để test:
    # 0 = bonafide, 1 = spoof
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_score = np.array([0.10, 0.20, 0.40, 0.65, 0.80, 0.90, 0.55, 0.70])

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=args.threshold,
        class_names=("bonafide", "spoof"),
    )

    print_metrics_summary(metrics)

    print("\nMetrics test OK.")


if __name__ == "__main__":
    main()