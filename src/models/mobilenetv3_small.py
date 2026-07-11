"""
mobilenetv3_small.py

Nhiệm vụ:
- Xây dựng MobileNetV3-Small cho bài toán Audio Deepfake Detection.
- Input: precomputed Log-Mel image feature [128, 128, 3].
- Backbone: MobileNetV3Small pretrained ImageNet.
- Classifier: binary classifier với sigmoid output.

Quy ước label:
    0 = bonafide / real
    1 = spoof / fake

Cách chạy test model:

    py -m src.models.mobilenetv3_small --config configs/mobilenetv3_small.yaml --summary --compile --forward-test

Nếu đang dùng .venv:

    python -m src.models.mobilenetv3_small --config configs/mobilenetv3_small.yaml --summary --compile --forward-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
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


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy block model config."""
    return config.get("model", {})


def get_backbone_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy block backbone config."""
    model_config = get_model_config(config)
    return model_config.get("backbone", {})


def get_classifier_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lấy block classifier config."""
    model_config = get_model_config(config)
    return model_config.get("classifier", {})


def get_input_shape(config: Dict[str, Any]) -> tuple[int, int, int]:
    """
    Lấy input shape.

    Ưu tiên:
    1. model.input_shape
    2. feature.input_shape
    """
    model_config = get_model_config(config)

    input_shape = model_config.get(
        "input_shape",
        config.get("feature", {}).get("input_shape", [128, 128, 3]),
    )

    if len(input_shape) != 3:
        raise ValueError(f"input_shape phải có 3 chiều, got: {input_shape}")

    input_shape = tuple(int(v) for v in input_shape)

    if input_shape[-1] != 3:
        raise ValueError(
            f"MobileNetV3 pretrained ImageNet cần input 3 channels. "
            f"Got input_shape={input_shape}"
        )

    if input_shape[0] < 32 or input_shape[1] < 32:
        raise ValueError(
            f"MobileNetV3 cần height/width >= 32. Got input_shape={input_shape}"
        )

    return input_shape


# ============================================================
# Model building
# ============================================================

def build_mobilenetv3_backbone(
    config: Dict[str, Any],
    input_shape: tuple[int, int, int],
    allow_random_fallback: bool = False,
) -> tf.keras.Model:
    """
    Build MobileNetV3-Small backbone.

    Args:
        config:
            Config dictionary từ mobilenetv3_small.yaml.

        input_shape:
            Input shape, ví dụ [128, 128, 3].

        allow_random_fallback:
            Nếu True và không load được ImageNet weights,
            sẽ fallback sang weights=None.
            Mặc định False để tránh vô tình train sai mục tiêu pretrained.

    Returns:
        tf.keras.Model backbone.
    """
    backbone_config = get_backbone_config(config)

    architecture = str(backbone_config.get("architecture", "MobileNetV3Small"))

    if architecture != "MobileNetV3Small":
        raise ValueError(
            f"Hiện file này chỉ hỗ trợ MobileNetV3Small. "
            f"Got architecture={architecture}"
        )

    weights = backbone_config.get("weights", "imagenet")
    include_top = bool(backbone_config.get("include_top", False))
    include_preprocessing = bool(backbone_config.get("include_preprocessing", True))
    pooling = backbone_config.get("pooling", "avg")
    alpha = float(backbone_config.get("alpha", 0.75))
    minimalistic = bool(backbone_config.get("minimalistic", False))
    dropout_rate = float(backbone_config.get("dropout_rate", 0.2))

    if include_top:
        raise ValueError(
            "Với bài toán binary classification, backbone phải dùng include_top=False. "
            "Classifier nhị phân sẽ được thêm ở phía sau."
        )

    try:
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            alpha=alpha,
            minimalistic=minimalistic,
            include_top=False,
            weights=weights,
            pooling=pooling,
            dropout_rate=dropout_rate,
            include_preprocessing=include_preprocessing,
        )

    except Exception as e:
        if not allow_random_fallback:
            raise RuntimeError(
                "Không build được MobileNetV3Small với weights cấu hình hiện tại. "
                "Nếu lỗi do không tải được ImageNet weights, hãy kiểm tra internet "
                "hoặc chạy với --allow-random-fallback để test nhanh."
            ) from e

        print("\nWARNING: Không load được pretrained weights.")
        print("Fallback sang weights=None để test model.")
        print(f"Original error: {repr(e)}")

        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            alpha=alpha,
            minimalistic=minimalistic,
            include_top=False,
            weights=None,
            pooling=pooling,
            dropout_rate=dropout_rate,
            include_preprocessing=include_preprocessing,
        )

    # Wrap lại để đặt tên ổn định cho backbone.
    backbone = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output,
        name="mobilenetv3_small_backbone",
    )

    base_trainable = bool(backbone_config.get("base_trainable", False))
    backbone.trainable = base_trainable

    return backbone


def build_mobilenetv3_small_from_config(
    config: Dict[str, Any],
    allow_random_fallback: bool = False,
) -> tf.keras.Model:
    """
    Build full MobileNetV3-Small binary classifier từ config.

    Kiến trúc:
        Input [128, 128, 3]
        → MobileNetV3Small backbone
        → Dropout
        → Dense
        → Dropout
        → Dense(1, sigmoid)
    """
    model_config = get_model_config(config)
    classifier_config = get_classifier_config(config)

    model_name = str(
        model_config.get(
            "name",
            "MobileNetV3Small_LogMel_AudioDeepfake",
        )
    )

    input_shape = get_input_shape(config)

    inputs = tf.keras.layers.Input(
        shape=input_shape,
        name="logmel_image_input",
    )

    backbone = build_mobilenetv3_backbone(
        config=config,
        input_shape=input_shape,
        allow_random_fallback=allow_random_fallback,
    )

    # Nếu backbone frozen, gọi training=False để BatchNorm trong pretrained backbone ổn định hơn.
    if backbone.trainable:
        x = backbone(inputs)
    else:
        x = backbone(inputs, training=False)

    dropout_1 = float(classifier_config.get("dropout_1", 0.30))
    dense_units = int(classifier_config.get("dense_units", 128))
    dense_activation = str(classifier_config.get("dense_activation", "relu"))
    dropout_2 = float(classifier_config.get("dropout_2", 0.20))
    output_units = int(classifier_config.get("output_units", 1))
    output_activation = str(classifier_config.get("output_activation", "sigmoid"))

    x = tf.keras.layers.Dropout(
        dropout_1,
        name="classifier_dropout_1",
    )(x)

    x = tf.keras.layers.Dense(
        dense_units,
        activation=dense_activation,
        name="classifier_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        dropout_2,
        name="classifier_dropout_2",
    )(x)

    outputs = tf.keras.layers.Dense(
        output_units,
        activation=output_activation,
        name="spoof_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    return model


# Alias ngắn để train script gọi cho tiện.
def build_model_from_config(
    config: Dict[str, Any],
    allow_random_fallback: bool = False,
) -> tf.keras.Model:
    """Alias cho build_mobilenetv3_small_from_config."""
    return build_mobilenetv3_small_from_config(
        config=config,
        allow_random_fallback=allow_random_fallback,
    )


# ============================================================
# Compile utilities
# ============================================================

def build_optimizer_from_config(config: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    """Build optimizer từ config."""
    training_config = config.get("training", {})
    optimizer_config = training_config.get("optimizer", {})

    optimizer_name = str(optimizer_config.get("name", "adam")).lower()
    learning_rate = float(optimizer_config.get("learning_rate", 3e-4))

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if optimizer_name == "sgd":
        momentum = float(optimizer_config.get("momentum", 0.9))
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
        )

    raise ValueError(f"Optimizer chưa hỗ trợ: {optimizer_name}")


def build_metrics_from_config(config: Dict[str, Any]) -> list[tf.keras.metrics.Metric]:
    """Build metrics từ config."""
    training_config = config.get("training", {})
    metric_names = training_config.get("metrics", ["accuracy", "roc_auc"])

    metrics: list[tf.keras.metrics.Metric] = []

    for metric_name in metric_names:
        metric_name = str(metric_name).lower()

        if metric_name == "accuracy":
            metrics.append(
                tf.keras.metrics.BinaryAccuracy(
                    name="accuracy",
                    threshold=0.5,
                )
            )

        elif metric_name == "roc_auc":
            metrics.append(
                tf.keras.metrics.AUC(
                    name="roc_auc",
                    curve="ROC",
                )
            )

        elif metric_name == "pr_auc":
            metrics.append(
                tf.keras.metrics.AUC(
                    name="pr_auc",
                    curve="PR",
                )
            )

        else:
            raise ValueError(f"Metric chưa hỗ trợ: {metric_name}")

    return metrics


def compile_mobilenetv3_model(
    model: tf.keras.Model,
    config: Dict[str, Any],
) -> tf.keras.Model:
    """
    Compile MobileNetV3 model.

    Loss:
        binary_crossentropy

    Metrics:
        accuracy, roc_auc
    """
    training_config = config.get("training", {})

    optimizer = build_optimizer_from_config(config)
    loss = str(training_config.get("loss", "binary_crossentropy"))
    metrics = build_metrics_from_config(config)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


# Alias để train script gọi cho tiện.
def compile_model(
    model: tf.keras.Model,
    config: Dict[str, Any],
) -> tf.keras.Model:
    """Alias cho compile_mobilenetv3_model."""
    return compile_mobilenetv3_model(model=model, config=config)


# ============================================================
# Backbone utilities
# ============================================================

def get_backbone(model: tf.keras.Model) -> Optional[tf.keras.Model]:
    """Tìm backbone MobileNetV3 trong full model."""
    try:
        layer = model.get_layer("mobilenetv3_small_backbone")
        if isinstance(layer, tf.keras.Model):
            return layer
    except ValueError:
        return None

    return None


def set_backbone_trainable(
    model: tf.keras.Model,
    trainable: bool,
    fine_tune_from_layer: Optional[int] = None,
) -> tf.keras.Model:
    """
    Bật/tắt trainable cho backbone.

    Args:
        model:
            Full model.

        trainable:
            True nếu muốn fine-tune backbone.

        fine_tune_from_layer:
            Nếu None: toàn bộ backbone trainable theo biến trainable.
            Nếu là số âm, ví dụ -30: chỉ mở 30 layer cuối.
            Nếu là số dương: mở từ layer index đó trở đi.
    """
    backbone = get_backbone(model)

    if backbone is None:
        raise ValueError("Không tìm thấy layer mobilenetv3_small_backbone trong model.")

    if not trainable:
        backbone.trainable = False
        return model

    backbone.trainable = True

    if fine_tune_from_layer is None:
        for layer in backbone.layers:
            layer.trainable = True
        return model

    num_layers = len(backbone.layers)

    if fine_tune_from_layer < 0:
        start_index = max(0, num_layers + fine_tune_from_layer)
    else:
        start_index = min(num_layers, fine_tune_from_layer)

    for idx, layer in enumerate(backbone.layers):
        layer.trainable = idx >= start_index

    return model


def print_backbone_trainable_summary(model: tf.keras.Model) -> None:
    """In số layer trainable/non-trainable trong backbone."""
    backbone = get_backbone(model)

    if backbone is None:
        print("Backbone: not found")
        return

    total_layers = len(backbone.layers)
    trainable_layers = sum(1 for layer in backbone.layers if layer.trainable)
    non_trainable_layers = total_layers - trainable_layers

    print("\n" + "=" * 70)
    print("BACKBONE TRAINABLE SUMMARY")
    print("=" * 70)
    print(f"Backbone name        : {backbone.name}")
    print(f"Total backbone layers: {total_layers}")
    print(f"Trainable layers     : {trainable_layers}")
    print(f"Non-trainable layers : {non_trainable_layers}")
    print(f"Backbone trainable   : {backbone.trainable}")


# ============================================================
# Model info utilities
# ============================================================

def get_model_parameter_count(model: tf.keras.Model) -> Dict[str, int]:
    """Đếm số params của model."""
    total_params = int(model.count_params())

    trainable_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    )

    non_trainable_params = int(
        np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def estimate_fp32_model_size_mb(model: tf.keras.Model) -> float:
    """Ước tính kích thước FP32 model dựa trên số params."""
    total_params = model.count_params()
    size_mb = total_params * 4 / (1024 ** 2)
    return float(size_mb)


def print_model_info(model: tf.keras.Model) -> None:
    """In thông tin model."""
    counts = get_model_parameter_count(model)
    size_mb = estimate_fp32_model_size_mb(model)

    print("\n" + "=" * 70)
    print("MODEL INFO")
    print("=" * 70)
    print(f"Model name          : {model.name}")
    print(f"Input shape         : {model.input_shape}")
    print(f"Output shape        : {model.output_shape}")
    print(f"Total params        : {counts['total_params']:,}")
    print(f"Trainable params    : {counts['trainable_params']:,}")
    print(f"Non-trainable params: {counts['non_trainable_params']:,}")
    print(f"Estimated FP32 size : {size_mb:.4f} MB")


def run_forward_pass_test(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int],
    batch_size: int = 4,
) -> None:
    """
    Test forward pass bằng input giả.

    Vì include_preprocessing=True trong MobileNetV3,
    input giả nên nằm trong khoảng [0, 255].
    """
    x = np.random.uniform(
        low=0.0,
        high=255.0,
        size=(batch_size, *input_shape),
    ).astype(np.float32)

    y = model.predict(x, verbose=0)

    print("\n" + "=" * 70)
    print("FORWARD PASS TEST")
    print("=" * 70)
    print(f"Input batch shape : {x.shape}")
    print(f"Output shape      : {y.shape}")
    print(f"Output min        : {float(np.min(y)):.6f}")
    print(f"Output max        : {float(np.max(y)):.6f}")

    if y.shape != (batch_size, 1):
        raise ValueError(
            f"Output shape không đúng. Expected {(batch_size, 1)}, got {y.shape}"
        )

    if np.isnan(y).any():
        raise ValueError("Output có NaN.")

    print("Forward pass OK.")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect MobileNetV3-Small model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mobilenetv3_small.yaml",
        help="Đường dẫn tới config YAML.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="In model.summary().",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model để kiểm tra optimizer/loss/metrics.",
    )

    parser.add_argument(
        "--forward-test",
        action="store_true",
        help="Chạy forward pass với input giả.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size cho forward-test.",
    )

    parser.add_argument(
        "--allow-random-fallback",
        action="store_true",
        help="Nếu không load được ImageNet weights thì fallback sang weights=None.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)

    print("Config file:", config_path)

    input_shape = get_input_shape(config)

    model = build_mobilenetv3_small_from_config(
        config=config,
        allow_random_fallback=args.allow_random_fallback,
    )

    if args.compile:
        model = compile_mobilenetv3_model(
            model=model,
            config=config,
        )

    print_model_info(model)
    print_backbone_trainable_summary(model)

    if args.summary:
        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        model.summary()

    if args.forward_test:
        run_forward_pass_test(
            model=model,
            input_shape=input_shape,
            batch_size=args.batch_size,
        )

    print("\nMobileNetV3-Small model OK.")


if __name__ == "__main__":
    main()