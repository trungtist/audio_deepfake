"""
cnn_baseline.py

Nhiệm vụ:
- Xây dựng mô hình CNN baseline nhẹ cho bài toán phát hiện Audio Deepfake.
- Input: Log-Mel Spectrogram shape [64, 251, 1].
- Output: xác suất audio thuộc lớp fake/spoof.

Cách chạy test từ thư mục gốc project:

    python -m src.models.cnn_baseline --config configs/cnn_baseline.yaml

Output mong muốn:
    Model summary
    Total parameters
    Estimated model size
    Test forward pass shape: (batch_size, 1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import layers, models


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


def set_random_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả ổn định hơn."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_tuple(value: Sequence[int] | Tuple[int, int], length: int = 2) -> Tuple[int, ...]:
    """
    Chuyển list trong YAML sang tuple.

    Ví dụ:
        [3, 3] -> (3, 3)
        [64, 251, 1] -> (64, 251, 1)
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Giá trị phải là list hoặc tuple, got: {type(value)}")

    if len(value) != length:
        raise ValueError(f"Độ dài không đúng. Expected {length}, got {len(value)}")

    return tuple(int(v) for v in value)


def get_activation_layer(activation: str) -> layers.Layer:
    """
    Tạo activation layer.

    Dùng layer rõ ràng thay vì Lambda để dễ export TFLite/ONNX hơn.
    """
    activation = activation.lower()

    if activation == "relu":
        return layers.ReLU()

    if activation == "relu6":
        return layers.ReLU(max_value=6.0)

    if activation == "sigmoid":
        return layers.Activation("sigmoid")

    if activation == "softmax":
        return layers.Activation("softmax")

    if activation == "linear":
        return layers.Activation("linear")

    raise ValueError(f"Activation chưa hỗ trợ: {activation}")


def build_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int] = (3, 3),
    padding: str = "same",
    use_bias: bool = False,
    batch_norm: bool = True,
    activation: str = "relu",
    pooling: Optional[str] = "max",
    pool_size: Optional[Tuple[int, int]] = (2, 2),
    block_name: str = "conv_block",
) -> tf.Tensor:
    """
    Xây dựng một block CNN:

        Conv2D
        → BatchNorm
        → ReLU
        → MaxPooling2D nếu có

    Args:
        x: tensor đầu vào.
        filters: số filter của Conv2D.
        kernel_size: kích thước kernel.
        padding: "same" hoặc "valid".
        use_bias: dùng bias trong Conv2D hay không.
        batch_norm: có dùng BatchNormalization không.
        activation: activation function.
        pooling: "max", "avg", hoặc None.
        pool_size: kích thước pooling.
        block_name: tên block.

    Returns:
        Tensor sau khi qua block.
    """
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        name=f"{block_name}_conv",
    )(x)

    if batch_norm:
        x = layers.BatchNormalization(name=f"{block_name}_bn")(x)

    x = get_activation_layer(activation)(x)

    if pooling is None:
        return x

    pooling = str(pooling).lower()

    if pool_size is None:
        raise ValueError(f"{block_name}: pool_size không được None khi pooling được bật")

    if pooling == "max":
        x = layers.MaxPooling2D(
            pool_size=pool_size,
            name=f"{block_name}_maxpool",
        )(x)
    elif pooling == "avg":
        x = layers.AveragePooling2D(
            pool_size=pool_size,
            name=f"{block_name}_avgpool",
        )(x)
    else:
        raise ValueError(f"{block_name}: pooling không hỗ trợ: {pooling}")

    return x


def build_light_cnn_baseline(
    input_shape: Tuple[int, int, int] = (64, 251, 1),
    model_name: str = "Light_CNN_LogMel_AudioDeepfake",
) -> tf.keras.Model:
    """
    Xây dựng CNN baseline mặc định, không phụ thuộc YAML.

    Kiến trúc:
        Input 64 x 251 x 1
        Conv 16 → BN → ReLU → MaxPool
        Conv 32 → BN → ReLU → MaxPool
        Conv 64 → BN → ReLU → MaxPool
        Conv 96 → BN → ReLU
        GlobalAveragePooling2D
        Dropout
        Dense 64
        Dropout
        Dense 1 Sigmoid

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="logmel_input")

    x = build_conv_block(
        x=inputs,
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        batch_norm=True,
        activation="relu",
        pooling="max",
        pool_size=(2, 2),
        block_name="block1",
    )

    x = build_conv_block(
        x=x,
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        batch_norm=True,
        activation="relu",
        pooling="max",
        pool_size=(2, 2),
        block_name="block2",
    )

    x = build_conv_block(
        x=x,
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        batch_norm=True,
        activation="relu",
        pooling="max",
        pool_size=(2, 2),
        block_name="block3",
    )

    x = build_conv_block(
        x=x,
        filters=96,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        batch_norm=True,
        activation="relu",
        pooling=None,
        pool_size=None,
        block_name="block4",
    )

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dropout(0.30, name="dropout_1")(x)

    x = layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer="he_normal",
        name="dense_64",
    )(x)

    x = layers.Dropout(0.20, name="dropout_2")(x)

    outputs = layers.Dense(
        units=1,
        activation="sigmoid",
        name="fake_probability",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    return model


def build_cnn_from_config(config: Dict[str, Any]) -> tf.keras.Model:
    """
    Xây dựng CNN baseline từ configs/cnn_baseline.yaml.

    Hàm này dùng trong train_cnn.py sau này.
    """
    project_config = config.get("project", {})
    model_config = config.get("model", {})

    seed = int(project_config.get("seed", 42))
    set_random_seed(seed)

    model_name = model_config.get("name", "Light_CNN_LogMel_AudioDeepfake")
    input_shape = to_tuple(model_config.get("input_shape", [64, 251, 1]), length=3)

    conv_blocks: List[Dict[str, Any]] = model_config.get("conv_blocks", [])

    if not conv_blocks:
        raise ValueError("Config thiếu model.conv_blocks")

    inputs = layers.Input(shape=input_shape, name="logmel_input")
    x = inputs

    for idx, block in enumerate(conv_blocks, start=1):
        block_name = f"block{idx}"

        filters = int(block["filters"])
        kernel_size = to_tuple(block.get("kernel_size", [3, 3]), length=2)
        padding = str(block.get("padding", "same"))
        use_bias = bool(block.get("use_bias", False))
        batch_norm = bool(block.get("batch_norm", True))
        activation = str(block.get("activation", "relu"))

        pooling = block.get("pooling", None)
        pool_size_value = block.get("pool_size", None)

        if pool_size_value is not None:
            pool_size = to_tuple(pool_size_value, length=2)
        else:
            pool_size = None

        x = build_conv_block(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=use_bias,
            batch_norm=batch_norm,
            activation=activation,
            pooling=pooling,
            pool_size=pool_size,
            block_name=block_name,
        )

    global_pooling = str(model_config.get("global_pooling", "global_average_pooling_2d"))

    if global_pooling == "global_average_pooling_2d":
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    elif global_pooling == "global_max_pooling_2d":
        x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    else:
        raise ValueError(f"global_pooling không hỗ trợ: {global_pooling}")

    classifier_config = model_config.get("classifier", {})

    dropout_1 = float(classifier_config.get("dropout_1", 0.30))
    dense_units = int(classifier_config.get("dense_units", 64))
    dense_activation = str(classifier_config.get("dense_activation", "relu"))
    dropout_2 = float(classifier_config.get("dropout_2", 0.20))
    output_units = int(classifier_config.get("output_units", 1))
    output_activation = str(classifier_config.get("output_activation", "sigmoid"))

    if dropout_1 > 0:
        x = layers.Dropout(dropout_1, name="dropout_1")(x)

    x = layers.Dense(
        units=dense_units,
        activation=dense_activation,
        kernel_initializer="he_normal",
        name=f"dense_{dense_units}",
    )(x)

    if dropout_2 > 0:
        x = layers.Dropout(dropout_2, name="dropout_2")(x)

    outputs = layers.Dense(
        units=output_units,
        activation=output_activation,
        name="fake_probability",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    return model


def compile_cnn_model(
    model: tf.keras.Model,
    config: Dict[str, Any],
) -> tf.keras.Model:
    """
    Compile model theo config.

    Giai đoạn train sẽ dùng hàm này.
    """
    training_config = config.get("training", {})
    optimizer_config = training_config.get("optimizer", {})

    optimizer_name = str(optimizer_config.get("name", "adam")).lower()
    learning_rate = float(optimizer_config.get("learning_rate", 1e-3))
    loss_name = str(training_config.get("loss", "binary_crossentropy"))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Optimizer chưa hỗ trợ: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss=loss_name,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )

    return model


def get_model_parameter_count(model: tf.keras.Model) -> Dict[str, int]:
    """Đếm tổng số tham số của model."""
    trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.trainable_weights])
    )
    non_trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    )
    total_params = trainable_params + non_trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def estimate_fp32_model_size_mb(model: tf.keras.Model) -> float:
    """
    Ước lượng kích thước model nếu lưu trọng số FP32.

    FP32 = 4 bytes / parameter.
    Đây chỉ là ước lượng trọng số, chưa tính overhead định dạng file.
    """
    counts = get_model_parameter_count(model)
    total_params = counts["total_params"]
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


def test_forward_pass(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int],
    batch_size: int = 8,
) -> None:
    """
    Test thử forward pass bằng input giả.

    Mục tiêu:
        Đảm bảo model nhận đúng input shape và output là [batch_size, 1].
    """
    x = np.random.randn(batch_size, *input_shape).astype(np.float32)
    y = model.predict(x, verbose=0)

    print("\n" + "=" * 70)
    print("FORWARD PASS TEST")
    print("=" * 70)
    print(f"Input batch shape : {x.shape}")
    print(f"Output shape      : {y.shape}")
    print(f"Output min        : {float(np.min(y)):.6f}")
    print(f"Output max        : {float(np.max(y)):.6f}")

    expected_output_shape = (batch_size, 1)

    if y.shape != expected_output_shape:
        raise ValueError(
            f"Output shape không đúng. "
            f"Expected: {expected_output_shape}, Got: {y.shape}"
        )

    print("Forward pass OK.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and test CNN baseline model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_baseline.yaml",
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size cho forward pass test.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.config)
    model = build_cnn_from_config(config)

    if args.compile:
        model = compile_cnn_model(model, config)

    if args.summary:
        model.summary()

    print_model_info(model)

    input_shape = to_tuple(config["model"]["input_shape"], length=3)

    test_forward_pass(
        model=model,
        input_shape=input_shape,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()