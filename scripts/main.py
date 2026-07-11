"""
scripts/main.py

Menu tổng để chạy pipeline Audio Deepfake Detection.

File này không xử lý trực tiếp dữ liệu/mô hình.
Nó chỉ gọi lại các module trong src/ bằng subprocess.

Hỗ trợ:
- CNN baseline        : configs/cnn_baseline.yaml
- CNN v2             : configs/cnn_v2.yaml
- MobileNetV3-Small  : configs/mobilenetv3_small.yaml

Cách chạy từ thư mục gốc project:

    python scripts/main.py

Ghi chú:
- Metadata gốc và Log-Mel được tạo một lần, dùng chung cho CNN baseline/CNN v2.
- MobileNetV3-Small dùng thêm bước precompute để chuyển Log-Mel thành feature 128x128x3.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODEL_CONFIGS = {
    "cnn_baseline": {
        "name": "CNN baseline",
        "config": "configs/cnn_baseline.yaml",
        "model_module": "src.models.cnn_baseline",
        "train_module": "src.training.train_cnn",
        "feature_type": "logmel",
    },
    "cnn_v2": {
        "name": "CNN v2 medium",
        "config": "configs/cnn_v2.yaml",
        "model_module": "src.models.cnn_baseline",
        "train_module": "src.training.train_cnn",
        "feature_type": "logmel",
    },
    "mobilenetv3_small": {
        "name": "MobileNetV3-Small",
        "config": "configs/mobilenetv3_small.yaml",
        "model_module": "src.models.mobilenetv3_small",
        "train_module": "src.training.train_mobilenetv3",
        "feature_type": "mobilenetv3_precomputed",
    },
}


CURRENT_MODEL_KEY = "cnn_baseline"


# ============================================================
# Common helpers
# ============================================================

def get_current_model() -> dict[str, str]:
    """Lấy thông tin model/config đang chọn."""
    return MODEL_CONFIGS[CURRENT_MODEL_KEY]


def get_current_config_path() -> str:
    """Lấy config path của model đang chọn."""
    return get_current_model()["config"]


def run_command(command: list[str]) -> None:
    """
    Chạy command bằng Python hiện tại.

    Dùng sys.executable để nếu đang ở .venv thì vẫn chạy đúng Python của .venv.
    """
    print("\n" + "=" * 80)
    print("RUNNING COMMAND")
    print("=" * 80)
    print(" ".join(command))
    print("=" * 80 + "\n")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        shell=False,
    )

    if result.returncode != 0:
        print("\n" + "=" * 80)
        print("COMMAND FAILED")
        print("=" * 80)
        print(f"Return code: {result.returncode}")
        print("Hãy kiểm tra log lỗi phía trên.")
    else:
        print("\n" + "=" * 80)
        print("COMMAND FINISHED OK")
        print("=" * 80)


def ask_split(default: str = "train", allow_all: bool = True) -> str:
    """Hỏi split."""
    valid = ["train", "dev", "eval"]

    if allow_all:
        valid.append("all")

    valid_text = "/".join(valid)
    split = input(f"Chọn split [{valid_text}], mặc định {default}: ").strip()

    if split == "":
        split = default

    if split not in valid:
        print(f"Split không hợp lệ. Dùng mặc định: {default}")
        split = default

    return split


def ask_int(prompt: str, default: int | None = None) -> int | None:
    """Hỏi số nguyên."""
    value = input(prompt).strip()

    if value == "":
        return default

    try:
        return int(value)
    except ValueError:
        print(f"Giá trị không hợp lệ. Dùng mặc định: {default}")
        return default


def ask_float(prompt: str, default: float | None = None) -> float | None:
    """Hỏi số thực."""
    value = input(prompt).strip()

    if value == "":
        return default

    try:
        return float(value)
    except ValueError:
        print(f"Giá trị không hợp lệ. Dùng mặc định: {default}")
        return default


def ask_yes_no(prompt: str, default_yes: bool = False) -> bool:
    """Hỏi yes/no."""
    suffix = "[Y/n]" if default_yes else "[y/N]"
    value = input(f"{prompt} {suffix}: ").strip().lower()

    if value == "":
        return default_yes

    return value == "y"


def ask_optional_text(prompt: str) -> str | None:
    """Hỏi chuỗi optional."""
    value = input(prompt).strip()

    if value == "":
        return None

    return value


def show_current_model() -> None:
    """In model/config hiện tại."""
    model = get_current_model()

    print("\n" + "=" * 80)
    print("CURRENT MODEL")
    print("=" * 80)
    print(f"Key     : {CURRENT_MODEL_KEY}")
    print(f"Name    : {model['name']}")
    print(f"Config  : {model['config']}")
    print(f"Train   : {model['train_module']}")
    print(f"Model   : {model['model_module']}")
    print(f"Feature : {model['feature_type']}")
    print("=" * 80)


def select_model() -> None:
    """Chọn model/config dùng cho các bước train/evaluate/validate."""
    global CURRENT_MODEL_KEY

    print("\n" + "=" * 80)
    print("CHỌN MODEL / CONFIG")
    print("=" * 80)

    keys = list(MODEL_CONFIGS.keys())

    for idx, key in enumerate(keys, start=1):
        item = MODEL_CONFIGS[key]
        current_mark = "  <-- current" if key == CURRENT_MODEL_KEY else ""
        print(f"{idx}. {key:18s} | {item['name']} | {item['config']}{current_mark}")

    value = input("Chọn model, Enter để giữ nguyên: ").strip()

    if value == "":
        return

    try:
        idx = int(value)
        if idx < 1 or idx > len(keys):
            raise ValueError
    except ValueError:
        print("Lựa chọn không hợp lệ. Giữ nguyên model hiện tại.")
        return

    CURRENT_MODEL_KEY = keys[idx - 1]
    show_current_model()


# ============================================================
# Shared data pipeline
# ============================================================

def prepare_metadata() -> None:
    """
    Tạo metadata gốc từ protocol ASVspoof.

    Metadata gốc dùng chung cho CNN baseline, CNN v2 và MobileNetV3.
    Khuyến nghị dùng config CNN baseline vì đây là config gốc chứa protocol paths.
    """
    config_path = "configs/cnn_baseline.yaml"

    command = [
        sys.executable,
        "-m",
        "src.data.make_metadata",
        "--config",
        config_path,
    ]

    run_command(command)


def extract_logmel() -> None:
    """
    Extract Log-Mel feature gốc.

    Log-Mel này dùng trực tiếp cho CNN baseline/CNN v2,
    đồng thời là source để precompute MobileNetV3 feature.
    """
    config_path = "configs/cnn_baseline.yaml"

    split = ask_split(default="train", allow_all=True)

    max_samples = ask_int(
        "Giới hạn số mẫu để test nhanh? Enter để chạy full: ",
        default=None,
    )

    force = ask_yes_no("Ghi đè file .npy đã tồn tại?", default_yes=False)

    command = [
        sys.executable,
        "-m",
        "src.features.logmel",
        "--config",
        config_path,
        "--split",
        split,
    ]

    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])

    if force:
        command.append("--force")

    run_command(command)


def precompute_mobilenetv3_features() -> None:
    """
    Tạo feature MobileNetV3-Small 128x128x3 từ Log-Mel đã có.
    """
    config_path = MODEL_CONFIGS["mobilenetv3_small"]["config"]

    split = ask_split(default="train", allow_all=True)

    max_samples = ask_int(
        "Giới hạn số mẫu để test nhanh? Enter để chạy full: ",
        default=None,
    )

    force = ask_yes_no("Ghi đè feature MobileNetV3 đã tồn tại?", default_yes=False)
    validate_only = ask_yes_no("Chỉ validate feature đã precompute?", default_yes=False)

    command = [
        sys.executable,
        "-m",
        "src.features.mobilenetv3_precompute",
        "--config",
        config_path,
        "--split",
        split,
    ]

    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])

    if force:
        command.append("--force")

    if validate_only:
        command.append("--validate-only")

    run_command(command)


# ============================================================
# Model pipeline
# ============================================================

def validate_dataset() -> None:
    """Validate dataset theo config đang chọn."""
    split = ask_split(default="train", allow_all=False)

    command = [
        sys.executable,
        "-m",
        "src.data.dataset_loader",
        "--config",
        get_current_config_path(),
        "--split",
        split,
        "--validate-files",
    ]

    run_command(command)


def build_model_test() -> None:
    """Build model, in summary, compile theo config đang chọn."""
    model = get_current_model()

    batch_size = ask_int("Batch size, mặc định 8: ", default=8)

    command = [
        sys.executable,
        "-m",
        model["model_module"],
        "--config",
        model["config"],
        "--summary",
        "--compile",
        "--batch-size",
        str(batch_size),
    ]

    if CURRENT_MODEL_KEY == "mobilenetv3_small":
        allow_fallback = ask_yes_no(
            "Cho phép fallback weights=None nếu không load được ImageNet?",
            default_yes=False,
        )
        if allow_fallback:
            command.append("--allow-random-fallback")

    run_command(command)


def train_debug() -> None:
    """Train debug theo model/config đang chọn."""
    model = get_current_model()

    epochs = ask_int("Số epoch debug, mặc định 5: ", default=5)
    inspect = ask_yes_no("Inspect batch trước khi train?", default_yes=True)

    command = [
        sys.executable,
        "-m",
        model["train_module"],
        "--config",
        model["config"],
        "--debug",
        "--epochs",
        str(epochs),
    ]

    if inspect:
        command.append("--inspect-batch")

    if CURRENT_MODEL_KEY == "mobilenetv3_small":
        fine_tune = ask_yes_no("Bật fine-tune stage 2?", default_yes=False)
        if fine_tune:
            command.append("--fine-tune")

        allow_fallback = ask_yes_no(
            "Cho phép fallback weights=None nếu không load được ImageNet?",
            default_yes=False,
        )
        if allow_fallback:
            command.append("--allow-random-fallback")

    run_command(command)


def train_full() -> None:
    """Train full theo model/config đang chọn."""
    show_current_model()

    confirm = ask_yes_no(
        "Train full có thể mất lâu. Bạn chắc chắn muốn chạy?",
        default_yes=False,
    )

    if not confirm:
        print("Đã hủy train full.")
        return

    model = get_current_model()

    command = [
        sys.executable,
        "-m",
        model["train_module"],
        "--config",
        model["config"],
    ]

    if CURRENT_MODEL_KEY == "mobilenetv3_small":
        fine_tune = ask_yes_no("Bật fine-tune stage 2?", default_yes=False)
        if fine_tune:
            command.append("--fine-tune")

        allow_fallback = ask_yes_no(
            "Cho phép fallback weights=None nếu không load được ImageNet?",
            default_yes=False,
        )
        if allow_fallback:
            command.append("--allow-random-fallback")

    run_command(command)


# ============================================================
# Evaluation
# ============================================================

def evaluate_debug() -> None:
    """Evaluate debug theo config đang chọn."""
    split = input("Chọn split evaluate [dev/eval/train], mặc định dev: ").strip()

    if split == "":
        split = "dev"

    max_samples = ask_int("Số mẫu evaluate debug, mặc định 500: ", default=500)

    threshold = ask_float(
        "Threshold, Enter để dùng config mặc định 0.5: ",
        default=None,
    )

    model_path = ask_optional_text(
        "Model path riêng, Enter để dùng checkpoint trong config: "
    )

    command = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--config",
        get_current_config_path(),
        "--split",
        split,
        "--max-samples",
        str(max_samples),
    ]

    if threshold is not None:
        command.extend(["--threshold", str(threshold)])

    if model_path is not None:
        command.extend(["--model-path", model_path])

    run_command(command)


def evaluate_full() -> None:
    """Evaluate full theo config đang chọn."""
    split = input("Chọn split evaluate full [dev/eval/train], mặc định dev: ").strip()

    if split == "":
        split = "dev"

    threshold = ask_float(
        "Threshold, Enter để dùng config mặc định 0.5: ",
        default=None,
    )

    model_path = ask_optional_text(
        "Model path riêng, Enter để dùng checkpoint trong config: "
    )

    output_dir = ask_optional_text(
        "Output dir riêng, Enter để dùng mặc định: "
    )

    command = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--config",
        get_current_config_path(),
        "--split",
        split,
    ]

    if threshold is not None:
        command.extend(["--threshold", str(threshold)])

    if model_path is not None:
        command.extend(["--model-path", model_path])

    if output_dir is not None:
        command.extend(["--output-dir", output_dir])

    run_command(command)


def evaluate_custom_threshold() -> None:
    """Evaluate với threshold tùy chọn."""
    split = input("Chọn split [dev/eval/train], mặc định dev: ").strip()

    if split == "":
        split = "dev"

    max_samples = ask_int(
        "Số mẫu evaluate, Enter để chạy full split: ",
        default=None,
    )

    threshold = ask_float(
        "Nhập threshold, ví dụ 0.759 hoặc 0.864: ",
        default=0.5,
    )

    model_path = ask_optional_text(
        "Model path riêng, Enter để dùng checkpoint trong config: "
    )

    output_dir = ask_optional_text(
        "Output dir riêng, Enter để dùng mặc định: "
    )

    command = [
        sys.executable,
        "-m",
        "src.evaluation.evaluate",
        "--config",
        get_current_config_path(),
        "--split",
        split,
        "--threshold",
        str(threshold),
    ]

    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])

    if model_path is not None:
        command.extend(["--model-path", model_path])

    if output_dir is not None:
        command.extend(["--output-dir", output_dir])

    run_command(command)


def export_tflite() -> None:
    """Export mô hình sang TFLite."""
    formats_input = ask_optional_text(
        "Nhập các format cách nhau bởi dấu cách [fp32, fp16, dynamic] hoặc Enter để dùng [fp32 fp16]: "
    )
    if formats_input:
        formats = formats_input.split()
    else:
        formats = ["fp32", "fp16"]
    
    command = [
        sys.executable,
        "-m",
        "src.deployment.export_tflite",
        "--config",
        get_current_config_path(),
        "--formats",
    ] + formats
    
    run_command(command)


def evaluate_tflite() -> None:
    """Evaluate mô hình TFLite."""
    split = input("Chọn split [dev/eval], mặc định eval: ").strip()
    if split == "":
        split = "eval"
        
    fmt = input("Chọn format [fp32/fp16/dynamic], mặc định fp16: ").strip()
    if fmt == "":
        fmt = "fp16"
        
    threshold = ask_float(
        "Nhập threshold (VD: 0.869), Enter để dùng mặc định 0.5: ",
        default=0.5,
    )
    
    command = [
        sys.executable,
        "-m",
        "src.deployment.evaluate_tflite",
        "--config",
        get_current_config_path(),
        "--split",
        split,
        "--format",
        fmt,
        "--threshold",
        str(threshold),
    ]
    
    run_command(command)


# ============================================================
# Menu
# ============================================================

def show_menu() -> None:
    model = get_current_model()

    print("\n" + "=" * 80)
    print("AUDIO DEEPFAKE DETECTION - PIPELINE MENU")
    print("=" * 80)
    print(f"Current model : {CURRENT_MODEL_KEY} | {model['name']}")
    print(f"Current config: {model['config']}")
    print("=" * 80)
    print("1.  Chọn model/config")
    print("2.  Prepare metadata gốc")
    print("3.  Extract Log-Mel feature")
    print("4.  Precompute MobileNetV3 feature 128x128x3")
    print("5.  Validate dataset / feature files")
    print("6.  Build model + summary + forward test")
    print("7.  Train debug")
    print("8.  Evaluate debug")
    print("9.  Train full")
    print("10. Evaluate full")
    print("11. Evaluate with custom threshold")
    print("12. Show current model/config")
    print("13. Export TFLite")
    print("14. Evaluate TFLite")
    print("0.  Exit")
    print("=" * 80)


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    show_current_model()

    while True:
        show_menu()

        choice = input("Chọn bước cần chạy: ").strip()

        if choice == "1":
            select_model()

        elif choice == "2":
            prepare_metadata()

        elif choice == "3":
            extract_logmel()

        elif choice == "4":
            precompute_mobilenetv3_features()

        elif choice == "5":
            validate_dataset()

        elif choice == "6":
            build_model_test()

        elif choice == "7":
            train_debug()

        elif choice == "8":
            evaluate_debug()

        elif choice == "9":
            train_full()

        elif choice == "10":
            evaluate_full()

        elif choice == "11":
            evaluate_custom_threshold()

        elif choice == "12":
            show_current_model()

        elif choice == "13":
            export_tflite()

        elif choice == "14":
            evaluate_tflite()

        elif choice == "0":
            print("Thoát.")
            break

        else:
            print("Lựa chọn không hợp lệ. Hãy chọn lại.")


if __name__ == "__main__":
    main()
