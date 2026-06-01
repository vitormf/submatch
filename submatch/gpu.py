from __future__ import annotations
import shutil


def check_gpu_mismatch() -> str | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.version.cuda is not None:
            return None
        if torch.backends.mps.is_available():
            return None
        if shutil.which("nvidia-smi") is None:
            return None
        return (
            "Warning: NVIDIA GPU detected but PyTorch was installed without CUDA support.\n"
            "Whisper will run on CPU, which is significantly slower.\n"
            "To enable GPU acceleration, reinstall PyTorch:\n\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n"
            "For other CUDA versions or AMD GPUs: https://pytorch.org/get-started/locally/"
        )
    except Exception:
        return None
