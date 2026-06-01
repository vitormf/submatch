import sys
from unittest.mock import patch, MagicMock

from submatch.gpu import check_gpu_mismatch

_CUDA_WARNING_PREFIX = "Warning: NVIDIA GPU detected"


def _mock_torch(cuda_version=None, mps_available=False):
    mock = MagicMock()
    mock.version.cuda = cuda_version
    mock.backends.mps.is_available.return_value = mps_available
    return mock


def test_no_warning_when_cuda_pytorch_installed():
    mock_torch = _mock_torch(cuda_version="12.4")
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            assert check_gpu_mismatch() is None


def test_warning_when_cpu_pytorch_and_nvidia_smi_present():
    mock_torch = _mock_torch(cuda_version=None, mps_available=False)
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("submatch.gpu.shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = check_gpu_mismatch()
    assert result is not None
    assert result.startswith(_CUDA_WARNING_PREFIX)
    assert "pip install torch" in result
    assert "https://download.pytorch.org/whl/cu124" in result


def test_no_warning_when_cpu_pytorch_but_no_nvidia_smi():
    mock_torch = _mock_torch(cuda_version=None, mps_available=False)
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("submatch.gpu.shutil.which", return_value=None):
            assert check_gpu_mismatch() is None


def test_no_warning_when_mps_available():
    mock_torch = _mock_torch(cuda_version=None, mps_available=True)
    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch("submatch.gpu.shutil.which", return_value="/usr/bin/nvidia-smi"):
            assert check_gpu_mismatch() is None


def test_no_warning_when_torch_import_fails():
    with patch.dict(sys.modules, {"torch": None}):
        assert check_gpu_mismatch() is None
