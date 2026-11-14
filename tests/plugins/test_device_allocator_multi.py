"""Test suite for the Device Allocator Multi module."""

from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.plugins.device_allocator import transform_device_allocator
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def base_semantics() -> MagicMock:
  """Create a configured `MagicMock` fixture for device allocator multi tests.

  Returns:
      Configured `base_semantics` fixture value (MagicMock).
  """
  mgr = MagicMock()
  variants = {
    "jax": {"requires_plugin": "device_allocator"},
    "mlx": {"requires_plugin": "device_allocator"},
    "tensorflow": {"requires_plugin": "device_allocator"},
  }
  mgr.get_definition.return_value = ("device", {"variants": variants})
  mgr.resolve_variant.side_effect = lambda aid, fw: variants.get(fw)
  mgr.is_verified.return_value = True
  return mgr


def get_rewriter(mgr: MagicMock, target: str) -> PivotRewriter:
  """Gets rewriter.

  Args:
      mgr: Semantics manager test instance (`MagicMock`).
      target: Target framework or dialect identifier string.

  Returns:
      Resulting `PivotRewriter` from get rewriter.
  """
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(mgr, cfg)


@pytest.fixture(autouse=True)
def setup_hooks() -> None:
  """Create a configured `None` fixture for device allocator multi tests."""
  hooks.register_hook("device_allocator")(transform_device_allocator)
  hooks.set_plugins_loaded(True)


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_jax_output(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verify `get_rewriter` emits `jax.devices('gpu')[0]` for JAX output.

  Args:
      mock_get: Mock object for get.
      base_semantics: Input `base_semantics` parameter value (`MagicMock`).
  """
  mock_get.side_effect = lambda n: JaxCoreAdapter() if n == "jax" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "jax")
  res: str = rewrite_code(rw, "d = torch.device('cuda')")
  assert "jax.devices('gpu')[0]" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_gpu(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verify `get_rewriter` emits `mx.Device(mx.gpu)` for MLX output gpu.

  Args:
      mock_get: Mock object for get.
      base_semantics: Input `base_semantics` parameter value (`MagicMock`).
  """
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "mlx")
  res: str = rewrite_code(rw, "d = torch.device('cuda')")
  assert "mx.Device(mx.gpu)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_cpu(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verify `get_rewriter` emits `mx.Device(mx.cpu)` for MLX output cpu.

  Args:
      mock_get: Mock object for get.
      base_semantics: Input `base_semantics` parameter value (`MagicMock`).
  """
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "mlx")
  res: str = rewrite_code(rw, "d = torch.device('cpu')")
  assert "mx.Device(mx.cpu)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_index(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verify `get_rewriter` emits `mx.Device(mx.gpu, 1)` for MLX output index.

  Args:
      mock_get: Mock object for get.
      base_semantics: Input `base_semantics` parameter value (`MagicMock`).
  """
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "mlx")
  res: str = rewrite_code(rw, "d = torch.device('cuda:1')")
  assert "mx.Device(mx.gpu, 1)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_tf_output_gpu(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verify `get_rewriter` emits `tf.device('GPU:0')` for TensorFlow output gpu.

  Args:
      mock_get: Mock object for get.
      base_semantics: Input `base_semantics` parameter value (`MagicMock`).
  """
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "tensorflow")
  res: str = rewrite_code(rw, "d = torch.device('cuda')")
  assert "tf.device('GPU:0')" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_tf_output_index(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verify `get_rewriter` emits `tf.device('GPU:2')` for TensorFlow output index.

  Args:
      mock_get: Mock object for get.
      base_semantics: Input `base_semantics` parameter value (`MagicMock`).
  """
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "tensorflow")
  res: str = rewrite_code(rw, "d = torch.device('cuda:2')")
  assert "tf.device('GPU:2')" in res
