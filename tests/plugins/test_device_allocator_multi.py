"""Test suite for the Device Allocator Multi module."""

import typing
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
  """Rewrites code."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def base_semantics() -> MagicMock:
  """Docstring."""
  mgr = MagicMock()
  variants: dict[str, typing.Any] = {
    "jax": {"requires_plugin": "device_allocator"},
    "mlx": {"requires_plugin": "device_allocator"},
    "tensorflow": {"requires_plugin": "device_allocator"},
  }
  mgr.get_definition.return_value = ("device", {"variants": variants})
  mgr.resolve_variant.side_effect = lambda aid, fw: variants.get(fw)
  mgr.is_verified.return_value = True
  return mgr


def get_rewriter(mgr: MagicMock, target: str) -> PivotRewriter:
  """Gets rewriter."""
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(mgr, cfg)


@pytest.fixture(autouse=True)
def setup_hooks() -> None:
  """Helper to setup hooks."""
  hooks._HOOKS["device_allocator"] = transform_device_allocator
  hooks._PLUGINS_LOADED = True


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_jax_output(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of JAX output."""
  mock_get.side_effect = lambda n: JaxCoreAdapter() if n == "jax" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "jax")
  res: str = rewrite_code(rw, "d = torch.device('cuda')")
  assert "jax.devices('gpu')[0]" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_gpu(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of MLX output gpu."""
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "mlx")
  res: str = rewrite_code(rw, "d = torch.device('cuda')")
  assert "mx.Device(mx.gpu)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_cpu(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of MLX output cpu."""
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "mlx")
  res: str = rewrite_code(rw, "d = torch.device('cpu')")
  assert "mx.Device(mx.cpu)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_index(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of MLX output index."""
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "mlx")
  res: str = rewrite_code(rw, "d = torch.device('cuda:1')")
  assert "mx.Device(mx.gpu, 1)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_tf_output_gpu(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of tf output gpu."""
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "tensorflow")
  res: str = rewrite_code(rw, "d = torch.device('cuda')")
  assert "tf.device('GPU:0')" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_tf_output_index(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of tf output index."""
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None
  rw: PivotRewriter = get_rewriter(base_semantics, "tensorflow")
  res: str = rewrite_code(rw, "d = torch.device('cuda:2')")
  assert "tf.device('GPU:2')" in res
