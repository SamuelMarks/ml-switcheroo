"""
Tests for Polygot Device Allocation using Adapter Delegation.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_allocator import transform_device_allocator

# Import Real Adapters to verify logic
from ml_switcheroo.frameworks.jax import JaxAdapter
from ml_switcheroo.frameworks.mlx import MLXAdapter
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def base_semantics():
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


# We use a factory to install the patch per-test invocation context
def get_rewriter(mgr, target):
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  # Fix: Use positional arguments for initialization
  return PivotRewriter(mgr, cfg)


@pytest.fixture(autouse=True)
def setup_hooks():
  hooks._HOOKS["device_allocator"] = transform_device_allocator
  hooks._PLUGINS_LOADED = True


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_jax_output(mock_get, base_semantics):
  mock_get.side_effect = lambda n: JaxAdapter() if n == "jax" else None

  rw = get_rewriter(base_semantics, "jax")
  res = rewrite_code(rw, "d = torch.device('cuda')")
  assert "jax.devices('gpu')[0]" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_gpu(mock_get, base_semantics):
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None

  rw = get_rewriter(base_semantics, "mlx")
  res = rewrite_code(rw, "d = torch.device('cuda')")
  assert "mx.Device(mx.gpu)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_cpu(mock_get, base_semantics):
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None

  rw = get_rewriter(base_semantics, "mlx")
  res = rewrite_code(rw, "d = torch.device('cpu')")
  assert "mx.Device(mx.cpu)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_mlx_output_index(mock_get, base_semantics):
  mock_get.side_effect = lambda n: MLXAdapter() if n == "mlx" else None

  rw = get_rewriter(base_semantics, "mlx")
  res = rewrite_code(rw, "d = torch.device('cuda:1')")
  # Check for correct logic (mx.gpu, 1)
  assert "mx.Device(mx.gpu, 1)" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_tf_output_gpu(mock_get, base_semantics):
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None

  rw = get_rewriter(base_semantics, "tensorflow")
  res = rewrite_code(rw, "d = torch.device('cuda')")
  assert "tf.device('GPU:0')" in res


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_tf_output_index(mock_get, base_semantics):
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None

  rw = get_rewriter(base_semantics, "tensorflow")
  res = rewrite_code(rw, "d = torch.device('cuda:2')")
  assert "tf.device('GPU:2')" in res
