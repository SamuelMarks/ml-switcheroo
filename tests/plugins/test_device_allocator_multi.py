"""
Tests for Polygot Device Allocation (JAX/TF/MLX).

Verifies that:
1. JAX output matches `jax.devices('gpu')[0]`.
2. MLX output matches `mx.Device(mx.gpu)`.
3. Tensorflow output matches `tf.device('GPU:0')`.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_allocator import transform_device_allocator


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def base_rewriter():
  hooks._HOOKS["device_allocator"] = transform_device_allocator
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()
  # Generic definition valid for all tests
  device_def = {
    "requires_plugin": "device_allocator",
    "std_args": ["type"],
    "variants": {
      "torch": {"api": "torch.device"},
      "jax": {"requires_plugin": "device_allocator"},
      "mlx": {"requires_plugin": "device_allocator"},
      "tensorflow": {"requires_plugin": "device_allocator"},
    },
  }

  mgr.get_definition.return_value = ("device", device_def)
  mgr.get_known_apis.return_value = {"device": device_def}
  mgr.is_verified.return_value = True

  return mgr


def get_rewriter(mgr, target):
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(semantics=mgr, config=cfg)


def test_jax_output(base_rewriter):
  rw = get_rewriter(base_rewriter, "jax")
  res = rewrite_code(rw, "d = torch.device('cuda')")
  assert "jax.devices('gpu')[0]" in res


def test_mlx_output_gpu(base_rewriter):
  rw = get_rewriter(base_rewriter, "mlx")
  res = rewrite_code(rw, "d = torch.device('cuda')")
  assert "mx.Device(mx.gpu)" in res


def test_mlx_output_cpu(base_rewriter):
  rw = get_rewriter(base_rewriter, "mlx")
  res = rewrite_code(rw, "d = torch.device('cpu')")
  assert "mx.Device(mx.cpu)" in res


def test_mlx_output_index(base_rewriter):
  rw = get_rewriter(base_rewriter, "mlx")
  res = rewrite_code(rw, "d = torch.device('cuda:1')")
  # Should append index arg
  clean = res.replace(" ", "")
  assert "mx.Device(mx.gpu,1)" in clean


def test_tf_output_gpu(base_rewriter):
  rw = get_rewriter(base_rewriter, "tensorflow")
  res = rewrite_code(rw, "d = torch.device('cuda')")
  assert "tf.device('GPU:0')" in res


def test_tf_output_index(base_rewriter):
  rw = get_rewriter(base_rewriter, "tensorflow")
  res = rewrite_code(rw, "d = torch.device('cuda:2')")
  assert "tf.device('GPU:2')" in res
