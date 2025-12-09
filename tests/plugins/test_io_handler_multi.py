"""
Tests for Multi-Backend IO Logic (Numpy, TF).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.io_handler import transform_io_calls

from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def base_semantics():
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  io_def = {"variants": {"numpy": {"requires_plugin": "io_handler"}, "tensorflow": {"requires_plugin": "io_handler"}}}

  mgr.get_definition.return_value = ("io", io_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: io_def["variants"].get(fw)
  mgr.is_verified.return_value = True
  return mgr


# Factory for rewriter with patched adapter
def get_rw(mgr, target):
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(mgr, cfg)


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_numpy_save(mock_get, base_semantics):
  mock_get.side_effect = lambda n: NumpyAdapter() if n == "numpy" else None

  rw = get_rw(base_semantics, "numpy")
  res = rewrite_code(rw, "def f():\n  torch.save(t, 'f')")

  assert "import numpy as np" in res
  assert "np.save(file='f', arr=t)" in res


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_numpy_load(mock_get, base_semantics):
  mock_get.side_effect = lambda n: NumpyAdapter() if n == "numpy" else None

  rw = get_rw(base_semantics, "numpy")
  res = rewrite_code(rw, "def f():\n  x = torch.load('f')")

  assert "np.load(file='f')" in res


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_tensorflow_save(mock_get, base_semantics):
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None

  rw = get_rewriter(base_semantics, "tensorflow")
  res = rewrite_code(rw, "def f():\n  torch.save(d, 'p')")

  assert "import tensorflow as tf" in res
  assert "tf.io.write_file('p', d)" in res


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_tensorflow_load(mock_get, base_semantics):
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None

  rw = get_rewriter(base_semantics, "tensorflow")
  res = rewrite_code(rw, "def f():\n  x = torch.load('p')")

  assert "tf.io.read_file('p')" in res


# Helper reused in tests but defined outside in snippet context usually
def get_rewriter(mgr, target):
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(mgr, cfg)
