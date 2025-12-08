"""
Tests for I/O Handler Multi-Backend Support.

Verifies that:
1. Numpy target rewrites to `np.save` / `np.load`.
2. TensorFlow target rewrites to `tf.io`.
3. Argument ordering is respected (swapped if needed).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.io_handler import transform_io_calls


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def base_rewriter():
  # Register Hook
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True

  # Semantics
  mgr = MagicMock()

  # Generic def allowing plugin usage for all targets
  io_def = {
    "requires_plugin": "io_handler",
    "std_args": ["obj", "f"],  # 'obj' is pos 0, 'f' is pos 1 in torch.save
    "variants": {
      "torch": {"api": "torch.save"},
      "jax": {"requires_plugin": "io_handler"},
      "numpy": {"requires_plugin": "io_handler"},
      "tensorflow": {"requires_plugin": "io_handler"},
    },
  }

  load_def = {
    "requires_plugin": "io_handler",
    "std_args": ["f"],
    "variants": {
      "torch": {"api": "torch.load"},
      "jax": {"requires_plugin": "io_handler"},
      "numpy": {"requires_plugin": "io_handler"},
      "tensorflow": {"requires_plugin": "io_handler"},
    },
  }

  def side_effect(name):
    if "save" in name:
      return "save", io_def
    if "load" in name:
      return "load", load_def
    return None

  mgr.get_definition.side_effect = side_effect
  mgr.get_known_apis.return_value = {"save": io_def, "load": load_def}
  mgr.is_verified.return_value = True
  return mgr


def get_rewriter(mgr, target):
  cfg = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(semantics=mgr, config=cfg)


# --- Numpy Tests ---


def test_numpy_save(base_rewriter):
  rw = get_rewriter(base_rewriter, "numpy")
  # Wrap in function to catch preamble
  code = "def f():\n    torch.save(tensor, 'file.npy')"
  res = rewrite_code(rw, code)

  assert "import numpy as np" in res
  assert "np.save('file.npy', tensor)" in res


def test_numpy_load(base_rewriter):
  rw = get_rewriter(base_rewriter, "numpy")
  code = "def f():\n    x = torch.load('file.npy')"
  res = rewrite_code(rw, code)

  assert "np.load('file.npy')" in res


# --- TensorFlow Tests ---


def test_tensorflow_save(base_rewriter):
  rw = get_rewriter(base_rewriter, "tensorflow")
  code = "def f():\n    torch.save(data, 'path')"
  res = rewrite_code(rw, code)

  assert "import tensorflow as tf" in res
  # Arguments order input: (obj, file) -> output: (file, obj)
  # tf.io.write_file(filename, contents)
  assert "tf.io.write_file('path', data)" in res


def test_tensorflow_load(base_rewriter):
  rw = get_rewriter(base_rewriter, "tensorflow")
  code = "def f():\n    x = torch.load('path')"
  res = rewrite_code(rw, code)

  assert "tf.io.read_file('path')" in res
