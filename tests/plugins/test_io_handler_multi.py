"""Test suite for the Io Handler Multi module."""

from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.numpy import NumpyAdapter
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.plugins.io_handler import transform_io_calls
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): Code rewriter.
      code (str): The code string.

  Returns:
      str: The rewritten code string.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def base_semantics() -> MagicMock:
  """Provides a mock base semantics for testing.

  Returns:
      MagicMock: Mock base semantics.
  """
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  io_def: dict[str, dict[str, dict[str, str]]] = {
    "variants": {"numpy": {"requires_plugin": "io_handler"}, "tensorflow": {"requires_plugin": "io_handler"}}
  }
  mgr.get_definition.return_value = ("io", io_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: io_def["variants"].get(fw)
  mgr.is_verified.return_value = True
  return mgr


def get_rw(mgr: MagicMock, target: str) -> PivotRewriter:
  """Gets rw.

  Args:
      mgr (MagicMock): Manager mock.
      target (str): Target framework string.

  Returns:
      PivotRewriter: The rewriter.
  """
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework=target)
  return PivotRewriter(mgr, cfg)


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_numpy_save(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of NumPy save.

  Args:
      mock_get (MagicMock): Get adapter mock.
      base_semantics (MagicMock): Base semantics mock fixture.
  """
  mock_get.side_effect = lambda n: NumpyAdapter() if n == "numpy" else None
  rw: PivotRewriter = get_rw(base_semantics, "numpy")
  res: str = rewrite_code(rw, "def f():\n  torch.save(t, 'f')")
  assert "import numpy as np" in res
  assert "np.save(file='f', arr=t)" in res


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_numpy_load(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of NumPy load.

  Args:
      mock_get (MagicMock): Get adapter mock.
      base_semantics (MagicMock): Base semantics mock fixture.
  """
  mock_get.side_effect = lambda n: NumpyAdapter() if n == "numpy" else None
  rw: PivotRewriter = get_rw(base_semantics, "numpy")
  res: str = rewrite_code(rw, "def f():\n  x = torch.load('f')")
  assert "np.load(file='f')" in res


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_tensorflow_save(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of TensorFlow save.

  Args:
      mock_get (MagicMock): Get adapter mock.
      base_semantics (MagicMock): Base semantics mock fixture.
  """
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None
  rw: PivotRewriter = get_rw(base_semantics, "tensorflow")
  res: str = rewrite_code(rw, "def f():\n  torch.save(d, 'p')")
  assert "import tensorflow as tf" in res
  assert "tf.io.write_file('p', d)" in res


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
def test_tensorflow_load(mock_get: MagicMock, base_semantics: MagicMock) -> None:
  """Verifies the behavior of TensorFlow load.

  Args:
      mock_get (MagicMock): Get adapter mock.
      base_semantics (MagicMock): Base semantics mock fixture.
  """
  mock_get.side_effect = lambda n: TensorFlowAdapter() if n == "tensorflow" else None
  rw: PivotRewriter = get_rw(base_semantics, "tensorflow")
  res: str = rewrite_code(rw, "def f():\n  x = torch.load('p')")
  assert "tf.io.read_file('p')" in res
