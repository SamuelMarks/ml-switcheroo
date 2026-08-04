"""Test suite for the Io Handler module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.io_handler import transform_io_calls
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def rewrite_code(rewriter, code: str) -> str:
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  io_def = {"variants": {"jax": {"requires_plugin": "io_handler"}}}
  mgr.get_definition.side_effect = lambda n: ("io", io_def) if n in ["torch.save", "torch.load"] else None
  mgr.resolve_variant.side_effect = lambda aid, fw: io_def["variants"].get(fw) if fw == "jax" else None
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_get.side_effect = lambda n: JaxCoreAdapter() if n == "jax" else None
    yield PivotRewriter(mgr, cfg)


def test_save_transform_positional(rewriter):
  """Verifies the behavior of save transform positional."""
  code = "def f():\n  torch.save(model, 'p')"
  res = rewrite_code(rewriter, code)
  assert "import orbax.checkpoint" in res
  assert "orbax.checkpoint.PyTreeCheckpointer().save" in res
  clean = res.replace(" ", "")
  assert "directory='p'" in clean
  assert "item=model" in clean


def test_save_transform_keywords(rewriter):
  """Verifies the behavior of save transform keywords."""
  code = "def f():\n  torch.save(f='p', obj=m)"
  res = rewrite_code(rewriter, code)
  clean = res.replace(" ", "")
  assert "directory='p'" in clean
  assert "item=m" in clean


def test_load_transform(rewriter):
  """Loads transform."""
  code = "def f():\n  x = torch.load('p')"
  res = rewrite_code(rewriter, code)
  assert "orbax.checkpoint.PyTreeCheckpointer().restore('p')" in res


def test_ignored_if_wrong_target(rewriter):
  """Verifies the behavior of ignored if wrong target."""
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=None):
    code = "torch.save(m, 'p')"
    assert "torch.save" in rewrite_code(rewriter, code)


def test_missing_func_name(rewriter):
  """Verifies behavior when _get_func_name returns None."""
  node = cst.Call(func=cst.SimpleString("'string'"))
  ctx = MagicMock()
  ctx.target_fw = "jax"
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_get.return_value = JaxCoreAdapter()
    res = transform_io_calls(node, ctx)
    assert res is node


def test_missing_serialization_syntax(rewriter):
  """Verifies behavior when adapter returns None for serialization syntax."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("save")),
    args=[cst.Arg(cst.Name("m")), cst.Arg(cst.SimpleString("'p'"))],
  )
  ctx = MagicMock()
  ctx.target_fw = "jax"
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.get_serialization_imports.return_value = []
    mock_adapter.get_serialization_syntax.return_value = None
    mock_get.return_value = mock_adapter
    res = transform_io_calls(node, ctx)
    assert res is node
