"""
Tests for IO Handler Plugin using Real JAX Adapter logic.

Verifies:
1. Delegation to Adapter for Preamble Injection.
2. Delegation to Adapter for Syntax Generation.
3. Correct handling of save/load args extraction.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.io_handler import transform_io_calls
from ml_switcheroo.frameworks.jax import JaxAdapter
from ml_switcheroo.frameworks.numpy import NumpyAdapter


def rewrite_code(rewriter, code: str) -> str:
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  # Generic def for save/load
  io_def = {"variants": {"jax": {"requires_plugin": "io_handler"}}}

  mgr.get_definition.side_effect = lambda n: ("io", io_def) if n in ["torch.save", "torch.load"] else None
  mgr.resolve_variant.side_effect = lambda aid, fw: io_def["variants"].get(fw) if fw == "jax" else None
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")

  # Patch get_adapter to return real JAX adapter
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_get.side_effect = lambda n: JaxAdapter() if n == "jax" else None
    yield PivotRewriter(mgr, cfg)


def test_save_transform_positional(rewriter):
  """
  Input: torch.save(model, 'p')
  Expect: orbax.checkpoint...save(directory='p', item=model)
  """
  code = "def f():\n  torch.save(model, 'p')"
  res = rewrite_code(rewriter, code)

  assert "import orbax.checkpoint" in res
  assert "orbax.checkpoint.PyTreeCheckpointer().save" in res
  # Check argument mapping
  clean = res.replace(" ", "")
  assert "directory='p'" in clean
  assert "item=model" in clean


def test_save_transform_keywords(rewriter):
  """
  Input: torch.save(obj=m, f='p')
  Expect: Correct mapping regardless of source order.
  """
  code = "def f():\n  torch.save(f='p', obj=m)"
  res = rewrite_code(rewriter, code)

  clean = res.replace(" ", "")
  assert "directory='p'" in clean
  assert "item=m" in clean


def test_load_transform(rewriter):
  """
  Input: torch.load('p')
  Expect: orbax...restore('p')
  """
  code = "def f():\n  x = torch.load('p')"
  res = rewrite_code(rewriter, code)

  assert "orbax.checkpoint.PyTreeCheckpointer().restore('p')" in res


def test_ignored_if_wrong_target(rewriter):
  # Reconfigure context
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"

  # Ensure adapter returns None in patch
  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=None):
    code = "torch.save(m, 'p')"
    assert "torch.save" in rewrite_code(rewriter, code)
