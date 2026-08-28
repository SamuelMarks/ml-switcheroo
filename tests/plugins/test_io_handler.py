"""Test suite for the Io Handler module."""

import pytest
import libcst as cst
from typing import Generator
from unittest.mock import MagicMock, patch
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.io_handler import transform_io_calls
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The code rewriter.
      code (str): The source code.

  Returns:
      str: Rewritten code.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter() -> Generator[PivotRewriter, None, None]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: The testing rewriter instance.
  """
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  io_def: dict[str, dict[str, dict[str, str]]] = {"variants": {"jax": {"requires_plugin": "io_handler"}}}
  mgr.get_definition.side_effect = lambda n: ("io", io_def) if n in ["torch.save", "torch.load"] else None
  mgr.resolve_variant.side_effect = lambda aid, fw: io_def["variants"].get(fw) if fw == "jax" else None
  mgr.is_verified.return_value = True
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_get.side_effect = lambda n: JaxCoreAdapter() if n == "jax" else None
    yield PivotRewriter(mgr, cfg)


def test_save_transform_positional(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of save transform positional.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "def f():\n  torch.save(model, 'p')"
  res: str = rewrite_code(rewriter, code)
  assert "import orbax.checkpoint" in res
  assert "orbax.checkpoint.PyTreeCheckpointer().save" in res
  clean: str = res.replace(" ", "")
  assert "directory='p'" in clean
  assert "item=model" in clean


def test_save_transform_keywords(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of save transform keywords.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "def f():\n  torch.save(f='p', obj=m)"
  res: str = rewrite_code(rewriter, code)
  clean: str = res.replace(" ", "")
  assert "directory='p'" in clean
  assert "item=m" in clean


def test_load_transform(rewriter: PivotRewriter) -> None:
  """Loads transform.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "def f():\n  x = torch.load('p')"
  res: str = rewrite_code(rewriter, code)
  assert "orbax.checkpoint.PyTreeCheckpointer().restore('p')" in res


def test_ignored_if_wrong_target(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignored if wrong target.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=None):
    code: str = "torch.save(m, 'p')"
    assert "torch.save" in rewrite_code(rewriter, code)


def test_missing_func_name(rewriter: PivotRewriter) -> None:
  """Verifies behavior when _get_func_name returns None.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.SimpleString("'string'"))
  ctx: MagicMock = MagicMock()
  ctx.target_fw = "jax"
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_get.return_value = JaxCoreAdapter()
    res: cst.CSTNode = transform_io_calls(node, ctx)
    assert res is node


def test_missing_serialization_syntax(rewriter: PivotRewriter) -> None:
  """Verifies behavior when adapter returns None for serialization syntax.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("save")),
    args=[cst.Arg(cst.Name("m")), cst.Arg(cst.SimpleString("'p'"))],
  )
  ctx: MagicMock = MagicMock()
  ctx.target_fw = "jax"
  with patch("ml_switcheroo.plugins.io_handler.get_adapter") as mock_get:
    mock_adapter: MagicMock = MagicMock()
    mock_adapter.get_serialization_imports.return_value = []
    mock_adapter.get_serialization_syntax.return_value = None
    mock_get.return_value = mock_adapter
    res: cst.CSTNode = transform_io_calls(node, ctx)
    assert res is node
