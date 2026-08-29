"""Test suite for the Io Handler module."""

from typing import Generator, List
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.plugins.io_handler import _get_arg, _get_func_name, transform_io_calls
from tests.conftest import TestRewriter as PivotRewriter


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


# --- Merged from test_io_handler_extra.py ---


def test_get_arg_wrong_keyword() -> None:
  """Gets argument wrong keyword."""
  arg: cst.Arg = cst.Arg(value=cst.Name("val"), keyword=cst.Name("wrong_name"))
  assert _get_arg([arg], 0, "obj") is None


# --- Merged from test_io_handler_missing.py ---


def test_get_func_name() -> None:
  """Gets function name."""
  assert _get_func_name(cst.Call(func=cst.Name("foo"))) == "foo"
  assert _get_func_name(cst.Call(func=cst.SimpleString("'bar'"))) is None


def test_get_arg() -> None:
  """Gets argument."""
  args: List[cst.Arg] = [cst.Arg(value=cst.Name("a"), keyword=cst.Name("a_kw"))]
  assert _get_arg(args, 1, "missing") is None


def test_transform_io_calls_misses() -> None:
  """Transforms I/O calls misses."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  node1: cst.Call = cst.Call(func=cst.Name("foo"))
  assert transform_io_calls(node1, ctx) == node1
  node_save: cst.Call = cst.Call(func=cst.Name("save"))
  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=None):
    assert transform_io_calls(node_save, ctx) == node_save

  class BadAdapter:
    def get_serialization_imports(self) -> List[str]:
      """Gets serialization imports.

      Returns:
          List[str]: List of serialization imports.
      """
      return []

    pass

  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=BadAdapter()):
    assert transform_io_calls(node_save, ctx) == node_save

  class GoodAdapter:
    def get_serialization_imports(self) -> List[str]:
      """Gets serialization imports.

      Returns:
          List[str]: List of serialization imports.
      """
      return []

    def format_save(self, obj: cst.BaseExpression, path: cst.BaseExpression) -> cst.Call:
      """Formats save.

      Args:
          obj (cst.BaseExpression): Object to save.
          path (cst.BaseExpression): Path to save to.

      Returns:
          cst.Call: The CST node for save call.
      """
      return cst.Call(func=cst.Name("good_save"))

    def format_load(self, path: cst.BaseExpression) -> cst.Call:
      """Formats load.

      Args:
          path (cst.BaseExpression): Path to load from.

      Returns:
          cst.Call: The CST node for load call.
      """
      return cst.Call(func=cst.Name("good_load"))

  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=GoodAdapter()):
    node_bad_save_1: cst.Call = cst.Call(func=cst.Name("save"), args=[cst.Arg(value=cst.Name("a"))])
    assert transform_io_calls(node_bad_save_1, ctx) == node_bad_save_1
    node_bad_save_2: cst.Call = cst.Call(func=cst.Name("save"), args=[])
    assert transform_io_calls(node_bad_save_2, ctx) == node_bad_save_2
    node_bad_load: cst.Call = cst.Call(func=cst.Name("load"), args=[])
    assert transform_io_calls(node_bad_load, ctx) == node_bad_load

    class RaiseAdapter:
      def get_serialization_imports(self) -> List[str]:
        """Gets serialization imports.

        Returns:
            List[str]: List of serialization imports.
        """
        return []

      def format_save(self, obj: cst.BaseExpression, path: cst.BaseExpression) -> cst.Call:
        """Formats save.

        Args:
            obj (cst.BaseExpression): Object to save.
            path (cst.BaseExpression): Path to save to.

        Raises:
            ValueError: Boom error.

        Returns:
            cst.Call: Function call.
        """
        raise ValueError("boom")

    with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=RaiseAdapter()):
      node_good_save: cst.Call = cst.Call(
        func=cst.Name("save"), args=[cst.Arg(value=cst.Name("obj")), cst.Arg(value=cst.Name("path"))]
      )
      assert transform_io_calls(node_good_save, ctx) == node_good_save
