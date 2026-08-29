"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.context_to_function_wrap import _create_dotted_name, transform_context_manager


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "b"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "a"


def test_transform_context_manager() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("no_grad"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  result: cst.CSTNode = transform_context_manager(node, ctx)
  ctx.inject_preamble.assert_called_with("import contextlib")
  assert isinstance(result, cst.Call)
  assert len(result.args) == 0
  assert isinstance(result.func, cst.Attribute)
  assert isinstance(result.func.value, cst.Name)
  assert result.func.value.value == "contextlib"
  assert result.func.attr.value == "nullcontext"
