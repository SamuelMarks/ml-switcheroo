"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.context_to_function_wrap import transform_context_manager, _create_dotted_name
from ml_switcheroo.core.hooks import HookContext


def test_create_dotted_name():
  """Docstring."""
  node = _create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"
  assert node.value.attr.value == "b"
  assert node.value.value.value == "a"


def test_transform_context_manager():
  """Docstring."""
  node = cst.Call(func=cst.Name("no_grad"))
  ctx = MagicMock(spec=HookContext)
  result = transform_context_manager(node, ctx)
  ctx.inject_preamble.assert_called_with("import contextlib")
  assert isinstance(result, cst.Call)
  assert len(result.args) == 0
  assert isinstance(result.func, cst.Attribute)
  assert result.func.value.value == "contextlib"
  assert result.func.attr.value == "nullcontext"
