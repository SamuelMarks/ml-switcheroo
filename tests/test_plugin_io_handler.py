"""Test suite for plugin io handler."""

import typing

from unittest.mock import MagicMock, patch

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.io_handler import _get_arg, _get_func_name, transform_io_calls

cst: typing.Any = _cst


def test_get_func_name() -> None:
  """Verify get func name behavior."""
  node1: typing.Any = cst.Call(func=cst.Name("load"), args=[])
  assert _get_func_name(node1) == "load"

  node2: typing.Any = cst.Call(func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("save")), args=[])
  assert _get_func_name(node2) == "save"

  node3: typing.Any = cst.Call(func=cst.Call(func=cst.Name("other"), args=[]), args=[])
  assert _get_func_name(node3) is None


def test_get_arg() -> None:
  """Verify get arg behavior."""
  arg_pos: typing.Any = cst.Arg(value=cst.Name("x"))
  arg_kw: typing.Any = cst.Arg(keyword=cst.Name("f"), value=cst.Name("y"))
  arg_kw2: typing.Any = cst.Arg(keyword=cst.Name("obj"), value=cst.Name("z"))
  args: typing.Any = [arg_pos, arg_kw, arg_kw2]

  assert _get_arg(args, 0, "obj") == arg_kw2  # by keyword priority
  assert _get_arg(args, 1, "f") == arg_kw
  assert _get_arg(args, 0, "nonexistent") == arg_pos  # fallback to pos 0
  assert _get_arg(args, 5, "missing") is None


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
@patch("ml_switcheroo.plugins.io_handler.capture_node_source")
def test_transform_io_calls(mock_capture: typing.Any, mock_get_adapter: typing.Any) -> None:
  """Verify transform io calls behavior.

  Args:
      mock_capture: Test argument or fixture.
      mock_get_adapter: Test argument or fixture.
  """
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"

  # 1. No adapter
  mock_get_adapter.return_value = None
  node: typing.Any = cst.Call(func=cst.Name("load"), args=[])
  assert transform_io_calls(node, ctx) == node

  # Setup adapter
  adapter: typing.Any = MagicMock()
  mock_get_adapter.return_value = adapter

  # 2. No func name
  node2: typing.Any = cst.Call(func=cst.Call(func=cst.Name("f"), args=[]), args=[])
  assert transform_io_calls(node2, ctx) == node2

  # 3. Not save or load
  node3: typing.Any = cst.Call(func=cst.Name("other_func"), args=[])
  assert transform_io_calls(node3, ctx) == node3

  # 4. Save missing args
  node_save_missing: typing.Any = cst.Call(func=cst.Name("save"), args=[cst.Arg(value=cst.Name("x"))])
  assert transform_io_calls(node_save_missing, ctx) == node_save_missing

  # 5. Load missing args
  node_load_missing: typing.Any = cst.Call(func=cst.Name("load"), args=[])
  assert transform_io_calls(node_load_missing, ctx) == node_load_missing

  # 6. Save valid
  node_save: typing.Any = cst.Call(
    func=cst.Name("save"), args=[cst.Arg(value=cst.Name("obj_var")), cst.Arg(value=cst.Name("file_var"))]
  )
  mock_capture.side_effect = lambda x: x.value
  adapter.get_serialization_imports.return_value = ["import os"]
  adapter.get_serialization_syntax.return_value = "jax.save(obj_var, file_var)"
  result: typing.Any = transform_io_calls(node_save, ctx)
  ctx.inject_preamble.assert_called_with("import os")
  adapter.get_serialization_syntax.assert_called_with("save", "file_var", "obj_var")
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "save"  # "jax.save" is parsed

  # 7. Load valid
  node_load: typing.Any = cst.Call(func=cst.Name("load"), args=[cst.Arg(value=cst.Name("file_var"))])
  adapter.get_serialization_syntax.return_value = "jax.load(file_var)"
  result2: typing.Any = transform_io_calls(node_load, ctx)
  adapter.get_serialization_syntax.assert_called_with("load", "file_var", None)
  assert isinstance(result2, cst.Call)

  # 8. Syntax returns None
  adapter.get_serialization_syntax.return_value = None
  assert transform_io_calls(node_load, ctx) == node_load

  # 9. Exception in parsing/syntax
  adapter.get_serialization_syntax.side_effect = Exception("error")
  assert transform_io_calls(node_load, ctx) == node_load
