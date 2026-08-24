"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.plugins.io_handler import _get_func_name, _get_arg, transform_io_calls
from ml_switcheroo.core.hooks import HookContext


def test_get_func_name():
  """Docstring."""
  node1 = cst.Call(func=cst.Name("load"), args=[])
  assert _get_func_name(node1) == "load"

  node2 = cst.Call(func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("save")), args=[])
  assert _get_func_name(node2) == "save"

  node3 = cst.Call(func=cst.Call(func=cst.Name("other"), args=[]), args=[])
  assert _get_func_name(node3) is None


def test_get_arg():
  """Docstring."""
  arg_pos = cst.Arg(value=cst.Name("x"))
  arg_kw = cst.Arg(keyword=cst.Name("f"), value=cst.Name("y"))
  arg_kw2 = cst.Arg(keyword=cst.Name("obj"), value=cst.Name("z"))
  args = [arg_pos, arg_kw, arg_kw2]

  assert _get_arg(args, 0, "obj") == arg_kw2  # by keyword priority
  assert _get_arg(args, 1, "f") == arg_kw
  assert _get_arg(args, 0, "nonexistent") == arg_pos  # fallback to pos 0
  assert _get_arg(args, 5, "missing") is None


@patch("ml_switcheroo.plugins.io_handler.get_adapter")
@patch("ml_switcheroo.plugins.io_handler.capture_node_source")
def test_transform_io_calls(mock_capture, mock_get_adapter):
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"

  # 1. No adapter
  mock_get_adapter.return_value = None
  node = cst.Call(func=cst.Name("load"), args=[])
  assert transform_io_calls(node, ctx) == node

  # Setup adapter
  adapter = MagicMock()
  mock_get_adapter.return_value = adapter

  # 2. No func name
  node2 = cst.Call(func=cst.Call(func=cst.Name("f"), args=[]), args=[])
  assert transform_io_calls(node2, ctx) == node2

  # 3. Not save or load
  node3 = cst.Call(func=cst.Name("other_func"), args=[])
  assert transform_io_calls(node3, ctx) == node3

  # 4. Save missing args
  node_save_missing = cst.Call(func=cst.Name("save"), args=[cst.Arg(value=cst.Name("x"))])
  assert transform_io_calls(node_save_missing, ctx) == node_save_missing

  # 5. Load missing args
  node_load_missing = cst.Call(func=cst.Name("load"), args=[])
  assert transform_io_calls(node_load_missing, ctx) == node_load_missing

  # 6. Save valid
  node_save = cst.Call(
    func=cst.Name("save"), args=[cst.Arg(value=cst.Name("obj_var")), cst.Arg(value=cst.Name("file_var"))]
  )
  mock_capture.side_effect = lambda x: x.value
  adapter.get_serialization_imports.return_value = ["import os"]
  adapter.get_serialization_syntax.return_value = "jax.save(obj_var, file_var)"
  result = transform_io_calls(node_save, ctx)
  ctx.inject_preamble.assert_called_with("import os")
  adapter.get_serialization_syntax.assert_called_with("save", "file_var", "obj_var")
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "save"  # "jax.save" is parsed

  # 7. Load valid
  node_load = cst.Call(func=cst.Name("load"), args=[cst.Arg(value=cst.Name("file_var"))])
  adapter.get_serialization_syntax.return_value = "jax.load(file_var)"
  result2 = transform_io_calls(node_load, ctx)
  adapter.get_serialization_syntax.assert_called_with("load", "file_var", None)
  assert isinstance(result2, cst.Call)

  # 8. Syntax returns None
  adapter.get_serialization_syntax.return_value = None
  assert transform_io_calls(node_load, ctx) == node_load

  # 9. Exception in parsing/syntax
  adapter.get_serialization_syntax.side_effect = Exception("error")
  assert transform_io_calls(node_load, ctx) == node_load
