import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.plugins.io_handler import _get_func_name, _get_arg, transform_io_calls
from ml_switcheroo.core.hooks import HookContext


def test_get_func_name():
  assert _get_func_name(cst.Call(func=cst.Name("foo"))) == "foo"
  assert _get_func_name(cst.Call(func=cst.SimpleString("'bar'"))) is None


def test_get_arg():
  args = [cst.Arg(value=cst.Name("a"), keyword=cst.Name("a_kw"))]
  assert _get_arg(args, 1, "missing") is None


def test_transform_io_calls_misses():
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"

  # Not save/load (line 58)
  node1 = cst.Call(func=cst.Name("foo"))
  assert transform_io_calls(node1, ctx) == node1

  # Save/load but missing adapter (line 62)
  node_save = cst.Call(func=cst.Name("save"))
  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=None):
    assert transform_io_calls(node_save, ctx) == node_save

  # Adapter doesn't have format_save (line 67)
  class BadAdapter:
    def get_serialization_imports(self):
      return []

    pass

  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=BadAdapter()):
    assert transform_io_calls(node_save, ctx) == node_save

  # Save with missing args (line 81, 86)
  class GoodAdapter:
    def get_serialization_imports(self):
      return []

    def format_save(self, obj, path):
      return cst.Call(func=cst.Name("good_save"))

    def format_load(self, path):
      return cst.Call(func=cst.Name("good_load"))

  with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=GoodAdapter()):
    node_bad_save_1 = cst.Call(func=cst.Name("save"), args=[cst.Arg(value=cst.Name("a"))])  # missing path
    assert transform_io_calls(node_bad_save_1, ctx) == node_bad_save_1

    node_bad_save_2 = cst.Call(func=cst.Name("save"), args=[])  # missing obj
    assert transform_io_calls(node_bad_save_2, ctx) == node_bad_save_2

    # Load missing arg (line 101)
    node_bad_load = cst.Call(func=cst.Name("load"), args=[])
    assert transform_io_calls(node_bad_load, ctx) == node_bad_load

    # Exception during adapter logic (line 105-106)
    class RaiseAdapter:
      def get_serialization_imports(self):
        return []

      def format_save(self, obj, path):
        raise ValueError("boom")

    with patch("ml_switcheroo.plugins.io_handler.get_adapter", return_value=RaiseAdapter()):
      node_good_save = cst.Call(
        func=cst.Name("save"), args=[cst.Arg(value=cst.Name("obj")), cst.Arg(value=cst.Name("path"))]
      )
      assert transform_io_calls(node_good_save, ctx) == node_good_save
