"""Docstring."""

from ml_switcheroo.core.ghost import GhostParam, GhostRef, GhostInspector
from unittest.mock import patch


def test_ghost_ref_has_arg():
  """Docstring."""
  ref = GhostRef(
    name="test",
    api_path="test.api",
    kind="function",
    params=[GhostParam(name="x", kind="POSITIONAL_OR_KEYWORD"), GhostParam(name="y", kind="POSITIONAL_OR_KEYWORD")],
  )
  assert ref.has_arg("x")
  assert ref.has_arg("y")
  assert not ref.has_arg("z")


def dummy_func(a, b=1, *args, c: int = 2, d=None):
  """Dummy docstring."""
  pass


class DummyClass:
  """dummy class docstring."""

  def __init__(self, x, y="test"):
    """Docstring."""
    pass


def test_ghost_inspector_func():
  """Docstring."""
  ref = GhostInspector.inspect(dummy_func, "dummy_func")
  assert ref.name == "dummy_func"
  assert ref.api_path == "dummy_func"
  assert ref.kind == "function"
  assert ref.docstring == "Dummy docstring."
  assert ref.has_varargs is True

  assert ref.has_arg("a")
  assert ref.has_arg("b")
  assert ref.has_arg("args")
  assert ref.has_arg("c")
  assert ref.has_arg("d")

  b_param = next(p for p in ref.params if p.name == "b")
  assert b_param.default == "1"

  c_param = next(p for p in ref.params if p.name == "c")
  assert c_param.annotation == "int"


def test_ghost_inspector_class():
  """Docstring."""
  ref = GhostInspector.inspect(DummyClass, "DummyClass")
  assert ref.name == "DummyClass"
  assert ref.kind == "class"
  assert ref.docstring == "dummy class docstring."
  assert ref.has_arg("x")
  assert ref.has_arg("y")
  assert not ref.has_arg("self")


@patch("ml_switcheroo.core.ghost.inspect.signature")
def test_ghost_inspector_c_extension_fallback(mock_sig):
  """Docstring."""
  mock_sig.side_effect = ValueError("no signature found")
  ref = GhostInspector.inspect(dummy_func, "dummy_func")
  assert ref.name == "dummy_func"
  assert ref.kind == "function"
  assert ref.has_varargs is True
  assert ref.has_arg("args")
  assert ref.has_arg("kwargs")


def test_ghost_inspector_sanitize_callable():
  """Docstring."""

  def func_with_callable_default(f=dummy_func):
    """Docstring."""
    pass

  ref = GhostInspector.inspect(func_with_callable_default, "func")
  f_param = next(p for p in ref.params if p.name == "f")
  assert f_param.default is None


class UnrepresentableStrOnly:
  """Docstring."""

  def __repr__(self):
    """Docstring."""
    return "custom repr without memory address"

  def __str__(self):
    """Docstring."""
    raise ValueError("Cannot stringify")


def test_ghost_inspector_unrepresentable_str_only():
  """Docstring."""

  def func_with_bad_str_default(x=UnrepresentableStrOnly()):
    """Docstring."""
    pass

  ref = GhostInspector.inspect(func_with_bad_str_default, "func")
  x_param = next(p for p in ref.params if p.name == "x")
  assert x_param.default == "<unrepresentable>"


class UnrepresentableStrAddress:
  """Docstring."""

  def __repr__(self):
    """Docstring."""
    return "custom repr without memory address"

  def __str__(self):
    """Docstring."""
    return "this has address at 0x1234"


def test_ghost_inspector_unrepresentable_str_address():
  """Docstring."""

  def func_with_bad_str_default(x=UnrepresentableStrAddress()):
    """Docstring."""
    pass

  ref = GhostInspector.inspect(func_with_bad_str_default, "func")
  x_param = next(p for p in ref.params if p.name == "x")
  assert x_param.default is None


def test_ghost_inspector_annotation_string():
  """Docstring."""
  MyType = int

  def func(x: "MyType"):
    """Docstring."""
    pass

  ref = GhostInspector.inspect(func, "func")
  x_param = next(p for p in ref.params if p.name == "x")
  assert x_param.annotation == "MyType"


def test_ghost_hydrate():
  """Docstring."""
  data = {"name": "test", "api_path": "a.b", "kind": "function", "params": [], "docstring": None, "has_varargs": False}
  ref = GhostInspector.hydrate(data)
  assert ref.name == "test"
  assert ref.api_path == "a.b"


def test_ghost_inspector_class_without_init():
  """Docstring."""

  class EmptyClass:
    """Docstring."""

    pass

  ref = GhostInspector.inspect(EmptyClass, "EmptyClass")
  assert ref.name == "EmptyClass"
  assert ref.kind == "class"
