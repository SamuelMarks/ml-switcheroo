"""Docstring."""

from typing import Callable, Dict, Optional, Union
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.ghost import GhostInspector, GhostParam, GhostRef


def test_ghost_ref_has_arg() -> None:
  """Docstring."""
  ref: GhostRef = GhostRef(
    name="test",
    api_path="test.api",
    kind="function",
    params=[GhostParam(name="x", kind="POSITIONAL_OR_KEYWORD"), GhostParam(name="y", kind="POSITIONAL_OR_KEYWORD")],
  )
  assert ref.has_arg("x")
  assert ref.has_arg("y")
  assert not ref.has_arg("z")


def dummy_func(a: int, b: int = 1, *args: tuple[int, ...], c: int = 2, d: Optional[str] = None) -> None:
  """Docstring."""
  pass


class DummyClass:
  """Docstring."""

  def __init__(self, x: int, y: str = "test") -> None:
    """Docstring."""
    pass


def test_ghost_inspector_func() -> None:
  """Docstring."""
  ref: GhostRef = GhostInspector.inspect(dummy_func, "dummy_func")
  assert ref.name == "dummy_func"
  assert ref.api_path == "dummy_func"
  assert ref.kind == "function"
  assert ref.docstring == "Docstring."
  assert ref.has_varargs is True

  assert ref.has_arg("a")
  assert ref.has_arg("b")
  assert ref.has_arg("args")
  assert ref.has_arg("c")
  assert ref.has_arg("d")

  b_param: GhostParam = next(p for p in ref.params if p.name == "b")
  assert b_param.default == "1"

  c_param: GhostParam = next(p for p in ref.params if p.name == "c")
  assert c_param.annotation == "int"


def test_ghost_inspector_class() -> None:
  """Docstring."""
  ref: GhostRef = GhostInspector.inspect(DummyClass, "DummyClass")
  assert ref.name == "DummyClass"
  assert ref.kind == "class"
  assert ref.docstring == "Docstring."
  assert ref.has_arg("x")
  assert ref.has_arg("y")
  assert not ref.has_arg("self")


@patch("ml_switcheroo.core.ghost.inspect.signature")
def test_ghost_inspector_c_extension_fallback(mock_sig: MagicMock) -> None:
  """Docstring."""
  mock_sig.side_effect = ValueError("no signature found")
  ref: GhostRef = GhostInspector.inspect(dummy_func, "dummy_func")
  assert ref.name == "dummy_func"
  assert ref.kind == "function"
  assert ref.has_varargs is True
  assert ref.has_arg("args")
  assert ref.has_arg("kwargs")


def test_ghost_inspector_sanitize_callable() -> None:
  """Docstring."""

  def func_with_callable_default(f: Callable[..., None] = dummy_func) -> None:
    pass

  ref: GhostRef = GhostInspector.inspect(func_with_callable_default, "func")
  f_param: GhostParam = next(p for p in ref.params if p.name == "f")
  assert f_param.default is None


class UnrepresentableStrOnly:
  """Docstring."""

  def __repr__(self) -> str:
    """Docstring."""
    return "custom repr without memory address"

  def __str__(self) -> str:
    """Docstring."""
    raise ValueError("Cannot stringify")


def test_ghost_inspector_unrepresentable_str_only() -> None:
  """Docstring."""

  def func_with_bad_str_default(x: UnrepresentableStrOnly = UnrepresentableStrOnly()) -> None:
    pass

  ref: GhostRef = GhostInspector.inspect(func_with_bad_str_default, "func")
  x_param: GhostParam = next(p for p in ref.params if p.name == "x")
  assert x_param.default == "<unrepresentable>"


class UnrepresentableStrAddress:
  """Docstring."""

  def __repr__(self) -> str:
    """Docstring."""
    return "custom repr without memory address"

  def __str__(self) -> str:
    """Docstring."""
    return "this has address at 0x1234"


def test_ghost_inspector_unrepresentable_str_address() -> None:
  """Docstring."""

  def func_with_bad_str_default(x: UnrepresentableStrAddress = UnrepresentableStrAddress()) -> None:
    pass

  ref: GhostRef = GhostInspector.inspect(func_with_bad_str_default, "func")
  x_param: GhostParam = next(p for p in ref.params if p.name == "x")
  assert x_param.default is None


def test_ghost_inspector_annotation_string() -> None:
  """Docstring."""
  MyType = int

  def func(x: "MyType") -> None:
    pass

  ref: GhostRef = GhostInspector.inspect(func, "func")
  x_param: GhostParam = next(p for p in ref.params if p.name == "x")
  assert x_param.annotation == "MyType"


def test_ghost_hydrate() -> None:
  """Docstring."""
  data: Dict[str, Union[str, bool, list[Dict[str, str]], None]] = {
    "name": "test",
    "api_path": "a.b",
    "kind": "function",
    "params": [],
    "docstring": None,
    "has_varargs": False,
  }
  ref: GhostRef = GhostInspector.hydrate(data)
  assert ref.name == "test"
  assert ref.api_path == "a.b"


def test_ghost_inspector_class_without_init() -> None:
  """Docstring."""

  class EmptyClass:
    pass

  ref: GhostRef = GhostInspector.inspect(EmptyClass, "EmptyClass")
  assert ref.name == "EmptyClass"
  assert ref.kind == "class"
