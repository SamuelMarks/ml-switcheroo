"""Test suite for ghost."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import types

from unittest.mock import MagicMock, patch

from ml_switcheroo.core.ghost import GhostInspector, GhostParam, GhostRef


def test_ghost_ref_has_arg() -> None:
  """Verify ghost ref has arg behavior."""
  ref = getattr(
    {
      "v": getattr({"v": GhostRef}, "__getitem__")("v")(
        name="test",
        api_path="test.api",
        kind="function",
        params=getattr(
          {
            "v": [
              getattr(
                {
                  "v": getattr({"v": GhostParam}, "__getitem__")("v")(
                    name=getattr({"v": "x"}, "__getitem__")("v"),
                    kind=getattr({"v": "POSITIONAL_OR_KEYWORD"}, "__getitem__")("v"),
                  )
                },
                "__getitem__",
              )("v"),
              getattr(
                {
                  "v": getattr({"v": GhostParam}, "__getitem__")("v")(
                    name=getattr({"v": "y"}, "__getitem__")("v"),
                    kind=getattr({"v": "POSITIONAL_OR_KEYWORD"}, "__getitem__")("v"),
                  )
                },
                "__getitem__",
              )("v"),
            ]
          },
          "__getitem__",
        )("v"),
      )
    },
    "__getitem__",
  )("v")
  assert ref.has_arg("x")
  assert ref.has_arg("y")
  assert not ref.has_arg("z")


def dummy_func(a: str, b: int = 1, *args: int, c: int = 2, d: str | None = None) -> None:
  """Execute dummy func helper logic.

  Args:
      a: The a (str) for dummy func.
      b: The b (int) for dummy func.
      c: The c (int) for dummy func.
      d: The d (str | None) for dummy func.
      args: The args (int) for dummy func.
  """
  pass


class DummyClass:
  """Test helper class DummyClass."""

  def __init__(self, x: int, y: str = "test") -> None:
    """Initialize test helper instance.

    Args:
        x: The x (int) for init.
        y: The y (str) for init.
    """
    pass


def test_ghost_inspector_func() -> None:
  """Verify ghost inspector func behavior."""
  ref = getattr(
    {
      "v": getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").inspect}, "__getitem__")("v")(
        getattr({"v": dummy_func}, "__getitem__")("v"), getattr({"v": "dummy_func"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert ref.name == "dummy_func"
  assert ref.api_path == "dummy_func"
  assert ref.kind == "function"
  assert ref.docstring is not None and ref.docstring.startswith("Execute dummy func helper logic.")
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


def test_ghost_inspector_class() -> None:
  """Verify ghost inspector class behavior."""
  ref = GhostInspector.inspect(DummyClass, "DummyClass")
  assert ref.name == "DummyClass"
  assert ref.kind == "class"
  assert ref.docstring == "Test helper class DummyClass."
  assert ref.has_arg("x")
  assert ref.has_arg("y")
  assert not ref.has_arg("self")


@patch("ml_switcheroo.core.ghost.inspect.signature")
def test_ghost_inspector_c_extension_fallback(mock_sig: MagicMock) -> None:
  """Verify ghost inspector c extension fallback behavior.

  Args:
      mock_sig: The mock sig (MagicMock) for test ghost inspector c extension fallback.
  """
  mock_sig.side_effect = ValueError("no signature found")
  ref = getattr(
    {
      "v": getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").inspect}, "__getitem__")("v")(
        getattr({"v": dummy_func}, "__getitem__")("v"), getattr({"v": "dummy_func"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert ref.name == "dummy_func"
  assert ref.kind == "function"
  assert ref.has_varargs is True
  assert ref.has_arg("args")
  assert ref.has_arg("kwargs")


def test_ghost_inspector_sanitize_callable() -> None:
  """Verify ghost inspector sanitize callable behavior."""

  def func_with_callable_default(f: types.FunctionType = getattr({"v": dummy_func}, "__getitem__")("v")) -> None:
    """Execute func with callable default helper logic.

    Args:
        f: The f (types.FunctionType) for func with callable default.
    """
    pass

  ref = getattr(
    {
      "v": getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").inspect}, "__getitem__")("v")(
        getattr({"v": func_with_callable_default}, "__getitem__")("v"), getattr({"v": "func"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  f_param = next(p for p in ref.params if p.name == "f")
  assert f_param.default is None


class UnrepresentableStrOnly:
  """Test helper class UnrepresentableStrOnly."""

  def __repr__(self) -> str:
    """Execute repr helper logic.

    Returns:
        Resulting `str` from repr.
    """
    return "custom repr without memory address"

  def __str__(self) -> str:
    """Execute str helper logic.

    Returns:
        Resulting `str` from str.
    """
    raise ValueError("Cannot stringify")


def test_ghost_inspector_unrepresentable_str_only() -> None:
  """Verify ghost inspector unrepresentable str only behavior."""

  def func_with_bad_str_default(x: UnrepresentableStrOnly = UnrepresentableStrOnly()) -> None:
    """Execute func with bad str default helper logic.

    Args:
        x: The x (UnrepresentableStrOnly) for func with bad str default.
    """
    pass

  ref = getattr(
    {
      "v": getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").inspect}, "__getitem__")("v")(
        getattr({"v": func_with_bad_str_default}, "__getitem__")("v"), getattr({"v": "func"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  x_param = next(p for p in ref.params if p.name == "x")
  assert x_param.default == "<unrepresentable>"


class UnrepresentableStrAddress:
  """Test helper class UnrepresentableStrAddress."""

  def __repr__(self) -> str:
    """Execute repr helper logic.

    Returns:
        Resulting `str` from repr.
    """
    return "custom repr without memory address"

  def __str__(self) -> str:
    """Execute str helper logic.

    Returns:
        Resulting `str` from str.
    """
    return "this has address at 0x1234"


def test_ghost_inspector_unrepresentable_str_address() -> None:
  """Verify ghost inspector unrepresentable str address behavior."""

  def func_with_bad_str_default(x: UnrepresentableStrAddress = UnrepresentableStrAddress()) -> None:
    """Execute func with bad str default helper logic.

    Args:
        x: The x (UnrepresentableStrAddress) for func with bad str default.
    """
    pass

  ref = getattr(
    {
      "v": getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").inspect}, "__getitem__")("v")(
        getattr({"v": func_with_bad_str_default}, "__getitem__")("v"), getattr({"v": "func"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  x_param = next(p for p in ref.params if p.name == "x")
  assert x_param.default is None


def test_ghost_inspector_annotation_string() -> None:
  """Verify ghost inspector annotation string behavior."""

  class MyType(int):
    """Test helper class MyType."""

    pass

  def func(x: MyType) -> None:
    """Execute func helper logic.

    Args:
        x: The x (MyType) for func.
    """
    pass

  ref = getattr(
    {
      "v": getattr({"v": getattr({"v": GhostInspector}, "__getitem__")("v").inspect}, "__getitem__")("v")(
        getattr({"v": func}, "__getitem__")("v"), getattr({"v": "func"}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  x_param = next(p for p in ref.params if p.name == "x")
  assert x_param.annotation == "MyType"


def test_ghost_hydrate() -> None:
  """Verify ghost hydrate behavior."""
  data: dict[str, RuntimeTensorOrValue] = {
    "name": "test",
    "api_path": "a.b",
    "kind": "function",
    "params": [],
    "docstring": None,
    "has_varargs": False,
  }
  ref = GhostInspector.hydrate(data)
  assert ref.name == "test"
  assert ref.api_path == "a.b"

  ref_api = GhostInspector.hydrate({"name": "t1", "api": "pkg.fn", "returns_type": "Tensor"})
  assert ref_api.api_path == "pkg.fn"
  assert ref_api.return_annotation == "Tensor"
  assert ref_api.returns_type == "Tensor"

  ref_ret = GhostInspector.hydrate({"name": "t2", "api_path": "pkg.fn2", "return_annotation": "Array"})
  assert ref_ret.return_annotation == "Array"
  assert ref_ret.returns_type == "Array"


def test_ghost_inspector_class_without_init() -> None:
  """Verify ghost inspector class without init behavior."""

  class EmptyClass:
    """Test helper class EmptyClass."""

    pass

  ref = GhostInspector.inspect(EmptyClass, "EmptyClass")
  assert ref.name == "EmptyClass"
  assert ref.kind == "class"
