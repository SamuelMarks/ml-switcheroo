"""Unit tests for `GhostInspector`, `GhostRef`, and `GhostParam` (`ml_switcheroo.core.ghost`)."""

import typing

import pytest
from ml_switcheroo_ir.schema.ghost import ParameterKind

from ml_switcheroo.core.ghost import GhostInspector, GhostParam, GhostRef
from ml_switcheroo.enums import InspectionObject, RuntimeTensorOrValue


def _sample_func(a: str, b: int = 10, *args: int, c: int = 2, opt: str | None = None) -> int:
  """Compute a dummy integer from positional, vararg, and keyword-only arguments.

  Args:
      a: String prefix argument.
      b: Integer argument with default `10`.
      *args: Variable integer arguments.
      c: Keyword-only integer argument with default `2`.
      opt: Optional string argument.

  Returns:
      Integer sum `b + c`.
  """
  _ = (a, args, opt)
  return b + c


class _SampleLayer:
  """Simulated neural layer class with typed `__init__` parameters."""

  def __init__(self, output_dim: int, activation: str = "relu") -> None:
    """Store `output_dim` and `activation`.

    Args:
        output_dim: Output feature dimension.
        activation: Activation name string.
    """
    self.output_dim = output_dim
    self.activation = activation


class _EmptyClass:
  """Class without custom `__init__`."""


def _noop_callback() -> None:
  """Default callable value used to verify callable-default sanitization."""


class _HexAddressStr:
  """Object whose `__str__` contains a simulated `0x...` memory address."""

  def __repr__(self) -> str:
    """Return address-free repr string.

    Returns:
        Safe repr string.
    """
    return "SafeRepr"

  def __str__(self) -> str:
    """Return string containing `0x` address token.

    Returns:
        String containing `0x1234`.
    """
    return "Instance at 0x1234"


class _ExplodingStr:
  """Object whose `__str__` raises `RuntimeError`."""

  def __repr__(self) -> str:
    """Return address-free repr string.

    Returns:
        Safe repr string.
    """
    return "SafeRepr"

  def __str__(self) -> str:
    """Raise `RuntimeError` during stringification.

    Returns:
        Never returns; raises `RuntimeError`.

    Raises:
        RuntimeError: Always raised.
    """
    raise RuntimeError("cannot stringify")


_HEX_DEFAULT = _HexAddressStr()
_EXPLODING_DEFAULT = _ExplodingStr()


def test_ghost_ref_and_hydrate_variants() -> None:
  """Verify `GhostRef.has_arg`, roundtrip serialization, and legacy `api` / `returns_type` hydration."""
  manual_ref = GhostRef(
    name="test",
    api_path="pkg.test",
    kind="function",
    params=[
      GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
      GhostParam(name="y", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
    ],
  )
  assert manual_ref.has_arg("x") is True
  assert manual_ref.has_arg("y") is True
  assert manual_ref.has_arg("z") is False

  live_ref = GhostInspector.inspect(_sample_func, "pkg.sample_func")
  raw_dump: dict[str, RuntimeTensorOrValue] = live_ref.model_dump()
  hydrated = GhostInspector.hydrate(raw_dump)
  assert hydrated == live_ref
  assert hydrated.has_arg("b") is True
  assert hydrated.has_arg("missing") is False

  legacy_api = GhostInspector.hydrate({"name": "fn1", "api": "pkg.fn1", "returns_type": "Tensor"})
  assert legacy_api.api_path == "pkg.fn1"
  assert legacy_api.return_annotation == "Tensor"
  assert legacy_api.returns_type == "Tensor"

  explicit_ret = GhostInspector.hydrate({"name": "fn2", "api_path": "pkg.fn2", "return_annotation": "Array"})
  assert explicit_ret.return_annotation == "Array"
  assert explicit_ret.returns_type == "Array"


def test_ghost_inspector_functions_classes_and_unbound_methods() -> None:
  """Verify `GhostInspector.inspect` on functions, classes with/without `__init__`, and `self`-prefixed functions."""
  fn_ref = GhostInspector.inspect(_sample_func, "pkg._sample_func")
  assert fn_ref.name == "_sample_func"
  assert fn_ref.kind == "function"
  assert fn_ref.has_varargs is True
  assert fn_ref.docstring is not None
  assert "Compute a dummy integer" in fn_ref.docstring
  assert {p.name for p in fn_ref.params} == {"a", "b", "args", "c", "opt"}

  b_param = next(p for p in fn_ref.params if p.name == "b")
  assert b_param.default == "10"
  c_param = next(p for p in fn_ref.params if p.name == "c")
  assert c_param.annotation == "int"
  opt_param = next(p for p in fn_ref.params if p.name == "opt")
  assert opt_param.annotation is not None
  assert "None" in opt_param.annotation

  cls_ref = GhostInspector.inspect(_SampleLayer, "pkg._SampleLayer")
  assert cls_ref.name == "_SampleLayer"
  assert cls_ref.kind == "class"
  assert [p.name for p in cls_ref.params] == ["output_dim", "activation"]
  assert cls_ref.params[1].default == "relu"

  empty_cls_ref = GhostInspector.inspect(_EmptyClass, "pkg._EmptyClass")
  assert empty_cls_ref.name == "_EmptyClass"
  assert empty_cls_ref.kind == "class"

  def _unbound_method(self: _EmptyClass, dim: int) -> None:
    """Unbound function carrying an explicit `self` parameter.

    Args:
        self: Receiver instance.
        dim: Dimension size.
    """
    _ = (self, dim)

  method_ref = GhostInspector.inspect(_unbound_method, "pkg.method")
  assert [p.name for p in method_ref.params] == ["dim"]


def test_ghost_inspector_sanitization_and_c_extension_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify default sanitization, nameless annotations, and `inspect.signature` `ValueError` fallbacks.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """

  def _fn_with_special_defaults(
    cb: typing.Callable[[], None] = _noop_callback,
    addr: _HexAddressStr = _HEX_DEFAULT,
    boom: _ExplodingStr = _EXPLODING_DEFAULT,
  ) -> None:
    """Function carrying callable, hex-address, and exploding-`__str__` defaults.

    Args:
        cb: Callable default.
        addr: Default stringifying with `0x`.
        boom: Default raising `RuntimeError` in `__str__`.
    """
    _ = (cb, addr, boom)

  special_ref = GhostInspector.inspect(_fn_with_special_defaults, "pkg.special")
  assert special_ref.params[0].default is None
  assert special_ref.params[1].default is None
  assert special_ref.params[2].default == "<unrepresentable>"

  class _NamelessAnnotation:
    """Annotation instance without `__name__`."""

    def __str__(self) -> str:
      """Return string representation `'CustomAnno'`.

      Returns:
          Literal `'CustomAnno'`.
      """
      return "CustomAnno"

  def _fn_custom_anno(x: int) -> None:
    """Function whose `__annotations__['x']` is replaced with a nameless object.

    Args:
        x: Annotated parameter.
    """
    _ = x

  _fn_custom_anno.__annotations__["x"] = _NamelessAnnotation()
  anno_ref = GhostInspector.inspect(_fn_custom_anno, "pkg.anno")
  assert anno_ref.params[0].annotation == "CustomAnno"

  # C-extension signature failure: function gets `*args, **kwargs`, class gets `[]`
  def _raise_sig_error(_obj: InspectionObject) -> typing.Never:
    """Simulate `inspect.signature` raising `ValueError` on C-builtins.

    Args:
        _obj: Inspected object.

    Returns:
        Never returns; raises `ValueError`.

    Raises:
        ValueError: Always raised.
    """
    raise ValueError("no signature found")

  monkeypatch.setattr("ml_switcheroo.core.ghost.inspect.signature", _raise_sig_error)
  c_fn_ref = GhostInspector.inspect(_sample_func, "pkg.c_fn")
  assert c_fn_ref.has_varargs is True
  assert [p.name for p in c_fn_ref.params] == ["args", "kwargs"]

  c_cls_ref = GhostInspector.inspect(_EmptyClass, "pkg.c_cls")
  assert c_cls_ref.params == []

  import math

  mod_ref = GhostInspector.inspect(math, "math")
  assert mod_ref.name == "math"
  assert mod_ref.has_varargs is True
