"""
Tests for Ghost Core (Introspection & Serialization).

Verifies that:
1. Live objects (Functions) are correctly inspected.
2. Live objects (Classes) have their __init__ signature extracted.
3. Serialization/Hydration (Ghost Mode) works seamlessly.
4. Edge cases (defaults, annotations, missing signatures) are handled.
"""

import pytest
from typing import Optional
from ml_switcheroo.core.ghost import GhostInspector, GhostRef


# --- Mock Objects for Inspection ---


def simple_func(x, y=10):
  """A simple function."""
  return x + y


def typed_func(x: int, opt: Optional[str] = None):
  pass


class SimpleClass:
  """A standard class."""

  def __init__(self, output_dim, activation="relu"):
    self.out = output_dim


class BuiltinLike:
  """A class mimicking built-ins that might fail signature inspection."""

  pass


# --- Tests ---


def test_inspect_simple_function():
  """Verify basic function inspection parameters."""
  ref = GhostInspector.inspect(simple_func, "test.simple_func")

  assert ref.name == "simple_func"
  assert ref.kind == "function"
  assert ref.api_path == "test.simple_func"
  assert ref.docstring == "A simple function."

  assert len(ref.params) == 2
  p0, p1 = ref.params

  assert p0.name == "x"
  assert p0.default is None

  assert p1.name == "y"
  assert p1.default == "10"


def test_inspect_class_init():
  """Verify class inspection looks at __init__ and skips self."""
  ref = GhostInspector.inspect(SimpleClass, "test.SimpleClass")

  assert ref.name == "SimpleClass"
  assert ref.kind == "class"
  assert ref.docstring == "A standard class."

  # Should detect 'output_dim' and 'activation'
  # Should SKIP 'self'
  assert len(ref.params) == 2
  assert ref.params[0].name == "output_dim"
  assert ref.params[1].name == "activation"
  assert ref.params[1].default == "relu"


def test_inspect_typed_signature():
  """Verify type hints are stringified."""
  ref = GhostInspector.inspect(typed_func, "test.typed")

  p0 = ref.params[0]
  assert p0.name == "x"
  assert p0.annotation == "int"

  p1 = ref.params[1]
  assert p1.name == "opt"
  # Representation of Optional[...] varies by python version, just check it exists stringified
  assert p1.annotation is not None
  assert "Optional" in p1.annotation or "Union" in p1.annotation or "None" in p1.annotation


def test_ghost_hydration_roundtrip():
  """Verify we can dump to dict and load back (Simulation of Cache)."""
  # 1. Inspect Live
  live_ref = GhostInspector.inspect(simple_func, "func")

  # 2. Serialize
  data = live_ref.model_dump()

  # 3. Hydrate
  ghost_ref = GhostInspector.hydrate(data)

  assert ghost_ref == live_ref
  assert ghost_ref.has_arg("y")
  assert not ghost_ref.has_arg("z")


def test_inspect_failure_handling(monkeypatch):
  """Verify safe fallback if signature inspection fails (e.g. C-extensions)."""

  # We patch signature to raise ValueError mimics C-ext failure
  def mock_sig(obj):
    raise ValueError("no signature found")

  monkeypatch.setattr("inspect.signature", mock_sig)

  ref = GhostInspector.inspect(BuiltinLike, "test.BuiltinLike")

  # Should return a valid Ref object with empty params
  assert ref.name == "BuiltinLike"
  assert ref.params == []
  # Docstring should still be captured if inspect.getdoc works
  assert "mimicking" in (ref.docstring or "")


def test_ghost_ref_helper_methods():
  """verify helper utility methods on the dataclass."""
  ref = GhostInspector.inspect(simple_func, "foo")

  assert ref.has_arg("x") is True
  assert ref.has_arg("non_existent") is False
