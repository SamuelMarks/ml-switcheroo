"""Tests."""

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_python_backend_base_class_resolution() -> None:
  """Test function."""
  b = PythonBackend(framework="paxml")

  class DummyTraits:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.module_base: str = "praxis.base_layer.BaseLayer"
      self.requires_super_init: bool = False
      self.forward_method: str = "__call__"
      self.init_method: str = "__init__"

  b.traits = DummyTraits()  # type: ignore
  assert b.compile(LogicalGraph("T"))

  b = PythonBackend(framework="keras")

  class DummyTraitsKeras:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.module_base: str = "keras.Layer"
      self.requires_super_init: bool = False
      self.forward_method: str = "call"
      self.init_method: str = "__init__"

  b.traits = DummyTraitsKeras()  # type: ignore
  assert b.compile(LogicalGraph("T"))


def test_python_backend_forward_init_fallback() -> None:
  """Test function."""
  b = PythonBackend(framework="torch")
  b._is_stateful = lambda x: False  # type: ignore
  b._is_stateful_layer = lambda x: False  # type: ignore

  class DummyTraits:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.module_base: str = "nn.Module"
      self.requires_super_init: bool = False
      self.forward_method: str = "forward"
      self.init_method: str = "__init__"

  b.traits = DummyTraits()  # type: ignore

  g = LogicalGraph("T")
  c: str = b.compile(g)
  assert "pass" in c
