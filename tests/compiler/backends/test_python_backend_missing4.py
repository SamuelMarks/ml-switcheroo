"""Tests."""

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_python_backend_base_class_resolution():
  """Test function."""
  b = PythonBackend(framework="paxml")

  class DummyTraits:
    def __init__(self):
      """Test function."""
      self.module_base = "praxis.base_layer.BaseLayer"
      self.requires_super_init = False
      self.forward_method = "__call__"
      self.init_method = "__init__"

  b.traits = DummyTraits()
  assert b.compile(LogicalGraph("T"))

  b = PythonBackend(framework="keras")

  class DummyTraitsKeras:
    def __init__(self):
      """Test function."""
      self.module_base = "keras.Layer"
      self.requires_super_init = False
      self.forward_method = "call"
      self.init_method = "__init__"

  b.traits = DummyTraitsKeras()
  assert b.compile(LogicalGraph("T"))


def test_python_backend_forward_init_fallback():
  """Test function."""
  b = PythonBackend(framework="torch")
  b._is_stateful = lambda x: False
  b._is_stateful_layer = lambda x: False

  class DummyTraits:
    def __init__(self):
      """Test function."""
      self.module_base = "nn.Module"
      self.requires_super_init = False
      self.forward_method = "forward"
      self.init_method = "__init__"

  b.traits = DummyTraits()

  g = LogicalGraph("T")
  c = b.compile(g)
  assert "pass" in c
