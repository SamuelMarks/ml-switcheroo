"""Tests."""

import libcst as cst
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


def test_python_backend_is_stateful_layer_fallbacks():
  """Test function."""
  b = PythonBackend()
  assert not b._is_stateful_layer(LogicalNode("n", "a.b.func_x"))


def test_python_backend_frameworks_base_class():
  """Test function."""
  backend = PythonBackend(framework="paxml")

  class DummyTraits:
    def __init__(self):
      """Test function."""
      self.module_base = "praxis.base_layer.BaseLayer"
      self.requires_super_init = True
      self.forward_method = "forward"

  backend.traits = DummyTraits()
  assert backend.compile(LogicalGraph("T"))

  backend = PythonBackend(framework="keras")

  class DummyTraitsKeras:
    def __init__(self):
      """Test function."""
      self.module_base = "keras.Layer"
      self.requires_super_init = True
      self.forward_method = "call"

  backend.traits = DummyTraitsKeras()
  assert backend.compile(LogicalGraph("T"))


def test_python_backend_layer_init_resolution():
  """Test function."""
  semantics = SemanticsManager()

  def mock_resolve(api, fw):
    """Test function."""
    if api == "Relu":
      if fw == "torch":
        return {"api": "torch.nn.functional.relu"}
      elif fw == "mlx":
        return {"api": "mlx.core.relu"}
    if api == "Linear":
      if fw == "torch":
        return {"api": "torch.nn.Linear"}
      if fw == "mlx":
        return {"api": "mlx.nn.Linear"}
    if api == "KerasDense":
      return {"api": "keras.layers.Dense"}
    if api == "PMLX":
      return {"api": "Linear"}  # test prefix fallback
    if api == "TFLayer":
      if fw == "tensorflow":
        return {"api": "Dense"}
    if api == "MLXSwiGLU":
      return {"api": "SwiGLU"}
    return None

  semantics.resolve_variant = mock_resolve

  b = PythonBackend(framework="torch", semantics=semantics)
  n_relu = LogicalNode("n_relu", "Relu")
  res = b._generate_layer_init(n_relu)
  assert "nn.Relu" in cst.Module(body=[res]).code

  n_linear = LogicalNode("n_linear", "Linear")
  res = b._generate_layer_init(n_linear)
  assert "nn.Linear" in cst.Module(body=[res]).code

  b = PythonBackend(framework="mlx", semantics=semantics)
  res = b._generate_layer_init(n_relu)
  assert "nn.Relu" in cst.Module(body=[res]).code

  res = b._generate_layer_init(n_linear)
  assert "nn.Linear" in cst.Module(body=[res]).code

  b = PythonBackend(framework="keras", semantics=semantics)
  res = b._generate_layer_init(LogicalNode("n1", "PMLX"))
  assert "keras.layers.Linear" in cst.Module(body=[res]).code

  b = PythonBackend(framework="tensorflow", semantics=semantics)
  res = b._generate_layer_init(LogicalNode("n1", "TFLayer"))
  assert "tf.keras.layers.Dense" in cst.Module(body=[res]).code

  b = PythonBackend(framework="mlx", semantics=semantics)
  res = b._generate_layer_init(LogicalNode("n1", "MLXSwiGLU"))
  assert "nn.silu" in cst.Module(body=[res]).code


def test_python_backend_forward_args():
  """Test function."""
  b = PythonBackend(framework="torch")

  def mock_is_stateful_layer(node):
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer

  n = LogicalNode("n", "func_x", metadata={"kwarg_a": "1"})
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  c = b.compile(g)
  assert "kwarg_a=1" in c
