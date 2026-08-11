"""Tests."""

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


def test_python_backend_frameworks():
  """Test function."""
  PythonBackend(framework="flax_nnx").compile(LogicalGraph("T"))

  semantics = SemanticsManager()

  def mock_resolve(api, fw):
    """Test function."""
    if api == "Relu":
      if fw == "torch":
        return {"api": "torch.nn.functional.relu"}
      elif fw == "mlx":
        return {"api": "mlx.core.relu"}
    return None

  semantics.resolve_variant = mock_resolve

  b = PythonBackend(framework="torch", semantics=semantics)
  c = b.compile(LogicalGraph("T", [LogicalNode("n1", "Relu")]))
  assert "self.n1 = nn.Relu" in c

  b = PythonBackend(framework="mlx", semantics=semantics)
  c = b.compile(LogicalGraph("T", [LogicalNode("n1", "Relu")]))
  assert "self.n1 = nn.Relu" in c

  b = PythonBackend(framework="torch")
  c = b.compile(LogicalGraph("T", [LogicalNode("n1", "Linear")]))
  assert "self.n1 = nn.Linear" in c

  b = PythonBackend(framework="mlx")
  c = b.compile(LogicalGraph("T", [LogicalNode("n1", "Linear")]))
  assert "self.n1 = nn.Linear" in c

  b = PythonBackend(framework="paxml")
  c = b.compile(LogicalGraph("T", [LogicalNode("n1", "Linear")]))
  assert "pl.Linear" in c

  b = PythonBackend(framework="keras")
  c = b.compile(LogicalGraph("T", [LogicalNode("n1", "Layer")]))
  assert "self.n1 = keras.layers.Layer" in c


def test_python_backend_sharding_and_metadata():
  """Test function."""
  b = PythonBackend(framework="torch")

  class FakeSharding:
    def __init__(self):
      """Test function."""
      self.axes = ["x"]

  n = LogicalNode("n", "func_x", metadata={"kwarg_a": "1"}, sharding=FakeSharding())

  def mock_is_stateful_layer(node):
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer

  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  c = b.compile(g)
  assert "kwarg_a=1" in c
  assert "distribute_tensor" in c


def test_python_backend_sharding_jax():
  """Test function."""
  b = PythonBackend(framework="jax")

  class FakeSharding:
    def __init__(self):
      """Test function."""
      self.axes = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())

  def mock_is_stateful_layer(node):
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  assert "with_sharding_constraint" in b.compile(g)


def test_python_backend_sharding_keras():
  """Test function."""
  b = PythonBackend(framework="keras")

  class FakeSharding:
    def __init__(self):
      """Test function."""
      self.axes = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())

  def mock_is_stateful_layer(node):
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  assert "keras.distribution.layout" in b.compile(g)


def test_python_backend_sharding_mlx():
  """Test function."""
  b = PythonBackend(framework="mlx")

  class FakeSharding:
    def __init__(self):
      """Test function."""
      self.axes = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())

  def mock_is_stateful_layer(node):
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  assert "mx.distributed.shard" in b.compile(g)


def test_python_backend_is_stateful_layer_fallbacks():
  """Test function."""
  b = PythonBackend()
  assert not b._is_stateful_layer(LogicalNode("n", "math.add"))
  # The dot check uses True if not starting with nn.
  # so "a.b.func_x" returns False? No, the code says:
  # if "." in node.kind and not node.kind.startswith("nn."): return False
  # Yes, it returns False.
  assert not b._is_stateful_layer(LogicalNode("n", "a.b.func_x"))
  assert not b._is_stateful_layer(LogicalNode("n", "math.add"))
