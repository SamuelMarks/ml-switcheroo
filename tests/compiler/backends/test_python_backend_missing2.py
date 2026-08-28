"""Tests."""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


def test_python_backend_frameworks() -> None:
  """Test function."""
  PythonBackend(framework="flax_nnx").compile(LogicalGraph("T"))

  semantics = SemanticsManager()

  def mock_resolve(api: str, fw: str) -> Optional[dict[str, Any]]:
    """Test function."""
    if api == "Relu":
      if fw == "torch":
        return {"api": "torch.nn.functional.relu"}
      elif fw == "mlx":
        return {"api": "mlx.core.relu"}
    return None

  semantics.resolve_variant = mock_resolve

  b = PythonBackend(framework="torch", semantics=semantics)
  c: str = b.compile(LogicalGraph("T", [LogicalNode("n1", "Relu")]))
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


def test_python_backend_sharding_and_metadata() -> None:
  """Test function."""
  b = PythonBackend(framework="torch")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", metadata={"kwarg_a": "1"}, sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer

  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  c: str = b.compile(g)
  assert "kwarg_a=1" in c
  assert "distribute_tensor" in c


def test_python_backend_sharding_jax() -> None:
  """Test function."""
  b = PythonBackend(framework="jax")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  assert "with_sharding_constraint" in b.compile(g)


def test_python_backend_sharding_keras() -> None:
  """Test function."""
  b = PythonBackend(framework="keras")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  assert "keras.distribution.layout" in b.compile(g)


def test_python_backend_sharding_mlx() -> None:
  """Test function."""
  b = PythonBackend(framework="mlx")

  class FakeSharding:
    """Docstring."""

    def __init__(self) -> None:
      """Test function."""
      self.axes: list[str] = ["x"]

  n = LogicalNode("n", "func_x", sharding=FakeSharding())  # type: ignore

  def mock_is_stateful_layer(node: LogicalNode) -> bool:
    """Test function."""
    return False

  b._is_stateful_layer = mock_is_stateful_layer
  g = LogicalGraph("T", nodes=[LogicalNode("i", "Input"), n], edges=[LogicalEdge("i", "n")])
  assert "mx.distributed.shard" in b.compile(g)


def test_python_backend_is_stateful_layer_fallbacks() -> None:
  """Test function."""
  b = PythonBackend()
  assert not b._is_stateful_layer(LogicalNode("n", "math.add"))
  assert not b._is_stateful_layer(LogicalNode("n", "a.b.func_x"))
  assert not b._is_stateful_layer(LogicalNode("n", "math.add"))
