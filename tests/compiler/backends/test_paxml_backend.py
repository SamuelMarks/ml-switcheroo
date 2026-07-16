"""Tests for PaxML Code Gen."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.backends.python import PythonBackend


def test_synthesize_paxml_chain() -> None:
  """Auto-generated doc."""
  backend = PythonBackend(framework="paxml")
  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("conv1", "Conv2d", {"out_channels": "16", "kernel_size": "(3, 3)"}),
      LogicalNode("relu", "ReLU"),
      LogicalNode("fc", "Linear", {"out_features": "10"}),
      LogicalNode("output", "Output"),
    ],
    edges=[
      LogicalEdge("x", "conv1"),
      LogicalEdge("conv1", "relu"),
      LogicalEdge("relu", "fc"),
      LogicalEdge("fc", "output"),
    ],
  )
  code = backend.generate(g, "PaxNet")

  assert "class PaxNet(BaseLayer):" in code
  assert "def setup(self):" in code
  assert "self.create_child('conv1', pl.Conv2d(kernel_size=(3, 3), out_channels=16))" in code
  assert "self.create_child('relu', pl.ReLU())" in code
  assert "self.create_child('fc', pl.Linear(out_features=10))" in code
  assert "def __call__(self, x):" in code
  assert "import praxis.layers as pl" in code
  assert "import praxis.layers.convolutions" in code
