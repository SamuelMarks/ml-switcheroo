"""
Tests for Graph Synthesizer.

Verifies that LogicalGraphs are correctly converted into valid Python/PyTorch source code.
Covers:
1. Basic Layer chaining.
2. Mix of Stateful Layers and Functional Operations.
3. Metadata argument rendering.
"""

import ast
import pytest
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.graph_synthesizer import GraphSynthesizer


@pytest.fixture
def synthesizer() -> GraphSynthesizer:
  return GraphSynthesizer()


def validate_python(code: str) -> None:
  """Ensures generated code is syntactically valid Python."""
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_synthesize_simple_chain(synthesizer: GraphSynthesizer) -> None:
  """
  Scenario: Input -> Conv2d -> Output.
  Expectation:
      class DecompiledNet(nn.Module):
          def __init__(self):
              super().__init__()
              self.conv1 = nn.Conv2d()
          def forward(self, x):
              x = self.conv1(x)
              return x
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    LogicalNode("conv1", "Conv2d"),
    LogicalNode("output", "Output"),
  ]
  g.edges = [
    LogicalEdge("x", "conv1"),
    LogicalEdge("conv1", "output"),
  ]

  code = synthesizer.generate(g, "SimpleNet")
  validate_python(code)

  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "x = self.conv1(x)" in code
  assert "return x" in code


def test_synthesize_functional_mix(synthesizer: GraphSynthesizer) -> None:
  """
  Scenario: Input -> Layer -> Functional -> Output.
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    LogicalNode("fc", "Linear"),
    # Functional Op
    LogicalNode("flat", "torch.flatten"),
    LogicalNode("out", "Output"),
  ]
  g.edges = [
    LogicalEdge("x", "fc"),
    LogicalEdge("fc", "flat"),
    LogicalEdge("flat", "out"),
  ]

  code = synthesizer.generate(g)
  validate_python(code)

  # Init should only have fc
  assert "self.fc = nn.Linear()" in code
  assert "self.flat" not in code
  assert "torch.flatten" not in code.split("__init__")[1].split("forward")[0]

  # Forward should have both
  assert "x = self.fc(x)" in code
  assert "x = torch.flatten(x)" in code


def test_synthesize_metadata_args(synthesizer: GraphSynthesizer) -> None:
  """
  Scenario: Nodes have argument metadata.
  Expectation: Arguments injected into init/call.
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    # Positional args simulated
    LogicalNode("c1", "Conv2d", {"arg_0": "1", "arg_1": "32", "kernel_size": "3"}),
    LogicalNode("out", "Output"),
  ]
  g.edges = [LogicalEdge("x", "c1"), LogicalEdge("c1", "out")]

  code = synthesizer.generate(g)
  validate_python(code)

  # Check for: nn.Conv2d(1, 32, kernel_size=3)
  # Exact ordering of kwargs depends on implementation, but positionals come first
  assert "nn.Conv2d(1, 32, kernel_size=3)" in code


def test_synthesize_custom_input_name(synthesizer: GraphSynthesizer) -> None:
  """
  Scenario: Input node has specific name 'img'.
  Expectation: forward argument is 'img'.
  """
  g = LogicalGraph()
  g.nodes = [LogicalNode("in_node", "Input", {"name": "img"}), LogicalNode("l1", "Linear"), LogicalNode("out", "Output")]
  g.edges = [LogicalEdge("in_node", "l1"), LogicalEdge("l1", "out")]

  code = synthesizer.generate(g)
  validate_python(code)

  assert "def forward(self, img):" in code
  assert "img = self.l1(img)" in code
  assert "return img" in code
