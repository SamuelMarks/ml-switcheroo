"""
Tests for TikZ Python Resynthesizer (Feature #5).

Verifies:
1.  **Class Construction**: Correct inheritance and init method.
2.  **Forward Logic**: Sequential chaining and variable tracking.
3.  **Framework Switching**: Torch vs JAX semantics (self.conv vs rngs).
4.  **Complex Graphs**: Branching/Merging logic.
"""

import pytest
import textwrap
import libcst as cst
from ml_switcheroo.core.tikz.synthesizer import GraphSynthesizer
from ml_switcheroo.core.tikz.analyser import LogicalGraph, LogicalNode, LogicalEdge


@pytest.fixture
def linear_graph():
  """
  Input -> Conv(k=3) -> Relu -> Output
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("input", "Input"),
    LogicalNode("conv1", "Conv2d", {"arg_0": "3", "arg_1": "32"}),
    LogicalNode("act", "ReLU", {}),
    LogicalNode("output", "Output"),
  ]
  g.edges = [
    LogicalEdge("input", "conv1"),
    LogicalEdge("conv1", "act"),
    LogicalEdge("act", "output"),
  ]
  return g


@pytest.fixture
def branching_graph():
  """
  Input -> A
  Input -> B
  Merge(A, B) -> Output
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("input", "Input"),
    LogicalNode("path_a", "Linear", {"out": "10"}),
    LogicalNode("path_b", "Linear", {"out": "10"}),
    LogicalNode("merge", "Add", {}),
    LogicalNode("output", "Output"),
  ]
  # Input feeds both
  g.edges = [
    LogicalEdge("input", "path_a"),
    LogicalEdge("input", "path_b"),
    LogicalEdge("path_a", "merge"),
    LogicalEdge("path_b", "merge"),
    LogicalEdge("merge", "output"),
  ]
  return g


def test_torch_linear_generation(linear_graph):
  """
  Expect standard PyTorch nn.Module structure.
  """
  gen = GraphSynthesizer(framework="torch")
  code = gen.generate(linear_graph, "MyNet")

  # 1. Imports
  assert "import torch" in code
  assert "import torch.nn as nn" in code

  # 2. Class
  assert "class MyNet(nn.Module):" in code

  # 3. Init
  assert "super().__init__()" in code
  assert "self.conv1 = nn.Conv2d(3, 32)" in code
  assert "self.act = nn.ReLU()" in code

  # 4. Forward
  assert "def forward(self, x):" in code
  assert "conv1 = self.conv1(x)" in code
  assert "act = self.act(conv1)" in code
  assert "return act" in code


def test_jax_linear_generation(linear_graph):
  """
  Expect Flax NNX structure (rngs).
  """
  gen = GraphSynthesizer(framework="jax")
  code = gen.generate(linear_graph, "MyJaxNet")

  # 1. Imports
  assert "from flax import nnx" in code

  # 2. Class
  assert "class MyJaxNet(nnx.Module):" in code

  # 3. Init (RNGs injection)
  assert "def __init__(self, rngs: nnx.Rngs):" in code
  # Heuristic: Conv2d gets rngs appended
  assert "self.conv1 = nnx.Conv2d(3, 32, rngs=rngs)" in code

  # 4. Call
  assert "def __call__(self, x):" in code
  assert "return act" in code


def test_branching_logic(branching_graph):
  """
  Verify multi-argument call construction.
  """
  gen = GraphSynthesizer(framework="torch")
  code = gen.generate(branching_graph, "BranchNet")

  # Check Merge Step
  # Should call merge with both inputs: merge = self.merge(path_a, path_b)
  # Note: Order depends on topo sort stability, checking presence of both
  assert "path_a = self.path_a(x)" in code
  assert "path_b = self.path_b(x)" in code

  # Check join
  # Regex loosely checking merge call structure
  import re

  assert re.search(r"merge = self.merge\(path_[ab], path_[ab]\)", code)


def test_metadata_cleaning():
  """Verify parsing of key-value metadata into args."""
  g = LogicalGraph()
  # Test keyword arg and string cleaning
  g.nodes = [LogicalNode("l1", "Test", {"kernel_size": "3", "padding": "1"})]

  gen = GraphSynthesizer(framework="torch")

  # We verify the output code string, as manipulating CST nodes directly in test
  # requires full module context wrapper.
  code = gen.generate(g, "TestNet")

  # Should look like nn.Test(kernel_size=3, padding=1)
  # Clean whitespace
  clean = code.replace(" ", "")
  assert "kernel_size=3" in clean
  assert "padding=1" in clean


def test_implicit_return_handling():
  """Verify return statement generated even without explicit Output node."""
  g = LogicalGraph()
  g.nodes = [LogicalNode("in", "Input"), LogicalNode("l1", "Layer")]
  g.edges = [LogicalEdge("in", "l1")]

  gen = GraphSynthesizer()
  code = gen.generate(g)

  assert "return l1" in code
