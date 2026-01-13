"""
Tests for LatexBackend.

Verifies:
1.  Python Code -> LogicalGraph -> LaTeX string flow.
2.  Identification of Layers (__init__) vs Ops (forward).
3.  Handling of positional and keyword arguments.
4.  Valid document structure.
"""

import pytest
import libcst as cst
from ml_switcheroo.compiler.backends.extras import LatexBackend
from ml_switcheroo.core.tikz.analyser import GraphExtractor


@pytest.fixture
def backend():
  return LatexBackend()


def parse_to_graph(code: str):
  """Helper to parse Python code into a LogicalGraph for the backend."""
  module = cst.parse_module(code)
  extractor = GraphExtractor()
  module.visit(extractor)
  return extractor.graph


def test_basic_model_emission(backend):
  """
  Scenario: Simple Linear model.
  """
  code = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""
  graph = parse_to_graph(code)
  # Ensure name extraction worked, or set manually
  graph.name = "Simple"

  latex = backend.compile(graph)

  # 1. Structure
  assert r"\documentclass" in latex
  assert r"\begin{DefModel}{Simple}" in latex

  # 2. Attribute (Memory)
  assert r"\Attribute{fc}{Linear}" in latex

  # 3. Input
  assert r"\Input{input}{[_]}" in latex

  # 4. StateOp
  assert r"\StateOp{op_fc}{fc}{input}{[_]}" in latex

  # 5. Return
  assert r"\Return{op_fc}" in latex

  # 6. Usage Comment
  assert "% [Requirement] midl.sty" in latex


def test_mixed_functional_emission(backend):
  """
  Scenario: Layer call then Functional Call (ReLU).
  """
  code = """
class Mix(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(1, 1)

    def forward(self, img):
        x = self.conv(img)
        x = F.relu(x)
        return x
"""
  graph = parse_to_graph(code)
  latex = backend.compile(graph)

  # Check Attribute
  assert r"\Attribute{conv}{Conv2d}" in latex

  # Check StateOp
  assert r"\StateOp{op_conv}{conv}{input}{[_]}" in latex

  # Check ComputeOp (functional relu)
  assert r"\Op{op_func_relu}{Relu}" in latex


def test_keyword_arguments_emission(backend):
  """
  Verify keyword args in Python propagate to LaTeX config.
  """
  code = """
class KArg(nn.Module):
    def __init__(self):
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x):
        return self.pool(x)
"""
  graph = parse_to_graph(code)
  latex = backend.compile(graph)

  assert r"\Attribute{pool}{MaxPool2d}" in latex
  assert "kernel_size=2" in latex
