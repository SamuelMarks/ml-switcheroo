"""
Tests for Provenance Tracking (AST-Graph Linkage).

Verifies that the GraphExtractor correctly maps Logical Nodes back to their
original LibCST source nodes via the `node_map` registry.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor


def extract(code: str) -> GraphExtractor:
  """Helper to parse code and run extractor."""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  return extractor


def test_provenance_input_args():
  """
  Scenario: Functional inputs via arguments.
  Expectation: `Input_x` maps to `cst.Param` node.
  """
  code = """
def forward(self, x):
    pass
"""
  ex = extract(code)
  assert "Input_x" in ex.node_map
  node = ex.node_map["Input_x"]
  assert isinstance(node, cst.Param)
  assert node.name.value == "x"


def test_provenance_layer_definition():
  """
  Scenario: Layer defined in __init__.
  Expectation: `conv` maps to `cst.Assign` node.
  """
  code = """
class Net:
    def __init__(self):
        self.conv = nn.Conv2d(1, 1, 1)
"""
  ex = extract(code)
  assert "conv" in ex.node_map
  node = ex.node_map["conv"]
  assert isinstance(node, cst.Assign)
  # Validate it's the right line
  assert "nn.Conv2d" in cst.Module([]).code_for_node(node.value)


def test_provenance_functional_call():
  """
  Scenario: Functional call `F.relu`.
  Expectation: `func_relu` maps to `cst.Call`.
  """
  code = """
def forward(self, x):
    return F.relu(x)
"""
  ex = extract(code)
  assert "func_relu" in ex.node_map
  node = ex.node_map["func_relu"]
  assert isinstance(node, cst.Call)
  assert node.func.attr.value == "relu"


def test_provenance_script_constant():
  """
  Scenario: Top level script constant.
  Expectation: `Input_x` maps to `cst.Assign`.
  """
  code = "x = 1"
  ex = extract(code)
  assert "Input_x" in ex.node_map
  node = ex.node_map["Input_x"]
  assert isinstance(node, cst.Assign)


def test_provenance_return_output():
  """
  Scenario: Return statement implies Output node.
  Expectation: `output` maps to `cst.Return` node.
  """
  code = """
def forward(self, x):
    return x
"""
  ex = extract(code)
  assert "output" in ex.node_map
  node = ex.node_map["output"]
  assert isinstance(node, cst.Return)


def test_provenance_implicit_script_input():
  """
  Scenario: Script calls function with var 'img'.
  Expectation: `Input_img` maps to `cst.Arg`.
  """
  code = "x = op(img)"
  ex = extract(code)
  assert "Input_img" in ex.node_map
  node = ex.node_map["Input_img"]
  assert isinstance(node, cst.Arg)
  assert node.value.value == "img"
