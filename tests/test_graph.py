"""Test suite for graph."""

import typing

import libcst as cst

from ml_switcheroo.core.graph import GraphExtractor


def test_graph_extractor_init_pass() -> None:
  """Verify graph extractor init pass behavior."""
  code: str = """
class GeneratedNet:
    def __init__(self):
        self.conv = nn.Conv2d(1, 20, 5)
        self.fc = nn.Linear(20, 10, bias=True)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert extractor.model_name == "GeneratedNet"
  assert "conv" in extractor.layer_registry
  assert "fc" in extractor.layer_registry
  assert extractor.layer_registry["conv"].op_type == "Conv2d"
  assert extractor.layer_registry["fc"].attributes["bias"] == "True"


def test_graph_extractor_forward_pass() -> None:
  """Verify graph extractor forward pass behavior."""
  code: str = """
class GeneratedNet:
    def __init__(self):
        self.conv = nn.Conv2d(1, 20, 5)

    def forward(self, x):
        y = self.conv(x)
        return y
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert extractor.model_name == "GeneratedNet"
  assert "conv" in extractor.layer_registry
  assert len(extractor.graph.edges) == 2

  edge_sources: typing.List[str] = [e.source for e in extractor.graph.edges]
  edge_targets: typing.List[str] = [e.target for e in extractor.graph.edges]

  assert "Input_x" in edge_sources
  assert "conv" in edge_targets

  assert "conv" in edge_sources
  assert "output" in edge_targets


def test_graph_extractor_external_function() -> None:
  """Verify graph extractor external function behavior."""
  code: str = """
class GeneratedNet:
    def forward(self, x):
        y = F.relu(x)
        return y
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert extractor.layer_registry["func_relu"].op_type == "F.relu"


def test_graph_extractor_return_call() -> None:
  """Verify graph extractor return call behavior."""
  code: str = """
class GeneratedNet:
    def forward(self, x):
        return self.fc(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert "Input_x" in extractor.layer_registry
  edge_targets: typing.List[str] = [e.target for e in extractor.graph.edges]
  assert "output" in edge_targets


def test_graph_extractor_top_level_flow() -> None:
  """Verify graph extractor top level flow behavior."""
  code: str = """
x = 1
y = F.relu(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert "Input_x" in extractor.layer_registry


def test_graph_extractor_top_level_expr() -> None:
  """Verify graph extractor top level expr behavior."""
  code: str = """
F.relu(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert "Input_x" in extractor.layer_registry


def test_graph_extractor_multiple_functional_calls() -> None:
  """Verify repeated functional calls in forward pass get distinct node IDs."""
  code: str = """
class MultiReluNet:
    def forward(self, x):
        h1 = F.relu(x)
        h2 = F.relu(h1)
        return F.relu(h2)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: GraphExtractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_relu" in extractor.layer_registry
  assert "func_relu_1" in extractor.layer_registry
  assert "func_relu_2" in extractor.layer_registry

  # Re-resolving the first call returns 'func_relu' without creating a new entry
  first_stmt = tree.body[0]
  assert isinstance(first_stmt, cst.ClassDef)
  fn_stmt = first_stmt.body.body[0]
  assert isinstance(fn_stmt, cst.FunctionDef)
  assign_stmt = fn_stmt.body.body[0]
  assert isinstance(assign_stmt, cst.SimpleStatementLine)
  assign_node = assign_stmt.body[0]
  assert isinstance(assign_node, cst.Assign) and isinstance(assign_node.value, cst.Call)
  assert extractor._resolve_layer_or_func_name(assign_node.value.func, context_node=assign_node) == "func_relu"
