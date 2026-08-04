"""Unit tests for the AST-based Graph Extraction Frontend.

This module verifies the behavior of the `GraphExtractor` class, which parses
Python Abstract Syntax Trees (AST) using LibCST to construct a `LogicalGraph`.
The tests validate extraction of:
- Layer initialization and metadata registration during `__init__`.
- Trace data flow and edge generation during `forward` or lifecycle methods.
- Integration of stateless/functional operations (e.g., `F.relu`).
- Module-level operations and variable assignments.
"""

import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor


def test_graph_extractor_init_pass():
  """Verifies that GraphExtractor successfully processes class instantiation.

  Specifically, it validates that layer definitions in the `__init__` method
  are registered with correct attributes, kinds (types), and both positional
  and keyword argument metadata.

  Args:
      None

  Returns:
      None
  """
  code = """
class MyModel:
    def __init__(self):
        self.conv = nn.Conv2d(3, 16)
        self.linear = nn.Linear(16, 10, bias=False)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph
  assert len(graph.nodes) == 2
  assert "conv" in extractor.layer_registry
  assert extractor.layer_registry["conv"].kind == "Conv2d"
  assert extractor.layer_registry["conv"].metadata["arg_0"] == "3"
  assert extractor.layer_registry["conv"].metadata["arg_1"] == "16"

  assert "linear" in extractor.layer_registry
  assert extractor.layer_registry["linear"].kind == "Linear"
  assert extractor.layer_registry["linear"].metadata["arg_0"] == "16"
  assert extractor.layer_registry["linear"].metadata["arg_1"] == "10"
  assert extractor.layer_registry["linear"].metadata["bias"] == "False"


def test_graph_extractor_forward_pass():
  """Verifies that GraphExtractor traces data flow during class execution.

  Specifically, it validates that edges are built between input variables,
  layer instances, and output returned values during the `forward` pass.

  Args:
      None

  Returns:
      None
  """
  code = """
class MyModel:
    def __init__(self):
        self.conv = nn.Conv2d()
        self.linear = nn.Linear()

    def forward(self, x):
        h = self.conv(x)
        out = self.linear(h)
        return out
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph

  edges = [(e.source, e.target) for e in graph.edges]
  assert ("Input_x", "conv") in edges
  assert ("conv", "linear") in edges
  assert ("linear", "output") in edges

  node_ids = [n.id for n in graph.nodes]
  assert "Input_x" in node_ids
  assert "conv" in node_ids
  assert "linear" in node_ids
  assert "output" in node_ids


def test_graph_extractor_functional_ops():
  """Verifies that stateless or functional calls are registered correctly.

  Specifically, it checks that expressions such as `F.relu(x)` are parsed
  as logical nodes and successfully connect to input and output flows.

  Args:
      None

  Returns:
      None
  """
  code = """
class MyModel:
    def forward(self, x):
        h = F.relu(x)
        return h
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph

  edges = [(e.source, e.target) for e in graph.edges]
  assert ("Input_x", "func_relu") in edges
  assert ("func_relu", "output") in edges


def test_graph_extractor_standalone_expr():
  """Verifies that standalone/functional statement calls are captured.

  Specifically, it checks that operations that don't assign to variables,
  such as `tl.store`, are mapped into functional call nodes and included in
  the nodes list of the logical graph.

  Args:
      None

  Returns:
      None
  """
  code = """
def kernel(x):
    tl.store(out, x)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph
  node_ids = [n.id for n in graph.nodes]
  assert "Input_x" in node_ids
  assert "func_store" in node_ids


def test_graph_extractor_return_call():
  """Verifies that a direct expression return statement is mapped correctly.

  Specifically, it checks that return statements wrapping an operation,
  e.g., `return F.relu(x)`, create appropriate data-flow edges leading
  directly to the final output node.

  Args:
      None

  Returns:
      None
  """
  code = """
class MyModel:
    def forward(self, x):
        return F.relu(x)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph
  edges = [(e.source, e.target) for e in graph.edges]
  assert ("Input_x", "func_relu") in edges
  assert ("func_relu", "output") in edges


def test_graph_extractor_direct_assignment():
  """Verifies graph extraction from top-level/module-level code assignments.

  Specifically, it checks that variables assigned and passed through functions
  outside class boundaries are correctly traced, linking inputs to operations.

  Args:
      None

  Returns:
      None
  """
  code = """
x = 5
y = torch.add(x, 2)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph
  node_ids = [n.id for n in graph.nodes]
  assert "Input_x" in node_ids
  assert "func_add" in node_ids

  edges = [(e.source, e.target) for e in graph.edges]
  assert ("Input_x", "func_add") in edges


def test_graph_extractor_var_from_expr():
  """Verifies variables mapped from functional expressions trace downstream.

  Specifically, it checks that intermediate output variables from expressions
  (like `h = F.relu(x)`) correctly pass as inputs to downstream operations,
  retaining full data-flow connectivity.

  Args:
      None

  Returns:
      None
  """
  code = """
class MyModel:
    def forward(self, x):
        h = F.relu(x)
        return torch.add(h, 2)
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  graph = extractor.graph

  edges = [(e.source, e.target) for e in graph.edges]
  assert ("func_relu", "func_add") in edges
  assert ("func_add", "output") in edges
