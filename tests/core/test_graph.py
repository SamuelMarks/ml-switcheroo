"""Test module."""

import typing

import libcst as cst

from ml_switcheroo.core.graph import GraphExtractor


def test_graph_extractor_init_pass() -> None:
  """Docstring."""
  code: str = """
class MyModel:
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)
        self.linear = nn.Linear(32, 10)
        self.dropout = Dropout(p=0.5)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert extractor.model_name == "MyModel"
  assert "conv1" in extractor.layer_registry
  assert "linear" in extractor.layer_registry

  conv_node = extractor.layer_registry["conv1"]
  assert conv_node.kind == "Conv2d"
  assert conv_node.metadata["arg_0"] == "16"
  assert conv_node.metadata["arg_1"] == "32"
  assert conv_node.metadata["kernel_size"] == "3"

  # Check graph population
  assert len(extractor.graph.nodes) == 3


def test_graph_extractor_forward_pass() -> None:
  """Docstring."""
  code: str = """
class MyModel:
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 32)
        self.relu = F.relu

    def forward(self, x):
        h = self.conv1(x)
        out = self.relu(h)
        return out
"""
  tree: cst.Module = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  # Input nodes
  assert "Input_x" in extractor.layer_registry

  # Check edges
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "conv1") in edges
  assert ("conv1", "relu") in edges
  assert ("relu", "output") in edges


def test_graph_extractor_direct_call_return() -> None:
  """Docstring."""
  code: str = """
class MyModel:
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 32)

    def forward(self, x):
        return self.conv1(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "conv1") in edges
  assert ("conv1", "output") in edges


def test_graph_extractor_top_level_data_flow() -> None:
  """Docstring."""
  code: str = """
x = 1
y = 2
z = add(x, y)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert "Input_x" in extractor.layer_registry
  assert "Input_y" in extractor.layer_registry
  assert "func_add" in extractor.layer_registry

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "func_add") in edges
  assert ("Input_y", "func_add") in edges


def test_graph_extractor_top_level_expr() -> None:
  """Docstring."""
  code: str = """
func(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)

  assert "func_func" in extractor.layer_registry
  assert "Input_x" in extractor.layer_registry

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "func_func") in edges


def test_graph_extractor_missing_nodes() -> None:
  """Docstring."""
  # Various branches that return None or False
  extractor = GraphExtractor()

  # Not self
  extractor._in_init = True
  code: str = "other.layer = nn.Linear()"
  cst.parse_module(code).visit(extractor)

  # Not Call
  code2: str = "self.layer = 1"
  cst.parse_module(code2).visit(extractor)

  # Data flow not call
  extractor._in_init = False
  extractor._in_forward = True
  code3: str = "x = 1"
  cst.parse_module(code3).visit(extractor)

  # Get var name complex
  code4: str = "x[0] = func(y[0])"
  cst.parse_module(code4).visit(extractor)


def test_graph_extractor_return_complex() -> None:
  """Docstring."""
  code: str = """
class MyModel:
    def forward(self, x):
        return x + 1
"""
  tree: cst.Module = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  # Shouldn't crash, should just return False in visit_Return since value is BinOp


def test_graph_extractor_missing_paths() -> None:
  """Docstring."""
  extractor = GraphExtractor()
  extractor._in_forward = True

  # Not Call (Line 278)
  code: str = "x = y"  # BinOp or Name depending on RHS. Name is caught by top_level block if depth=0, but _scope_depth is 0 in tests unless we mock
  extractor._scope_depth = 1  # ensure we bypass top level data flow
  cst.parse_module(code).visit(extractor)

  # Missing args kwargs (Line 331)
  code2: str = "x = func(kwarg=1)"
  cst.parse_module(code2).visit(extractor)

  # Not context_node call (Line 319) - this is covered if we pass just cst.Call
  call_node = typing.cast(
    cst.Call,
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("func(1)").body[0]).body[0]).value,
  )
  extractor._resolve_layer_or_func_name(call_node.func, context_node=call_node)

  # Unresolved func name (Line 340, 360)
  # E.g. a complex func node that get_full_name can't resolve
  call_node_bad = typing.cast(
    cst.Call,
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("func()[0]()").body[0]).body[0]).value,
  )
  extractor._analyze_call_expression(call_node_bad, [])


def test_graph_extractor_context_node_call() -> None:
  """Docstring."""
  extractor = GraphExtractor()
  call_node = typing.cast(
    cst.Call,
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("some_func(1)").body[0]).body[0]).value,
  )
  # Since some_func is not in layer_registry, it will be added, hitting line 319
  extractor._resolve_layer_or_func_name(call_node.func, context_node=call_node)
  assert "func_some_func" in extractor.layer_registry
