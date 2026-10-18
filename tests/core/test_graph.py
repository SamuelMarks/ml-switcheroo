"""Test suite for graph."""

import typing

import libcst as cst

from ml_switcheroo.core.graph import GraphExtractor


def test_graph_extractor_init_pass() -> None:
  """Verify graph extractor init pass behavior."""
  code: str = """
class MyModel:
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)
        self.linear = nn.Linear(32, 10)
        self.dropout = Dropout(p=0.5)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: typing.Any = GraphExtractor()
  tree.visit(extractor)

  assert extractor.model_name == "MyModel"
  assert "conv1" in extractor.layer_registry
  assert "linear" in extractor.layer_registry

  conv_node: typing.Any = extractor.layer_registry["conv1"]
  assert conv_node.op_type == "Conv2d"
  assert conv_node.attributes["arg_0"] == "16"
  assert conv_node.attributes["arg_1"] == "32"
  assert conv_node.attributes["kernel_size"] == "3"

  # Check graph population
  assert len(extractor.graph.nodes) == 3


def test_graph_extractor_forward_pass() -> None:
  """Verify graph extractor forward pass behavior."""
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
  extractor: typing.Any = GraphExtractor()
  tree.visit(extractor)

  # Input nodes
  assert "Input_x" in extractor.layer_registry

  # Check edges
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "conv1") in edges
  assert ("conv1", "relu") in edges
  assert ("relu", "output") in edges


def test_graph_extractor_direct_call_return() -> None:
  """Verify graph extractor direct call return behavior."""
  code: str = """
class MyModel:
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 32)

    def forward(self, x):
        return self.conv1(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: typing.Any = GraphExtractor()
  tree.visit(extractor)

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "conv1") in edges
  assert ("conv1", "output") in edges


def test_graph_extractor_top_level_data_flow() -> None:
  """Verify graph extractor top level data flow behavior."""
  code: str = """
x = 1
y = 2
z = add(x, y)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: typing.Any = GraphExtractor()
  tree.visit(extractor)

  assert "Input_x" in extractor.layer_registry
  assert "Input_y" in extractor.layer_registry
  assert "func_add" in extractor.layer_registry

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "func_add") in edges
  assert ("Input_y", "func_add") in edges


def test_graph_extractor_top_level_expr() -> None:
  """Verify graph extractor top level expr behavior."""
  code: str = """
func(x)
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: typing.Any = GraphExtractor()
  tree.visit(extractor)

  assert "func_func" in extractor.layer_registry
  assert "Input_x" in extractor.layer_registry

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in extractor.graph.edges]
  assert ("Input_x", "func_func") in edges


def test_graph_extractor_missing_nodes() -> None:
  """Verify graph extractor missing nodes behavior."""
  # Various branches that return None or False
  extractor: typing.Any = GraphExtractor()

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
  _res: typing.Any = cst.parse_module(code4).visit(extractor)
  assert isinstance(_res, cst.Module)
  assert "func_func" in extractor.layer_registry
  assert extractor.layer_registry["func_func"].op_type == "func"


def test_graph_extractor_return_complex() -> None:
  """Verify graph extractor return complex behavior."""
  code: str = """
class MyModel:
    def forward(self, x):
        return x + 1
"""
  tree: cst.Module = cst.parse_module(code)
  extractor: typing.Any = GraphExtractor()
  _res: typing.Any = tree.visit(extractor)
  assert isinstance(_res, cst.Module)
  assert "Input_x" in extractor.graph.nodes
  assert "output" not in extractor.graph.nodes
  assert extractor.graph.edges == []
  # Shouldn't crash, should just return False in visit_Return since value is BinOp


def test_graph_extractor_missing_paths() -> None:
  """Verify graph extractor missing paths behavior."""
  extractor: typing.Any = GraphExtractor()
  extractor._in_forward = True

  # Not Call (Line 278)
  code: str = "x = y"  # BinOp or Name depending on RHS. Name is caught by top_level block if depth=0, but _scope_depth is 0 in tests unless we mock
  extractor._scope_depth = 1  # ensure we bypass top level data flow
  cst.parse_module(code).visit(extractor)

  # Missing args kwargs (Line 331)
  code2: str = "x = func(kwarg=1)"
  cst.parse_module(code2).visit(extractor)

  # Not context_node call (Line 319) - this is covered if we pass just cst.Call
  call_node: typing.Any = typing.cast(
    cst.Call,
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("func(1)").body[0]).body[0]).value,
  )
  extractor._resolve_layer_or_func_name(call_node.func, context_node=call_node)

  # Unresolved func name (Line 340, 360)
  # E.g. a complex func node that get_full_name can't resolve
  call_node_bad: typing.Any = typing.cast(
    cst.Call,
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("func()[0]()").body[0]).body[0]).value,
  )
  _res: typing.Any = extractor._analyze_call_expression(call_node_bad, [])
  assert _res is None
  assert "func_func" in extractor.layer_registry
  assert extractor.layer_registry["func_func"].attributes == {"kwarg": "1"}


def test_graph_extractor_context_node_call() -> None:
  """Verify graph extractor context node call behavior."""
  extractor: typing.Any = GraphExtractor()
  call_node: typing.Any = typing.cast(
    cst.Call,
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("some_func(1)").body[0]).body[0]).value,
  )
  # Since some_func is not in layer_registry, it will be added, hitting line 319
  extractor._resolve_layer_or_func_name(call_node.func, context_node=call_node)
  assert "func_some_func" in extractor.layer_registry
