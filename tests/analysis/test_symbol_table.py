"""Test module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.analysis.symbol_table import SymbolTable, SymbolTableAnalyzer
from ml_switcheroo.analysis.symbol_types import ModuleType, Scope, SymbolType, TensorType, UnionType
from ml_switcheroo.semantics.manager import SemanticsManager


def test_symbol_table_basic() -> None:
  """Docstring."""
  table: SymbolTable = SymbolTable()
  node: cst.Name = cst.Name("test")
  sym: TensorType = TensorType(framework="torch")
  table.record_type(node, sym)
  assert table.get_type(node) == sym
  assert table.get_type(cst.Name("other")) is None


def test_symbol_table_analyzer_imports() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
import torch
import torch.nn as nn
from jax import numpy as jnp
from jax import *
from jax import lax

torch.add(1, 2)
nn.Conv2d(1, 1, 1)
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)

  scope: Scope = analyzer.current_scope
  assert scope.get("torch") == ModuleType(path="torch", name="Module")
  assert scope.get("nn") == ModuleType(path="torch.nn", name="Module")
  assert scope.get("jnp") == ModuleType(path="jax.numpy", name="Module")
  assert scope.get("lax") == ModuleType(path="jax.lax", name="Module")


def test_symbol_table_analyzer_assignments() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  # Fake semantics to return Tensor for torch.randn
  semantics._key_origins = {"torch.randn": "neural"}
  semantics.data = {"torch.randn": {"return_type": "Tensor", "variants": {"torch": {"api": "torch.randn"}}}}
  semantics._reverse_index = {"torch.randn": ("torch.randn", semantics.data["torch.randn"])}

  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
import torch
x = torch.randn(10)
self.y = x
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)

  scope: Scope = analyzer.current_scope
  assert scope.get("x") == TensorType(framework="torch")

  # We can check that the table has self.y recorded
  # It's an Attribute, we need to find the node.
  assigns: list[cst.CSTNode] = [
    node
    for node in analyzer.table._node_types.keys()
    if isinstance(node, cst.Attribute) and getattr(node.attr, "value", "") == "y"
  ]
  assert len(assigns) == 1
  assert analyzer.table.get_type(assigns[0]) == TensorType(framework="torch")


def test_symbol_table_analyzer_scopes() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
x = 1
class MyClass:
    y = 2
    def my_func():
        z = 3
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)
  # The analyzer restores the scope after visiting.
  assert analyzer.current_scope.name == "global"


def test_symbol_table_analyzer_control_flow_if() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
import torch
import jax

if True:
    x = torch.randn()
else:
    x = jax.numpy.zeros()
"""
  # Need to mock the semantics to recognize torch.randn and jax.numpy.zeros
  semantics.data = {"torch.randn": {"return_type": "Tensor"}, "jax.numpy.zeros": {"return_type": "Tensor"}}
  semantics._reverse_index = {
    "torch.randn": ("torch.randn", semantics.data["torch.randn"]),
    "jax.numpy.zeros": ("jax.numpy.zeros", semantics.data["jax.numpy.zeros"]),
  }
  semantics._key_origins = {"torch.randn": "neural", "jax.numpy.zeros": "array"}

  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)

  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)
  assert len(sym.types) == 2
  assert any(getattr(t, "framework", "") == "torch" for t in sym.types)
  assert any(getattr(t, "framework", "") == "jax" for t in sym.types)


def test_symbol_table_analyzer_control_flow_loops() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
import torch
for i in range(10):
    x = torch.randn()

while True:
    y = torch.randn()
"""
  semantics.data = {"torch.randn": {"return_type": "Tensor"}}
  semantics._reverse_index = {"torch.randn": ("torch.randn", semantics.data["torch.randn"])}
  semantics._key_origins = {"torch.randn": "neural"}

  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)

  # After loops, variables defined inside might exist.
  # Start state didn't have x, end state has x = Tensor.
  # Merge should keep Tensor (optimistic).
  assert analyzer.current_scope.get("x") == TensorType(framework="torch")
  assert analyzer.current_scope.get("y") == TensorType(framework="torch")


def test_symbol_table_analyzer_ifexp() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
import torch
import jax
x = torch.randn() if True else jax.numpy.zeros()
"""
  semantics.data = {"torch.randn": {"return_type": "Tensor"}, "jax.numpy.zeros": {"return_type": "Tensor"}}
  semantics._reverse_index = {
    "torch.randn": ("torch.randn", semantics.data["torch.randn"]),
    "jax.numpy.zeros": ("jax.numpy.zeros", semantics.data["jax.numpy.zeros"]),
  }
  semantics._key_origins = {"torch.randn": "neural", "jax.numpy.zeros": "array"}

  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)

  # Find the IfExp node in the table
  ifexp_nodes: list[cst.CSTNode] = [node for node in analyzer.table._node_types.keys() if isinstance(node, cst.IfExp)]
  assert len(ifexp_nodes) == 1
  sym: SymbolType | None = analyzer.table.get_type(ifexp_nodes[0])
  assert isinstance(sym, UnionType)


def test_symbol_table_analyzer_call_methods() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
import torch
x = torch.randn()
y = x.view()
"""
  semantics.data = {"torch.randn": {"return_type": "Tensor"}, "torch.Tensor.view": {"return_type": "Tensor"}}
  semantics._reverse_index = {
    "torch.randn": ("torch.randn", semantics.data["torch.randn"]),
    "torch.Tensor.view": ("torch.Tensor.view", semantics.data["torch.Tensor.view"]),
  }
  semantics._key_origins = {"torch.randn": "neural", "torch.Tensor.view": "array"}

  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)

  assert analyzer.current_scope.get("y") == TensorType(framework="torch")


def test_make_union() -> None:
  """Docstring."""
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  t1: TensorType = TensorType(framework="torch")
  t2: TensorType = TensorType(framework="torch")

  # Identical
  res: SymbolType = analyzer._make_union(t1, t2)
  assert res == t1

  # Different
  t3: TensorType = TensorType(framework="jax")
  res2: SymbolType = analyzer._make_union(t1, t3)
  assert isinstance(res2, UnionType)
  assert len(res2.types) == 2

  # Nested Union
  res3: SymbolType = analyzer._make_union(res2, t3)
  assert isinstance(res3, UnionType)
  assert len(res3.types) == 2  # Deduplicated!


def test_symbol_table_analyzer_missing_branches() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  # Test For/While with orelse
  code: str = """
for i in range(1):
    x = 1
else:
    y = 2

while False:
    z = 3
else:
    w = 4

# Test IfExp with missing parts (impossible in valid Python, but we mock the types)
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)


def test_symbol_table_analyzer_ifexp_partials() -> None:
  """Docstring."""
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())

  ifexp: cst.IfExp = cst.IfExp(body=cst.Name("a"), test=cst.Name("b"), orelse=cst.Name("c"))

  t: TensorType = TensorType(framework="torch")

  # only t1
  analyzer.table.record_type(ifexp.body, t)
  analyzer.leave_IfExp(ifexp)
  assert analyzer.table.get_type(ifexp) == t

  # clean table
  analyzer.table = SymbolTable()
  # only t2
  analyzer.table.record_type(ifexp.orelse, t)
  analyzer.leave_IfExp(ifexp)
  assert analyzer.table.get_type(ifexp) == t


def test_symbol_table_analyzer_merge_states_missing_b() -> None:
  """Docstring."""
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  state_a: dict[str, SymbolType] = {"x": TensorType(framework="torch")}
  state_b: dict[str, SymbolType] = {}
  res: dict[str, SymbolType] = analyzer._merge_states(state_a, state_b)
  assert "x" in res


def test_symbol_table_analyzer_importfrom_edge() -> None:
  """Docstring."""
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  code: str = "from . import something"
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)
  # no module


def test_symbol_table_analyzer_union_call() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  semantics.data = {"torch.Tensor.view": {"return_type": "Tensor"}}
  semantics._reverse_index = {"torch.Tensor.view": ("torch.Tensor.view", semantics.data["torch.Tensor.view"])}
  semantics._key_origins = {"torch.Tensor.view": "array"}

  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = "x.view()"
  tree: cst.Module = cst.parse_module(code)

  # Manually inject a UnionType for x
  expr_stmt = tree.body[0]
  if isinstance(expr_stmt, cst.SimpleStatementLine):
    expr = expr_stmt.body[0]
    if isinstance(expr, cst.Expr):
      call_val = expr.value
      if isinstance(call_val, cst.Call):
        func = call_val.func
        if isinstance(func, cst.Attribute):
          x_node: cst.BaseExpression = func.value
          u: UnionType = UnionType(types=[TensorType(framework="torch"), ModuleType(path="sys")])
          analyzer.table.record_type(x_node, u)

  tree.visit(analyzer)

  if isinstance(tree.body[0], cst.SimpleStatementLine):
    expr = tree.body[0].body[0]
    if isinstance(expr, cst.Expr):
      call_node: cst.BaseExpression = expr.value
      assert analyzer.table.get_type(call_node) == TensorType(framework="torch")


def test_symbol_table_analyzer_loose_lookup() -> None:
  """Docstring."""
  semantics: SemanticsManager = SemanticsManager()
  # Provide a definition for 'view' but not 'torch.Tensor.view'
  semantics.data = {"view": {"return_type": "Tensor"}}
  semantics._reverse_index = {"view": ("view", semantics.data["view"])}
  semantics._key_origins = {"view": "array"}

  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = "x.view()"
  tree: cst.Module = cst.parse_module(code)

  # Inject TensorType for x
  expr_stmt = tree.body[0]
  if isinstance(expr_stmt, cst.SimpleStatementLine):
    expr = expr_stmt.body[0]
    if isinstance(expr, cst.Expr):
      call_val = expr.value
      if isinstance(call_val, cst.Call):
        func = call_val.func
        if isinstance(func, cst.Attribute):
          x_node: cst.BaseExpression = func.value
          analyzer.table.record_type(x_node, TensorType(framework="torch"))

  tree.visit(analyzer)

  if isinstance(tree.body[0], cst.SimpleStatementLine):
    expr = tree.body[0].body[0]
    if isinstance(expr, cst.Expr):
      call_node: cst.BaseExpression = expr.value
      assert analyzer.table.get_type(call_node) == TensorType(framework="torch")


def test_make_union_len_one() -> None:
  """Docstring."""
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  t1: TensorType = TensorType(framework="torch")
  u2: UnionType = UnionType([t1])
  # The union of t1 and u2 should return t1
  res: SymbolType = analyzer._make_union(t1, u2)
  assert isinstance(res, TensorType)


# --- Merged from test_symbol_table_extra.py ---


def analyze(code: str) -> SymbolTableAnalyzer:
  """Analyze code for symbol table."""
  tree: cst.Module = cst.parse_module(code)
  sm: MagicMock = MagicMock()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(sm)
  tree.visit(analyzer)
  return analyzer


def test_missing_symbol_table_coverage() -> None:
  """Docstring."""
  # Test try/except blocks
  code: str = """
try:
    x = 1
except Exception as e:
    x = 2
finally:
    y = 3
    """
  analyze(code)

  # Test boolean ops
  code_bool: str = """
x = True and False or True
    """
  analyze(code_bool)

  # Test unary ops
  code_unary: str = """
x = not True
y = -1
    """
  analyze(code_unary)

  # Test with/async with
  code_with: str = """
with open('file.txt') as f:
    x = 1
    """
  analyze(code_with)


def test_global_scope_access() -> None:
  """Docstring."""
  code: str = """
global_var = 1
def func():
    return global_var
    """
  analyze(code)


def test_class_def_nested() -> None:
  """Docstring."""
  code: str = """
class Outer:
    class Inner:
        def __init__(self):
            self.x = 1
    """
  analyze(code)


def test_lambda() -> None:
  """Docstring."""
  code: str = """
f = lambda x: x + 1
    """
  analyze(code)


def test_list_comp() -> None:
  """Docstring."""
  code: str = """
l = [x for x in range(10)]
    """
  analyze(code)
