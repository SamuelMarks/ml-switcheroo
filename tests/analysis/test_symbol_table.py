"""Test module for the SymbolTable and SymbolTableAnalyzer components.

This module validates the correctness of the symbol table logic, ensuring that
variables, imports, assignments, and control flow blocks (like if/else and loops)
are accurately tracked and typed within a Control Flow Graph context. It also verifies
the resolution of types for neural network and tensor operations using the
`SemanticsManager`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.analysis.symbol_table import SymbolTable, SymbolTableAnalyzer
from ml_switcheroo.analysis.symbol_types import ModuleType, Scope, SymbolType, TensorType, UnionType
from ml_switcheroo.semantics.manager import SemanticsManager


def test_symbol_table_basic() -> None:
  """Test basic reading and writing to the SymbolTable structure.

  Verifies that we can assign a specific `TensorType` to a CST Node and
  retrieve it successfully, and that querying an unrecorded node returns None.
  """
  table: SymbolTable = SymbolTable()
  node: cst.Name = cst.Name("test")
  sym: TensorType = TensorType(framework="torch")
  table.record_type(node, sym)
  assert table.get_type(node) == sym
  assert table.get_type(cst.Name("other")) is None


def test_symbol_table_analyzer_imports() -> None:
  """Test that import statements are correctly recorded as ModuleTypes in the scope.

  Verifies that standard imports (e.g. `import torch`) and aliased imports
  (e.g. `import torch.nn as nn` or `from jax import numpy as jnp`) correctly populate
  the symbol table's current scope with fully-qualified module paths.
  """
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
  """Test that standard assignments are typed correctly.

  Verifies that the analyzer resolves function calls like `torch.randn` via the
  SemanticsManager, and records the resulting `TensorType` against the assigned
  variables (e.g., `x` and `self.y`).
  """
  semantics: SemanticsManager = SemanticsManager()
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

  assigns: list[cst.CSTNode] = [
    node
    for node in analyzer.table._node_types.keys()
    if isinstance(node, cst.Attribute) and getattr(node.attr, "value", "") == "y"
  ]
  assert len(assigns) == 1
  assert analyzer.table.get_type(assigns[0]) == TensorType(framework="torch")


def test_symbol_table_analyzer_scopes() -> None:
  """Test that the analyzer correctly manages block scopes (e.g. classes and functions).

  Verifies that entering a class or function definition pushes a new scope,
  and that the analyzer restores the original `global` scope upon leaving the block.
  """
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
  assert analyzer.current_scope.name == "global"


def test_symbol_table_analyzer_control_flow_if() -> None:
  """Test variable type merging at the end of an if/else block.

  Verifies that if a variable is assigned a PyTorch tensor in the `if` branch
  and a JAX array in the `else` branch, the resulting scope type is correctly
  recorded as a `UnionType` representing both frameworks.
  """
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
  """Test variable type tracking across `for` and `while` loop iterations.

  Verifies that the analyzer properly tracks types generated inside loops
  and bubbles them up to the surrounding scope safely.
  """
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

  assert analyzer.current_scope.get("x") == TensorType(framework="torch")
  assert analyzer.current_scope.get("y") == TensorType(framework="torch")


def test_symbol_table_analyzer_ifexp() -> None:
  """Test type unioning in inline if-expressions (ternary operators).

  Verifies that the type of `a if cond else b` correctly yields a `UnionType`
  if the branches resolve to different tensor types.
  """
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

  ifexp_nodes: list[cst.CSTNode] = [node for node in analyzer.table._node_types.keys() if isinstance(node, cst.IfExp)]
  assert len(ifexp_nodes) == 1
  sym: SymbolType | None = analyzer.table.get_type(ifexp_nodes[0])
  assert isinstance(sym, UnionType)


def test_symbol_table_analyzer_call_methods() -> None:
  """Test method type resolution on identified variables.

  Verifies that if `x` is recognized as a PyTorch Tensor, a method call like `x.view()`
  is correctly resolved to `torch.Tensor.view` and properly typed in the resulting assignment.
  """
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
  """Test the `_make_union` utility function behavior.

  Verifies that unioning identical types collapses to a single type,
  unioning different types creates a proper `UnionType`, and unioning
  existing `UnionType`s flattens and deduplicates the resulting types.
  """
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  t1: TensorType = TensorType(framework="torch")
  t2: TensorType = TensorType(framework="torch")

  res: SymbolType = analyzer._make_union(t1, t2)
  assert res == t1

  t3: TensorType = TensorType(framework="jax")
  res2: SymbolType = analyzer._make_union(t1, t3)
  assert isinstance(res2, UnionType)
  assert len(res2.types) == 2

  res3: SymbolType = analyzer._make_union(res2, t3)
  assert isinstance(res3, UnionType)
  assert len(res3.types) == 2


def test_symbol_table_analyzer_missing_branches() -> None:
  """Test symbol analysis over various complex control flows.

  Ensures the analyzer safely traverses `for...else` and `while...else`
  blocks without halting or raising AST evaluation errors.
  """
  semantics: SemanticsManager = SemanticsManager()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = """
for i in range(1):
    x = 1
else:
    y = 2

while False:
    z = 3
else:
    w = 4
"""
  tree: cst.Module = cst.parse_module(code)
  _res = tree.visit(analyzer)
  assert isinstance(_res, cst.Module)
  assert analyzer.current_scope.name == "global"


def test_symbol_table_analyzer_ifexp_partials() -> None:
  """Test edge-cases for IfExp type recording.

  Verifies that if only the body or only the `orelse` branch of a ternary
  operator has a recordable type, the analyzer safely defaults the entire
  expression to that single resolved type instead of crashing.
  """
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())

  ifexp: cst.IfExp = cst.IfExp(body=cst.Name("a"), test=cst.Name("b"), orelse=cst.Name("c"))
  t: TensorType = TensorType(framework="torch")

  analyzer.table.record_type(ifexp.body, t)
  analyzer.leave_IfExp(ifexp)
  assert analyzer.table.get_type(ifexp) == t

  analyzer.table = SymbolTable()
  analyzer.table.record_type(ifexp.orelse, t)
  analyzer.leave_IfExp(ifexp)
  assert analyzer.table.get_type(ifexp) == t


def test_symbol_table_analyzer_merge_states_missing_b() -> None:
  """Test scope state merging when a variable only exists in one branch.

  Verifies that if variable `x` exists in state `a` but not state `b` (e.g. defined in
  an `if` block but not in `else`), the resulting merged scope optimistically retains `x`.
  """
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  state_a: dict[str, SymbolType] = {"x": TensorType(framework="torch")}
  state_b: dict[str, SymbolType] = {}
  res: dict[str, SymbolType] = analyzer._merge_states(state_a, state_b)
  assert "x" in res


def test_symbol_table_analyzer_importfrom_edge() -> None:
  """Test analyzer safety when handling blank relative imports.

  Verifies that statements like `from . import something` do not cause
  module resolution errors or crashes within the analyzer.
  """
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  code: str = "from . import something"
  tree: cst.Module = cst.parse_module(code)
  _res = tree.visit(analyzer)
  assert isinstance(_res, cst.Module)
  assert analyzer.current_scope.symbols == {}


def test_symbol_table_analyzer_union_call() -> None:
  """Test method resolution when the caller variable is a UnionType.

  Verifies that if a variable `x` is determined to be a Union containing a PyTorch
  Tensor, calling `x.view()` successfully resolves the PyTorch branch of the union
  and types the return value appropriately.
  """
  semantics: SemanticsManager = SemanticsManager()
  semantics.data = {"torch.Tensor.view": {"return_type": "Tensor"}}
  semantics._reverse_index = {"torch.Tensor.view": ("torch.Tensor.view", semantics.data["torch.Tensor.view"])}
  semantics._key_origins = {"torch.Tensor.view": "array"}

  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = "x.view()"
  tree: cst.Module = cst.parse_module(code)

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
  """Test loose fallback resolution for methods in the SemanticManager.

  Verifies that if a fully qualified method name like `torch.Tensor.view` is not found,
  the analyzer can successfully fall back to looking up the base operation name `view`.
  """
  semantics: SemanticsManager = SemanticsManager()
  semantics.data = {"view": {"return_type": "Tensor"}}
  semantics._reverse_index = {"view": ("view", semantics.data["view"])}
  semantics._key_origins = {"view": "array"}

  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(semantics)

  code: str = "x.view()"
  tree: cst.Module = cst.parse_module(code)

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
  """Test _make_union deduplication edge cases.

  Verifies that attempting to union a base `TensorType` with a `UnionType`
  that only contains that same `TensorType` correctly collapses down to
  just the base `TensorType`.
  """
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  t1: TensorType = TensorType(framework="torch")
  u2: UnionType = UnionType([t1])
  res: SymbolType = analyzer._make_union(t1, u2)
  assert isinstance(res, TensorType)


def analyze(code: str) -> SymbolTableAnalyzer:
  """Helper to run the SymbolTableAnalyzer on a snippet of code.

  Args:
      code (str): Source code to parse and analyze.

  Returns:
      SymbolTableAnalyzer: The populated analyzer instance.
  """
  tree: cst.Module = cst.parse_module(code)
  sm: MagicMock = MagicMock()
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(sm)
  tree.visit(analyzer)
  return analyzer


def test_missing_symbol_table_coverage() -> None:
  """Test parsing of additional control flow blocks.

  Ensures the analyzer safely traverses `try/except/finally`, complex boolean logic,
  unary operators, and `with` blocks without raising errors.
  """
  code: str = """
try:
    x = 1
except Exception as e:
    x = 2
finally:
    y = 3
    """
  analyze(code)

  code_bool: str = """
x = True and False or True
    """
  analyze(code_bool)

  code_unary: str = """
x = not True
y = -1
    """
  analyze(code_unary)

  code_with: str = """
with open('file.txt') as f:
    x = 1
    """
  _res = analyze(code_with)
  assert isinstance(_res, SymbolTableAnalyzer)
  assert _res.current_scope.name == "global"


def test_global_scope_access() -> None:
  """Test scope fallthrough for undefined local variables.

  Verifies that if a function scope does not contain a variable (`global_var`),
  the analyzer accurately traverses up the scope stack to find it in the parent scopes.
  """
  code: str = """
global_var = 1
def func():
    return global_var
    """
  _res = analyze(code)
  assert isinstance(_res, SymbolTableAnalyzer)
  assert _res.current_scope.name == "global"


def test_class_def_nested() -> None:
  """Test scope management for nested classes.

  Verifies that the analyzer can accurately push and pop nested class definition
  scopes without mixing attribute states.
  """
  code: str = """
class Outer:
    class Inner:
        def __init__(self):
            self.x = 1
    """
  _res = analyze(code)
  assert isinstance(_res, SymbolTableAnalyzer)
  assert _res.current_scope.name == "global"


def test_lambda() -> None:
  """Test traversal safety for lambda functions.

  Verifies that lambda nodes are safely visited by the AST analyzer.
  """
  code: str = """
f = lambda x: x + 1
    """
  _res = analyze(code)
  assert isinstance(_res, SymbolTableAnalyzer)
  assert _res.current_scope.name == "global"


def test_list_comp() -> None:
  """Test traversal safety for list comprehensions.

  Verifies that list comprehensions are safely visited by the AST analyzer.
  """
  code: str = """
l = [x for x in range(10)]
    """
  _res = analyze(code)
  assert isinstance(_res, SymbolTableAnalyzer)
  assert _res.current_scope.name == "global"


def test_symbol_table_missing_branches() -> None:
  """Test specific missing edge-case branch coverages in SymbolTableAnalyzer.

  Verifies the handling of `If` blocks lacking `Else` branches and function calls
  on variables that do not carry Tensor or Union type data (e.g. built-in floats).
  """
  code = """
import torch
if True:
    x = torch.randn(1)
"""
  tree = cst.parse_module(code)
  semantics = SemanticsManager()
  builder = SymbolTableAnalyzer(semantics)
  tree.visit(builder)

  code2 = """
import math
y = math.pi
y.conjugate()
"""
  tree2 = cst.parse_module(code2)
  builder2 = SymbolTableAnalyzer(SemanticsManager())
  _res = tree2.visit(builder2)
  assert isinstance(_res, cst.Module)
  assert "math" in builder2.current_scope.symbols


def test_symbol_table_412() -> None:
  """Test the non-Tensor call branch in detail.

  Manually injects an empty base `SymbolType` and verifies the analyzer handles
  calling a method (`x.conjugate()`) on it without throwing an attribute error.
  """
  analyzer = SymbolTableAnalyzer(SemanticsManager())
  analyzer.current_scope.symbols["x"] = SymbolType()
  node = cst.ensure_type(cst.parse_expression("x.conjugate()"), cst.Call)
  analyzer.visit_Call(node)
  assert analyzer.table.get_type(node) is None


@pytest.fixture
def analyzer() -> SymbolTableAnalyzer:
  """Fixture providing a mocked SymbolTableAnalyzer.

  Mocks the `SemanticsManager` to resolve `randn` and `view` methods to returning
  a `TensorType` for testing type propagation without loading full framework rules.

  Returns:
      SymbolTableAnalyzer: The configured analyzer ready for AST traversal.
  """
  sem: MagicMock = MagicMock()

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock side effect to resolve specific API signatures.

    Args:
        name (str): The fully qualified API name to look up.

    Returns:
        Optional[Tuple[str, Dict]]: The mocked return type definition if recognized.
    """
    if "randn" in name or "view" in name:
      return ("op", {"return_type": "Tensor"})
    return None

  sem.get_definition.side_effect = get_def
  res_analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(sem)
  res_analyzer.table = SymbolTable()
  return res_analyzer


def _analyze_with(code: str, analyzer: SymbolTableAnalyzer) -> cst.Module:
  """Helper function to parse code and execute the analyzer over it.

  Args:
      code (str): The source code to parse.
      analyzer (SymbolTableAnalyzer): The configured analyzer instance to execute.

  Returns:
      cst.Module: The parsed CST tree.
  """
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)
  return tree


def test_for_else(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution across a `for...else` construct.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
x = torch.nn
for i in range(10):
    pass
else:
    x = torch.randn(1)
"""
  _analyze_with(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution for variables modified inside a `while` loop.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
x = torch.nn
while True:
    x = torch.randn(1)
"""
  _analyze_with(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop_else(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution across a `while...else` construct.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
x = torch.nn
while True:
    pass
else:
    x = torch.randn(1)
"""
  _analyze_with(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_ifexp_partial(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution for inline ternary `if` expressions with partial types.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
x = torch.randn(1) if True else untyped_func()
y = untyped_func() if True else torch.randn(1)
"""
  _analyze_with(code, analyzer)
  assert isinstance(analyzer.current_scope.get("x"), TensorType)
  assert isinstance(analyzer.current_scope.get("y"), TensorType)


def test_merge_states_b_only(analyzer: SymbolTableAnalyzer) -> None:
  """Test state merging when a variable is only defined in one branch of an `if` block.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
if True:
    pass
else:
    z = torch.randn(1)
"""
  _analyze_with(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("z")
  assert isinstance(sym, TensorType)


def test_make_union_same(analyzer: SymbolTableAnalyzer) -> None:
  """Test union creation with identical base types.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  t1: TensorType = TensorType("Tensor", "torch")
  res: SymbolType = analyzer._make_union(t1, t1)
  assert res == t1


def test_make_union_nested(analyzer: SymbolTableAnalyzer) -> None:
  """Test deduplication when unioning a base type with an existing UnionType.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  t1: TensorType = TensorType("Tensor", "torch")
  m1: ModuleType = ModuleType("Module", "torch.nn")
  u1: UnionType = UnionType([t1, m1])
  res: SymbolType = analyzer._make_union(u1, t1)
  assert isinstance(res, UnionType)
  assert len(res.types) == 2


def test_make_union_dedup_single(analyzer: SymbolTableAnalyzer) -> None:
  """Test that union deduplication correctly unwraps single-item unions.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  t1: TensorType = TensorType("Tensor", "torch")
  t2: TensorType = TensorType("Tensor", "torch")
  u1: UnionType = UnionType([t1])
  res: SymbolType = analyzer._make_union(u1, t2)
  assert isinstance(res, TensorType)


def test_import_from(analyzer: SymbolTableAnalyzer) -> None:
  """Test parsing of specific module members and local aliasing.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
from torch import nn, optim as opt
from . import local_module
"""
  _analyze_with(code, analyzer)
  assert isinstance(analyzer.current_scope.get("nn"), ModuleType)
  assert isinstance(analyzer.current_scope.get("opt"), ModuleType)
  assert analyzer.current_scope.get("local_module") is None


def test_assign_untyped(analyzer: SymbolTableAnalyzer) -> None:
  """Test assignment of untyped function calls.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
untyped_var = untyped_func()
"""
  _analyze_with(code, analyzer)
  assert analyzer.current_scope.get("untyped_var") is None


def test_assign_attribute(analyzer: SymbolTableAnalyzer) -> None:
  """Test symbol tracking for object attributes.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
class A:
    def __init__(self):
        self.x = torch.randn(1)
"""
  tree: cst.Module = _analyze_with(code, analyzer)

  class AttrVisitor(cst.CSTVisitor):
    """Helper visitor to locate attribute assignment targets in the AST."""

    def __init__(self) -> None:
      """Initializes the AttrVisitor instance."""
      self.nodes: list[cst.Attribute] = []

    def visit_Attribute(self, node: cst.Attribute) -> None:
      """Store references to the 'x' attribute.

      Args:
          node: CST or AST node under inspection or transformation.
      """
      if getattr(node.attr, "value", "") == "x":
        self.nodes.append(node)

  v: AttrVisitor = AttrVisitor()
  tree.visit(v)
  assert len(v.nodes) > 0
  assert isinstance(analyzer.table.get_type(v.nodes[0]), TensorType)


def test_call_on_tensor(analyzer: SymbolTableAnalyzer) -> None:
  """Test method type resolution on identified variables.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """
import torch
x = torch.randn(1)
y = x.view()
"""
  assert isinstance(analyzer.semantics, MagicMock)
  analyzer.semantics.get_definition.side_effect = lambda n: (
    ("op", {"return_type": "Tensor"}) if "view" in n or "randn" in n else None
  )
  _analyze_with(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ifexp_both_unknown(analyzer: SymbolTableAnalyzer) -> None:
  """Test inline if-expression fallback on entirely unknown branches.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "x = unknown() if True else unknown2()"
  _analyze_with(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_assign_subscript(analyzer: SymbolTableAnalyzer) -> None:
  """Test safety when assigning directly to list/dict subscripts.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = """import torch
my_list[0] = torch.randn(1)"""
  _res = _analyze_with(code, analyzer)
  assert isinstance(_res, cst.Module)
  assert "my_list" not in analyzer.current_scope.symbols


def test_call_non_tensor_return(analyzer: SymbolTableAnalyzer) -> None:
  """Test that operations returning scalar/non-tensor data are not recorded.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  assert isinstance(analyzer.semantics, MagicMock)
  analyzer.semantics.get_key_origins.return_value = {}
  code: str = """import torch
x = torch.get_int()"""
  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "int"}) if "get_int" in n else None
  _analyze_with(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_import_star(analyzer: SymbolTableAnalyzer) -> None:
  """Test safety when encountering `from module import *` syntax.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "from torch import *"
  _res = _analyze_with(code, analyzer)
  assert isinstance(_res, cst.Module)
  assert analyzer.current_scope.symbols == {}


def test_union_no_tensor(analyzer: SymbolTableAnalyzer) -> None:
  """Test method calls on UnionTypes that do not contain a resolving TensorType.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  x_node: cst.BaseExpression = cst.parse_expression("x")
  u_type: UnionType = UnionType([ModuleType("Module", "torch.nn")])
  analyzer.table.record_type(x_node, u_type)
  call_node: cst.Call = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))
  analyzer.leave_Call(call_node)
  assert analyzer.table.get_type(call_node) is None


def test_call_on_non_tensor_non_union(analyzer: SymbolTableAnalyzer) -> None:
  """Test method calls on types that are neither TensorType nor UnionType.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  x_node: cst.BaseExpression = cst.parse_expression("x")
  m_type: ModuleType = ModuleType(path="sys")
  analyzer.table.record_type(x_node, m_type)
  call_node: cst.Call = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("some_method")))
  analyzer.leave_Call(call_node)
  assert analyzer.table.get_type(call_node) is None
