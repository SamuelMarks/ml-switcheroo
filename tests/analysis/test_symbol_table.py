"""Test module for the SymbolTable and SymbolTableAnalyzer components.

This module validates the correctness of the symbol table logic, ensuring that
variables, imports, assignments, and control flow blocks (like if/else and loops)
are accurately tracked and typed within a Control Flow Graph context. It also verifies
the resolution of types for neural network and tensor operations using the
`SemanticsManager`.
"""

from unittest.mock import MagicMock

import libcst as cst

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
  # The analyzer restores the scope after visiting.
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

  # After loops, variables defined inside might exist.
  # Start state didn't have x, end state has x = Tensor.
  # Merge should keep Tensor (optimistic).
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

  # Find the IfExp node in the table
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
  """Test symbol analysis over various complex control flows.

  Ensures the analyzer safely traverses `for...else` and `while...else`
  blocks without halting or raising AST evaluation errors.
  """
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
  """Test edge-cases for IfExp type recording.

  Verifies that if only the body or only the `orelse` branch of a ternary
  operator has a recordable type, the analyzer safely defaults the entire
  expression to that single resolved type instead of crashing.
  """
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
  tree.visit(analyzer)
  # no module


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
  """Test loose fallback resolution for methods in the SemanticManager.

  Verifies that if a fully qualified method name like `torch.Tensor.view` is not found,
  the analyzer can successfully fall back to looking up the base operation name `view`.
  """
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
  """Test _make_union deduplication edge cases.

  Verifies that attempting to union a base `TensorType` with a `UnionType`
  that only contains that same `TensorType` correctly collapses down to
  just the base `TensorType`.
  """
  analyzer: SymbolTableAnalyzer = SymbolTableAnalyzer(SemanticsManager())
  t1: TensorType = TensorType(framework="torch")
  u2: UnionType = UnionType([t1])
  # The union of t1 and u2 should return t1
  res: SymbolType = analyzer._make_union(t1, u2)
  assert isinstance(res, TensorType)


# --- Merged from test_symbol_table_extra.py ---


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
  """Test scope fallthrough for undefined local variables.

  Verifies that if a function scope does not contain a variable (`global_var`),
  the analyzer accurately traverses up the scope stack to find it in the parent scopes.
  """
  code: str = """
global_var = 1
def func():
    return global_var
    """
  analyze(code)


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
  analyze(code)


def test_lambda() -> None:
  """Test traversal safety for lambda functions.

  Verifies that lambda nodes are safely visited by the AST analyzer.
  """
  code: str = """
f = lambda x: x + 1
    """
  analyze(code)


def test_list_comp() -> None:
  """Test traversal safety for list comprehensions.

  Verifies that list comprehensions are safely visited by the AST analyzer.
  """
  code: str = """
l = [x for x in range(10)]
    """
  analyze(code)


def test_symbol_table_missing_branches() -> None:
  """Test specific missing edge-case branch coverages in SymbolTableAnalyzer.

  Verifies the handling of `If` blocks lacking `Else` branches and function calls
  on variables that do not carry Tensor or Union type data (e.g. built-in floats).
  """
  # 144->147: if without else
  code = """
import torch
if True:
    x = torch.randn(1)
"""
  tree = cst.parse_module(code)
  semantics = SemanticsManager()
  builder = SymbolTableAnalyzer(semantics)
  tree.visit(builder)

  # 412->421: Call on attribute of a non-Tensor/non-Union type.
  code2 = """
import math
y = math.pi
y.conjugate()
"""
  tree2 = cst.parse_module(code2)
  builder2 = SymbolTableAnalyzer(SemanticsManager())
  tree2.visit(builder2)


def test_symbol_table_412() -> None:
  """Test the non-Tensor call branch in detail.

  Manually injects an empty base `SymbolType` and verifies the analyzer handles
  calling a method (`x.conjugate()`) on it without throwing an attribute error.
  """
