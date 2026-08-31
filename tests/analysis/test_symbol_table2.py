"""Test suite for the Symbol Table module covering complex control flow and type union edge cases.

This module provides exhaustive testing for `SymbolTableAnalyzer`, particularly focusing
on variable type resolution across loops (for/while with else clauses), inline if-expressions,
nested unions, and complex module import scenarios. It complements `test_symbol_table.py`
by targeting specific, deeply nested branch logic.
"""

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.analysis.symbol_table import SymbolTable, SymbolTableAnalyzer
from ml_switcheroo.analysis.symbol_types import ModuleType, SymbolType, TensorType, UnionType


@pytest.fixture
def analyzer() -> SymbolTableAnalyzer:
  """Fixture providing a mocked SymbolTableAnalyzer.

  Mocks the `SemanticsManager` to resolve `randn` and `view` methods to returning
  a `TensorType` for testing type propagation without loading full framework rules.

  Returns:
      SymbolTableAnalyzer: The configured analyzer ready for AST traversal.
  """
  sem: MagicMock = MagicMock()

  from typing import Dict, Optional, Tuple

  def get_def(name: str) -> Optional[Tuple[str, Dict]]:
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
  res_analyzer.source_fw = "torch"
  return res_analyzer


def test_for_else(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution across a `for...else` construct.

  Verifies that if a variable is assigned a type within the `else` block of a `for` loop,
  the resulting scope type correctly unions the prior state with the new assignment.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nimport torch\nx = torch.nn\nfor i in range(10):\n    pass\nelse:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution for variables modified inside a `while` loop.

  Verifies that variables assigned a Tensor type inside a while loop merge correctly
  with their pre-loop state, resulting in a UnionType if the states differ.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nimport torch\nx = torch.nn\nwhile True:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop_else(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution across a `while...else` construct.

  Verifies that variables assigned inside the `else` block of a while loop correctly
  contribute to the final merged scope state.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nimport torch\nx = torch.nn\nwhile True:\n    pass\nelse:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_ifexp_partial(analyzer: SymbolTableAnalyzer) -> None:
  """Test type resolution for inline ternary `if` expressions with partial types.

  Verifies that if only one branch of an inline `if` expression yields a known
  tensor type (and the other is untyped/unknown), the analyzer safely infers the
  known type for the entire expression.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = (
    "\nimport torch\nx = torch.randn(1) if True else untyped_func()\ny = untyped_func() if True else torch.randn(1)\n"
  )
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("x"), TensorType)
  assert isinstance(analyzer.current_scope.get("y"), TensorType)


def test_merge_states_b_only(analyzer: SymbolTableAnalyzer) -> None:
  """Test state merging when a variable is only defined in one branch of an `if` block.

  Verifies that a variable assigned exclusively in an `else` branch correctly
  survives the post-block state merge.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nimport torch\nif True:\n    pass\nelse:\n    z = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("z")
  assert isinstance(sym, TensorType)


def test_make_union_same(analyzer: SymbolTableAnalyzer) -> None:
  """Test union creation with identical base types.

  Verifies that unioning two instances of the identical `TensorType` collapses
  down safely into a single `TensorType` rather than creating a redundant Union.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  t1: TensorType = TensorType("Tensor", "torch")
  res: SymbolType = analyzer._make_union(t1, t1)
  assert res == t1


def test_make_union_nested(analyzer: SymbolTableAnalyzer) -> None:
  """Test deduplication when unioning a base type with an existing UnionType.

  Verifies that if a UnionType already contains a specific `TensorType`, unioning
  it again with that same `TensorType` does not increase the size of the union.

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

  Verifies that if deduplication reduces a UnionType down to a single element,
  it collapses completely back into that base element type.

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

  Verifies that `from x import y as z` syntax correctly registers `z` as a
  ModuleType, while local relative imports (`from . import`) are safely ignored.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nfrom torch import nn, optim as opt\nfrom . import local_module\n"
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("nn"), ModuleType)
  assert isinstance(analyzer.current_scope.get("opt"), ModuleType)
  assert analyzer.current_scope.get("local_module") is None


def test_assign_untyped(analyzer: SymbolTableAnalyzer) -> None:
  """Test assignment of untyped function calls.

  Verifies that calls to unknown functions do not register phantom variables
  in the symbol table scope.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nuntyped_var = untyped_func()\n"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("untyped_var") is None


def test_assign_attribute(analyzer: SymbolTableAnalyzer) -> None:
  """Test symbol tracking for object attributes.

  Verifies that assigning a resolved Tensor type to `self.x` correctly records
  the type directly against the CST Attribute node in the SymbolTable.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nimport torch\nclass A:\n    def __init__(self):\n        self.x = torch.randn(1)\n"
  tree: cst.Module = analyze(code, analyzer)

  class AttrVisitor(cst.CSTVisitor):
    """Helper visitor to locate attribute assignment targets in the AST."""

    def __init__(self) -> None:
      """Initializes the AttrVisitor instance."""
      self.nodes: list[cst.Attribute] = []

    def visit_Attribute(self, node: cst.Attribute) -> None:
      """Store references to the 'x' attribute."""
      if getattr(node.attr, "value", "") == "x":
        self.nodes.append(node)

  v: AttrVisitor = AttrVisitor()
  tree.visit(v)
  assert len(v.nodes) > 0
  assert isinstance(analyzer.table.get_type(v.nodes[0]), TensorType)


def test_call_on_tensor(analyzer: SymbolTableAnalyzer) -> None:
  """Test method type resolution on identified variables.

  Verifies that invoking `.view()` on a previously resolved Tensor correctly
  propagates the Tensor type to the newly assigned variable.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "\nimport torch\nx = torch.randn(1)\ny = x.view()\n"
  analyzer.semantics.get_definition.side_effect = lambda n: (
    ("op", {"return_type": "Tensor"}) if "view" in n or "randn" in n else None
  )
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ifexp_both_unknown(analyzer: SymbolTableAnalyzer) -> None:
  """Test inline if-expression fallback on entirely unknown branches.

  Verifies that if neither branch of a ternary operation can be resolved,
  the result correctly remains untyped in the scope.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "x = unknown() if True else unknown2()"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_assign_subscript(analyzer: SymbolTableAnalyzer) -> None:
  """Test safety when assigning directly to list/dict subscripts.

  Verifies that the analyzer does not crash when parsing `my_list[0] = Tensor`
  syntax. Subscript targeting is currently not fully tracked, but must not fault.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "import torch\nmy_list[0] = torch.randn(1)"
  analyze(code, analyzer)


def test_call_non_tensor_return(analyzer: SymbolTableAnalyzer) -> None:
  """Test that operations returning scalar/non-tensor data are not recorded.

  Verifies that a known method marked as returning `int` does not pollute the
  tensor-focused symbol table.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  analyzer.semantics.get_key_origins.return_value = {}
  code: str = "import torch\nx = torch.get_int()"
  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "int"}) if "get_int" in n else None
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_import_star(analyzer: SymbolTableAnalyzer) -> None:
  """Test safety when encountering `from module import *` syntax.

  Verifies that wildcard imports are safely bypassed without causing analysis faults.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  code: str = "from torch import *"
  analyze(code, analyzer)


def test_union_no_tensor(analyzer: SymbolTableAnalyzer) -> None:
  """Test method calls on UnionTypes that do not contain a resolving TensorType.

  Verifies that attempting to invoke a tensor method (like `.view()`) on a Union
  that only contains Module types correctly results in an unresolved return type.

  Args:
      analyzer (SymbolTableAnalyzer): The mocked analyzer fixture.
  """
  x_node: cst.BaseExpression = cst.parse_expression("x")
  u_type: UnionType = UnionType([ModuleType("Module", "torch.nn")])
  analyzer.table.record_type(x_node, u_type)
  call_node: cst.Call = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))
  analyzer.leave_Call(call_node)
  assert analyzer.table.get_type(call_node) is None


def analyze(code: str, analyzer: SymbolTableAnalyzer) -> cst.Module:
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
