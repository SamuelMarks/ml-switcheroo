"""Test suite for the Symbol Table2 module."""

import libcst as cst
import pytest
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_types import TensorType, ModuleType, UnionType, SymbolType
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer, SymbolTable


@pytest.fixture
def analyzer() -> SymbolTableAnalyzer:
  """Provides a mock analyzer for testing."""
  sem: MagicMock = MagicMock()

  from typing import Optional, Tuple, Dict

  def get_def(name: str) -> Optional[Tuple[str, Dict]]:
    """Gets def.

    Args:
        name: ...
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
  """Verifies the behavior of for else.

  Args:
      analyzer: ...
  """
  code: str = "\nimport torch\nx = torch.nn\nfor i in range(10):\n    pass\nelse:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of while loop.

  Args:
      analyzer: ...
  """
  code: str = "\nimport torch\nx = torch.nn\nwhile True:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop_else(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of while loop else.

  Args:
      analyzer: ...
  """
  code: str = "\nimport torch\nx = torch.nn\nwhile True:\n    pass\nelse:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_ifexp_partial(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of ifexp partial.

  Args:
      analyzer: ...
  """
  code: str = (
    "\nimport torch\nx = torch.randn(1) if True else untyped_func()\ny = untyped_func() if True else torch.randn(1)\n"
  )
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("x"), TensorType)
  assert isinstance(analyzer.current_scope.get("y"), TensorType)


def test_merge_states_b_only(analyzer: SymbolTableAnalyzer) -> None:
  """Merges states b only.

  Args:
      analyzer: ...
  """
  code: str = "\nimport torch\nif True:\n    pass\nelse:\n    z = torch.randn(1)\n"
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("z")
  assert isinstance(sym, TensorType)


def test_make_union_same(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of make union same.

  Args:
      analyzer: ...
  """
  t1: TensorType = TensorType("Tensor", "torch")
  res: SymbolType = analyzer._make_union(t1, t1)
  assert res == t1


def test_make_union_nested(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of make union nested.

  Args:
      analyzer: ...
  """
  t1: TensorType = TensorType("Tensor", "torch")
  m1: ModuleType = ModuleType("Module", "torch.nn")
  u1: UnionType = UnionType([t1, m1])
  res: SymbolType = analyzer._make_union(u1, t1)
  assert isinstance(res, UnionType)
  assert len(res.types) == 2


def test_make_union_dedup_single(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of make union dedup single.

  Args:
      analyzer: ...
  """
  t1: TensorType = TensorType("Tensor", "torch")
  t2: TensorType = TensorType("Tensor", "torch")
  u1: UnionType = UnionType([t1])
  res: SymbolType = analyzer._make_union(u1, t2)
  assert isinstance(res, TensorType)


def test_import_from(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of import from.

  Args:
      analyzer: ...
  """
  code: str = "\nfrom torch import nn, optim as opt\nfrom . import local_module\n"
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("nn"), ModuleType)
  assert isinstance(analyzer.current_scope.get("opt"), ModuleType)
  assert analyzer.current_scope.get("local_module") is None


def test_assign_untyped(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of assign untyped.

  Args:
      analyzer: ...
  """
  code: str = "\nuntyped_var = untyped_func()\n"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("untyped_var") is None


def test_assign_attribute(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of assign attribute.

  Args:
      analyzer: ...
  """
  code: str = "\nimport torch\nclass A:\n    def __init__(self):\n        self.x = torch.randn(1)\n"
  tree: cst.Module = analyze(code, analyzer)

  class AttrVisitor(cst.CSTVisitor):
    """Test suite for the Attr Visitor component."""

    def __init__(self) -> None:
      """Initializes the AttrVisitor instance."""
      self.nodes: list[cst.Attribute] = []

    def visit_Attribute(self, node: cst.Attribute) -> None:
      """Helper to visit Attribute.

      Args:
          node: ...
      """
      if getattr(node.attr, "value", "") == "x":
        self.nodes.append(node)

  v: AttrVisitor = AttrVisitor()
  tree.visit(v)
  assert len(v.nodes) > 0
  assert isinstance(analyzer.table.get_type(v.nodes[0]), TensorType)


def test_call_on_tensor(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of call on tensor.

  Args:
      analyzer: ...
  """
  code: str = "\nimport torch\nx = torch.randn(1)\ny = x.view()\n"
  analyzer.semantics.get_definition.side_effect = lambda n: (
    ("op", {"return_type": "Tensor"}) if "view" in n or "randn" in n else None
  )
  analyze(code, analyzer)
  sym: SymbolType | None = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ifexp_both_unknown(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of ifexp both unknown.

  Args:
      analyzer: ...
  """
  code: str = "x = unknown() if True else unknown2()"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_assign_subscript(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of assign subscript.

  Args:
      analyzer: ...
  """
  code: str = "import torch\nmy_list[0] = torch.randn(1)"
  analyze(code, analyzer)


def test_call_non_tensor_return(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of call non tensor return.

  Args:
      analyzer: ...
  """
  analyzer.semantics.get_key_origins.return_value = {}
  code: str = "import torch\nx = torch.get_int()"
  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "int"}) if "get_int" in n else None
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_import_star(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of import star.

  Args:
      analyzer: ...
  """
  code: str = "from torch import *"
  analyze(code, analyzer)


def test_union_no_tensor(analyzer: SymbolTableAnalyzer) -> None:
  """Verifies the behavior of union no tensor.

  Args:
      analyzer: ...
  """
  x_node: cst.BaseExpression = cst.parse_expression("x")
  u_type: UnionType = UnionType([ModuleType("Module", "torch.nn")])
  analyzer.table.record_type(x_node, u_type)
  call_node: cst.Call = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))
  analyzer.leave_Call(call_node)
  assert analyzer.table.get_type(call_node) is None


def analyze(code: str, analyzer: SymbolTableAnalyzer) -> cst.Module:
  """Analyzes .

  Args:
      code: ...
      analyzer: ...
  """
  tree: cst.Module = cst.parse_module(code)
  tree.visit(analyzer)
  return tree
