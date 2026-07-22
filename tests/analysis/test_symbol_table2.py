"""Test suite for the Symbol Table2 module."""

import libcst as cst
import pytest
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_types import TensorType, ModuleType, UnionType
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer, SymbolTable


@pytest.fixture
def analyzer():
  """Provides a mock analyzer for testing."""
  sem = MagicMock()

  def get_def(name):
    """Gets def."""
    if "randn" in name or "view" in name:
      return ("op", {"return_type": "Tensor"})
    return None

  sem.get_definition.side_effect = get_def
  analyzer = SymbolTableAnalyzer(sem)
  analyzer.table = SymbolTable()
  analyzer.source_fw = "torch"
  return analyzer


def test_for_else(analyzer):
  """Verifies the behavior of for else."""
  code = "\nimport torch\nx = torch.nn\nfor i in range(10):\n    pass\nelse:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop(analyzer):
  """Verifies the behavior of while loop."""
  code = "\nimport torch\nx = torch.nn\nwhile True:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop_else(analyzer):
  """Verifies the behavior of while loop else."""
  code = "\nimport torch\nx = torch.nn\nwhile True:\n    pass\nelse:\n    x = torch.randn(1)\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_ifexp_partial(analyzer):
  """Verifies the behavior of ifexp partial."""
  code = (
    "\nimport torch\nx = torch.randn(1) if True else untyped_func()\ny = untyped_func() if True else torch.randn(1)\n"
  )
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("x"), TensorType)
  assert isinstance(analyzer.current_scope.get("y"), TensorType)


def test_merge_states_b_only(analyzer):
  """Merges states b only."""
  code = "\nimport torch\nif True:\n    pass\nelse:\n    z = torch.randn(1)\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("z")
  assert isinstance(sym, TensorType)


def test_make_union_same(analyzer):
  """Verifies the behavior of make union same."""
  t1 = TensorType("Tensor", "torch")
  res = analyzer._make_union(t1, t1)
  assert res == t1


def test_make_union_nested(analyzer):
  """Verifies the behavior of make union nested."""
  t1 = TensorType("Tensor", "torch")
  m1 = ModuleType("Module", "torch.nn")
  u1 = UnionType([t1, m1])
  res = analyzer._make_union(u1, t1)
  assert isinstance(res, UnionType)
  assert len(res.types) == 2


def test_make_union_dedup_single(analyzer):
  """Verifies the behavior of make union dedup single."""
  t1 = TensorType("Tensor", "torch")
  t2 = TensorType("Tensor", "torch")
  u1 = UnionType([t1])
  res = analyzer._make_union(u1, t2)
  assert isinstance(res, TensorType)


def test_import_from(analyzer):
  """Verifies the behavior of import from."""
  code = "\nfrom torch import nn, optim as opt\nfrom . import local_module\n"
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("nn"), ModuleType)
  assert isinstance(analyzer.current_scope.get("opt"), ModuleType)
  assert analyzer.current_scope.get("local_module") is None


def test_assign_untyped(analyzer):
  """Verifies the behavior of assign untyped."""
  code = "\nuntyped_var = untyped_func()\n"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("untyped_var") is None


def test_assign_attribute(analyzer):
  """Verifies the behavior of assign attribute."""
  code = "\nimport torch\nclass A:\n    def __init__(self):\n        self.x = torch.randn(1)\n"
  tree = analyze(code, analyzer)

  class AttrVisitor(cst.CSTVisitor):
    """Test suite for the Attr Visitor component."""

    def __init__(self):
      """Initializes the AttrVisitor instance."""
      self.nodes = []

    def visit_Attribute(self, node):
      """Helper to visit Attribute."""
      if node.attr.value == "x":
        self.nodes.append(node)

  v = AttrVisitor()
  tree.visit(v)
  assert len(v.nodes) > 0
  assert isinstance(analyzer.table.get_type(v.nodes[0]), TensorType)


def test_call_on_tensor(analyzer):
  """Verifies the behavior of call on tensor."""
  code = "\nimport torch\nx = torch.randn(1)\ny = x.view()\n"
  analyzer.semantics.get_definition.side_effect = (
    lambda n: ("op", {"return_type": "Tensor"}) if "view" in n or "randn" in n else None
  )
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ifexp_both_unknown(analyzer):
  """Verifies the behavior of ifexp both unknown."""
  code = "x = unknown() if True else unknown2()"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_assign_subscript(analyzer):
  """Verifies the behavior of assign subscript."""
  code = "import torch\nmy_list[0] = torch.randn(1)"
  analyze(code, analyzer)


def test_call_non_tensor_return(analyzer):
  """Verifies the behavior of call non tensor return."""
  analyzer.semantics.get_key_origins.return_value = {}
  code = "import torch\nx = torch.get_int()"
  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "int"}) if "get_int" in n else None
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_import_star(analyzer):
  """Verifies the behavior of import star."""
  code = "from torch import *"
  analyze(code, analyzer)


def test_union_no_tensor(analyzer):
  """Verifies the behavior of union no tensor."""
  x_node = cst.parse_expression("x")
  u_type = UnionType([ModuleType("Module", "torch.nn")])
  analyzer.table.record_type(x_node, u_type)
  call_node = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))
  analyzer.leave_Call(call_node)
  assert analyzer.table.get_type(call_node) is None


def analyze(code, analyzer):
  """Analyzes ."""
  tree = cst.parse_module(code)
  tree.visit(analyzer)
  return tree
