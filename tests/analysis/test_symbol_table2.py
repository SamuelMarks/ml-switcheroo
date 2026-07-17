"""Auto-generated doc."""

import libcst as cst
import pytest
from unittest.mock import MagicMock

from ml_switcheroo.analysis.symbol_types import TensorType, ModuleType, UnionType
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer, SymbolTable


@pytest.fixture
def analyzer():
  """Auto-generated doc."""
  sem = MagicMock()

  def get_def(name):
    """Auto-generated doc."""
    if "randn" in name or "view" in name:
      return ("op", {"return_type": "Tensor"})
    return None

  sem.get_definition.side_effect = get_def

  analyzer = SymbolTableAnalyzer(sem)
  analyzer.table = SymbolTable()
  analyzer.source_fw = "torch"
  return analyzer


def test_for_else(analyzer):
  """Test For loop with an else branch."""
  code = """
import torch
x = torch.nn
for i in range(10):
    pass
else:
    x = torch.randn(1)
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop(analyzer):
  """Test While loop execution."""
  code = """
import torch
x = torch.nn
while True:
    x = torch.randn(1)
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_while_loop_else(analyzer):
  """Test While loop with else branch."""
  code = """
import torch
x = torch.nn
while True:
    pass
else:
    x = torch.randn(1)
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)


def test_ifexp_partial(analyzer):
  """Test ternary expression (IfExp) when only one side has a recognized type."""
  code = """
import torch
x = torch.randn(1) if True else untyped_func()
y = untyped_func() if True else torch.randn(1)
"""
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("x"), TensorType)
  assert isinstance(analyzer.current_scope.get("y"), TensorType)


def test_merge_states_b_only(analyzer):
  """Test merging states when a key is present only in the second state."""
  code = """
import torch
if True:
    pass
else:
    z = torch.randn(1)
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("z")
  assert isinstance(sym, TensorType)


def test_make_union_same(analyzer):
  """Test _make_union with two identical types."""
  t1 = TensorType("Tensor", "torch")
  res = analyzer._make_union(t1, t1)
  assert res == t1


def test_make_union_nested(analyzer):
  """Test _make_union flattening nested UnionTypes."""
  t1 = TensorType("Tensor", "torch")
  m1 = ModuleType("Module", "torch.nn")
  u1 = UnionType([t1, m1])
  res = analyzer._make_union(u1, t1)
  assert isinstance(res, UnionType)
  assert len(res.types) == 2


def test_make_union_dedup_single(analyzer):
  """Test _make_union deduplicating to a single type."""
  t1 = TensorType("Tensor", "torch")
  t2 = TensorType("Tensor", "torch")
  u1 = UnionType([t1])
  res = analyzer._make_union(u1, t2)
  assert isinstance(res, TensorType)


def test_import_from(analyzer):
  """Test tracking ImportFrom statements including without a module."""
  code = """
from torch import nn, optim as opt
from . import local_module
"""
  analyze(code, analyzer)
  assert isinstance(analyzer.current_scope.get("nn"), ModuleType)
  assert isinstance(analyzer.current_scope.get("opt"), ModuleType)
  assert analyzer.current_scope.get("local_module") is None


def test_assign_untyped(analyzer):
  """Test Assign when the right hand side is an unknown type."""
  code = """
untyped_var = untyped_func()
"""
  analyze(code, analyzer)
  assert analyzer.current_scope.get("untyped_var") is None


def test_assign_attribute(analyzer):
  """Test Assign to an attribute records the type."""
  code = """
import torch
class A:
    def __init__(self):
        self.x = torch.randn(1)
"""
  tree = analyze(code, analyzer)

  class AttrVisitor(cst.CSTVisitor):
    """Visitor to find Attribute nodes."""

    def __init__(self):
      """Initializes the visitor with an empty list of nodes."""
      self.nodes = []

    def visit_Attribute(self, node):
      """Records Attribute nodes matching 'x'."""
      if node.attr.value == "x":
        self.nodes.append(node)

  v = AttrVisitor()
  tree.visit(v)
  assert len(v.nodes) > 0
  assert isinstance(analyzer.table.get_type(v.nodes[0]), TensorType)


def test_call_on_tensor(analyzer):
  """Test resolving a method call on a Tensor receiver."""
  code = """
import torch
x = torch.randn(1)
y = x.view()
"""
  analyzer.semantics.get_definition.side_effect = lambda n: (
    ("op", {"return_type": "Tensor"}) if "view" in n or "randn" in n else None
  )

  analyze(code, analyzer)
  sym = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ifexp_both_unknown(analyzer):
  """Test IfExp when both sides are unknown types."""
  code = "x = unknown() if True else unknown2()"
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_assign_subscript(analyzer):
  """Test assigning to a Subscript (neither Name nor Attribute)."""
  code = "import torch\nmy_list[0] = torch.randn(1)"
  analyze(code, analyzer)
  # No crash, and doesn't record to scope since it's not a Name/Attribute


def test_call_non_tensor_return(analyzer):
  """Test Call that returns a non-tensor type."""
  analyzer.semantics.get_key_origins.return_value = {}
  code = "import torch\nx = torch.get_int()"
  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "int"}) if "get_int" in n else None
  analyze(code, analyzer)
  assert analyzer.current_scope.get("x") is None


def test_import_star(analyzer):
  """Test ImportStar."""
  code = "from torch import *"
  analyze(code, analyzer)


def test_union_no_tensor(analyzer):
  """Test calling a method on a Union that contains no TensorType."""
  x_node = cst.parse_expression("x")
  u_type = UnionType([ModuleType("Module", "torch.nn")])
  analyzer.table.record_type(x_node, u_type)
  call_node = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))
  analyzer.leave_Call(call_node)
  assert analyzer.table.get_type(call_node) is None


def analyze(code, analyzer):
  """Test case for analyze."""
  tree = cst.parse_module(code)
  tree.visit(analyzer)
  return tree
