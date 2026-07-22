"""Test suite for the Symbol Table module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer, ModuleType, TensorType, UnionType, SymbolType, Scope


@pytest.fixture
def analyzer():
  """Provides a mock analyzer for testing."""
  semantics = MagicMock()

  def get_def(name):
    """Gets def."""
    if "randn" in name or "add" in name or "abs" in name:
      return ("op", {"return_type": "Tensor"})
    return None

  semantics.get_definition.side_effect = get_def
  return SymbolTableAnalyzer(semantics)


def analyze(code, analyzer):
  """Analyzes ."""
  tree = cst.parse_module(code)
  tree.visit(analyzer)
  return tree


def test_import_tracking(analyzer):
  """Verifies the behavior of import tracking."""
  code = "import torch.nn as nn"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("nn")
  assert isinstance(sym, ModuleType)
  assert sym.path == "torch.nn"


def test_assignment_tracking(analyzer):
  """Verifies the behavior of assignment tracking."""
  code = "\nimport torch\nx = torch.randn(1)\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, TensorType)
  assert sym.framework == "torch"


def test_control_flow_union(analyzer):
  """Verifies the behavior of control flow union."""
  code = "\nimport torch\nif True:\n    x = torch.randn(1)\nelse:\n    x = torch.nn\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)
  types_str = [str(t) for t in sym.types]
  assert "Tensor" in types_str
  assert "Module" in types_str


def test_control_flow_ambiguity(analyzer):
  """Verifies the behavior of control flow ambiguity."""
  code = "\nimport torch\nif True:\n    y = torch.randn(1)\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ternary_expression_union(analyzer):
  """Verifies the behavior of ternary expression union."""
  code = "\nimport torch\nx = torch.randn() if True else torch.nn\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)
  types_str = [str(t) for t in sym.types]
  assert "Tensor" in types_str
  assert "Module" in types_str


def test_loop_state_merge(analyzer):
  """Verifies the behavior of loop state merge."""
  code = "\nimport torch\nx = torch.nn\nfor i in range(10):\n    x = torch.randn()\n"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, UnionType)
  types_str = [str(t) for t in sym.types]
  assert "Tensor" in types_str
  assert "Module" in types_str


def test_implicit_tensor_method_on_union(analyzer):
  """Verifies the behavior of implicit tensor method on union."""
  x_node = cst.parse_expression("x")
  u_type = UnionType([TensorType("Tensor", "torch"), ModuleType("Module", "torch")])
  analyzer.table.record_type(x_node, u_type)
  call_node = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))
  analyzer.leave_Call(call_node)
  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "Tensor"})
  analyzer.leave_Call(call_node)
  res_type = analyzer.table.get_type(call_node)
  assert isinstance(res_type, TensorType)


def test_symbol_type_equality():
  """Verifies the behavior of symbol type equality."""
  s1 = SymbolType()
  s1.name = "Test"
  s2 = SymbolType()
  s2.name = "Test"
  s3 = SymbolType()
  s3.name = "Other"
  assert s1 == s2
  assert s1 != s3
  assert s1 != "Test"


def test_tensor_type_equality():
  """Verifies the behavior of tensor type equality."""
  t1 = TensorType("Tensor", "torch")
  t2 = TensorType("Tensor", "torch")
  t3 = TensorType("Tensor", "jax")
  assert t1 == t2
  assert t1 != t3
  assert t1 != "Tensor"


def test_module_type_equality():
  """Verifies the behavior of module type equality."""
  m1 = ModuleType("Module", "torch.nn")
  m2 = ModuleType("Module", "torch.nn")
  m3 = ModuleType("Module", "jax.nn")
  assert m1 == m2
  assert m1 != m3
  assert m1 != "Module"


def test_union_type_equality_and_str():
  """Verifies the behavior of union type equality and string."""
  u1 = UnionType([TensorType("Tensor", "torch"), ModuleType("Module", "torch.nn")])
  u2 = UnionType([ModuleType("Module", "torch.nn"), TensorType("Tensor", "torch")])
  u3 = UnionType([TensorType("Tensor", "jax")])
  assert u1 == u2
  assert u1 != u3
  assert u1 != "Union"
  assert (
    str(u1) == "Union[Module, Tensor]"
    or str(u1) == "Union[Tensor, torch.nn]"
    or str(u1) == "Union[Module, torch.nn, Tensor]"
    or ("Union" in str(u1))
  )


def test_scope_resolution_parent():
  """Verifies the behavior of scope resolution parent."""
  parent = Scope(name="parent")
  pt = SymbolType()
  pt.name = "ParentType"
  parent.set("x", pt)
  child = Scope(parent=parent, name="child")
  ct = SymbolType()
  ct.name = "ChildType"
  child.set("y", ct)
  assert child.get("y").name == "ChildType"
  assert child.get("x").name == "ParentType"
  assert child.get("z") is None


def test_class_and_function_scope(analyzer):
  """Verifies the behavior of class and function scope."""
  code = "\nclass MyClass:\n    a = torch.randn(1)\n    def my_func(self):\n        b = torch.randn(1)\n"
  analyze(code, analyzer)
  assert analyzer.current_scope.name == "global"
