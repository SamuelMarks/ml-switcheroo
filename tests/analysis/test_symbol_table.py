"""
Tests for Symbol Table Analysis.
Updated to include Control Flow Inference tests.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer, ModuleType, TensorType, UnionType


@pytest.fixture
def analyzer():
  """Test case for analyzer."""
  # Mock Semantics
  semantics = MagicMock()

  # Mock definition that returns a Tensor return type
  def get_def(name):
    """Test case for get_def."""
    if "randn" in name or "add" in name or "abs" in name:
      return ("op", {"return_type": "Tensor"})
    return None

  semantics.get_definition.side_effect = get_def
  return SymbolTableAnalyzer(semantics)


def analyze(code, analyzer):
  """Test case for analyze."""
  tree = cst.parse_module(code)
  # Run visitor
  tree.visit(analyzer)
  return tree


def test_import_tracking(analyzer):
  """Test case for test_import_tracking."""
  code = "import torch.nn as nn"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("nn")
  assert isinstance(sym, ModuleType)
  assert sym.path == "torch.nn"


def test_assignment_tracking(analyzer):
  """Test case for test_assignment_tracking."""
  code = """ 
import torch
x = torch.randn(1) 
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")
  assert isinstance(sym, TensorType)
  assert sym.framework == "torch"


def test_control_flow_union(analyzer):
  """
  Scenario:
  if cond:
      x = torch.randn()  (Tensor)
  else:
      x = 5              (Assuming 5 is untracked/None or we can simulate int if we had IntType)

  Since we don't track ints, let's use:
  if cond:
      x = torch.randn()
  else:
      x = torch.nn      (Module)

  Result: x is Union[Tensor, Module]
  """
  code = """ 
import torch
if True: 
    x = torch.randn(1) 
else: 
    x = torch.nn
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")

  assert isinstance(sym, UnionType)
  types_str = [str(t) for t in sym.types]
  assert "Tensor" in types_str
  assert "Module" in types_str


def test_control_flow_ambiguity(analyzer):
  """
  Scenario: Variable defined in IF but not ELSE.
  Result: Should retain type from IF branch (Optimistic typing).
  """
  code = """ 
import torch
if True: 
    y = torch.randn(1) 
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("y")
  assert isinstance(sym, TensorType)


def test_ternary_expression_union(analyzer):
  """
  Scenario: x = torch.randn() if C else torch.nn
  Result: Expr type is Union[Tensor, Module]. x gets that type.
  """
  code = """ 
import torch
x = torch.randn() if True else torch.nn
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")

  assert isinstance(sym, UnionType)
  types_str = [str(t) for t in sym.types]
  assert "Tensor" in types_str
  assert "Module" in types_str


def test_loop_state_merge(analyzer):
  """
  Scenario:
  x = torch.nn
  for i in range(10):
      x = torch.randn()

  Start state: x is Module.
  Loop body: x becomes Tensor.
  Merge: x is Union[Module, Tensor].
  """
  code = """ 
import torch
x = torch.nn
for i in range(10): 
    x = torch.randn() 
"""
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("x")

  assert isinstance(sym, UnionType)
  types_str = [str(t) for t in sym.types]
  assert "Tensor" in types_str
  assert "Module" in types_str


def test_implicit_tensor_method_on_union(analyzer):
  """
  Scenario: x is Union[Tensor, Module]. Call x.view().
  Expected: Inferred as Tensor method call due to presence of Tensor in Union.
  """
  # Manually inject a union type into the table for a node
  x_node = cst.parse_expression("x")
  u_type = UnionType([TensorType("Tensor", "torch"), ModuleType("Module", "torch")])
  analyzer.table.record_type(x_node, u_type)

  # Now simulate visiting a call x.view()
  # We construct a synthetic call node that wraps x_node
  call_node = cst.Call(func=cst.Attribute(value=x_node, attr=cst.Name("view")))

  # Run leave_Call logic
  analyzer.leave_Call(call_node)

  # Check if call node got resolved to Tensor (because .view -> Reshape -> Tensor return)
  # We need semantics to return definition for 'view' or 'Reshape'
  # The mock returns 'op' with return_type='Tensor' for anything not explicitly filtered.
  # And leave_Call logic checks 'Tensor' substring in API path.

  # Our mock get_def handles 'randn', 'add'. Let's update it or rely on fallback.
  # api path generated will be 'torch.Tensor.view'.
  # get_definition receives 'torch.Tensor.view'.
  # The Mock in fixture returns None for 'view'.
  # Fallback checks 'view' (leaf).
  # We need to update fixture mock to support 'view'.

  analyzer.semantics.get_definition.side_effect = lambda n: ("op", {"return_type": "Tensor"})

  analyzer.leave_Call(call_node)

  res_type = analyzer.table.get_type(call_node)
  assert isinstance(res_type, TensorType)


from ml_switcheroo.analysis.symbol_table import SymbolType, TensorType, ModuleType, UnionType, Scope


def test_symbol_type_equality():
  """Test SymbolType equality."""
  s1 = SymbolType("Test")
  s2 = SymbolType("Test")
  s3 = SymbolType("Other")
  assert s1 == s2
  assert s1 != s3
  assert s1 != "Test"


def test_tensor_type_equality():
  """Test TensorType equality."""
  t1 = TensorType("Tensor", "torch")
  t2 = TensorType("Tensor", "torch")
  t3 = TensorType("Tensor", "jax")
  assert t1 == t2
  assert t1 != t3
  assert t1 != "Tensor"


def test_module_type_equality():
  """Test ModuleType equality."""
  m1 = ModuleType("Module", "torch.nn")
  m2 = ModuleType("Module", "torch.nn")
  m3 = ModuleType("Module", "jax.nn")
  assert m1 == m2
  assert m1 != m3
  assert m1 != "Module"


def test_union_type_equality_and_str():
  """Test UnionType equality and string representation."""
  u1 = UnionType([TensorType("Tensor", "torch"), ModuleType("Module", "torch.nn")])
  u2 = UnionType([ModuleType("Module", "torch.nn"), TensorType("Tensor", "torch")])
  u3 = UnionType([TensorType("Tensor", "jax")])
  assert u1 == u2
  assert u1 != u3
  assert u1 != "Union"
  assert str(u1) == "Union[Module, Tensor]"


def test_scope_resolution_parent():
  """Test Scope resolution with a parent scope."""
  parent = Scope(name="parent")
  parent.set("x", SymbolType("ParentType"))
  child = Scope(parent=parent, name="child")
  child.set("y", SymbolType("ChildType"))

  assert child.get("y").name == "ChildType"
  assert child.get("x").name == "ParentType"
  assert child.get("z") is None


def test_class_and_function_scope(analyzer):
  """Test traversing class and function definitions creates and drops scopes."""
  code = """
class MyClass:
    a = torch.randn(1)
    def my_func(self):
        b = torch.randn(1)
"""
  analyze(code, analyzer)
  assert analyzer.current_scope.name == "global"


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
