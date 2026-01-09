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
  # Mock Semantics
  semantics = MagicMock()

  # Mock definition that returns a Tensor return type
  def get_def(name):
    if "randn" in name or "add" in name or "abs" in name:
      return ("op", {"return_type": "Tensor"})
    return None

  semantics.get_definition.side_effect = get_def
  return SymbolTableAnalyzer(semantics)


def analyze(code, analyzer):
  tree = cst.parse_module(code)
  # Run visitor
  tree.visit(analyzer)
  return tree


def test_import_tracking(analyzer):
  code = "import torch.nn as nn"
  analyze(code, analyzer)
  sym = analyzer.current_scope.get("nn")
  assert isinstance(sym, ModuleType)
  assert sym.path == "torch.nn"


def test_assignment_tracking(analyzer):
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
