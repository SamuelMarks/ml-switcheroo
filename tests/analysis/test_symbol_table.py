"""
Tests for Symbol Table Analysis.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.analysis.symbol_table import SymbolTableAnalyzer, ModuleType, TensorType


@pytest.fixture
def analyzer():
  # Mock Semantics
  semantics = MagicMock()
  # Mock definition that returns a Tensor return type
  semantics.get_definition.side_effect = lambda n: (
    ("randn", {"return_type": "Tensor"}) if "randn" in n else ("add", {"return_type": "Tensor"}) if "add" in n else None
  )
  return SymbolTableAnalyzer(semantics)


def analyze(code, analyzer):
  tree = cst.parse_module(code)
  tree.visit(analyzer)
  return tree


def test_import_tracking(analyzer):
  code = "import torch.nn as nn"
  analyze(code, analyzer)

  # Check 'nn' symbol in root scope
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


def test_method_chaining(analyzer):
  code = """
import torch
x = torch.randn(1)
y = x.add(1)
"""
  analyze(code, analyzer)

  # x is known Tensor
  # y = x.add(...) -> x is Tensor, so x.add is resolved.
  # Semantics mock says 'add' returns Tensor. So y is Tensor.

  sym_y = analyzer.current_scope.get("y")
  assert isinstance(sym_y, TensorType)
