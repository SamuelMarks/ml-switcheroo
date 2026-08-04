"""Test suite for the Audit module."""

import libcst as cst
from typing import Any
from ml_switcheroo.analysis.audit import CoverageScanner


class MockSemantics:
  """Mock Semantics class for testing purposes."""

  def get_definition(self, fqn: str) -> Any:
    """Mock implementation of get definition."""
    if fqn == "torch.add":
      return ("Add", {"variants": {"torch": {"api": "torch.add"}}})
    if fqn == "torch.sub":
      return ("Sub", {"variants": {"torch": {"api": "torch.something_else"}}})
    if fqn == "torch.mul":
      return ("Mul", {"variants": {"torch": None, "jax": {"api": "jax.mul"}, "torch2": {"api": "torch.mul"}}})
    return None


def test_coverage_scanner():
  """Verifies the behavior of coverage scanner."""
  semantics = MockSemantics()
  scanner = CoverageScanner(semantics, {"torch", "jax"})
  code = "\nimport torch\nimport torch as th\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom torch import *\nfrom sys import exit\n\ntorch.add(1, 2)\ntorch.sub(1, 2)\ntorch.mul(1, 2)\nth.add(1, 2)\nnn.Linear(10, 10)\nF.relu(x)\njax.numpy.sum(x)\nunknown_fw.foo()\ntorch.float32\nexit()\n"
  tree = cst.parse_module(code)
  tree.visit(scanner)
  assert scanner.results["torch.add"] == (True, "torch")
  assert scanner.results["torch.nn.Linear"] == (False, "torch")
  assert scanner.results["torch.nn.functional.relu"] == (False, "torch")
  assert scanner.results["torch.float32"] == (False, "torch")
  assert "exit" not in scanner.results


def test_coverage_scanner_unresolvable_fqn():
  """Verifies the behavior of coverage scanner when fqn cannot be resolved."""
  from ml_switcheroo.analysis.audit import CoverageScanner

  semantics = MockSemantics()
  scanner = CoverageScanner(semantics, {"torch"})

  # A complex Call node where get_full_name returns None (e.g. function is a lambda or complex expression)
  code = "(lambda x: x)(1, 2)"
  tree = cst.parse_module(code)
  tree.visit(scanner)
  assert len(scanner.results) == 0


def test_coverage_scanner_unresolvable_attribute():
  """Verifies the behavior of coverage scanner when attribute cannot be resolved."""
  from ml_switcheroo.analysis.audit import CoverageScanner

  semantics = MockSemantics()
  scanner = CoverageScanner(semantics, {"torch"})

  # Attribute on a non-Name/Attribute node
  code = "[].append(1)"
  tree = cst.parse_module(code)
  tree.visit(scanner)
  assert len(scanner.results) == 0


def test_coverage_scanner_relative_import():
  """Verifies the behavior of coverage scanner with relative imports."""
  from ml_switcheroo.analysis.audit import CoverageScanner

  semantics = MockSemantics()
  scanner = CoverageScanner(semantics, {"torch"})
  code = "from . import foo\nfrom .foo import bar\n"
  tree = cst.parse_module(code)
  tree.visit(scanner)
  assert len(scanner.results) == 0
