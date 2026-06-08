import libcst as cst
from typing import Any

from ml_switcheroo.analysis.audit import CoverageScanner


class MockSemantics:
  def get_definition(self, fqn: str) -> Any:
    if fqn == "torch.add":
      return ("Add", {"variants": {"torch": {"api": "torch.add"}}})
    return None


def test_coverage_scanner():
  semantics = MockSemantics()
  scanner = CoverageScanner(semantics, {"torch", "jax"})  # type: ignore

  code = """
import torch
import torch as th
from torch import nn
from torch.nn import functional as F
from sys import exit

torch.add(1, 2)
th.add(1, 2)
nn.Linear(10, 10)
F.relu(x)
jax.numpy.sum(x)
unknown_fw.foo()
torch.float32
exit()
"""
  tree = cst.parse_module(code)
  tree.visit(scanner)

  assert scanner.results["torch.add"] == (True, "torch")
  assert scanner.results["torch.nn.Linear"] == (False, "torch")
  assert scanner.results["torch.nn.functional.relu"] == (False, "torch")
  assert scanner.results["torch.float32"] == (False, "torch")
  assert "exit" not in scanner.results


def test_import_from_no_module():
  semantics = MockSemantics()
  scanner = CoverageScanner(semantics, {"torch"})  # type: ignore
  tree = cst.parse_module("from . import module")
  tree.visit(scanner)
