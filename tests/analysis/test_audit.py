"""Test module."""

import libcst as cst
from ml_switcheroo.analysis.audit import CoverageScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def test_coverage_scanner_import_resolution() -> None:
  """Test element."""
  semantics: SemanticsManager = SemanticsManager()
  scanner: CoverageScanner = CoverageScanner(semantics, {"torch", "jax"})

  code: str = """
import torch
import torch.nn as nn
from jax import numpy as jnp
from jax import *
from jax import lax

torch.add(1, 2)
nn.Conv2d(1, 1, 1)
jnp.sum([1, 2])
lax.add(1, 2)
torch.float32
unknown.call()
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(scanner)

  # Check alias map
  assert scanner._alias_map["torch"] == "torch"
  assert scanner._alias_map["nn"] == "torch.nn"
  assert scanner._alias_map["jnp"] == "jax.numpy"
  assert scanner._alias_map["lax"] == "jax.lax"

  # Check results
  assert "torch.add" in scanner.results
  assert "torch.nn.Conv2d" in scanner.results
  assert "jax.numpy.sum" in scanner.results
  assert "jax.lax.add" in scanner.results
  assert "torch.float32" in scanner.results
  assert "unknown.call" not in scanner.results


def test_coverage_scanner_resolve_fqn() -> None:
  """Test element."""
  semantics: SemanticsManager = SemanticsManager()
  scanner: CoverageScanner = CoverageScanner(semantics, {"torch"})

  code: str = """
import torch as t
from torch import nn

t.sum()
nn.Conv2d()
non_name_call(1)
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(scanner)

  assert scanner.results["torch.sum"][1] == "torch"
  assert scanner.results["torch.nn.Conv2d"][1] == "torch"


def test_coverage_scanner_edge_cases() -> None:
  """Test element."""
  semantics: SemanticsManager = SemanticsManager()
  scanner: CoverageScanner = CoverageScanner(semantics, {"torch"})

  # Test line 69 (not node.module)
  code_no_module: str = "from . import something"
  tree_no_module: cst.Module = cst.parse_module(code_no_module)
  tree_no_module.visit(scanner)

  # Test line 156 (no raw name)
  scanner._check_node(cst.Pass())

  # Test line 166 (root only alias resolution)
  # _check_node calls _resolve_fqn on node.func (for Calls) or node (for Attributes)
  # A standalone name 't' will be an Attribute or Name in Expr, but CoverageScanner
  # only visits Call and Attribute. Let's make an attribute call that evaluates to just the root.
  # Actually, a Call node like t() where t is an alias for torch.
  code_alias_only: str = """
import torch as t
t()
"""
  tree_alias_only: cst.Module = cst.parse_module(code_alias_only)
  tree_alias_only.visit(scanner)
  assert scanner.results["torch"][1] == "torch"
