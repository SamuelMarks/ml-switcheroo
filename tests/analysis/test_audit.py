"""Test module for the semantic API CoverageScanner.

This module contains unit tests verifying the correctness of `CoverageScanner` in tracking
and identifying API usage within CST trees. It tests alias resolution (e.g. `import torch as t`),
FQNs extraction, and interaction with the `SemanticsManager`.
"""

import libcst as cst

from ml_switcheroo.analysis.audit import CoverageScanner
from ml_switcheroo.semantics.manager import SemanticsManager


def test_coverage_scanner_import_resolution() -> None:
  """Test that CoverageScanner correctly resolves standard and aliased imports.

  This test verifies that standard modules (`torch`), aliased modules (`import torch.nn as nn`),
  and `from` imports (`from jax import numpy as jnp`) correctly populate the scanner's internal
  alias map and are recognized when they are later invoked in the code.
  """
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
  """Test the correct generation of Fully Qualified Names (FQNs) based on aliases.

  This verifies that the scanner can map method calls back to their canonical FQNs.
  For example, if `torch` is imported as `t`, `t.sum()` should resolve to `torch.sum`.
  """
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
  """Test edge cases such as relative imports, unrecognized node types, and single root calls.

  This test ensures the scanner degrades gracefully when handling:
  - Relative `from . import` statements where `node.module` may be empty.
  - Invalid nodes passed to `_check_node`.
  - Calling a root module alias directly (e.g., `t()` when `import torch as t`).
  """
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


def test_coverage_scanner_variants_no_match() -> None:
  """Test that scanner identifies the root module even when semantics mapping returns a variant.

  This verifies that if we have a mocked `SemanticsManager` returning definitions, the
  scanner correctly attributes the API call to the framework tracked by the scanner.
  """
  from unittest.mock import MagicMock

  semantics = MagicMock()
  semantics.get_definition.return_value = ("abstract_id", {"variants": {"jax": {"api": "jax.other"}}})
  scanner = CoverageScanner(semantics, {"jax"})

  code = """
import jax
jax.something()
"""
  tree = cst.parse_module(code)
  tree.visit(scanner)
  assert scanner.results["jax.something"] == (True, "jax")
