"""
Tests for Alias-Aware Semantic Lookup in BaseRewriter.

Verifies that:
1. `import as` aliases are resolved correctly (e.g., `import torch as t; t.abs` -> `torch.abs`).
2. `from ... import` bindings are resolved (e.g., `from torch import nn; nn.Linear` -> `torch.nn.Linear`).
3. Standard imports without aliases bind the root correctly.
4. Relative imports are ignored (safety fallback).
5. Scoping rules are respected (last import wins in map, strict linear flow).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


class MockAliasSemantics(SemanticsManager):
  """
  Mock Manager with explicit definitions for alias testing.
  """

  def __init__(self):
    # Skip init to avoid file loading
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}
    self.framework_configs = {}

    # 1. abs: torch.abs -> jax.numpy.abs
    self._inject("abs", "torch.abs", "jax.numpy.abs")

    # 2. Linear: torch.nn.Linear -> flax.nnx.Linear
    self._inject("Linear", "torch.nn.Linear", "flax.nnx.Linear")

    # 3. functional.relu: torch.nn.functional.relu -> jax.nn.relu
    self._inject("relu", "torch.nn.functional.relu", "jax.nn.relu")

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(self, name, s_api, t_api):
    self.data[name] = {
      "variants": {"torch": {"api": s_api}, "jax": {"api": t_api}},
      "std_args": ["x"],
    }
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter():
  semantics = MockAliasSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter, code):
  """Parses code, applies rewriter, and returns generated source."""
  tree = cst.parse_module(code)
  # PivotRewriter subclasses BaseRewriter, so alias tracking happens in visitor walks
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_import_as_alias(rewriter):
  """
  Scenario: `import torch as t; t.abs(x)`
  Expectation: `t` resolves to `torch`, lookup `torch.abs` succeeds -> `jax.numpy.abs(x)`.
  """
  code = """ 
import torch as t
y = t.abs(x) 
"""
  result = rewrite_code(rewriter, code)

  # Check that t.abs became jax.numpy.abs
  assert "jax.numpy.abs(x)" in result
  # Imports might be pruned or kept by ImportFixer later, but rewriter just transforms Call
  # Note: ImportFixer is separate. Rewriter preserves structure.


def test_from_import_binding(rewriter):
  """
  Scenario: `from torch import nn; nn.Linear(1, 2)`
  Expectation: `nn` resolves to `torch.nn`, lookup `torch.nn.Linear` succeeds -> `flax.nnx.Linear(1, 2)`.
  """
  code = """ 
from torch import nn
layer = nn.Linear(1, 2) 
"""
  result = rewrite_code(rewriter, code)

  assert "flax.nnx.Linear(1, 2)" in result


def test_from_import_as_binding(rewriter):
  """
  Scenario: `from torch import nn as n; n.Linear(1, 2)`
  Expectation: `n` -> `torch.nn` -> lookup `torch.nn.Linear`.
  """
  code = """ 
from torch import nn as n
layer = n.Linear(1, 2) 
"""
  result = rewrite_code(rewriter, code)

  assert "flax.nnx.Linear(1, 2)" in result


def test_deep_import_chains(rewriter):
  """
  Scenario: `import torch.nn.functional as F; F.relu(x)`
  Expectation: `F` -> `torch.nn.functional` -> lookup `torch.nn.functional.relu`.
  """
  code = """ 
import torch.nn.functional as F
y = F.relu(x) 
"""
  result = rewrite_code(rewriter, code)

  assert "jax.nn.relu(x)" in result


def test_standard_import_no_alias(rewriter):
  """
  Scenario: `import torch; torch.abs(x)`
  Expectation: `torch` -> `torch` -> lookup `torch.abs`.
  """
  code = """ 
import torch
y = torch.abs(x) 
"""
  result = rewrite_code(rewriter, code)

  assert "jax.numpy.abs(x)" in result


def test_relative_import_ignored(rewriter):
  """
  Scenario: `from . import utils; utils.abs(x)`
  Expectation: Relative import ignored, map not updated. `utils.abs` looked up as-is (fails/ignored).
  """
  code = """ 
from . import utils
# utils.abs in this context is likely local, so it shouldn't match torch.abs
y = utils.abs(x) 
"""
  result = rewrite_code(rewriter, code)

  # Should NOT be rewritten to jax
  assert "utils.abs(x)" in result
  assert "jax.numpy.abs" not in result


def test_alias_redefinition(rewriter):
  """
  Scenario: Alias redefined in file. Assumes linear execution flow for updating map.
  """
  code = """ 
import torch as t
y1 = t.abs(x) 

import numpy as t
y2 = t.abs(x) 
"""
  # NOTE: LibCST visits nodes structurally.
  # The visit_Import happens when encountered.
  # So line 1 map: t->torch.
  # Line 2 call: t.abs -> torch.abs (rewritten).
  # Line 3 map: t->numpy.
  # Line 4 call: t.abs -> numpy.abs (not in semantics -> ignored).

  result = rewrite_code(rewriter, code)

  assert "y1 = jax.numpy.abs(x)" in result
  assert "y2 = t.abs(x)" in result  # Not rewritten because map changed to numpy


def test_alias_shadowing_imported_name(rewriter):
  """
  Scenario: `from torch import nn` binds 'nn'.
  But `nn` usually resolves to `torch.nn` via explicit logic.
  """
  code = """ 
from torch import nn
l = nn.Linear(1, 2) 
"""
  result = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear" in result
