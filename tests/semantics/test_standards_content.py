"""
Tests for Standards Dictionary Content Integrity.

Verifies that:
1. INTERNAL_OPS contains expected functional primitives (vmap, grad).
2. Standard arguments match expected signatures.
3. No syntax errors in the static dictionary definition.
"""

import pytest
from ml_switcheroo.semantics.standards_internal import INTERNAL_OPS

def test_functional_ops_merged():
  """Verify that functional transforms from the orphan script are present."""
  assert "vmap" in INTERNAL_OPS
  assert "grad" in INTERNAL_OPS
  assert "value_and_grad" in INTERNAL_OPS
  assert "jit" in INTERNAL_OPS

def test_vmap_signature():
  """Verify vmap standard arguments align with JAX/Torch abstraction."""
  args = INTERNAL_OPS["vmap"]["std_args"]
  assert "func" in args
  assert "in_axes" in args
  assert "out_axes" in args

def test_grad_signature():
  """Verify grad arguments."""
  args = INTERNAL_OPS["grad"]["std_args"]
  assert "func" in args
  assert "argnums" in args

def test_optimizer_standards():
  """Verify existing optimizers are preserved."""
  assert "Adam" in INTERNAL_OPS
  assert "lr" in INTERNAL_OPS["Adam"]["std_args"]

def test_vision_standards():
  """Verify vision transforms are preserved."""
  assert "Resize" in INTERNAL_OPS
  assert "interpolation" in INTERNAL_OPS["Resize"]["std_args"]
