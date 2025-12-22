"""
Tests for ODL Schema Extension: Symbolic Shape Specification.

Verifies:
1. ParameterDef supports `shape_spec` field.
2. Defaults are None.
3. Integration with other constraints (rank/dtype).
"""

import pytest
from ml_switcheroo.core.dsl import ParameterDef


def test_symbolic_shape_initialization():
  """
  Verify shape_spec field exists and accepts symbolic strings.
  Test Case from specification.
  """
  p = ParameterDef(name="x", shape_spec="[N, N]")
  assert p.shape_spec == "[N, N]"
  assert "N" in p.shape_spec
  assert p.name == "x"


def test_shape_spec_defaults_none():
  """Verify default value is None (unconstrained shape)."""
  p = ParameterDef(name="y")
  assert p.shape_spec is None


def test_shape_spec_complex_format():
  """Verify acceptance of complex dimension strings."""
  spec = "[Batch, Heads, Seq, Dim]"
  p = ParameterDef(name="attn_mask", shape_spec=spec)
  assert p.shape_spec == spec


def test_shape_spec_integration_with_rank():
  """
  Verify `shape_spec` can coexist with `rank`.
  """
  # Rank 4 tensor with specific NCHW symbolic logic
  p = ParameterDef(name="image", rank=4, shape_spec="[N, C, H, W]")
  assert p.rank == 4
  assert "H" in p.shape_spec


def test_shape_spec_integration_with_dtype():
  """
  Verify `shape_spec` can coexist with `dtype`.
  """
  p = ParameterDef(name="mask", dtype="bool", shape_spec="[B, T]")
  assert p.dtype == "bool"
  assert p.shape_spec == "[B, T]"
