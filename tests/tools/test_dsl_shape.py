"""Test suite for the Dsl Shape module."""

from ml_switcheroo.core.dsl import ParameterDef


def test_symbolic_shape_initialization():
  """Verifies the behavior of symbolic shape initialization."""
  p = ParameterDef(name="x", shape_spec="[N, N]")
  assert p.shape_spec == "[N, N]"
  assert "N" in p.shape_spec
  assert p.name == "x"


def test_shape_spec_defaults_none():
  """Verifies the behavior of shape spec defaults none."""
  p = ParameterDef(name="y")
  assert p.shape_spec is None


def test_shape_spec_complex_format():
  """Verifies the behavior of shape spec complex format."""
  spec = "[Batch, Heads, Seq, Dim]"
  p = ParameterDef(name="attn_mask", shape_spec=spec)
  assert p.shape_spec == spec


def test_shape_spec_integration_with_rank():
  """Verifies the behavior of shape spec integration with rank."""
  p = ParameterDef(name="image", rank=4, shape_spec="[N, C, H, W]")
  assert p.rank == 4
  assert "H" in p.shape_spec


def test_shape_spec_integration_with_dtype():
  """Verifies the behavior of shape spec integration with dtype."""
  p = ParameterDef(name="mask", dtype="bool", shape_spec="[B, T]")
  assert p.dtype == "bool"
  assert p.shape_spec == "[B, T]"
