"""Test suite for the Dsl Shape module."""

from ml_switcheroo.core.dsl import ParameterDef


def test_symbolic_shape_initialization() -> None:
  """Verify `ParameterDef` emits `N` for symbolic shape initialization."""
  p: ParameterDef = ParameterDef(name="x", shape_spec="[N, N]")
  assert p.shape_spec == "[N, N]"
  assert "N" in p.shape_spec
  assert p.name == "x"


def test_shape_spec_defaults_none() -> None:
  """Verify `ParameterDef` handles shape spec defaults none as expected."""
  p: ParameterDef = ParameterDef(name="y")
  assert p.shape_spec is None


def test_shape_spec_complex_format() -> None:
  """Verify `ParameterDef` handles shape spec complex format as expected."""
  spec: str = "[Batch, Heads, Seq, Dim]"
  p: ParameterDef = ParameterDef(name="attn_mask", shape_spec=spec)
  assert p.shape_spec == spec


def test_shape_spec_integration_with_rank() -> None:
  """Verify `ParameterDef` returns `4` for shape spec integration with rank."""
  p: ParameterDef = ParameterDef(name="image", rank=4, shape_spec="[N, C, H, W]")
  assert p.rank == 4
  assert p.shape_spec is not None
  assert "H" in p.shape_spec


def test_shape_spec_integration_with_dtype() -> None:
  """Verify `ParameterDef` handles shape spec integration with dtype as expected."""
  p: ParameterDef = ParameterDef(name="mask", dtype="bool", shape_spec="[B, T]")
  assert p.dtype == "bool"
  assert p.shape_spec == "[B, T]"
