"""Test suite for the Dsl Shape module."""

from tests.conftest import _test_getattr

from ml_switcheroo.core.dsl import ParameterDef


def test_symbolic_shape_initialization() -> None:
  """Verifies the behavior of symbolic shape initialization."""
  p: ParameterDef = ParameterDef(name="x", shape_spec="[N, N]")
  assert _test_getattr(p, "shape_spec") == "[N, N]"
  assert "N" in _test_getattr(p, "shape_spec")
  assert _test_getattr(p, "name") == "x"


def test_shape_spec_defaults_none() -> None:
  """Verifies the behavior of shape spec defaults none."""
  p: ParameterDef = ParameterDef(name="y")
  assert _test_getattr(p, "shape_spec") is None


def test_shape_spec_complex_format() -> None:
  """Verifies the behavior of shape spec complex format."""
  spec: str = "[Batch, Heads, Seq, Dim]"
  p: ParameterDef = ParameterDef(name="attn_mask", shape_spec=spec)
  assert _test_getattr(p, "shape_spec") == spec


def test_shape_spec_integration_with_rank() -> None:
  """Verifies the behavior of shape spec integration with rank."""
  p: ParameterDef = ParameterDef(name="image", rank=4, shape_spec="[N, C, H, W]")
  assert _test_getattr(p, "rank") == 4
  assert "H" in _test_getattr(p, "shape_spec")


def test_shape_spec_integration_with_dtype() -> None:
  """Verifies the behavior of shape spec integration with dtype."""
  p: ParameterDef = ParameterDef(name="mask", dtype="bool", shape_spec="[B, T]")
  assert _test_getattr(p, "dtype") == "bool"
  assert _test_getattr(p, "shape_spec") == "[B, T]"
