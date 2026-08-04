"""Test suite for the Standards Sass module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_neural_ops_sass_variants() -> None:
  """Verifies the behavior of neural ops SASS variants."""
  mgr = SemanticsManager()
  variant = mgr.resolve_variant("Conv2d", "sass")
  assert variant is not None
  assert variant["api"] == "Macro.Conv2d"

  variant_linear = mgr.resolve_variant("Linear", "sass")
  assert variant_linear is not None
  assert variant_linear["api"] == "Macro.Linear"


def test_math_ops_sass_variants() -> None:
  """Verifies the behavior of math ops SASS variants."""
  mgr = SemanticsManager()

  variant_add = mgr.resolve_variant("Add", "sass")
  assert variant_add is not None
  assert variant_add["api"] == "FADD"

  variant_mul = mgr.resolve_variant("Mul", "sass")
  assert variant_mul is not None
  assert variant_mul["api"] == "FMUL"
