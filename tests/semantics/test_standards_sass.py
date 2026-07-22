"""Test suite for the Standards Sass module."""

from ml_switcheroo.frameworks.sass import SassAdapter


def test_neural_ops_sass_variants() -> None:
  """Verifies the behavior of neural ops SASS variants."""
  adapter = SassAdapter()
  defs = adapter.definitions
  assert "Conv2d" in defs
  assert defs["Conv2d"].api == "Macro.Conv2d"
  assert "Linear" in defs
  assert defs["Linear"].api == "Macro.Linear"


def test_math_ops_sass_variants() -> None:
  """Verifies the behavior of math ops SASS variants."""
  adapter = SassAdapter()
  defs = adapter.definitions
  assert "Add" in defs
  assert defs["Add"].api == "FADD"
  assert "Mul" in defs
  assert defs["Mul"].api == "FMUL"
  assert "Clamp" in defs
  assert defs["Clamp"].api == "MNMX"
  assert "Abs" in defs
  assert defs["Abs"].api in ["IABS", "FABS"]
