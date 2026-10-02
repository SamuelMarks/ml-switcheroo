"""Test suite for the Standards Sass module."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_neural_ops_sass_variants() -> None:
  """Verifies the behavior of neural ops NVIDIA_SASS variants."""
  mgr = SemanticsManager()
  variant = mgr.resolve_variant("Conv2d", "nvidia_sass")
  assert variant is not None
  assert variant["api"] == "Macro.Conv2d"

  variant_linear = mgr.resolve_variant("Linear", "nvidia_sass")
  assert variant_linear is not None
  assert variant_linear["api"] == "Macro.Linear"


def test_math_ops_sass_variants() -> None:
  """Verifies the behavior of math ops NVIDIA_SASS variants."""
  mgr = SemanticsManager()

  variant_add = mgr.resolve_variant("Add", "nvidia_sass")
  assert variant_add is not None
  assert variant_add["api"] == "FADD"

  variant_mul = mgr.resolve_variant("Mul", "nvidia_sass")
  assert variant_mul is not None
  assert variant_mul["api"] == "FMUL"


def test_no_dummy_instructions_in_nvidia_sass() -> None:
  """Verifies that no DUMMY instructions exist in the NVIDIA SASS ISA."""
  import json
  import os

  path = os.path.join(os.path.dirname(__file__), "../../src/ml_switcheroo/semantics/nvidia_sass_isa.json")
  with open(path, "r") as f:
    data = json.load(f)
  for k in data.keys():
    assert not k.startswith("DUMMY_NVIDIA_SASS_INST"), f"Found dummy instruction: {k}"
  assert "FFMA" in data
  assert "LDG" in data
