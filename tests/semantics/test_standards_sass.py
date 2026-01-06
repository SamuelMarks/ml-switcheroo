"""
Tests for SASS Definition Wiring.

Verifies that SASS-specific operation variants are correctly exposed via the
Framework Adapter layer, enforcing the architectural decision to decouple
implementation details from the central standards Hub.
"""

from ml_switcheroo.frameworks.sass import SassAdapter


def test_neural_ops_sass_variants() -> None:
  """
  Verify that Neural Network macros are registered in the SassAdapter definitions.
  These mappings (1-to-N) are injected programmatically by the adapter.
  """
  adapter = SassAdapter()
  defs = adapter.definitions

  assert "Conv2d" in defs
  assert defs["Conv2d"].api == "Macro.Conv2d"

  assert "Linear" in defs
  assert defs["Linear"].api == "Macro.Linear"


def test_math_ops_sass_variants() -> None:
  """
  Verify that Math Opcodes (1-to-1) are registered in the SassAdapter definitions.
  """
  adapter = SassAdapter()
  defs = adapter.definitions

  # Add -> FADD
  assert "Add" in defs
  assert defs["Add"].api == "FADD"

  # Mul -> FMUL
  assert "Mul" in defs
  assert defs["Mul"].api == "FMUL"

  # Clamp -> MNMX (MinMax)
  assert "Clamp" in defs
  assert defs["Clamp"].api == "MNMX"

  # Abs -> IABS or FABS depending on implementation choice, adapter uses IABS/Generic
  assert "Abs" in defs
  assert defs["Abs"].api in ["IABS", "FABS"]
