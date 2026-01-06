"""
Tests for SASS Framework Adapter Wiring.

Verifies:
1.  **Macro Registration**: `PythonToSassEmitter` injects `expand_conv2d`/`expand_linear` into `SassSynthesizer`.
2.  **Logic Continuity**: The `SassSynthesizer` instantiated by the emitter is correctly configured.
"""

from ml_switcheroo.frameworks.sass import PythonToSassEmitter, SassAdapter
from ml_switcheroo.core.sass.macros import expand_conv2d, expand_linear


def test_emitter_wires_macros() -> None:
  """
  Verify that PythonToSassEmitter populates the macro_registry on init.
  """
  emitter = PythonToSassEmitter()

  # Access the internal synthesizer instance
  synth = emitter.synth

  # Check registration
  assert "Conv2d" in synth.macro_registry
  assert synth.macro_registry["Conv2d"] == expand_conv2d

  assert "Linear" in synth.macro_registry
  assert synth.macro_registry["Linear"] == expand_linear


def test_adapter_creates_valid_emitter() -> None:
  """
  Verify SassAdapter.create_emitter() returns a properly configured instance.
  """
  adapter = SassAdapter()
  emitter = adapter.create_emitter()

  assert isinstance(emitter, PythonToSassEmitter)
  assert "Conv2d" in emitter.synth.macro_registry
