"""Test suite for the Sass Wiring module."""

from unittest.mock import MagicMock

from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer


def test_synthesizer_wires_macros() -> None:
  """Verifies the behavior of synthesizer wires macros."""
  semantics_mock = MagicMock()
  synth = NvidiaSassSynthesizer(semantics_mock)
  assert "Conv2d" in synth.macro_registry
  assert synth.macro_registry["Conv2d"] == expand_conv2d
  assert "Linear" in synth.macro_registry
  assert synth.macro_registry["Linear"] == expand_linear


def test_backend_wires_synthesizer() -> None:
  """Verifies the behavior of backend wires synthesizer."""
  semantics_mock = MagicMock()
  backend = NvidiaSassBackend(semantics=semantics_mock)
  assert isinstance(backend.synthesizer, NvidiaSassSynthesizer)
  assert "Conv2d" in backend.synthesizer.macro_registry
