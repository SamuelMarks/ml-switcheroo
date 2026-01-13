from unittest.mock import MagicMock
from ml_switcheroo.compiler.backends.sass.synthesizer import SassSynthesizer
from ml_switcheroo.compiler.backends.sass import SassBackend
from ml_switcheroo.compiler.backends.sass.macros import expand_conv2d, expand_linear


def test_synthesizer_wires_macros() -> None:
  """
  Verify that SassSynthesizer populates the macro_registry on init.
  """
  semantics_mock = MagicMock()
  synth = SassSynthesizer(semantics_mock)

  # Check registration matches macros
  assert "Conv2d" in synth.macro_registry
  assert synth.macro_registry["Conv2d"] == expand_conv2d

  assert "Linear" in synth.macro_registry
  assert synth.macro_registry["Linear"] == expand_linear


def test_backend_wires_synthesizer() -> None:
  """
  Verify SassBackend initializes the synthesizer correctly.
  """
  semantics_mock = MagicMock()
  backend = SassBackend(semantics=semantics_mock)

  assert isinstance(backend.synthesizer, SassSynthesizer)
  assert "Conv2d" in backend.synthesizer.macro_registry
