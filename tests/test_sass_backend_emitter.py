"""Test module."""

from ml_switcheroo.core.compiler.backends.sass.emitter import SassEmitter
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment


def test_sass_emitter_emit(mocker):
  """Test element."""
  emitter = SassEmitter()
  nodes = [SassComment(text="test")]
  mock_printer = mocker.patch("ml_switcheroo.core.compiler.backends.sass.emitter.SassPrinter")
  instance = mock_printer.return_value
  instance.emit.return_value = "    // test\n"

  result = emitter.emit(nodes)
  instance.emit.assert_called_once_with(nodes)
  assert result == "    // test\n"


def test_sass_emitter_integration():
  """Test element."""
  emitter = SassEmitter()
  nodes = [SassComment(text="integration")]
  result = emitter.emit(nodes)
  assert "integration" in result
