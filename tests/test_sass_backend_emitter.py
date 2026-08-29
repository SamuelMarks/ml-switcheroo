"""Test module."""

from typing import Any, List

from ml_switcheroo.core.compiler.backends.sass.emitter import SassEmitter
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment, SassNode


def test_sass_emitter_emit(mocker: Any) -> None:
  """Docstring."""
  emitter: SassEmitter = SassEmitter()
  nodes: List[SassNode] = [SassComment(text="test")]
  mock_printer: Any = mocker.patch("ml_switcheroo.core.compiler.backends.sass.emitter.SassPrinter")
  instance: Any = mock_printer.return_value
  instance.emit.return_value = "    // test\n"

  result: str = emitter.emit(nodes)
  instance.emit.assert_called_once_with(nodes)
  assert result == "    // test\n"


def test_sass_emitter_integration() -> None:
  """Docstring."""
  emitter: SassEmitter = SassEmitter()
  nodes: List[SassNode] = [SassComment(text="integration")]
  result: str = emitter.emit(nodes)
  assert "integration" in result
