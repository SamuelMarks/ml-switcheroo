"""Test module."""

from typing import Any, List
from unittest.mock import patch

from ml_switcheroo.core.compiler.backends.nvidia_sass.emitter import NvidiaSassEmitter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassNode


def test_nvidia_sass_emitter_emit() -> None:
  """Docstring."""
  emitter: NvidiaSassEmitter = NvidiaSassEmitter()
  nodes: List[NvidiaSassNode] = [NvidiaSassComment(text="test")]
  with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.emitter.NvidiaSassPrinter") as mock_printer:
    instance: Any = mock_printer.return_value
    instance.emit.return_value = "    // test\n"

    result: str = emitter.emit(nodes)
    instance.emit.assert_called_once_with(nodes)
    assert result == "    // test\n"


def test_nvidia_sass_emitter_integration() -> None:
  """Docstring."""
  emitter: NvidiaSassEmitter = NvidiaSassEmitter()
  nodes: List[NvidiaSassNode] = [NvidiaSassComment(text="integration")]
  result: str = emitter.emit(nodes)
  assert "integration" in result
