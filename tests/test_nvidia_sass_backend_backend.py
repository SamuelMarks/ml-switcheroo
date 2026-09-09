"""Test module."""

from typing import List
from unittest.mock import patch

from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassNode
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.semantics.manager import SemanticsManager


def test_nvidia_sass_backend_init() -> None:
  """Docstring."""
  backend: NvidiaSassBackend = NvidiaSassBackend()
  assert getattr(backend, "synthesizer") is not None
  assert getattr(backend, "emitter") is not None


def test_nvidia_sass_backend_init_with_semantics() -> None:
  """Docstring."""
  sem: SemanticsManager = SemanticsManager()
  backend: NvidiaSassBackend = NvidiaSassBackend(semantics=sem)
  assert backend.synthesizer.semantics == sem


def test_nvidia_sass_backend_compile() -> None:
  """Docstring."""
  backend: NvidiaSassBackend = NvidiaSassBackend()
  graph: LogicalGraph = LogicalGraph()

  nodes: List[NvidiaSassNode] = [NvidiaSassComment(text="test")]
  with (
    patch.object(backend.synthesizer, "from_graph", return_value=nodes) as mock_synthesizer,
    patch.object(backend.emitter, "emit", return_value="    // test") as mock_emitter,
  ):
    result: str = backend.compile(graph)
    mock_synthesizer.assert_called_once_with(graph)
    mock_emitter.assert_called_once_with(nodes)
    assert result == "    // test"
