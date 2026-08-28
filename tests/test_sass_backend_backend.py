"""Test module."""

from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment, SassNode
from typing import Any, List


def test_sass_backend_init() -> None:
  """Test element."""
  backend: SassBackend = SassBackend()
  assert getattr(backend, "synthesizer") is not None
  assert getattr(backend, "emitter") is not None


def test_sass_backend_init_with_semantics() -> None:
  """Test element."""
  sem: SemanticsManager = SemanticsManager()
  backend: SassBackend = SassBackend(semantics=sem)
  assert backend.synthesizer.semantics == sem


def test_sass_backend_compile(mocker: Any) -> None:
  """Test element."""
  backend: SassBackend = SassBackend()
  graph: LogicalGraph = LogicalGraph()
  # mock synthesizer and emitter
  mock_synthesizer: Any = mocker.patch.object(backend.synthesizer, "from_graph")
  mock_emitter: Any = mocker.patch.object(backend.emitter, "emit")

  nodes: List[SassNode] = [SassComment(text="test")]
  mock_synthesizer.return_value = nodes
  mock_emitter.return_value = "    // test"

  result: str = backend.compile(graph)
  mock_synthesizer.assert_called_once_with(graph)
  mock_emitter.assert_called_once_with(nodes)
  assert result == "    // test"
