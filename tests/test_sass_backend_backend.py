"""Test module."""

from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment


def test_sass_backend_init():
  """Test element."""
  backend = SassBackend()
  assert backend.synthesizer is not None
  assert backend.emitter is not None


def test_sass_backend_init_with_semantics():
  """Test element."""
  sem = SemanticsManager()
  backend = SassBackend(semantics=sem)
  assert backend.synthesizer.semantics == sem


def test_sass_backend_compile(mocker):
  """Test element."""
  backend = SassBackend()
  graph = LogicalGraph()
  # mock synthesizer and emitter
  mock_synthesizer = mocker.patch.object(backend.synthesizer, "from_graph")
  mock_emitter = mocker.patch.object(backend.emitter, "emit")

  nodes = [SassComment(text="test")]
  mock_synthesizer.return_value = nodes
  mock_emitter.return_value = "    // test"

  result = backend.compile(graph)
  mock_synthesizer.assert_called_once_with(graph)
  mock_emitter.assert_called_once_with(nodes)
  assert result == "    // test"
