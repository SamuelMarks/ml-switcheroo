"""Test suite for nvidia sass backend backend."""

from unittest.mock import patch

from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassNode
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.semantics.manager import SemanticsManager


def test_nvidia_sass_backend_init() -> None:
  """Verify `NvidiaSassBackend` handles nvidia SASS backend init as expected."""
  backend: NvidiaSassBackend = NvidiaSassBackend()
  assert backend.synthesizer is not None
  assert backend.emitter is not None
  assert isinstance(backend.synthesizer.semantics, SemanticsManager)
  emitted: str = backend.emitter.emit([NvidiaSassComment(text="init")])
  assert "// init" in emitted
  assert [line.strip() for line in emitted.splitlines() if line.strip()][-1] == "// init"


def test_nvidia_sass_backend_init_with_semantics() -> None:
  """Verify `SemanticsManager` handles nvidia SASS backend init with semantics as expected."""
  sem: SemanticsManager = SemanticsManager()
  backend: NvidiaSassBackend = NvidiaSassBackend(semantics=sem)
  assert backend.synthesizer.semantics == sem


def test_nvidia_sass_backend_compile() -> None:
  """Verify `NvidiaSassBackend` handles nvidia SASS backend compile as expected."""
  backend: NvidiaSassBackend = NvidiaSassBackend()
  graph: LogicalGraph = LogicalGraph()

  nodes: list[NvidiaSassNode] = [NvidiaSassComment(text="test")]
  with (
    patch.object(backend.synthesizer, "from_graph", return_value=nodes) as mock_synthesizer,
    patch.object(backend.emitter, "emit", return_value="    // test") as mock_emitter,
  ):
    result: str = backend.compile(graph)
    mock_synthesizer.assert_called_once_with(graph)
    mock_emitter.assert_called_once_with(nodes)
    assert result == "    // test"


def test_nvidia_sass_backend_validate_instruction() -> None:
  """Test validate_instruction method of NvidiaSassBackend."""
  backend = NvidiaSassBackend()
  report = backend.validate_instruction("FADD")
  assert report is not None

  with patch("ml_ecosystem_snapshots.grounding.hardware.validate_sass_instruction", side_effect=RuntimeError("Failed")):
    assert backend.validate_instruction("FADD") is None
