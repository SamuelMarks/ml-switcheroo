"""Test suite for the Engine Routing module."""

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_managers() -> MagicMock:
  """Execute mock managers helper logic.

  Returns:
      Configured `mock_managers` fixture value (MagicMock).
  """
  sem = MagicMock(spec=SemanticsManager)
  sem.get_framework_config.return_value = {}
  return sem


def test_routing_to_compiler_pipeline_target_isa(mock_managers: MagicMock) -> None:
  """Verifies the behavior of routing to compiler pipeline target isa.

  Args:
      mock_managers: The mock managers (MagicMock) for test routing to compiler pipeline target isa.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass")
  engine = ASTEngine(mock_managers, config)
  setattr(engine, "_run_compiler_pipeline", MagicMock())
  setattr(engine, "_run_rewriter_pipeline", MagicMock())
  engine.run("code")
  getattr(
    {
      "v": getattr({"v": getattr({"v": engine}, "__getitem__")("v")._run_compiler_pipeline}, "__getitem__")(
        "v"
      ).assert_called_once
    },
    "__getitem__",
  )("v")()
  getattr(
    {
      "v": getattr({"v": getattr({"v": engine}, "__getitem__")("v")._run_rewriter_pipeline}, "__getitem__")(
        "v"
      ).assert_not_called
    },
    "__getitem__",
  )("v")()


def test_routing_to_compiler_pipeline_source_isa(mock_managers: MagicMock) -> None:
  """Verifies the behavior of routing to compiler pipeline source isa.

  Args:
      mock_managers: The mock managers (MagicMock) for test routing to compiler pipeline source isa.
  """
  config = RuntimeConfig(source_framework="rdna", target_framework="torch")
  engine = ASTEngine(mock_managers, config)
  setattr(engine, "_run_compiler_pipeline", MagicMock())
  setattr(engine, "_run_rewriter_pipeline", MagicMock())
  engine.run("code")
  getattr(
    {
      "v": getattr({"v": getattr({"v": engine}, "__getitem__")("v")._run_compiler_pipeline}, "__getitem__")(
        "v"
      ).assert_called_once
    },
    "__getitem__",
  )("v")()
  getattr(
    {
      "v": getattr({"v": getattr({"v": engine}, "__getitem__")("v")._run_rewriter_pipeline}, "__getitem__")(
        "v"
      ).assert_not_called
    },
    "__getitem__",
  )("v")()


def test_routing_to_rewriter_pipeline_high_level(mock_managers: MagicMock) -> None:
  """Verifies the behavior of routing to rewriter pipeline high level.

  Args:
      mock_managers: The mock managers (MagicMock) for test routing to rewriter pipeline high level.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(mock_managers, config)
  setattr(engine, "_run_compiler_pipeline", MagicMock())
  setattr(engine, "_run_rewriter_pipeline", MagicMock())
  engine.run("code")
  getattr(
    {
      "v": getattr({"v": getattr({"v": engine}, "__getitem__")("v")._run_rewriter_pipeline}, "__getitem__")(
        "v"
      ).assert_called_once
    },
    "__getitem__",
  )("v")()
  getattr(
    {
      "v": getattr({"v": getattr({"v": engine}, "__getitem__")("v")._run_compiler_pipeline}, "__getitem__")(
        "v"
      ).assert_not_called
    },
    "__getitem__",
  )("v")()


def test_python_frontend_invoked_in_compiler_pipeline(mock_managers: MagicMock) -> None:
  """Verifies the behavior of python frontend invoked in compiler pipeline.

  Args:
      mock_managers: The mock managers (MagicMock) for test python frontend invoked in compiler pipeline.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass")
  engine = ASTEngine(mock_managers, config)
  with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFront:
    MockFront.return_value.parse_to_graph.return_value = LogicalGraph()
    with patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend:
      mock_get_backend.return_value = MagicMock()
      engine.run("x=1")
      MockFront.assert_called_with("x=1")
