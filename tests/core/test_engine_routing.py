"""Test suite for the Engine Routing module."""

import typing

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def mock_managers() -> typing.Any:
  """Execute mock managers helper logic.

  Returns:
      Computed test result or fixture value.
  """
  sem: typing.Any = MagicMock(spec=SemanticsManager)
  sem.get_framework_config.return_value = {}
  return sem


def test_routing_to_compiler_pipeline_target_isa(mock_managers: MagicMock) -> None:
  """Verifies the behavior of routing to compiler pipeline target isa.

  Args:
      mock_managers: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass")
  engine: typing.Any = ASTEngine(mock_managers, config)
  engine._run_compiler_pipeline = MagicMock()
  engine._run_rewriter_pipeline = MagicMock()
  engine.run("code")
  engine._run_compiler_pipeline.assert_called_once()
  engine._run_rewriter_pipeline.assert_not_called()


def test_routing_to_compiler_pipeline_source_isa(mock_managers: MagicMock) -> None:
  """Verifies the behavior of routing to compiler pipeline source isa.

  Args:
      mock_managers: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(source_framework="rdna", target_framework="torch")
  engine: typing.Any = ASTEngine(mock_managers, config)
  engine._run_compiler_pipeline = MagicMock()
  engine._run_rewriter_pipeline = MagicMock()
  engine.run("code")
  engine._run_compiler_pipeline.assert_called_once()
  engine._run_rewriter_pipeline.assert_not_called()


def test_routing_to_rewriter_pipeline_high_level(mock_managers: MagicMock) -> None:
  """Verifies the behavior of routing to rewriter pipeline high level.

  Args:
      mock_managers: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine: typing.Any = ASTEngine(mock_managers, config)
  engine._run_compiler_pipeline = MagicMock()
  engine._run_rewriter_pipeline = MagicMock()
  engine.run("code")
  engine._run_rewriter_pipeline.assert_called_once()
  engine._run_compiler_pipeline.assert_not_called()


def test_python_frontend_invoked_in_compiler_pipeline(mock_managers: MagicMock) -> None:
  """Verifies the behavior of python frontend invoked in compiler pipeline.

  Args:
      mock_managers: Test argument or fixture.
  """
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="nvidia_sass")
  engine: typing.Any = ASTEngine(mock_managers, config)
  with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFront:
    MockFront.return_value.parse_to_graph.return_value = LogicalGraph()
    with patch("ml_switcheroo.core.engine.get_backend_class") as mock_get_backend:
      mock_get_backend.return_value = MagicMock()
      engine.run("x=1")
      MockFront.assert_called_with("x=1")
