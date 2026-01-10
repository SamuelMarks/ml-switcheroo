"""
Tests for ASTEngine Pipeline Routing.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.compiler.ir import LogicalGraph


@pytest.fixture
def mock_managers():
  sem = MagicMock(spec=SemanticsManager)
  sem.get_framework_config.return_value = {}
  return sem


def test_routing_to_compiler_pipeline_target_isa(mock_managers):
  """Target SASS -> Compiler Pipeline."""
  config = RuntimeConfig(source_framework="torch", target_framework="sass")
  engine = ASTEngine(mock_managers, config)
  engine._run_compiler_pipeline = MagicMock()
  engine._run_rewriter_pipeline = MagicMock()
  engine.run("code")
  engine._run_compiler_pipeline.assert_called_once()
  engine._run_rewriter_pipeline.assert_not_called()


def test_routing_to_compiler_pipeline_source_isa(mock_managers):
  """Source RDNA -> Compiler Pipeline."""
  config = RuntimeConfig(source_framework="rdna", target_framework="torch")
  engine = ASTEngine(mock_managers, config)
  engine._run_compiler_pipeline = MagicMock()
  engine._run_rewriter_pipeline = MagicMock()
  engine.run("code")
  engine._run_compiler_pipeline.assert_called_once()
  engine._run_rewriter_pipeline.assert_not_called()


def test_routing_to_rewriter_pipeline_high_level(mock_managers):
  """Torch -> JAX -> Rewriter Pipeline."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(mock_managers, config)
  engine._run_compiler_pipeline = MagicMock()
  engine._run_rewriter_pipeline = MagicMock()
  engine.run("code")
  engine._run_rewriter_pipeline.assert_called_once()
  engine._run_compiler_pipeline.assert_not_called()


def test_python_frontend_invoked(mock_managers):
  """Verify python source ingestion logic."""
  config = RuntimeConfig(source_framework="torch", target_framework="sass")
  engine = ASTEngine(mock_managers, config)
  with patch("ml_switcheroo.core.engine.PythonFrontend") as MockFront:
    MockFront.return_value.parse_to_graph.return_value = LogicalGraph()
    with patch("ml_switcheroo.core.engine.get_backend_class") as mock_get:
      mock_get.return_value = MagicMock()
      engine.run("x=1")
      MockFront.assert_called_with("x=1")
