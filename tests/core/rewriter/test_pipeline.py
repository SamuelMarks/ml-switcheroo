"""Test suite for the Pipeline module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.rewriter.context import RewriterContext


class MockPass(RewriterPass):
  """Mock Pass class for testing purposes."""

  def __init__(self, label: str) -> None:
    """Initializes the MockPass instance."""
    self.label = label

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Mock implementation of transform."""
    header = list(module.header)
    header.append(cst.EmptyLine(comment=cst.Comment(f"# Pass: {self.label}"), newline=cst.Newline()))
    return module.with_changes(header=header)


def test_pipeline_execution_sequence() -> None:
  """Verifies the behavior of pipeline execution sequence."""
  ctx = MagicMock(spec=RewriterContext)
  pass1 = MockPass("A")
  pass2 = MockPass("B")
  pipeline = RewriterPipeline([pass1, pass2])
  module = cst.parse_module("x = 1")
  result = pipeline.run(module, ctx)
  code = result.code
  assert "# Pass: A" in code
  assert "# Pass: B" in code
  idx_a = code.find("# Pass: A")
  idx_b = code.find("# Pass: B")
  assert idx_a < idx_b


def test_pipeline_empty() -> None:
  """Verifies the behavior of pipeline empty."""
  ctx = MagicMock(spec=RewriterContext)
  pipeline = RewriterPipeline([])
  module = cst.parse_module("x = 1")
  result = pipeline.run(module, ctx)
  assert result.code == module.code


def test_interface_enforcement() -> None:
  """Verifies the behavior of interface enforcement."""
  with pytest.raises(TypeError):

    class InvalidPass(RewriterPass):
      """Test suite for the Invalid Pass component."""

      pass

    InvalidPass()
