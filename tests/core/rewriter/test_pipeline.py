"""
Tests for the Rewriter Pipeline Infrastructure.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.rewriter.context import RewriterContext


class MockPass(RewriterPass):
  """Simple pass that injects a comment to verify execution."""

  def __init__(self, label: str) -> None:
    self.label = label

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Appends a comment to the module header."""
    header = list(module.header)
    header.append(
      cst.EmptyLine(
        comment=cst.Comment(f"# Pass: {self.label}"),
        newline=cst.Newline(),
      )
    )
    return module.with_changes(header=header)


def test_pipeline_execution_sequence() -> None:
  """
  Verify that passes are executed in the order provided.
  """
  # Setup
  ctx = MagicMock(spec=RewriterContext)
  pass1 = MockPass("A")
  pass2 = MockPass("B")
  pipeline = RewriterPipeline([pass1, pass2])

  module = cst.parse_module("x = 1")

  # Run
  result = pipeline.run(module, ctx)
  code = result.code

  # Verify Output
  assert "# Pass: A" in code
  assert "# Pass: B" in code

  # Verify Order (A before B)
  idx_a = code.find("# Pass: A")
  idx_b = code.find("# Pass: B")
  assert idx_a < idx_b


def test_pipeline_empty() -> None:
  """
  Verify pipeline works with no passes (Identity).
  """
  ctx = MagicMock(spec=RewriterContext)
  pipeline = RewriterPipeline([])
  module = cst.parse_module("x = 1")
  result = pipeline.run(module, ctx)

  assert result.code == module.code


def test_interface_enforcement() -> None:
  """
  Verify abstract base class enforcement.
  """
  with pytest.raises(TypeError):
    # Should fail if transform not implemented
    class InvalidPass(RewriterPass):
      pass

    InvalidPass()
