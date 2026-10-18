"""Test suite for the Pipeline module."""

import typing

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline


class MockPass(RewriterPass):
  """Test helper class MockPass."""

  def __init__(self, label: str) -> None:
    """Initializes the MockPass instance.

    Args:
        label: Test argument or fixture.
    """
    self.label = label

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Mock implementation of transform.

    Args:
        module: Test argument or fixture.
        context: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    header: typing.Any = list(module.header)
    header.append(cst.EmptyLine(comment=cst.Comment(f"# Pass: {self.label}"), newline=cst.Newline()))
    return module.with_changes(header=header)


def test_pipeline_execution_sequence() -> None:
  """Verifies the behavior of pipeline execution sequence."""
  ctx: typing.Any = MagicMock(spec=RewriterContext)
  pass1: typing.Any = MockPass("A")
  pass2: typing.Any = MockPass("B")
  pipeline: typing.Any = RewriterPipeline([pass1, pass2])
  module: typing.Any = cst.parse_module("x = 1")
  result: typing.Any = pipeline.run(module, ctx)
  code: typing.Any = result.code
  assert "# Pass: A" in code
  assert "# Pass: B" in code
  idx_a: typing.Any = code.find("# Pass: A")
  idx_b: typing.Any = code.find("# Pass: B")
  assert idx_a < idx_b


def test_pipeline_empty() -> None:
  """Verifies the behavior of pipeline empty."""
  ctx: typing.Any = MagicMock(spec=RewriterContext)
  pipeline: typing.Any = RewriterPipeline([])
  module: typing.Any = cst.parse_module("x = 1")
  result: typing.Any = pipeline.run(module, ctx)
  assert result.code == module.code


def test_interface_enforcement() -> None:
  """Verifies the behavior of interface enforcement."""
  with pytest.raises(TypeError):

    class InvalidPass(RewriterPass):
      """Test helper class InvalidPass."""

      pass

    cls: typing.Any = InvalidPass
    cls()


def test_interface_base_call() -> None:
  """Verifies calling RewriterPass.transform directly executes base method."""
  ctx: typing.Any = MagicMock(spec=RewriterContext)
  pass_inst: typing.Any = MockPass("Base")
  res: typing.Any = RewriterPass.transform(pass_inst, cst.parse_module("x = 1"), ctx)
  assert res is None
