"""Test module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline


def test_rewriter_pipeline() -> None:
  """Docstring."""
  # Setup
  pass1: MagicMock = MagicMock(spec=RewriterPass)
  pass2: MagicMock = MagicMock(spec=RewriterPass)
  pipeline: RewriterPipeline = RewriterPipeline([pass1, pass2])

  module_mock: MagicMock = MagicMock(spec=cst.Module)
  context_mock: MagicMock = MagicMock(spec=RewriterContext)

  pass1.transform.return_value = "module_1"
  pass2.transform.return_value = "module_2"

  # Run
  result: cst.CSTNode = pipeline.run(module_mock, context_mock)

  # Assert
  assert result == "module_2"
  pass1.transform.assert_called_once_with(module_mock, context_mock)
  pass2.transform.assert_called_once_with("module_1", context_mock)
