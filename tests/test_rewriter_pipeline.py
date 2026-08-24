"""Test module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.context import RewriterContext


def test_rewriter_pipeline():
  """Test element."""
  # Setup
  pass1 = MagicMock(spec=RewriterPass)
  pass2 = MagicMock(spec=RewriterPass)
  pipeline = RewriterPipeline([pass1, pass2])

  module_mock = MagicMock(spec=cst.Module)
  context_mock = MagicMock(spec=RewriterContext)

  pass1.transform.return_value = "module_1"
  pass2.transform.return_value = "module_2"

  # Run
  result = pipeline.run(module_mock, context_mock)

  # Assert
  assert result == "module_2"
  pass1.transform.assert_called_once_with(module_mock, context_mock)
  pass2.transform.assert_called_once_with("module_1", context_mock)
