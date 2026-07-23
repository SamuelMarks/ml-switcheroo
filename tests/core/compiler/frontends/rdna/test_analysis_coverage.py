"""Tests for RDNA analysis coverage."""

from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer


def test_analyze_block_empty():
  """Test analyzing an empty block."""
  res = RdnaAnalyzer.analyze_block("Conv2d", [])
  assert res == {}
