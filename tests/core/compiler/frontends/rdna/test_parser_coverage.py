"""Tests for RDNA parser coverage."""

from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser


def test_rdna_parser_empty_lines():
  """Test RDNA parser with empty lines."""
  parser = RdnaParser("v_mov_b32 v0, v1\n\n  \n")
  nodes = parser.parse()
  assert len(nodes) == 1
