"""Test parser coverage."""

from ml_switcheroo.core.tikz.parser import TikzParser


def test_parser_error():
  """Test parser error handling."""
  parser = TikzParser(r"\begin{tikzpicture} \invalid \end{tikzpicture}")
  try:
    parser.parse()
    assert False
  except ValueError as e:
    assert "Failed to parse TikZ" in str(e)
