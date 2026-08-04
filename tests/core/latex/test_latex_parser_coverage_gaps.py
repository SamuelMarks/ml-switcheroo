"""Latex parser coverage tests."""

from ml_switcheroo.core.latex.parser import LatexParser


def test_latex_parser_coverage_gaps():
  """Test latex parser coverage gaps."""
  source = r"""
    \% Escaped comment
    % Regular comment
    \begin{DefModel}{MyModel{Nested}}
    \\ % Non-alpha macro
    \DummyMacro{layer}[[dim=10]]{in=10,out=10}
    \Attribute{nested_layer}{Linear}{in={10}, out=10}
    \end{DefModel}
    """
  parser = LatexParser(source)
  from libcst._nodes.base import CSTValidationError

  try:
    _ = parser.parse()
  except CSTValidationError:
    pass
