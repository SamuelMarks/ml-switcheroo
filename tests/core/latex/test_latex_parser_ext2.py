"""Latex parser coverage tests."""

from ml_switcheroo.core.latex.parser import LatexParser


def test_latex_parser_coverage_gaps() -> None:
  """Verify latex parser coverage gaps behavior."""
  source: str = r"""
    \% Escaped comment
    % Regular comment
    \begin{DefModel}{MyModel{Nested}}
    \\ % Non-alpha macro
    \DummyMacro{layer}[[dim=10]]{in=10,out=10}
    \Attribute{nested_layer}{Linear}{in={10}, out=10}
    \end{DefModel}
    """
  parser = LatexParser(source)
  import pytest
  from libcst._nodes.base import CSTValidationError

  with pytest.raises(CSTValidationError):
    parser.parse()

  parser_skip = LatexParser("\\begin{DefModel}{ValidModel}\x00\\Return{x}\\end{DefModel}")
  mod = parser_skip.parse()
  assert "class ValidModel" in mod.code

  from ml_switcheroo.core.markup.parser import RecursiveDescentParser, Token

  tok = Token("TEST", "val")
  assert repr(tok) == "Token(TEST, 'val')"
  rd = RecursiveDescentParser([tok])
  assert rd.match("OTHER") is False
  assert rd.match("TEST") is True
  assert rd.advance() is None
  with pytest.raises(ValueError, match="Expected TEST"):
    rd.expect("TEST")
