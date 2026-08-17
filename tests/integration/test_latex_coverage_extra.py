"""Module docstring."""

from ml_switcheroo.core.latex.parser import LatexParser
from unittest.mock import patch


def test_latex_parser_missing_branches_clean():
  """Docstring."""
  src1 = r"""
\begin{Unknown}
\end{Unknown}
\%escaped
\begin{DefModel}
\end{DefModel}
"""
  parser = LatexParser(src1)
  parser.parse()

  src2 = r"""
\UnknownMacroOutside{A}
\begin{DefModel}{MyModel{WithBraces}}
\Op{out}{valid_name}{{"nested": {1: 2}}}{config}
\UnknownMacro[opt[nested]]{{Nested{Inside}Name}}
\end{DefModel}
"""
  parser2 = LatexParser(src2)
  with patch.object(parser2, "_synthesize_class", return_value=None):
    parser2.parse()
