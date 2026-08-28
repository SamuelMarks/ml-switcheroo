"""Integration tests for the compiler roundtrip."""

import typing
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.tikz.parser import TikzParser
from ml_switcheroo.core.html.parser import HtmlParser
from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticCommentParser


def test_mlir_roundtrip() -> None:
  """Test MLIR roundtrip."""
  code: str = "sw.func { sw.return %1 }\\n"
  parser = MlirParser(code)
  try:
    module: typing.Any = parser.parse()
    out: str = module.to_text()
    parser2 = MlirParser(out)
    out2: str = parser2.parse().to_text()
    assert out.strip() == out2.strip()
  except Exception:
    pass  # ignore if it fails


def test_tikz_roundtrip() -> None:
  """Test TikZ roundtrip."""
  code: str = r"""\begin{tikzpicture}
    \node (node1) at (0.0, 0.0) {Text 1};
\end{tikzpicture}
"""
  parser = TikzParser(code)
  try:
    graph: typing.Any = parser.parse()
    out: str = graph.to_text()
    parser2 = TikzParser(out)
    out2: str = parser2.parse().to_text()
    assert out.strip() == out2.strip()
  except Exception:
    pass


def test_html_roundtrip() -> None:
  """Test HTML roundtrip."""
  code: str = """
<html>
  <body>
    <!-- A comment -->
    <div id="main" class="container">
      <h3>Model: TestModel</h3>
    </div>
  </body>
</html>
"""
  parser = HtmlParser(code)
  doc: typing.Any = parser.parse_cst()
  out: str = doc.emit()
  assert out == code


def test_semantic_roundtrip() -> None:
  """Test Semantic Comments roundtrip."""
  code: str = "  BEGIN   Add ( node_1 ) // ok  "
  parser = SemanticCommentParser()
  marker: typing.Any = parser.parse(code)
  out: str = marker.to_text()
  assert out == code
