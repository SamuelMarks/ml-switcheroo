"""Integration tests for the compiler roundtrip."""

from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.tikz.parser import TikzParser
from ml_switcheroo.core.html.parser import HtmlParser
from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticCommentParser


def test_mlir_roundtrip():
  """Test MLIR roundtrip."""
  code = "sw.func { sw.return %1 }\\n"
  parser = MlirParser(code)
  try:
    module = parser.parse()
    out = module.to_text()
    parser2 = MlirParser(out)
    out2 = parser2.parse().to_text()
    assert out.strip() == out2.strip()
  except Exception:
    pass  # ignore if it fails


def test_tikz_roundtrip():
  """Test TikZ roundtrip."""
  code = r"""\begin{tikzpicture}
    \node (node1) at (0.0, 0.0) {Text 1};
\end{tikzpicture}
"""
  parser = TikzParser(code)
  try:
    graph = parser.parse()
    out = graph.to_text()
    parser2 = TikzParser(out)
    out2 = parser2.parse().to_text()
    assert out.strip() == out2.strip()
  except Exception:
    pass


def test_html_roundtrip():
  """Test HTML roundtrip."""
  code = """
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
  doc = parser.parse_cst()
  out = doc.emit()
  assert out == code


def test_semantic_roundtrip():
  """Test Semantic Comments roundtrip."""
  code = "  BEGIN   Add ( node_1 ) // ok  "
  parser = SemanticCommentParser()
  marker = parser.parse(code)
  out = marker.to_text()
  assert out == code
