"""Integration tests for the compiler roundtrip."""

from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticCommentParser
from ml_switcheroo.core.html.parser import HtmlParser
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.tikz.parser import TikzParser


def test_mlir_roundtrip() -> None:
  """Verify mlir roundtrip behavior."""
  code: str = "sw.func { sw.return %1 }\n"
  parser = MlirParser(code)
  module = parser.parse()
  out: str = module.to_text()
  parser2 = MlirParser(out)
  out2: str = parser2.parse().to_text()
  assert out.strip() == out2.strip()


def test_tikz_roundtrip() -> None:
  """Verify tikz roundtrip behavior."""
  code: str = "\\begin{tikzpicture}\n    \\node (node1) at (0.0, 0.0) {Text 1};\n\\end{tikzpicture}\n"
  parser = TikzParser(code)
  graph = parser.parse()
  out: str = graph.to_text()
  parser2 = TikzParser(out)
  out2: str = parser2.parse().to_text()

  # Normalize whitespace before asserting to ignore indentation issues
  def normalize(s: str) -> str:
    """Execute normalize test helper.

    Args:
        s: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return "\\n".join(line.strip() for line in s.strip().splitlines())

  assert normalize(out) == normalize(out2)


def test_html_roundtrip() -> None:
  """Verify html roundtrip behavior."""
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
  doc = parser.parse_cst()
  out: str = doc.emit()
  assert out == code


def test_semantic_roundtrip() -> None:
  """Verify semantic roundtrip behavior."""
  code: str = "  BEGIN   Add ( node_1 ) // ok  "
  parser = SemanticCommentParser()
  marker = parser.parse(code)
  out: str = marker.to_text()  # type: ignore[union-attr]
  assert out == code
