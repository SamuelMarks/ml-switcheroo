"""Tests for HTML parser."""

from ml_switcheroo.core.html.parser import HtmlParser, InternalHtmlParser, GridExtractor


def test_html_parser_unmatched_endtag():
  """Test unmatched end tag parsing."""
  parser = InternalHtmlParser()
  parser.feed("</div>")
  assert len(parser.root_children) == 0


def test_html_parser_h3_non_text_child():
  """Test h3 non-text child."""
  html = "<h3><span>Not a text node directly</span>Not Model</h3>"
  parser = HtmlParser(source=html)
  doc = parser.parse_cst()
  assert doc.model_name == "Model"
  extractor = GridExtractor()
  extractor.extract(doc)
  assert extractor.model_name == "Model"


def test_html_parser_div_non_class_attr():
  """Test div with non-class attribute."""
  html = "<div id='test' class=''>Content</div>"
  parser = HtmlParser(source=html)
  doc = parser.parse_cst()
  assert len(doc.children) > 0
  extractor = GridExtractor()
  extractor.extract(doc)


def test_html_parser_process_box_non_text_child():
  """Test processing box with non-text child."""
  html = "<div class='box'><span><b>Bold</b></span><code><i>Italic</i></code><p>Ignored</p></div>"
  parser = HtmlParser(source=html)
  doc = parser.parse_cst()
  assert len(doc.children) > 0
  extractor = GridExtractor()
  extractor.extract(doc)
  assert len(extractor.attrs) == 0  # No valid header/code extracted because they were non-text
