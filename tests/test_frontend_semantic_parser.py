"""Test module."""

import pytest
from typing import Optional
from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  Trivia,
  SemanticMarker,
  SemanticInput,
  SemanticBegin,
  SemanticEnd,
  SemanticUnmapped,
  SemanticReturn,
  SemanticCommentParser,
  _opt_trivia,
)


def test_trivia() -> None:
  """Test element."""
  t: Trivia = Trivia("  ")
  assert t.to_text() == "  "


def test_semantic_marker_base() -> None:
  """Test element."""
  marker: SemanticMarker = SemanticMarker()
  with pytest.raises(NotImplementedError):
    marker.to_text()


def test_opt_trivia() -> None:
  """Test element."""
  assert _opt_trivia(None) is None
  res: Optional[Trivia] = _opt_trivia(" test ")
  assert res is not None
  assert res.text == " test "


class TestSemanticParser:
  """Test element."""

  @pytest.fixture
  def parser(self) -> SemanticCommentParser:
    """Test element."""
    return SemanticCommentParser()

  def test_parse_input(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    text: str = "  Input input_name -> something"
    marker: Optional[SemanticMarker] = parser.parse(text)
    assert isinstance(marker, SemanticInput)
    assert marker.name == "input_name"
    assert marker.leading_trivia == "  "
    assert marker.kw_input_trivia.text == " "
    assert marker.name_trivia.text == " "
    assert marker.tail.text == " something"
    assert marker.to_text() == text

  def test_parse_begin(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    text: str = "BEGIN loop ( id_1 ) "
    marker: Optional[SemanticMarker] = parser.parse(text)
    assert isinstance(marker, SemanticBegin)
    assert marker.kind == "loop"
    assert marker.id == "id_1"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_end(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    text: str = "\tEND loop ( id_1 )\n"
    marker: Optional[SemanticMarker] = parser.parse(text)
    assert isinstance(marker, SemanticEnd)
    assert marker.kind == "loop"
    assert marker.id == "id_1"
    assert marker.leading_trivia == "\t"
    assert marker.to_text() == text

  def test_parse_unmapped(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    text: str = "Unmapped Op: jnp.add ( id_42 )"
    marker: Optional[SemanticMarker] = parser.parse(text)
    assert isinstance(marker, SemanticUnmapped)
    assert marker.api == "jnp.add"
    assert marker.id == "id_42"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_unmapped_tail(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    text: str = "Unmapped Op: jnp.add ( id_42 ) trailing"
    marker: Optional[SemanticMarker] = parser.parse(text)
    assert isinstance(marker, SemanticUnmapped)
    assert marker.api == "jnp.add"
    assert marker.id == "id_42"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_return(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    text: str = "Return: output_1, output_2"
    marker: Optional[SemanticMarker] = parser.parse(text)
    assert isinstance(marker, SemanticReturn)
    assert marker.tail.text == " output_1, output_2"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_invalid(self, parser: SemanticCommentParser) -> None:
    """Test element."""
    assert parser.parse("invalid string") is None
    assert parser.parse("Input") is None
    assert parser.parse("BEGIN loop ( )") is None
    assert parser.parse("END ()") is None
    assert parser.parse("Unmapped Op: (id_1)") is None

  def test_parse_exception_handled(self, parser: SemanticCommentParser, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test element."""
    monkeypatch.setattr(parser.parser, "parse", lambda x: 1 / 0)
    assert parser.parse("BEGIN loop ( id_1 ) ") is None


def test_to_text_with_no_optional_trivia() -> None:
  """Test element."""
  marker_input: SemanticInput = SemanticInput(name="inp")
  assert marker_input.to_text() == "Inputinp->"

  marker_begin: SemanticBegin = SemanticBegin(kind="k", id="i")
  assert marker_begin.to_text() == "BEGINk(i)"

  marker_end: SemanticEnd = SemanticEnd(kind="k", id="i")
  assert marker_end.to_text() == "ENDk(i)"

  marker_unmapped: SemanticUnmapped = SemanticUnmapped(api="api", id="i")
  assert marker_unmapped.to_text() == "UnmappedOp:api(i)"

  marker_return: SemanticReturn = SemanticReturn()
  assert marker_return.to_text() == "Return:"
