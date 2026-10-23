"""Test suite for frontend semantic parser."""

from lark import Token
import pytest

from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticBegin,
  SemanticCommentParser,
  SemanticEnd,
  SemanticInput,
  SemanticMarker,
  SemanticReturn,
  SemanticUnmapped,
  Trivia,
  _opt_trivia,
)


def test_trivia() -> None:
  """Verify trivia behavior."""
  t = Trivia("  ")
  assert t.to_text() == "  "


def test_semantic_marker_base() -> None:
  """Verify semantic marker base behavior."""
  marker = SemanticMarker()
  with pytest.raises(NotImplementedError):
    marker.to_text()


def test_opt_trivia() -> None:
  """Verify opt trivia behavior."""
  assert _opt_trivia(None) is None
  res = _opt_trivia(Token("TRIVIA", " test "))
  assert res is not None
  assert res.text == " test "


class TestSemanticParser:
  """Test helper class TestSemanticParser."""

  @pytest.fixture
  def parser(self) -> SemanticCommentParser:
    """Execute parser helper logic.

    Returns:
        Configured `parser` fixture value (SemanticCommentParser).
    """
    return SemanticCommentParser()

  def test_parse_input(self, parser: SemanticCommentParser) -> None:
    """Verify parse input behavior.

    Args:
        parser: Parser instance under test.
    """
    text = "  Input input_name -> something"
    marker = parser.parse(text)
    assert isinstance(marker, SemanticInput)
    assert marker.name == "input_name"
    assert marker.leading_trivia == "  "
    assert marker.kw_input_trivia is not None
    assert marker.kw_input_trivia.text == " "
    assert marker.name_trivia is not None
    assert marker.name_trivia.text == " "
    assert marker.tail is not None
    assert marker.tail.text == " something"
    assert marker.to_text() == text

  def test_parse_begin(self, parser: SemanticCommentParser) -> None:
    """Verify parse begin behavior.

    Args:
        parser: Parser instance under test.
    """
    text = "BEGIN loop ( id_1 ) "
    marker = parser.parse(text)
    assert isinstance(marker, SemanticBegin)
    assert marker.kind == "loop"
    assert marker.id == "id_1"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_end(self, parser: SemanticCommentParser) -> None:
    """Verify parse end behavior.

    Args:
        parser: Parser instance under test.
    """
    text = "\tEND loop ( id_1 )\n"
    marker = parser.parse(text)
    assert isinstance(marker, SemanticEnd)
    assert marker.kind == "loop"
    assert marker.id == "id_1"
    assert marker.leading_trivia == "\t"
    assert marker.to_text() == text

  def test_parse_unmapped(self, parser: SemanticCommentParser) -> None:
    """Verify parse unmapped behavior.

    Args:
        parser: Parser instance under test.
    """
    text = "Unmapped Op: jnp.add ( id_42 )"
    marker = parser.parse(text)
    assert isinstance(marker, SemanticUnmapped)
    assert marker.api == "jnp.add"
    assert marker.id == "id_42"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_unmapped_tail(self, parser: SemanticCommentParser) -> None:
    """Verify parse unmapped tail behavior.

    Args:
        parser: Parser instance under test.
    """
    text = "Unmapped Op: jnp.add ( id_42 ) trailing"
    marker = parser.parse(text)
    assert isinstance(marker, SemanticUnmapped)
    assert marker.api == "jnp.add"
    assert marker.id == "id_42"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_return(self, parser: SemanticCommentParser) -> None:
    """Verify parse return behavior.

    Args:
        parser: Parser instance under test.
    """
    text = "Return: output_1, output_2"
    marker = parser.parse(text)
    assert isinstance(marker, SemanticReturn)
    assert marker.tail is not None
    assert marker.tail.text == " output_1, output_2"
    assert marker.leading_trivia == ""
    assert marker.to_text() == text

  def test_parse_invalid(self, parser: SemanticCommentParser) -> None:
    """Verify parse invalid behavior.

    Args:
        parser: Parser instance under test.
    """
    assert parser.parse("invalid string") is None
    assert parser.parse("Input") is None
    assert parser.parse("BEGIN loop ( )") is None
    assert parser.parse("END ()") is None
    assert parser.parse("Unmapped Op: (id_1)") is None

  def test_parse_exception_handled(self, parser: SemanticCommentParser, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify parse exception handled behavior.

    Args:
        parser: Parser instance under test.
        monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
    """
    monkeypatch.setattr(parser.parser, "parse", lambda x: 1 / 0)
    assert parser.parse("BEGIN loop ( id_1 ) ") is None


def test_to_text_with_no_optional_trivia() -> None:
  """Verify to text with no optional trivia behavior."""
  marker_input = SemanticInput(name="inp")
  assert marker_input.to_text() == "Inputinp->"

  marker_begin = SemanticBegin(kind="k", id="i")
  assert marker_begin.to_text() == "BEGINk(i)"

  marker_end = SemanticEnd(kind="k", id="i")
  assert marker_end.to_text() == "ENDk(i)"

  marker_unmapped = SemanticUnmapped(api="api", id="i")
  assert marker_unmapped.to_text() == "UnmappedOp:api(i)"

  marker_return = SemanticReturn()
  assert marker_return.to_text() == "Return:"
