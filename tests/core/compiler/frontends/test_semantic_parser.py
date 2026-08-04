"""Test suite for semantic comment parser."""

from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticCommentParser,
  SemanticInput,
  SemanticBegin,
  SemanticEnd,
  SemanticUnmapped,
  SemanticReturn,
  SemanticMarker,
  Trivia,
)
import pytest


def test_trivia_to_text() -> None:
  """Test Trivia to_text method."""
  t = Trivia("foo")
  assert t.text == "foo"


def test_semantic_marker_abstract() -> None:
  """Test abstract method."""
  m = SemanticMarker()
  with pytest.raises(NotImplementedError):
    m.to_text()


def test_semantic_parser_input() -> None:
  """Test parsing of an input semantic comment."""
  parser = SemanticCommentParser()
  res = parser.parse("Input my_var -> R0")
  assert isinstance(res, SemanticInput)
  assert res.name == "my_var"


def test_semantic_parser_begin() -> None:
  """Test parsing of a begin semantic comment."""
  parser = SemanticCommentParser()
  res = parser.parse("BEGIN Add (node_1)")
  assert isinstance(res, SemanticBegin)
  assert res.kind == "Add"
  assert res.id == "node_1"


def test_semantic_parser_end() -> None:
  """Test parsing of an end semantic comment."""
  parser = SemanticCommentParser()
  res = parser.parse("END Add (node_1) // comment")
  assert isinstance(res, SemanticEnd)
  assert res.kind == "Add"
  assert res.id == "node_1"


def test_semantic_parser_unmapped() -> None:
  """Test parsing of an unmapped semantic comment."""
  parser = SemanticCommentParser()
  res = parser.parse("Unmapped Op: torch.add (node_2)")
  assert isinstance(res, SemanticUnmapped)
  assert res.api == "torch.add"
  assert res.id == "node_2"


def test_semantic_parser_return() -> None:
  """Test parsing of a return semantic comment."""
  parser = SemanticCommentParser()
  res = parser.parse("Return: R0")
  assert isinstance(res, SemanticReturn)


def test_semantic_parser_invalid() -> None:
  """Test parsing of an invalid semantic comment."""
  parser = SemanticCommentParser()
  res = parser.parse("Just a regular comment")
  assert res is None


def test_semantic_parser_to_text_roundtrip() -> None:
  """Test roundtrip of to_text."""
  parser = SemanticCommentParser()
  cases = [
    "Input my_var -> R0",
    "  Input my_var -> R0  ",
    "BEGIN Add (node_1)",
    " BEGIN   Add ( node_1 ) // ok  ",
    "END Add (node_1) // comment",
    "  END  Mul ( node_2 ) ",
    "Unmapped Op: torch.add (node_2)",
    "   Unmapped Op:  torch.add  ( node_3 ) ",
    "Return: R0",
    " Return: R0 ",
  ]
  for c in cases:
    res = parser.parse(c)
    assert res is not None, f"Failed to parse {c}"
    assert res.to_text() == c, f"Roundtrip failed for {c}: {res.to_text()} != {c}"
