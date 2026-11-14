"""Unit tests for `SemanticCommentParser` (`ml_switcheroo.core.compiler.frontends.semantic_parser`)."""

from unittest.mock import Mock

import pytest
from lark import Token

import ml_switcheroo.core.compiler.frontends.semantic_parser as sp_mod
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
  _SemanticTransformer,
)
from ml_switcheroo.exceptions import SyntaxParseError


def test_semantic_markers_trivia_roundtrips_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticCommentParser` roundtrips all marker types, trivialess defaults, and error fallbacks.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert Trivia("  ").to_text() == "  "
  assert _opt_trivia(None) is None
  tok_triv = _opt_trivia(Token("TRIVIA", " test "))
  assert tok_triv is not None
  assert tok_triv.text == " test "

  with pytest.raises(NotImplementedError):
    SemanticMarker().to_text()

  # Trivialess defaults
  assert SemanticInput(name="inp").to_text() == "Inputinp->"
  assert SemanticBegin(kind="k", id="i").to_text() == "BEGINk(i)"
  assert SemanticEnd(kind="k", id="i").to_text() == "ENDk(i)"
  assert SemanticUnmapped(api="api", id="i").to_text() == "UnmappedOp:api(i)"
  assert SemanticReturn().to_text() == "Return:"

  monkeypatch.setattr(sp_mod, "_CACHED_PARSER", None)
  parser = SemanticCommentParser()
  _ = SemanticCommentParser()

  with pytest.raises(SyntaxParseError, match="Expected SemanticMarker"):
    _SemanticTransformer().marker_start([None, Token("BAD", "x")])

  inp = parser.parse("  Input input_name -> something")
  assert isinstance(inp, SemanticInput)
  assert inp.name == "input_name"

  beg = parser.parse("BEGIN Conv2d(block1)")
  assert isinstance(beg, SemanticBegin)
  assert (beg.kind, beg.id) == ("Conv2d", "block1")

  end = parser.parse(f"{chr(9)}END loop ( id_1 ){chr(10)}")
  assert isinstance(end, SemanticEnd)
  assert (end.kind, end.id) == ("loop", "id_1")

  unm = parser.parse(" Unmapped Op: torch.nn.functional.relu(node2) ")
  assert isinstance(unm, SemanticUnmapped)
  assert (unm.api, unm.id) == ("torch.nn.functional.relu", "node2")

  ret = parser.parse("Return: output_1, output_2")
  assert isinstance(ret, SemanticReturn)
  assert ret.tail is not None

  roundtrip_cases = (
    "Input x ->",
    "  Input my_var -> R0  ",
    "BEGIN Add (node_1)",
    " BEGIN   Add ( node_1 ) // ok  ",
    "END Add (node_1) // comment",
    "  END  Mul ( node_2 ) ",
    "Unmapped Op: torch.add (node_2)",
    "   Unmapped Op:  torch.add  ( node_3 ) trailing",
    "Return:",
    " Return: R0 ",
  )
  for case in roundtrip_cases:
    parsed = parser.parse(case)
    assert parsed is not None
    assert parsed.to_text() == case

  for invalid in ("Invalid comment", "Input", "BEGIN Conv2d block1)", "BEGIN loop ( )", "END ()", "Unmapped Op: (id_1)"):
    assert parser.parse(invalid) is None

  monkeypatch.setattr(parser.parser, "parse", Mock(side_effect=RuntimeError("boom")))
  assert parser.parse("BEGIN loop ( id_1 ) ") is None
