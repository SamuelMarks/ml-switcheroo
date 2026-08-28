"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser, RdnaTransformer
from lark import Token
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaDirective, RdnaModifier
from typing import List


def test_parser_empty_line() -> None:
  """Test element."""
  parser: RdnaParser = RdnaParser("  \n  v_add_f32 v0, v1, v2")
  parser.parse()


def test_parser_modifier() -> None:
  """Test element."""
  transformer: RdnaTransformer = RdnaTransformer()
  res: RdnaModifier = transformer.modifier([Token("MODIFIER", "row_mask:0xf")])
  assert res.name == "row_mask:0xf"


def test_parser_eof_trivia() -> None:
  """Test element."""
  parser: RdnaParser = RdnaParser("v_add_f32 v0, v1, v2 ; eof comment")
  parser.parse()


def test_param_children() -> None:
  """Test element."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaTransformer

  transformer: RdnaTransformer = RdnaTransformer()

  class DummyToken:
    def __init__(self) -> None:
      self.children: List[Token] = [Token("A", "b"), Token("B", "c")]

  # We call directive directly
  # children = [ DOT, Token("IDENTIFIER", "name"), param_list ]
  # param_list is a list of parameters
  res: RdnaDirective = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "my_dir"), [DummyToken()]])
  assert res.name == "my_dir"
  assert "bc" in getattr(res, "params")[0]
