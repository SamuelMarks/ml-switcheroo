"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser, RdnaTransformer
from lark import Token


def test_parser_empty_line():
  """Test element."""
  parser = RdnaParser("  \n  v_add_f32 v0, v1, v2")
  parser.parse()


def test_parser_modifier():
  """Test element."""
  transformer = RdnaTransformer()
  res = transformer.modifier([Token("MODIFIER", "row_mask:0xf")])
  assert res.name == "row_mask:0xf"


def test_parser_eof_trivia():
  """Test element."""
  parser = RdnaParser("v_add_f32 v0, v1, v2 ; eof comment")
  parser.parse()


def test_param_children():
  """Test element."""
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaTransformer

  transformer = RdnaTransformer()

  class DummyToken:
    def __init__(self):
      self.children = [Token("A", "b"), Token("B", "c")]

  # We call directive directly
  # children = [ DOT, Token("IDENTIFIER", "name"), param_list ]
  # param_list is a list of parameters
  res = transformer.directive([Token("DOT", "."), Token("IDENTIFIER", "my_dir"), [DummyToken()]])
  assert res.name == "my_dir"
  assert "bc" in res.params[0]
