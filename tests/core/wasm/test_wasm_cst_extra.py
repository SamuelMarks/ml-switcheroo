"""Test module."""

import pytest
from ml_switcheroo.core.wasm.cst import WatParser


def test_wasm_parser_empty_peek_consume():
  """Test function."""
  parser = WatParser("")
  assert parser._peek() == ""
  with pytest.raises(ValueError, match="Unexpected EOF"):
    parser._consume()


def test_wasm_parser_consume_mismatch():
  """Test function."""
  parser = WatParser("(module)")
  with pytest.raises(ValueError, match="Expected func, got module"):
    parser._consume("(")
    parser._consume("func")


def test_wasm_parser_skip_unrecognized_blocks():
  """Test function."""
  text = '(module (import "env" "memory" (memory 1)) (func))'
  parser = WatParser(text)
  mod = parser.parse()
  assert len(mod.functions) == 1


def test_wasm_parser_func_unrecognized_block():
  """Test function."""
  text = "(module (func (unrecognized_stuff) nop))"
  parser = WatParser(text)
  mod = parser.parse()
  assert len(mod.functions[0].body) == 2  # unrecognized_stuff and nop


def test_wasm_parser_func_sexpr_instruction():
  """Test function."""
  text = "(module (func (local.get $x)))"
  parser = WatParser(text)
  mod = parser.parse()
  assert mod.functions[0].body[0].opcode == "local.get"
  assert mod.functions[0].body[0].args == ["$x"]


def test_wasm_parser_comments():
  """Test function."""
  text = "(module ;; comment here\n (func))"
  parser = WatParser(text)
  mod = parser.parse()
  assert len(mod.functions) == 1


def test_wasm_parser_param_local_no_name():
  """Test function."""
  text = "(module (func (param f32) (local i32) nop))"
  parser = WatParser(text)
  mod = parser.parse()
  assert mod.functions[0].params[0].type_id == "f32"
  assert mod.functions[0].params[0].name == ""
  assert mod.functions[0].locals[0].type_id == "i32"
  assert mod.functions[0].locals[0].name == ""


def test_wasm_parser_fallback_token():
  """Test function."""
  text = "(module @)"
  parser = WatParser(text)
  assert "@" in parser.tokens


def test_wasm_parser_bad_string():
  """Test function."""
  # an unclosed string might hit the fallback logic in _tokenize
  # but re.match handles the pattern, if it fails, it's not a match.
  text = '(module "unclosed)'
  parser = WatParser(text)
  assert '"' in parser.tokens
