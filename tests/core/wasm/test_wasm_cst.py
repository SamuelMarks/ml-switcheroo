"""Test suite for the WASM CST module."""

import typing

import pytest

from ml_switcheroo.core.wasm.cst import WatFunc, WatInstr, WatLocal, WatModule, WatNode, WatParam, WatParser, WatResult


def test_wat_node_base() -> None:
  """Docstring."""
  node = WatNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_wat_param() -> None:
  """Docstring."""
  param = WatParam("x", "f32")
  assert param.to_text() == "(param $x f32)"


def test_wat_result() -> None:
  """Docstring."""
  result = WatResult("f32")
  assert result.to_text() == "(result f32)"


def test_wat_local() -> None:
  """Docstring."""
  local = WatLocal("temp", "f32")
  assert local.to_text() == "(local $temp f32)"


def test_wat_instr() -> None:
  """Docstring."""
  instr1 = WatInstr("f32.add")
  assert instr1.to_text() == "f32.add"

  instr2 = WatInstr("local.get", ["$x"])
  assert instr2.to_text(indent=1) == "  local.get $x"


def test_wat_func() -> None:
  """Docstring."""
  func = WatFunc(
    name="add_two",
    export=True,
    params=[WatParam("a", "f32"), WatParam("b", "f32")],
    results=[WatResult("f32")],
    locals=[WatLocal("c", "f32")],
    body=[
      WatInstr("local.get", ["$a"]),
      WatInstr("local.get", ["$b"]),
      WatInstr("f32.add"),
      WatInstr("local.set", ["$c"]),
      WatInstr("local.get", ["$c"]),
    ],
  )
  text = func.to_text()
  assert '(func $add_two (export "add_two") (param $a f32) (param $b f32) (result f32)' in text
  assert "  (local $c f32)" in text
  assert "  local.get $a" in text
  assert "  f32.add" in text


def test_wat_module() -> None:
  """Docstring."""
  mod = WatModule(functions=[WatFunc(name="main", body=[WatInstr("nop")])])
  text = mod.to_text()
  assert "(module\n" in text
  assert "  (func $main\n" in text
  assert "    nop\n" in text
  assert "  )\n" in text
  assert ")\n" in text


def test_wat_parser_roundtrip() -> None:
  """Docstring."""
  from ml_switcheroo.core.wasm.cst import WatParser

  text = """(module
  (func $main (export "main") (param $x i32) (param $y f32) (result f32)
    (local $z f32)
    local.get $x
    local.get $y
    f32.add
    local.set $z
    local.get $z
  )
)
"""
  parser = WatParser(text)
  mod = parser.parse()
  assert mod.to_text() == text


# --- Merged from test_wasm_cst_extra.py ---


def test_wasm_parser_empty_peek_consume() -> None:
  """Docstring."""
  parser = WatParser("")
  assert parser._peek() == ""
  with pytest.raises(ValueError, match="Unexpected EOF"):
    parser._consume()


def test_wasm_parser_consume_mismatch() -> None:
  """Docstring."""
  parser = WatParser("(module)")
  with pytest.raises(ValueError, match="Expected func, got module"):
    parser._consume("(")
    parser._consume("func")


def test_wasm_parser_skip_unrecognized_blocks() -> None:
  """Docstring."""
  text: str = '(module (import "env" "memory" (memory 1)) (func))'
  parser = WatParser(text)
  mod: typing.Any = parser.parse()
  assert len(mod.functions) == 1


def test_wasm_parser_func_unrecognized_block() -> None:
  """Docstring."""
  text: str = "(module (func (unrecognized_stuff) nop))"
  parser = WatParser(text)
  mod: typing.Any = parser.parse()
  assert len(mod.functions[0].body) == 2  # unrecognized_stuff and nop


def test_wasm_parser_func_sexpr_instruction() -> None:
  """Docstring."""
  text: str = "(module (func (local.get $x)))"
  parser = WatParser(text)
  mod: typing.Any = parser.parse()
  assert mod.functions[0].body[0].opcode == "local.get"
  assert mod.functions[0].body[0].args == ["$x"]


def test_wasm_parser_comments() -> None:
  """Docstring."""
  text: str = "(module ;; comment here\n (func))"
  parser = WatParser(text)
  mod: typing.Any = parser.parse()
  assert len(mod.functions) == 1


def test_wasm_parser_param_local_no_name() -> None:
  """Docstring."""
  text: str = "(module (func (param f32) (local i32) nop))"
  parser = WatParser(text)
  mod: typing.Any = parser.parse()
  assert mod.functions[0].params[0].type_id == "f32"
  assert mod.functions[0].params[0].name == ""
  assert mod.functions[0].locals[0].type_id == "i32"
  assert mod.functions[0].locals[0].name == ""


def test_wasm_parser_fallback_token() -> None:
  """Docstring."""
  text: str = "(module @)"
  parser = WatParser(text)
  assert "@" in parser.tokens


def test_wasm_parser_bad_string() -> None:
  """Docstring."""
  # an unclosed string might hit the fallback logic in _tokenize
  # but re.match handles the pattern, if it fails, it's not a match.
  text: str = '(module "unclosed)'
  parser = WatParser(text)
  assert '"' in parser.tokens
