"""Test suite for the WASM CST module."""

import pytest
from ml_switcheroo.core.wasm.cst import (
  WatNode,
  WatParam,
  WatResult,
  WatLocal,
  WatInstr,
  WatFunc,
  WatModule,
)


def test_wat_node_base() -> None:
  """Tests WatNode base class."""
  node = WatNode()
  with pytest.raises(NotImplementedError):
    node.to_text()


def test_wat_param() -> None:
  """Tests WatParam."""
  param = WatParam("x", "f32")
  assert param.to_text() == "(param $x f32)"


def test_wat_result() -> None:
  """Tests WatResult."""
  result = WatResult("f32")
  assert result.to_text() == "(result f32)"


def test_wat_local() -> None:
  """Tests WatLocal."""
  local = WatLocal("temp", "f32")
  assert local.to_text() == "(local $temp f32)"


def test_wat_instr() -> None:
  """Tests WatInstr."""
  instr1 = WatInstr("f32.add")
  assert instr1.to_text() == "f32.add"

  instr2 = WatInstr("local.get", ["$x"])
  assert instr2.to_text(indent=1) == "  local.get $x"


def test_wat_func() -> None:
  """Tests WatFunc."""
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
  """Tests WatModule."""
  mod = WatModule(functions=[WatFunc(name="main", body=[WatInstr("nop")])])
  text = mod.to_text()
  assert "(module\n" in text
  assert "  (func $main\n" in text
  assert "    nop\n" in text
  assert "  )\n" in text
  assert ")\n" in text


def test_wat_parser_roundtrip() -> None:
  """Tests parsing and roundtrip emitting."""
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
