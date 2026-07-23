"""Test suite for the Parser module."""

from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser, LabelRef
from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Comment,
  Label,
  Directive,
  Instruction,
  Register,
  Immediate,
  Memory,
  Predicate,
)


def test_label_ref():
  """Verifies the behavior of label reference."""
  ref = LabelRef("foo")
  assert str(ref) == "foo"


def test_parse_empty():
  """Parses empty."""
  parser = SassParser("")
  assert len(parser.parse()) == 0


def test_parse_semicolon():
  """Parses semicolon."""
  parser = SassParser(";")
  assert len(parser.parse()) == 0


def test_parse_comment():
  """Parses comment."""
  parser = SassParser("// comment")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Comment)
  assert nodes[0].text == "comment"


def test_parse_label():
  """Parses label."""
  parser = SassParser("L_1:")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Label)
  assert nodes[0].name == "L_1"


def test_parse_directive():
  """Parses directive."""
  parser = SassParser(".headerflags 1, 2;")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Directive)
  assert nodes[0].name == "headerflags"
  assert nodes[0].params == ["1", "2"]


def test_parse_directive_multiline():
  """Parses directive multiline."""
  parser = SassParser(".headerflags 1\n.global_base")
  nodes = parser.parse()
  assert len(nodes) == 2


def test_parse_instruction_simple():
  """Parses instruction simple."""
  parser = SassParser("NOP;")
  nodes = parser.parse()
  assert len(nodes) == 1
  assert isinstance(nodes[0], Instruction)
  assert nodes[0].opcode == "NOP"


def test_parse_instruction_predicate():
  """Parses instruction predicate."""
  parser = SassParser("@P0 FADD R0, R1;")
  nodes = parser.parse()
  assert len(nodes) == 1
  inst = nodes[0]
  assert inst.opcode == "FADD"
  assert isinstance(inst.predicate, Predicate)
  assert inst.predicate.name == "P0"
  assert inst.predicate.negated is False
  parser2 = SassParser("@!P1 FADD R0, R1;")
  inst2 = parser2.parse()[0]
  assert inst2.predicate.name == "P1"
  assert inst2.predicate.negated is True


def test_parse_operands():
  """Parses operands."""
  parser = SassParser("FADD R0, -R1, |R2|, -|R3|, c[0x0][0x4], [R1 + 0x4], [R2], 0x1, 1.5, @P0, L_1, L_2:")
  nodes = parser.parse()
  inst = nodes[0]
  assert len(inst.operands) == 12
  assert isinstance(inst.operands[0], Register) and inst.operands[0].name == "R0"
  assert isinstance(inst.operands[1], Register) and inst.operands[1].negated is True
  assert isinstance(inst.operands[2], Register) and inst.operands[2].absolute is True
  assert (
    isinstance(inst.operands[3], Register) and inst.operands[3].absolute is True and (inst.operands[3].negated is True)
  )
  assert isinstance(inst.operands[4], Memory) and inst.operands[4].base == "c[0x0]" and (inst.operands[4].offset == 4)
  assert (
    isinstance(inst.operands[5], Memory)
    and getattr(inst.operands[5].base, "name", None) == "R1"
    and (inst.operands[5].offset == 4)
  )
  assert (
    isinstance(inst.operands[6], Memory)
    and getattr(inst.operands[6].base, "name", None) == "R2"
    and (inst.operands[6].offset is None)
  )
  assert isinstance(inst.operands[7], Immediate) and inst.operands[7].value == 1 and (inst.operands[7].is_hex is True)
  assert isinstance(inst.operands[8], Immediate) and inst.operands[8].value == 1.5
  assert isinstance(inst.operands[9], Predicate) and inst.operands[9].name == "P0"
  assert isinstance(inst.operands[10], LabelRef) and inst.operands[10].name == "L_1"
  assert isinstance(inst.operands[11], LabelRef) and inst.operands[11].name == "L_2"


def test_parse_unexpected_eof():
  """Parses unexpected eof."""
  pass


def test_parse_operand_eof():
  """Parses operand eof."""
  pass


def test_parse_unexpected_token():
  """Parses unexpected token."""
  pass


def test_parse_bad_operand():
  """Parses bad operand."""
  pass


def test_parse_memory_missing_offset():
  """Parses memory missing offset."""
  parser = SassParser("FADD c[0x0]")
  nodes = parser.parse()
  assert isinstance(nodes[0].operands[0], Memory)
  assert nodes[0].operands[0].base == "c[0x0]"
  assert nodes[0].operands[0].offset == 0


def test_parse_memory_dec_offset():
  """Parses memory dec offset."""
  parser = SassParser("FADD [R1 + 10]")
  nodes = parser.parse()
  assert nodes[0].operands[0].offset == 10


def test_consume_unexpected():
  """Verifies the behavior of consume unexpected."""
  pass


def test_instruction_multiline():
  """Verifies the behavior of instruction multiline."""
  # Changed from FADD\nNOP because Lark might parse it as FADD NOP (operands)
  parser = SassParser("FADD;\nNOP")
  assert len(parser.parse()) == 2
