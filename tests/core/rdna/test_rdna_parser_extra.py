"""Test suite for the RDNA Parser Extra module."""

from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser


def test_rdna_parser_missing():
  """Verifies the behavior of RDNA parser missing."""
  parser = RdnaParser(".text\n.global main")
  parser.parse()
  parser = RdnaParser("v_add_f32 v0, v1, v2\n.text")
  parser.parse()


def test_rdna_parser_comments_and_empty():
  """Tests comments and empty lines."""
  parser = RdnaParser("; comment\n  \n;  \n")
  nodes = parser.parse()
  assert len(nodes) == 2
  assert nodes[0].text == "comment"


def test_rdna_parser_directive_params():
  """Tests directives with multiple params."""
  parser = RdnaParser(".reqntid 128, 1, 1\n")
  nodes = parser.parse()
  assert nodes[0].name == "reqntid"
  assert nodes[0].params == ["128", "1", "1"]


def test_rdna_parser_memory_and_labels():
  """Tests memory and labels."""
  parser = RdnaParser("label_target:\nv_add_f32 v0, v1, v2")
  nodes = parser.parse()
  assert nodes[0].name == "label_target"
  assert nodes[1].opcode == "v_add_f32"


def test_rdna_parser_registers():
  """Tests registers."""
  parser = RdnaParser("s_mov_b32 s0, s[1:2]\nv_mov_b32 v0, v[1:2]")
  nodes = parser.parse()
  assert nodes[0].operands[0].index == 0
  assert nodes[0].operands[1].index == 1
  assert nodes[0].operands[1].count == 2
  assert nodes[1].operands[0].index == 0
  assert nodes[1].operands[1].index == 1
  assert nodes[1].operands[1].count == 2


def test_rdna_parser_immediates_and_modifiers():
  """Tests immediate and register variations."""
  parser = RdnaParser("v_add_f32 v0, -1.5, -0x10\nglobal_load_dword v0, v[1:2], off, offset:0x10")
  nodes = parser.parse()
  assert nodes[0].operands[1].value == -1.5
  assert nodes[0].operands[2].value == -16
  assert nodes[0].operands[2].is_hex is True
  assert nodes[1].operands[2].name == "off"
  assert nodes[1].operands[3].name == "offset:0x10"


def test_rdna_parser_infinite_loop_prevention():
  """Tests malformed memory to hit while loops, and unparseable operands to hit consume(1)."""
  parser = RdnaParser("v_add_f32 v0, - \n")
  nodes = parser.parse()
  assert nodes[0].operands[1].name == "-"  # the '-' gets consumed as a generic LabelRef
  assert RdnaParser("/").parse()[0].opcode == ""


def test_rdna_parser_empty_and_semicolon():
  """Tests edge cases for early returns."""
  assert RdnaParser("").parse() == []
  assert RdnaParser("   \n ").parse() == []
  # cover line 266 (peek returns None at EOF)
  assert RdnaParser("c").parse()[0].opcode == "c"
  assert RdnaParser("/").parse()[0].opcode == ""


def test_rdna_parser_instruction_break_conditions():
  """Tests breaking instruction parsing early."""
  parser = RdnaParser("v_add_f32 v0, v1 ; comment here\n")
  nodes = parser.parse()
  assert nodes[0].opcode == "v_add_f32"
  assert len(nodes[0].operands) == 2
  assert RdnaParser("v_add_f32 v0, \n").parse()[0].opcode == "v_add_f32"
