"""Test suite for the Gap module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.nodes import (
  SGPR,
  VGPR,
  Immediate,
  LabelRef,
  Label,
  Directive,
  Comment,
  Instruction,
  Memory,
  Modifier,
  c_SGPR,
  c_VGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.tokens import RdnaLexer
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.nodes import (
  Register,
  Predicate,
  Immediate as SassImmediate,
  Memory as SassMemory,
  Instruction as SassInstruction,
  Label as SassLabel,
  Directive as SassDirective,
  Comment as SassComment,
)
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.tokens import SassLexer


def test_rdna_nodes():
  """Verifies the behavior of RDNA nodes."""
  s0 = c_SGPR(0)
  assert str(s0) == "s0"
  v0 = c_VGPR(0)
  assert str(v0) == "v0"
  s_range = SGPR(0, 4)
  assert str(s_range) == "s[0:3]"
  v_range = VGPR(10, 2)
  assert str(v_range) == "v[10:11]"
  imm = Immediate(42)
  assert str(imm) == "42"
  imm_hex = Immediate(42, is_hex=True)
  assert str(imm_hex) == "0x2a"
  label_ref = LabelRef("L1")
  assert str(label_ref) == "L1"
  mod = Modifier("glc")
  assert str(mod) == "glc"
  mem1 = Memory(s0)
  assert str(mem1) == "s0"
  mem2 = Memory(v0, offset=4)
  assert str(mem2) == "v0 offset:4"
  inst1 = Instruction("s_mov_b32", [s0, imm])
  assert str(inst1) == "s_mov_b32 s0, 42"
  inst2 = Instruction("s_waitcnt", [])
  assert str(inst2) == "s_waitcnt"
  label = Label("L_START")
  assert str(label) == "L_START:"
  dir1 = Directive("text")
  assert str(dir1) == ".text"
  dir2 = Directive("global_base", ["0"])
  assert str(dir2) == ".global_base 0"
  comment = Comment("hello")
  assert str(comment) == "; hello"


def test_sass_nodes():
  """Verifies the behavior of SASS nodes."""
  r0 = Register("R0")
  assert str(r0) == "R0"
  r0_neg = Register("R0", negated=True)
  assert str(r0_neg) == "-R0"
  r0_abs = Register("R0", absolute=True)
  assert str(r0_abs) == "|R0|"
  p0 = Predicate("P0")
  assert str(p0) == "P0"
  p0_neg = Predicate("P0", negated=True)
  assert str(p0_neg) == "!P0"
  imm = SassImmediate(42)
  assert str(imm) == "42"
  imm_hex = SassImmediate(42, is_hex=True)
  assert str(imm_hex) == "0x2a"
  imm_float = SassImmediate(42.0, is_hex=True)
  assert str(imm_float) == "0x2a"
  mem1 = SassMemory("c[0x0]", offset=4)
  assert str(mem1) == "c[0x0][0x4]"
  mem2 = SassMemory("c[0x0]")
  assert str(mem2) == "c[0x0][0x0]"
  mem3 = SassMemory(r0, offset=8)
  assert str(mem3) == "[R0 + 0x8]"
  mem4 = SassMemory(r0)
  assert str(mem4) == "[R0]"
  inst1 = SassInstruction("FADD", [r0, r0], predicate=p0)
  assert str(inst1) == "@P0 FADD R0, R0;"
  inst2 = SassInstruction("NOP")
  assert str(inst2) == "NOP ;"
  label = SassLabel("L1")
  assert str(label) == "L1:"
  SassDirective("text")
  dir2 = SassDirective("headerflags")
  assert str(dir2) == ".headerflags"
  dir3 = SassDirective("section", ["0x1"])
  assert str(dir3) == ".section 0x1"
  comment = SassComment("hello")
  assert str(comment) == "// hello"


def test_rdna_lexer_parser():
  """Verifies the behavior of RDNA lexer parser."""
  code = "\n    ; test code\n    .text\n    .amdgcn_target amdgcn\n    L1:\n    s_mov_b32 s0, 0x1\n    v_add_f32 v0, v[1:2], 1.5\n    s_waitcnt vmcnt(0) lgkmcnt(0)\n    s_cmp_lt_i32 s0, 10\n    s_cmp_lt_i32 s1, 15\n    v_mov_b32 v0, exec\n    s_branch L1\n    "
  parser = RdnaParser(code)
  nodes = parser.parse()
  assert len(nodes) > 0
  with pytest.raises(SyntaxError, match="Unexpected token at line 1: 0x1"):
    RdnaParser("0x1").parse()
  with pytest.raises(SyntaxError, match="Unknown operand type"):
    RdnaParser("s_mov_b32 :").parse()
  with pytest.raises(ValueError):
    list(RdnaLexer().tokenize("@bad"))
  with pytest.raises(SyntaxError, match="Unexpected End of File"):
    RdnaParser("s_mov_b32 s[").parse()


def test_rdna_analysis():
  """Verifies the behavior of RDNA analysis."""
  insts = [Instruction("s_cmp_lt_i32", [c_SGPR(0), Immediate(3)]), Instruction("s_cmp_lt_i32", [c_SGPR(1), Immediate(5)])]
  meta = RdnaAnalyzer.analyze_block("Conv2d", insts)
  assert meta["k"] == 5
  meta2 = RdnaAnalyzer.analyze_block("Linear", insts)
  assert meta2["in_features"] == 5
  meta3 = RdnaAnalyzer.analyze_block("Other", insts)
  assert meta3 == {}
  meta_empty = RdnaAnalyzer.analyze_block("Conv2d", [])
  assert meta_empty == {}


def test_sass_lexer_parser():
  """Verifies the behavior of SASS lexer parser."""
  code = "\n    // comment\n    .headerflags\n    .section .text;\n    L1:\n    @P0 FADD R0, R1, 0x1;\n    @!P0 ISETP.LT.AND P1, PT, R0, 10, PT;\n    IADD3 RZ, R0, 0x0, RZ;\n    MOV R1, c[0x0][0x4];\n    MOV R1, c[0x0];\n    LDG R1, [R2 + 0x8];\n    LDG R1, [R2];\n    BRA L1;\n    "
  parser = SassParser(code)
  nodes = parser.parse()
  assert len(nodes) > 0
  with pytest.raises(ValueError):
    list(SassLexer().tokenize("??"))
  with pytest.raises(SyntaxError, match="Unexpected token"):
    SassParser("0x1").parse()
  with pytest.raises(SyntaxError, match="Unknown operand type"):
    SassParser("FADD .text;").parse()
  with pytest.raises(SyntaxError, match="Unexpected EOF expecting operand"):
    parser = SassParser("")
    parser._parse_operand()


def test_sass_analysis():
  """Verifies the behavior of SASS analysis."""
  insts = [
    SassInstruction("ISETP.LT.AND", [Predicate("P1"), Predicate("PT"), Register("R0"), SassImmediate(7), Predicate("PT")])
  ]
  meta = SassAnalyzer.analyze_block("Conv2d", insts)
  assert meta["kernel_size"] == 7
  meta2 = SassAnalyzer.analyze_block("Linear", insts)
  assert meta2["in_features"] == 7
  meta_empty = SassAnalyzer.analyze_block("Conv2d", [])
  assert meta_empty == {}
