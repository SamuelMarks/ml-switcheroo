import pytest
from ml_switcheroo.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.compiler.frontends.rdna.nodes import (
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
from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.compiler.frontends.rdna.tokens import RdnaLexer

from ml_switcheroo.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.compiler.frontends.sass.nodes import (
  Register,
  Predicate,
  Immediate as SassImmediate,
  Memory as SassMemory,
  Instruction as SassInstruction,
  Label as SassLabel,
  Directive as SassDirective,
  Comment as SassComment,
)
from ml_switcheroo.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.compiler.frontends.sass.tokens import SassLexer


def test_rdna_nodes():
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

  dir1 = SassDirective("text")
  dir2 = SassDirective("headerflags")
  assert str(dir2) == ".headerflags"
  dir3 = SassDirective("section", ["0x1"])
  assert str(dir3) == ".section 0x1"

  comment = SassComment("hello")
  assert str(comment) == "// hello"


def test_rdna_lexer_parser():
  code = """
    ; test code
    .text
    .amdgcn_target amdgcn
    L1:
    s_mov_b32 s0, 0x1
    v_add_f32 v0, v[1:2], 1.5
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_cmp_lt_i32 s0, 10
    s_cmp_lt_i32 s1, 15
    v_mov_b32 v0, exec
    s_branch L1
    """

  parser = RdnaParser(code)
  nodes = parser.parse()
  assert len(nodes) > 0

  # Coverage for syntax errors
  with pytest.raises(SyntaxError, match="Unexpected token at line 1: 0x1"):
    RdnaParser("0x1").parse()

  with pytest.raises(SyntaxError, match="Unknown operand type"):
    # Put something that doesn't match expected operand types
    RdnaParser("s_mov_b32 :").parse()

  with pytest.raises(ValueError):
    list(RdnaLexer().tokenize("@bad"))

  # We want to trigger unexpected EOF in _parse_operand:
  # `s_mov_b32 ` doesn't trigger it because EOF breaks the instruction loop.
  # What triggers it? A token that makes it enter operand parsing but then fails?
  # No, _parse_operand does `token = self._peek()`, if not token `raise SyntaxError`.
  # How to reach `not token` in `_parse_operand`?
  # If the instruction parses an operand but EOF is hit...
  # Wait, the while loop in `_parse_instruction` has `peek = self._peek()`, `if not peek: break`.
  # So `_parse_operand` is never called when `peek` is None!
  # Except if `_parse_register_range` or something consumes tokens and hits EOF.
  with pytest.raises(SyntaxError, match="Unexpected End of File"):
    RdnaParser("s_mov_b32 s[").parse()


def test_rdna_analysis():
  insts = [
    Instruction("s_cmp_lt_i32", [c_SGPR(0), Immediate(3)]),
    Instruction("s_cmp_lt_i32", [c_SGPR(1), Immediate(5)]),
  ]
  meta = RdnaAnalyzer.analyze_block("Conv2d", insts)
  assert meta["k"] == 5

  meta2 = RdnaAnalyzer.analyze_block("Linear", insts)
  assert meta2["in_features"] == 5

  meta3 = RdnaAnalyzer.analyze_block("Other", insts)
  assert meta3 == {}

  meta_empty = RdnaAnalyzer.analyze_block("Conv2d", [])
  assert meta_empty == {}


def test_sass_lexer_parser():
  code = """
    // comment
    .headerflags
    .section .text;
    L1:
    @P0 FADD R0, R1, 0x1;
    @!P0 ISETP.LT.AND P1, PT, R0, 10, PT;
    IADD3 RZ, R0, 0x0, RZ;
    MOV R1, c[0x0][0x4];
    MOV R1, c[0x0];
    LDG R1, [R2 + 0x8];
    LDG R1, [R2];
    BRA L1;
    """

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
    # to hit this, we must call _parse_operand directly at EOF
    parser = SassParser("")
    parser._parse_operand()


def test_sass_analysis():
  insts = [
    SassInstruction(
      "ISETP.LT.AND", [Predicate("P1"), Predicate("PT"), Register("R0"), SassImmediate(7), Predicate("PT")]
    ),
  ]
  meta = SassAnalyzer.analyze_block("Conv2d", insts)
  assert meta["kernel_size"] == 7

  meta2 = SassAnalyzer.analyze_block("Linear", insts)
  assert meta2["in_features"] == 7

  meta_empty = SassAnalyzer.analyze_block("Conv2d", [])
  assert meta_empty == {}


def test_rdna_lifter():
  code = """
    ; Input x ->
    ; BEGIN Conv2d (n1)
    s_cmp_lt_i32 s0, 3
    ; END Conv2d (n1)
    ; Unmapped Op: torch.flatten (n2)
    ; Return:
    """
  nodes = RdnaParser(code).parse()
  graph = RdnaLifter().lift(nodes)
  assert "x" in [n.id for n in graph.nodes]
  assert "n1" in [n.id for n in graph.nodes]
  assert "n2" in [n.id for n in graph.nodes]
  assert "output" in [n.id for n in graph.nodes]
  # To cover empty seen_ids branch in commit_node
  # we can pass another Return
  nodes.append(Comment("Return:"))
  graph2 = RdnaLifter().lift(nodes)
  assert "output" in [n.id for n in graph2.nodes]


def test_sass_lifter():
  code = """
    // Input x ->
    // BEGIN Conv2d (n1)
    ISETP.LT.AND P1, PT, R0, 3, PT;
    // END Conv2d (n1)
    // Unmapped Op: torch.flatten (n2)
    FADD R0, R1, R2;
    // Return:
    """
  nodes = SassParser(code).parse()
  graph = SassLifter().lift(nodes)
  assert "x" in [n.id for n in graph.nodes]
  assert "n1" in [n.id for n in graph.nodes]
  assert "n2" in [n.id for n in graph.nodes]
  assert "output" in [n.id for n in graph.nodes]


def test_sass_parser_unreachable():
  from ml_switcheroo.compiler.frontends.sass.parser import SassParser

  # 106: _parse_line with no token
  parser = SassParser("")
  assert parser._parse_line() is None

  # 109-110: lonely semicolon
  parser = SassParser(";")
  assert parser._parse_line() is None

  # 152-153: directive param on next line
  parser = SassParser(".headerflags\nparam")
  # Actually wait, `\nparam` won't parse as param because line number is higher.
  # It will break out of the while loop and leave `param` for the next parse.
  assert len(parser.parse()) == 2


def test_sass_parser_directives():
  from ml_switcheroo.compiler.frontends.sass.parser import SassParser

  # 155-159: directive with multiple params on same line
  parser = SassParser(".headerflags param1, param2;")
  nodes = parser.parse()
  assert len(nodes[0].params) == 2

  # 144: next_t is None
  # this happens when file ends abruptly while parsing directive params
  # However the tokenizer usually yields an EOF token?
  # No, EOF is not a token in this lexer (unless we checked).
  # Wait, _is_eof() is `pos >= len(tokens)`.
  # Let's test just EOF inside directive.
  parser2 = SassParser(".headerflags ")
  parser2.parse()


def test_sass_parser_consume():
  from ml_switcheroo.compiler.frontends.sass.parser import SassParser
  from ml_switcheroo.compiler.frontends.sass.tokens import TokenType

  parser = SassParser("")
  with pytest.raises(SyntaxError, match="Unexpected End of File"):
    parser._consume()

  parser = SassParser("FADD")
  with pytest.raises(SyntaxError, match="Expected"):
    parser._consume(TokenType.COMMA)


def test_sass_parser_predicate_operand():
  from ml_switcheroo.compiler.frontends.sass.parser import SassParser

  # 223-225: predicate as operand
  # @P0 FADD P1, R1, R2; -> wait, predicate in instruction usually is a predicate token.
  parser = SassParser("@P1")
  op = parser._parse_operand()
  assert op.__class__.__name__ == "Predicate"


def test_sass_parser_str():
  from ml_switcheroo.compiler.frontends.sass.parser import LabelRef

  n = LabelRef(name="test")
  assert str(n) == "test"


def test_sass_lifter_duplicates():
  from ml_switcheroo.compiler.frontends.sass.lifter import SassLifter
  from ml_switcheroo.compiler.frontends.sass.parser import SassParser

  # create duplicate BEGIN to trigger line 60
  code = """
    // Unmapped Op: torch.nn.Linear (n1)
    // Unmapped Op: torch.nn.Linear (n1)
    """
  nodes = SassParser(code).parse()
  graph = SassLifter().lift(nodes)
  assert len(graph.nodes) == 1


def test_sass_node_abstract():
  from ml_switcheroo.compiler.frontends.sass.nodes import SassNode

  class DummySassNode(SassNode):
    def __str__(self):
      return super().__str__() or "test"

  assert str(DummySassNode()) == "test"


def test_sass_token_identifier_fallback():
  from ml_switcheroo.compiler.frontends.sass.tokens import SassLexer, TokenType

  # line 130-134
  # "UR" starts with "UR", but doesn't have a digit, and isn't "RZ" or "PT".
  tokens = list(SassLexer().tokenize("ZFOO"))
  assert tokens[0].kind == TokenType.IDENTIFIER


def test_rdna_parser_consume():
  from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser
  from ml_switcheroo.compiler.frontends.rdna.tokens import TokenType

  parser = RdnaParser("v_add_f32")
  with pytest.raises(SyntaxError, match="Expected"):
    parser._consume(TokenType.COMMA)


def test_rdna_parser_eof_checks():
  from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser

  # 86: _parse_line with no token
  parser = RdnaParser("")
  assert parser._parse_line() is None

  # 164: _parse_operand at EOF
  with pytest.raises(SyntaxError, match="Unexpected EOF expecting operand"):
    parser = RdnaParser("")
    parser._parse_operand()


def test_rdna_lifter_duplicates_and_unmapped():
  from ml_switcheroo.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser

  code = """
    ; Unmapped Op: torch.add (a)
    ; Unmapped Op: torch.add (a)
    v_add_f32 v0, v1, v2
    """
  nodes = RdnaParser(code).parse()
  graph = RdnaLifter().lift(nodes)
  # The duplicate 'a' will be skipped at line 52.
  # The `v_add_f32` will be processed without a block id at lines 116-120.
  assert len(graph.nodes) > 0


def test_rdna_node_abstract():
  from ml_switcheroo.compiler.frontends.rdna.nodes import RdnaNode, Operand

  class DummyRdnaNode(RdnaNode):
    def __str__(self):
      return super().__str__() or "rdna"

  assert str(DummyRdnaNode()) == "rdna"

  class DummyOp(Operand):
    pass

  assert str(DummyOp()) == ""
