"""Tests for SASS CST functionality."""

from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.nodes import Register


def test_cst_basic_trivia():
  """Verifies that simple leading whitespace and trailing comments are preserved."""
  code = "    FADD R0, R1, R2; // simple add\n"
  parser = SassParser(code)
  nodes = parser.parse()

  assert len(nodes) == 2  # Instruction and Comment
  instr = nodes[0]
  assert instr.leading_trivia == "    "
  # Comment is parsed as separate node
  comment = nodes[1]
  assert comment.leading_trivia == " "
  assert comment.trailing_trivia == "\n"

  res = "".join(str(n) for n in nodes)
  assert res == code


def test_cst_complex_formatting():
  """Verifies multi-line comments, aligned operands, and empty lines."""
  code = """
  // Header
  .section

  @P0 MOV     R0,  R1 ;
"""
  parser = SassParser(code)
  nodes = parser.parse()

  res = "".join(str(n) for n in nodes)
  assert res == code


def test_cst_ast_manipulation():
  """Verifies modifying operands retains surrounding trivia."""
  code = "    MOV  R0, R1 ; // move"
  parser = SassParser(code)
  nodes = parser.parse()

  instr = nodes[0]
  old_trivia = instr.operands[1].leading_trivia
  # Change R1 to R2
  instr.operands[1] = Register(name="R2")
  instr.operands[1].leading_trivia = old_trivia

  res = "".join(str(n) for n in nodes)
  assert res == "    MOV  R0, R2 ; // move"
