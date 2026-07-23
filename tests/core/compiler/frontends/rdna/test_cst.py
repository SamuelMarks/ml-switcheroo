"""Tests for RDNA CST functionality."""

from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.nodes import VGPR


def test_cst_basic_trivia():
  """Verifies that simple leading whitespace and trailing comments are preserved."""
  code = "    v_add_f32 v0, v1, v2 ; simple add\n"
  parser = RdnaParser(code)
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
  ; Header
  .rodata

  v_mov_b32     v0,  v1
"""
  parser = RdnaParser(code)
  nodes = parser.parse()

  res = "".join(str(n) for n in nodes)
  assert res == code


def test_cst_ast_manipulation():
  """Verifies modifying operands retains surrounding trivia."""
  code = "    v_mov_b32  v0, v1  ; move"
  parser = RdnaParser(code)
  nodes = parser.parse()

  instr = nodes[0]
  # Change v1 to v2
  old_trivia = instr.operands[1].leading_trivia
  instr.operands[1] = VGPR(index=2, count=1)
  instr.operands[1].leading_trivia = old_trivia

  res = "".join(str(n) for n in nodes)
  assert res == "    v_mov_b32  v0, v2  ; move"
