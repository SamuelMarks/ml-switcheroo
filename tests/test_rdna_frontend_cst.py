"""Test module."""

import pytest
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaLabelRef,
  RdnaSGPR,
  RdnaVGPR,
  c_SGPR,
  c_VGPR,
  RdnaImmediate,
  RdnaModifier,
  RdnaMemory,
  RdnaInstruction,
  RdnaLabel,
  RdnaDirective,
  RdnaComment,
  RdnaModule,
)
from ml_switcheroo.core.cst.base import Trivia


def with_trivia(node):
  """Test element."""
  node.leading_trivia = [Trivia(" ")]
  node.trailing_trivia = [Trivia("\n")]
  return node


def test_cst_label_ref():
  """Test element."""
  node = RdnaLabelRef(name="label1")
  assert node.to_text() == "label1"
  node = with_trivia(node)
  assert node.to_text() == " label1\n"


def test_cst_sgpr():
  """Test element."""
  node = RdnaSGPR(index=0)
  assert node.to_text() == "s0"
  node = RdnaSGPR(index=1, count=4)
  assert node.to_text() == "s[1:4]"
  node = with_trivia(node)
  assert node.to_text() == " s[1:4]\n"
  assert c_SGPR(5).to_text() == "s5"


def test_cst_vgpr():
  """Test element."""
  node = RdnaVGPR(index=2)
  assert node.to_text() == "v2"
  node = RdnaVGPR(index=2, count=3)
  assert node.to_text() == "v[2:4]"
  node = with_trivia(node)
  assert node.to_text() == " v[2:4]\n"
  assert c_VGPR(3).to_text() == "v3"


def test_cst_immediate():
  """Test element."""
  node = RdnaImmediate(value=42)
  assert node.to_text() == "42"
  node = RdnaImmediate(value=255, is_hex=True)
  assert node.to_text() == "0xff"
  node = with_trivia(node)
  assert node.to_text() == " 0xff\n"


def test_cst_modifier():
  """Test element."""
  node = RdnaModifier(name="off")
  assert node.to_text() == "off"
  node = with_trivia(node)
  assert node.to_text() == " off\n"


def test_cst_memory():
  """Test element."""
  node = RdnaMemory(base=c_VGPR(1))
  assert node.to_text() == "v1"
  node = RdnaMemory(base=c_VGPR(1), offset=12)
  assert node.to_text() == "v1 offset:12"

  # testing base as string if needed, although base should be RdnaSGPR/RdnaVGPR
  node = RdnaMemory(base="string_base", offset=0)
  assert node.to_text() == "string_base"

  node = with_trivia(RdnaMemory(base=c_VGPR(2), offset=4))
  assert node.to_text() == " v2 offset:4\n"


def test_cst_instruction():
  """Test element."""
  node = RdnaInstruction(opcode="v_add_f32", operands=[c_VGPR(0), c_VGPR(1), c_VGPR(2)])
  assert node.to_text() == "v_add_f32 v0, v1, v2"

  # instruction with leading trivia on second operand
  op2 = c_VGPR(1)
  op2.leading_trivia = [Trivia(" ")]
  node = RdnaInstruction(opcode="v_add_f32", operands=[c_VGPR(0), op2])
  assert node.to_text() == "v_add_f32 v0 v1"

  node = with_trivia(node)
  assert node.to_text() == " v_add_f32 v0 v1\n"

  # invalid opcode
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    RdnaInstruction(opcode="v_add f32", operands=[])


def test_cst_label():
  """Test element."""
  node = RdnaLabel(name="loop_start")
  assert node.to_text() == "loop_start:"
  node = with_trivia(node)
  assert node.to_text() == " loop_start:\n"


def test_cst_directive():
  """Test element."""
  node = RdnaDirective(name="text")
  assert node.to_text() == ".text"
  node = RdnaDirective(name="globl", params=["main"])
  assert node.to_text() == ".globl main"
  node = with_trivia(node)
  assert node.to_text() == " .globl main\n"


def test_cst_comment():
  """Test element."""
  node = RdnaComment(text="this is a comment")
  assert node.to_text() == "; this is a comment"
  node = with_trivia(node)
  assert node.to_text() == " ; this is a comment\n"


def test_cst_module():
  """Test element."""
  node = RdnaModule(statements=[RdnaComment(text="1"), RdnaComment(text="2")])
  assert node.to_text() == "; 1; 2"
  node = with_trivia(node)
  assert node.to_text() == " ; 1; 2\n"
