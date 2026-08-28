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
  RdnaNode,
)
from ml_switcheroo.core.cst.base import Trivia


def with_trivia(node: RdnaNode) -> RdnaNode:
  """Test element."""
  node.leading_trivia = [Trivia(" ")]
  node.trailing_trivia = [Trivia("\n")]
  return node


def test_cst_label_ref() -> None:
  """Test element."""
  node: RdnaNode = RdnaLabelRef(name="label1")
  assert node.to_text() == "label1"
  node = with_trivia(node)
  assert node.to_text() == " label1\n"


def test_cst_sgpr() -> None:
  """Test element."""
  node: RdnaNode = RdnaSGPR(index=0)
  assert node.to_text() == "s0"
  node2: RdnaNode = RdnaSGPR(index=1, count=4)
  assert node2.to_text() == "s[1:4]"
  node3: RdnaNode = with_trivia(node2)
  assert node3.to_text() == " s[1:4]\n"
  assert c_SGPR(5).to_text() == "s5"


def test_cst_vgpr() -> None:
  """Test element."""
  node: RdnaNode = RdnaVGPR(index=2)
  assert node.to_text() == "v2"
  node2: RdnaNode = RdnaVGPR(index=2, count=3)
  assert node2.to_text() == "v[2:4]"
  node3: RdnaNode = with_trivia(node2)
  assert node3.to_text() == " v[2:4]\n"
  assert c_VGPR(3).to_text() == "v3"


def test_cst_immediate() -> None:
  """Test element."""
  node: RdnaNode = RdnaImmediate(value=42)
  assert node.to_text() == "42"
  node2: RdnaNode = RdnaImmediate(value=255, is_hex=True)
  assert node2.to_text() == "0xff"
  node3: RdnaNode = with_trivia(node2)
  assert node3.to_text() == " 0xff\n"


def test_cst_modifier() -> None:
  """Test element."""
  node: RdnaNode = RdnaModifier(name="off")
  assert node.to_text() == "off"
  node2: RdnaNode = with_trivia(node)
  assert node2.to_text() == " off\n"


def test_cst_memory() -> None:
  """Test element."""
  node: RdnaNode = RdnaMemory(base=c_VGPR(1))
  assert node.to_text() == "v1"
  node2: RdnaNode = RdnaMemory(base=c_VGPR(1), offset=12)
  assert node2.to_text() == "v1 offset:12"

  # testing base as string if needed, although base should be RdnaSGPR/RdnaVGPR
  node3: RdnaNode = RdnaMemory(base="string_base", offset=0)
  assert node3.to_text() == "string_base"

  node4: RdnaNode = with_trivia(RdnaMemory(base=c_VGPR(2), offset=4))
  assert node4.to_text() == " v2 offset:4\n"


def test_cst_instruction() -> None:
  """Test element."""
  node: RdnaNode = RdnaInstruction(opcode="v_add_f32", operands=[c_VGPR(0), c_VGPR(1), c_VGPR(2)])
  assert node.to_text() == "v_add_f32 v0, v1, v2"

  # instruction with leading trivia on second operand
  op2: RdnaNode = c_VGPR(1)
  op2.leading_trivia = [Trivia(" ")]
  node2: RdnaNode = RdnaInstruction(opcode="v_add_f32", operands=[c_VGPR(0), op2])
  assert node2.to_text() == "v_add_f32 v0 v1"

  node3: RdnaNode = with_trivia(node2)
  assert node3.to_text() == " v_add_f32 v0 v1\n"

  # invalid opcode
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    RdnaInstruction(opcode="v_add f32", operands=[])


def test_cst_label() -> None:
  """Test element."""
  node: RdnaNode = RdnaLabel(name="loop_start")
  assert node.to_text() == "loop_start:"
  node2: RdnaNode = with_trivia(node)
  assert node2.to_text() == " loop_start:\n"


def test_cst_directive() -> None:
  """Test element."""
  node: RdnaNode = RdnaDirective(name="text")
  assert node.to_text() == ".text"
  node2: RdnaNode = RdnaDirective(name="globl", params=["main"])
  assert node2.to_text() == ".globl main"
  node3: RdnaNode = with_trivia(node2)
  assert node3.to_text() == " .globl main\n"


def test_cst_comment() -> None:
  """Test element."""
  node: RdnaNode = RdnaComment(text="this is a comment")
  assert node.to_text() == "; this is a comment"
  node2: RdnaNode = with_trivia(node)
  assert node2.to_text() == " ; this is a comment\n"


def test_cst_module() -> None:
  """Test element."""
  node: RdnaNode = RdnaModule(statements=[RdnaComment(text="1"), RdnaComment(text="2")])
  assert node.to_text() == "; 1; 2"
  node2: RdnaNode = with_trivia(node)
  assert node2.to_text() == " ; 1; 2\n"
