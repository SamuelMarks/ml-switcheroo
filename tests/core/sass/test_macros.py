import pytest
from unittest.mock import MagicMock

from ml_switcheroo.compiler.backends.sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.compiler.frontends.sass.nodes import Instruction, Label, Register, Comment, Predicate


class MockAllocator:
  """Mock implementation of RegisterAllocatorProtocol."""

  def __init__(self) -> None:
    self.counter = 10

  def get_register(self, var_name: str) -> Register:
    if var_name == "output":
      return Register("R0")
    return Register("R_VAR")

  def allocate_temp(self) -> Register:
    name = f"R{self.counter}"
    self.counter += 1
    return Register(name)


def test_expand_conv2d_structure() -> None:
  """
  Verify Conv2d generates nested loops and FFMA logic.
  """
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "conv1", {"k": 3})

  # 1. Check Labels
  labels = [n for n in nodes if isinstance(n, Label)]
  assert len(labels) == 2
  assert labels[0].name == "L_KY_conv1"
  assert labels[1].name == "L_KX_conv1"

  # 2. Check Instructions
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]

  assert "MOV" in opcodes
  assert "IMAD" in opcodes
  assert "IADD3" in opcodes
  assert "LDG.E.F32" in opcodes
  assert opcodes.count("LDG.E.F32") >= 2
  assert "FFMA" in opcodes
  assert "ISETP.LT.AND" in opcodes
  assert "BRA" in opcodes

  # 3. Check Predicate Usage
  branches = [n for n in nodes if isinstance(n, Instruction) and n.opcode == "BRA"]
  assert branches[0].predicate.name == "P0"


def test_expand_linear_structure() -> None:
  """
  Verify Linear generates GEMM loop.
  """
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 512, "bias": True})

  labels = [n for n in nodes if isinstance(n, Label)]
  assert len(labels) == 1
  assert labels[0].name == "L_GEMM_fc1"

  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]
  assert "LDG.E.F32" in opcodes
  assert "FFMA" in opcodes
  assert "IADD3" in opcodes
  assert "FADD" in opcodes


def test_expand_linear_no_bias() -> None:
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 128})
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]
  assert "FADD" not in opcodes
  assert "FFMA" in opcodes


def test_comment_generation() -> None:
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "layer1", {})
  text = [n.text for n in nodes if isinstance(n, Comment)]
  assert "BEGIN Conv2d (layer1)" in text
  assert "END Conv2d (layer1)" in text
