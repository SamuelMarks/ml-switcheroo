"""
Tests for SASS Macro Expansion Logic.

Verifies that:
1.  `expand_conv2d` produces the correct nested loop structure.
2.  `expand_linear` produces a dot product loop.
3.  Register allocation inside macros works via the protocol.
4.  Instructions match expected opcodes (IMAD, FFMA, LDG).
"""

import pytest
from unittest.mock import MagicMock

from ml_switcheroo.core.sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.sass.nodes import Instruction, Label, Register, Comment, Predicate


class MockAllocator:
  """Mock implementation of RegisterAllocatorProtocol."""

  def __init__(self) -> None:
    self.counter = 10  # Start high to distinguish from R0-R9 used in snippets

  def get_register(self, var_name: str) -> Register:
    """Returns a fixed register for output, specific one for inputs."""
    if var_name == "output":
      return Register("R0")
    return Register("R_VAR")

  def allocate_temp(self) -> Register:
    """Allocates R10, R11..."""
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

  # Initialization
  assert "MOV" in opcodes

  # Address Calc
  assert "IMAD" in opcodes
  assert "IADD3" in opcodes

  # Memory
  assert "LDG.E.F32" in opcodes
  assert opcodes.count("LDG.E.F32") >= 2  # Image + Weight

  # Math
  assert "FFMA" in opcodes

  # Branching
  assert "ISETP.LT.AND" in opcodes
  assert "BRA" in opcodes

  # 3. Check Predicate Usage
  branches = [n for n in nodes if isinstance(n, Instruction) and n.opcode == "BRA"]
  assert branches[0].predicate.name == "P0"


def test_expand_linear_structure() -> None:
  """
  Verify Linear generates GEMM loop and pointer increments.
  """
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 512, "bias": True})

  # 1. Label
  labels = [n for n in nodes if isinstance(n, Label)]
  assert len(labels) == 1
  assert labels[0].name == "L_GEMM_fc1"

  # 2. Key Instructions
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]

  # Core loop
  assert "LDG.E.F32" in opcodes
  assert "FFMA" in opcodes

  # Ptr Increment
  assert "IADD3" in opcodes

  # Bias Addition (enabled in metadata)
  assert "FADD" in opcodes


def test_expand_linear_no_bias() -> None:
  """
  Verify Bias logic is skipped if metadata says so.
  """
  alloc = MockAllocator()
  # Implicit bias=False by omission
  nodes = expand_linear(alloc, "fc1", {"in_features": 128})

  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]

  # FADD used for bias addition specifically in this macro
  assert "FADD" not in opcodes
  # Only FFMA for main loop
  assert "FFMA" in opcodes


def test_comment_generation() -> None:
  """
  Verify comments delineate blocks.
  """
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "layer1", {})

  text = [n.text for n in nodes if isinstance(n, Comment)]
  assert "BEGIN Conv2d (layer1)" in text
  assert "END Conv2d (layer1)" in text
