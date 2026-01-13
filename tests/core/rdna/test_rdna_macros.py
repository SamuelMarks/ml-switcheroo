import pytest
from ml_switcheroo.compiler.backends.rdna.macros import expand_conv2d, expand_linear
from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Instruction,
  Label,
  SGPR,
  VGPR,
  Comment,
)


class MockAllocator:
  """Mock implementation of RegisterAllocatorProtocol."""

  def __init__(self) -> None:
    self.v_counter = 0
    self.s_counter = 0

  def get_vector_register(self, var_name: str) -> VGPR:
    """Returns v0 for output, vX for others."""
    if var_name == "conv1":
      return VGPR(0)
    idx = self.v_counter
    self.v_counter += 1
    return VGPR(idx)

  def get_scalar_register(self, var_name: str) -> SGPR:
    idx = self.s_counter
    self.s_counter += 1
    return SGPR(idx)

  def allocate_vector_temp(self) -> VGPR:
    """Allocates next VGPR."""
    idx = self.v_counter
    self.v_counter += 1
    return VGPR(idx)

  def allocate_scalar_temp(self) -> SGPR:
    """Allocates next SGPR."""
    idx = self.s_counter
    self.s_counter += 1
    return SGPR(idx)


def test_expand_conv2d_structure() -> None:
  """
  Verify Conv2d generates nested loops, loads, and waitcnt.
  """
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "conv1", {"k": 3})

  # 1. Check Labels (2 loops = 2 labels)
  labels = [n for n in nodes if isinstance(n, Label)]
  assert len(labels) == 2
  assert labels[0].name == "L_KY_conv1"
  assert labels[1].name == "L_KX_conv1"

  # 2. Check Instructions
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]

  # Init
  assert "v_mov_b32" in opcodes
  assert "s_mov_b32" in opcodes

  # Loads & Barrier
  assert "global_load_dword" in opcodes
  assert opcodes.count("global_load_dword") >= 2
  assert "s_waitcnt" in opcodes

  # Math
  assert "v_fmac_f32" in opcodes

  # Control Flow
  assert "s_add_i32" in opcodes
  assert "s_cmp_lt_i32" in opcodes
  assert "s_cbranch_scc1" in opcodes

  # 3. Check Register Usage Types
  # Loop counters must be SGPRs (Scalar alu s_add_i32)
  s_add_instrs = [n for n in nodes if isinstance(n, Instruction) and n.opcode == "s_add_i32"]
  for inst in s_add_instrs:
    # Dest and Src0 should be SGPRs
    assert isinstance(inst.operands[0], SGPR)


def test_expand_linear_structure() -> None:
  """
  Verify Linear generates GEMM loop.
  """
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 256})

  # 1. Label
  labels = [n for n in nodes if isinstance(n, Label)]
  assert len(labels) == 1
  assert labels[0].name == "L_GEMM_fc1"

  # 2. Key Instructions
  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]

  assert "global_load_dword" in opcodes
  assert "s_waitcnt" in opcodes
  assert "v_fmac_f32" in opcodes
  # Pointer increment for addresses is Vector Add (v_add_u32)
  assert "v_add_u32" in opcodes
  # Loop counter is Scalar (s_add_i32)
  assert "s_add_i32" in opcodes

  # 3. Bias is optional (Check missing)
  assert "v_add_f32" not in opcodes  # Bias add uses v_add_f32 in this macro


def test_expand_linear_with_bias() -> None:
  """Verify bias addition."""
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"bias": True})

  opcodes = [n.opcode for n in nodes if isinstance(n, Instruction)]

  # Should have final add
  assert "v_add_f32" in opcodes
  comment_texts = [n.text for n in nodes if isinstance(n, Comment)]
  assert "Add Bias" in comment_texts


def test_macros_generate_comments() -> None:
  """Verify block delimiters."""
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "L1", {})

  comments = [n.text for n in nodes if isinstance(n, Comment)]
  assert "BEGIN Conv2d (L1)" in comments
  assert "END Conv2d (L1)" in comments
