"""Test suite for the Rdna Macros module."""

import typing

from ml_switcheroo.core.compiler.backends.rdna.macros import (
  RdnaSGPR,
  RdnaVGPR,
  RegisterAllocatorProtocol,
  expand_adam,
  expand_conv2d,
  expand_conv3d,
  expand_conv_general_dilated,
  expand_dropout,
  expand_flatten,
  expand_l,
  expand_linear,
  expand_relu,
  expand_reshape,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaInstruction, RdnaLabel, RdnaSGPR, RdnaVGPR  # noqa: F811


class MockAllocator:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockAllocator instance."""
    self.v_counter = 0
    self.s_counter = 0

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Mock implementation of get vector register."""
    if var_name == "conv1":
      return RdnaVGPR(index=0)
    idx = self.v_counter
    self.v_counter += 1
    return RdnaVGPR(idx)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Mock implementation of get scalar register."""
    idx = self.s_counter
    self.s_counter += 1
    return RdnaSGPR(idx)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Mock implementation of allocate vector temp."""
    idx = self.v_counter
    self.v_counter += 1
    return RdnaVGPR(idx)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Mock implementation of allocate scalar temp."""
    idx = self.s_counter
    self.s_counter += 1
    return RdnaSGPR(idx)


def test_expand_conv2d_structure() -> None:
  """Verifies the behavior of expand conv2d structure."""
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "conv1", {"k": 3})
  labels = [n for n in nodes if isinstance(n, RdnaLabel)]
  assert len(labels) == 2
  assert labels[0].name == "L_KY_conv1"
  assert labels[1].name == "L_KX_conv1"
  opcodes = [n.opcode for n in nodes if isinstance(n, RdnaInstruction)]
  assert "v_mov_b32" in opcodes
  assert "s_mov_b32" in opcodes
  assert "global_load_dword" in opcodes
  assert opcodes.count("global_load_dword") >= 2
  assert "s_waitcnt" in opcodes
  assert "v_fmac_f32" in opcodes
  assert "s_add_i32" in opcodes
  assert "s_cmp_lt_i32" in opcodes
  assert "s_cbranch_scc1" in opcodes
  s_add_instrs = [n for n in nodes if isinstance(n, RdnaInstruction) and n.opcode == "s_add_i32"]
  for inst in s_add_instrs:
    assert isinstance(inst.operands[0], RdnaSGPR)


def test_expand_linear_structure() -> None:
  """Verifies the behavior of expand linear structure."""
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 256})
  labels = [n for n in nodes if isinstance(n, RdnaLabel)]
  assert len(labels) == 1
  assert labels[0].name == "L_GEMM_fc1"
  opcodes = [n.opcode for n in nodes if isinstance(n, RdnaInstruction)]
  assert "global_load_dword" in opcodes
  assert "s_waitcnt" in opcodes
  assert "v_fmac_f32" in opcodes
  assert "v_add_u32" in opcodes
  assert "s_add_i32" in opcodes
  assert "v_add_f32" not in opcodes


def test_expand_linear_with_bias() -> None:
  """Verifies the behavior of expand linear with bias."""
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"bias": True})
  opcodes = [n.opcode for n in nodes if isinstance(n, RdnaInstruction)]
  assert "v_add_f32" in opcodes
  comment_texts = [n.text for n in nodes if isinstance(n, RdnaComment)]
  assert "Add Bias" in comment_texts


def test_macros_generate_comments() -> None:
  """Verifies the behavior of macros generate comments."""
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "L1", {})
  comments = [n.text for n in nodes if isinstance(n, RdnaComment)]
  assert "BEGIN Conv2d (L1)" in comments
  assert "END Conv2d (L1)" in comments


# --- Merged from test_rdna_macros_extra.py ---


class MockAllocatorExtra(RegisterAllocatorProtocol):
  """Mock allocator."""

  def __init__(self) -> None:
    """Init."""
    self.vc: int = 0
    self.sc: int = 0

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Get vector register."""
    return RdnaVGPR(index=0)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Get scalar register."""
    return RdnaSGPR(index=0)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Allocate vector temp."""
    self.vc += 1
    return RdnaVGPR(index=self.vc)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Allocate scalar temp."""
    self.sc += 1
    return RdnaSGPR(index=self.sc)


def test_expand_all_macros() -> None:
  """Docstring."""
  alloc = MockAllocator()
  funcs: list[typing.Callable[..., typing.Any]] = [
    expand_conv2d,
    expand_linear,
    expand_relu,
    expand_flatten,
    expand_reshape,
    expand_conv3d,
    expand_dropout,
    expand_variable,
    expand_transpose,
    expand_conv_general_dilated,
    expand_adam,
    expand_l,
  ]
  for func in funcs:
    nodes: list[typing.Any] = func(alloc, "node1", {})
    assert len(nodes) > 0
