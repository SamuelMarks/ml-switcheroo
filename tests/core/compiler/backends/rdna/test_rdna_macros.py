"""Unit tests for RDNA procedural macro expansions (`ml_switcheroo.core.compiler.backends.rdna.macros`)."""

import inspect

import ml_switcheroo.core.compiler.backends.rdna.macros as rdna_macros
from ml_switcheroo.core.compiler.backends.rdna.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaInstruction,
  RdnaLabel,
  RdnaSGPR,
  RdnaVGPR,
)


class _RdnaMacroAllocator:
  """Typed dual-pool RDNA register allocator stub for macro tests."""

  def __init__(self) -> None:
    """Initialize VGPR and SGPR counters."""
    self.v_counter = 0
    self.s_counter = 0

  def get_register(self, var_name: str) -> RdnaVGPR:
    """Allocate or return VGPR for `var_name`.

    Args:
        var_name: Variable name.

    Returns:
        `RdnaVGPR` instance.
    """
    if var_name == "conv1":
      return RdnaVGPR(index=0)
    idx = self.v_counter
    self.v_counter += 1
    return RdnaVGPR(idx)

  def get_scalar_register(self, _var_name: str) -> RdnaSGPR:
    """Allocate SGPR for `_var_name`.

    Args:
        _var_name: Scalar variable name.

    Returns:
        `RdnaSGPR` instance.
    """
    idx = self.s_counter
    self.s_counter += 1
    return RdnaSGPR(idx)

  def allocate_temp(self) -> RdnaVGPR:
    """Allocate temporary VGPR.

    Returns:
        `RdnaVGPR` instance.
    """
    idx = self.v_counter
    self.v_counter += 1
    return RdnaVGPR(idx)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Allocate temporary SGPR.

    Returns:
        `RdnaSGPR` instance.
    """
    idx = self.s_counter
    self.s_counter += 1
    return RdnaSGPR(idx)


def test_rdna_macros_conv2d_linear_bias_and_all_expanders() -> None:
  """Verify `expand_conv2d`, `expand_linear` (with/without bias), comments, and every `expand_*` macro."""
  alloc = _RdnaMacroAllocator()
  conv_nodes = expand_conv2d(alloc, "conv1", {"k": 3, "row_stride_bytes": 64})
  assert [n.name for n in conv_nodes if isinstance(n, RdnaLabel)] == ["L_KY_conv1", "L_KX_conv1"]
  conv_ops = [n.opcode for n in conv_nodes if isinstance(n, RdnaInstruction)]
  assert {
    "v_mov_b32",
    "s_mov_b32",
    "global_load_dword",
    "s_waitcnt",
    "v_fmac_f32",
    "s_add_i32",
    "s_cmp_lt_i32",
    "s_cbranch_scc1",
  } <= set(conv_ops)
  assert conv_ops.count("global_load_dword") >= 2
  for inst in (n for n in conv_nodes if isinstance(n, RdnaInstruction) and n.opcode == "s_add_i32"):
    assert isinstance(inst.operands[0], RdnaSGPR)

  comments = [n.text for n in conv_nodes if isinstance(n, RdnaComment)]
  assert "BEGIN Conv2d (conv1)" in comments
  assert "END Conv2d (conv1)" in comments

  lin_nodes = expand_linear(_RdnaMacroAllocator(), "fc1", {"in_features": 256, "runtime_comment": "resolved"})
  assert [n.name for n in lin_nodes if isinstance(n, RdnaLabel)] == ["L_GEMM_fc1"]
  lin_ops = [n.opcode for n in lin_nodes if isinstance(n, RdnaInstruction)]
  assert {"global_load_dword", "s_waitcnt", "v_fmac_f32", "v_add_u32", "s_add_i32"} <= set(lin_ops)
  assert "v_add_f32" not in lin_ops

  lin_bias = expand_linear(_RdnaMacroAllocator(), "fc1", {"bias": True})
  assert "v_add_f32" in [n.opcode for n in lin_bias if isinstance(n, RdnaInstruction)]
  assert "Add Bias" in [n.text for n in lin_bias if isinstance(n, RdnaComment)]

  for name, fn in inspect.getmembers(rdna_macros, inspect.isfunction):
    if name.startswith("expand_"):
      out = fn(_RdnaMacroAllocator(), "node1", {"k": 3, "d_in": 10, "d_out": 10})
      assert len(out) > 0

  import sys

  import pytest

  from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter

  emitter = RdnaEmitter()
  _ = emitter.validate_instruction("v_fmac_f32")
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "ml_ecosystem_snapshots.grounding.hardware", None)
    assert emitter.validate_instruction("v_fmac_f32") is None
