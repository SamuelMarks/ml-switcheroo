"""Unit tests for NVIDIA SASS procedural macro expansions (`ml_switcheroo.core.compiler.backends.nvidia_sass.macros`)."""

import inspect

import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as sass_macros
from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassRegister,
)


class _SassMacroAllocator:
  """Typed NVIDIA SASS register allocator stub for macro tests."""

  def __init__(self) -> None:
    """Initialize temporary register counter."""
    self.counter = 10

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Return `R0` for `'output'` or `R_VAR_<var_name>`.

    Args:
        var_name: Variable name.

    Returns:
        `NvidiaSassRegister` instance.
    """
    if var_name == "output":
      return NvidiaSassRegister(name="R0")
    return NvidiaSassRegister(name=f"R_VAR_{var_name}")

  def allocate_temp(self) -> NvidiaSassRegister:
    """Allocate temporary `R<counter>` register.

    Returns:
        `NvidiaSassRegister` instance.
    """
    reg = NvidiaSassRegister(name=f"R{self.counter}")
    self.counter += 1
    return reg


def test_nvidia_sass_macros_conv2d_linear_bias_and_all_expanders() -> None:
  """Verify `expand_conv2d`, `expand_linear` (with/without bias), comments, and every `expand_*` macro."""
  conv_nodes = expand_conv2d(_SassMacroAllocator(), "conv1", {"k": 3})
  assert [n.name for n in conv_nodes if isinstance(n, NvidiaSassLabel)] == ["L_KY_conv1", "L_KX_conv1"]
  conv_ops = [n.opcode for n in conv_nodes if isinstance(n, NvidiaSassInstruction)]
  assert {"MOV", "IMAD", "IADD3", "LDG.E.F32", "FFMA", "ISETP.LT.AND", "BRA"} <= set(conv_ops)
  assert conv_ops.count("LDG.E.F32") >= 2
  branches = [n for n in conv_nodes if isinstance(n, NvidiaSassInstruction) and n.opcode == "BRA"]
  assert branches[0].predicate is not None
  assert branches[0].predicate.name == "P0"

  comments = [n.text for n in conv_nodes if isinstance(n, NvidiaSassComment)]
  assert "BEGIN Conv2d (conv1)" in comments
  assert "END Conv2d (conv1)" in comments

  lin_bias = expand_linear(
    _SassMacroAllocator(),
    "fc1",
    {"in_features": 512, "bias": True, "runtime_comment": "resolved"},
  )
  assert [n.name for n in lin_bias if isinstance(n, NvidiaSassLabel)] == ["L_GEMM_fc1"]
  lin_bias_ops = [n.opcode for n in lin_bias if isinstance(n, NvidiaSassInstruction)]
  assert {"LDG.E.F32", "FFMA", "IADD3", "FADD"} <= set(lin_bias_ops)

  lin_nobias_ops = [
    n.opcode
    for n in expand_linear(_SassMacroAllocator(), "fc1", {"in_features": 128})
    if isinstance(n, NvidiaSassInstruction)
  ]
  assert "FFMA" in lin_nobias_ops
  assert "FADD" not in lin_nobias_ops

  for name, fn in inspect.getmembers(sass_macros, inspect.isfunction):
    if name.startswith("expand_"):
      out = fn(_SassMacroAllocator(), "test_node", {"k": 3, "seq_len": 5, "in_features": 64, "elements": 32})
      assert len(out) > 0
