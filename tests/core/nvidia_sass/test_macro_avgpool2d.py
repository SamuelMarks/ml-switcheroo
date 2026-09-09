"""Test suite for the AvgPool2d NVIDIA_SASS Macro."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_avgpool2d
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassInstruction,
  NvidiaSassLabel,
)


def test_nvidia_sass_macro_avgpool2d() -> None:
  """Verifies that expand_avgpool2d generates correct NVIDIA_SASS loops."""
  allocator = RegisterAllocator()
  node_id = "pool1"
  metadata = {"kernel_size": 2}

  nodes = expand_avgpool2d(allocator, node_id, metadata)

  # Basic checks
  assert len(nodes) > 10

  # Check for BEGIN and END comments
  comments = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN AvgPool2d ({node_id})" in comments
  assert f"END AvgPool2d ({node_id})" in comments

  # Check for labels
  labels = [n.name for n in nodes if isinstance(n, NvidiaSassLabel)]
  assert f"L_KY_{node_id}" in labels
  assert f"L_KX_{node_id}" in labels

  # Check for FADD and FMUL
  opcodes = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "FADD" in opcodes
  assert "FMUL" in opcodes
  assert "LDG.E.F32" in opcodes
  assert "ISETP.LT.AND" in opcodes
