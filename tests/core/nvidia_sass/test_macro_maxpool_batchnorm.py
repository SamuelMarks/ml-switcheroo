"""Test suite for macro maxpool batchnorm."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_batchnorm2d, expand_maxpool2d
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassInstruction
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as GenericRegisterAllocator

RegisterAllocator = GenericRegisterAllocator


def test_nvidia_sass_macro_maxpool2d() -> None:
  """Verifies that expand_maxpool2d generates correct NVIDIA_SASS loops."""
  allocator = RegisterAllocator(256, lambda x: None)
  node_id = "pool2"
  metadata = {"kernel_size": 2}

  nodes = expand_maxpool2d(allocator, node_id, metadata)

  # Basic checks
  assert len(nodes) > 10

  comments = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN MaxPool2d ({node_id})" in comments
  assert f"END MaxPool2d ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "FMAX" in opcodes
  assert "LDG.E.F32" in opcodes


def test_nvidia_sass_macro_batchnorm2d() -> None:
  """Verifies that expand_batchnorm2d generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator(256, lambda x: None)
  node_id = "bn1"
  metadata = {"eps": 1e-5}

  nodes = expand_batchnorm2d(allocator, node_id, metadata)

  assert len(nodes) > 5

  comments = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN BatchNorm2d ({node_id})" in comments
  assert f"END BatchNorm2d ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "MUFU" in opcodes
  assert "FFMA" in opcodes
