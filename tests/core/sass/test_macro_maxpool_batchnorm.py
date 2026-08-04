"""Test suite for the MaxPool2d and BatchNorm2d SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_maxpool2d, expand_batchnorm2d
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassComment


def test_sass_macro_maxpool2d() -> None:
  """Verifies that expand_maxpool2d generates correct SASS loops."""
  allocator = RegisterAllocator()
  node_id = "pool2"
  metadata = {"kernel_size": 2}

  nodes = expand_maxpool2d(allocator, node_id, metadata)

  # Basic checks
  assert len(nodes) > 10

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN MaxPool2d ({node_id})" in comments
  assert f"END MaxPool2d ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FMAX" in opcodes
  assert "LDG.E.F32" in opcodes


def test_sass_macro_batchnorm2d() -> None:
  """Verifies that expand_batchnorm2d generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "bn1"
  metadata = {"eps": 1e-5}

  nodes = expand_batchnorm2d(allocator, node_id, metadata)

  assert len(nodes) > 5

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN BatchNorm2d ({node_id})" in comments
  assert f"END BatchNorm2d ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "MUFU" in opcodes
  assert "FFMA" in opcodes
