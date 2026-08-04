"""Test suite for the AvgPool2d SASS Macro."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_avgpool2d
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassLabel, SassComment


def test_sass_macro_avgpool2d() -> None:
  """Verifies that expand_avgpool2d generates correct SASS loops."""
  allocator = RegisterAllocator()
  node_id = "pool1"
  metadata = {"kernel_size": 2}

  nodes = expand_avgpool2d(allocator, node_id, metadata)

  # Basic checks
  assert len(nodes) > 10

  # Check for BEGIN and END comments
  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN AvgPool2d ({node_id})" in comments
  assert f"END AvgPool2d ({node_id})" in comments

  # Check for labels
  labels = [n.name for n in nodes if isinstance(n, SassLabel)]
  assert f"L_KY_{node_id}" in labels
  assert f"L_KX_{node_id}" in labels

  # Check for FADD and FMUL
  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FADD" in opcodes
  assert "FMUL" in opcodes
  assert "LDG.E.F32" in opcodes
  assert "ISETP.LT.AND" in opcodes
