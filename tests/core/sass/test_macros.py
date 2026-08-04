"""Test suite for the Macros module."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_conv2d, expand_linear
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassLabel, SassRegister, SassComment


class MockAllocator:
  """Mock Allocator class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockAllocator instance."""
    self.counter = 10

  def get_register(self, var_name: str) -> SassRegister:
    """Mock implementation of get register."""
    if var_name == "output":
      return SassRegister(name="R0")
    return SassRegister(name="R_VAR")

  def allocate_temp(self) -> SassRegister:
    """Mock implementation of allocate temp."""
    name = f"R{self.counter}"
    self.counter += 1
    return SassRegister(name)


def test_expand_conv2d_structure() -> None:
  """Verifies the behavior of expand conv2d structure."""
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "conv1", {"k": 3})
  labels = [n for n in nodes if isinstance(n, SassLabel)]
  assert len(labels) == 2
  assert labels[0].name == "L_KY_conv1"
  assert labels[1].name == "L_KX_conv1"
  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "MOV" in opcodes
  assert "IMAD" in opcodes
  assert "IADD3" in opcodes
  assert "LDG.E.F32" in opcodes
  assert opcodes.count("LDG.E.F32") >= 2
  assert "FFMA" in opcodes
  assert "ISETP.LT.AND" in opcodes
  assert "BRA" in opcodes
  branches = [n for n in nodes if isinstance(n, SassInstruction) and n.opcode == "BRA"]
  assert branches[0].predicate.name == "P0"


def test_expand_linear_structure() -> None:
  """Verifies the behavior of expand linear structure."""
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 512, "bias": True})
  labels = [n for n in nodes if isinstance(n, SassLabel)]
  assert len(labels) == 1
  assert labels[0].name == "L_GEMM_fc1"
  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "LDG.E.F32" in opcodes
  assert "FFMA" in opcodes
  assert "IADD3" in opcodes
  assert "FADD" in opcodes


def test_expand_linear_no_bias() -> None:
  """Verifies the behavior of expand linear no bias."""
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "fc1", {"in_features": 128})
  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FADD" not in opcodes
  assert "FFMA" in opcodes


def test_comment_generation() -> None:
  """Verifies the behavior of comment generation."""
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "layer1", {})
  text = [n.text for n in nodes if isinstance(n, SassComment)]
  assert "BEGIN Conv2d (layer1)" in text
  assert "END Conv2d (layer1)" in text
