"""Test suite for the Dropout and MatMul SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_dropout, expand_linear
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassComment


def test_sass_macro_dropout() -> None:
  """Verifies that expand_dropout generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "drop1"
  metadata = {"p": 0.5}

  nodes = expand_dropout(allocator, node_id, metadata)

  assert len(nodes) > 5

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN Dropout ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FSETP.GE.AND" in opcodes
  assert "FMUL" in opcodes


def test_sass_macro_matmul() -> None:
  """Verifies that MatMul maps to linear correctly."""
  allocator = RegisterAllocator()
  node_id = "mm1"
  metadata = {"in_features": 64}

  nodes = expand_linear(allocator, node_id, metadata)
  assert len(nodes) > 5


def test_sass_analyzer_dropout_matmul():
  """Verifies analyzer handles Dropout and MatMul."""
  instructions = []
  metadata_drop = SassAnalyzer.analyze_block("Dropout", instructions)
  assert len(metadata_drop) == 0

  metadata_mm = SassAnalyzer.analyze_block("MatMul", instructions)
  assert len(metadata_mm) == 0
