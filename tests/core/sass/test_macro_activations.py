"""Test suite for the Activation SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_sigmoid, expand_tanh, expand_gelu
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassComment


def test_sass_macro_sigmoid() -> None:
  """Verifies that expand_sigmoid generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "sig1"
  metadata = {}

  nodes = expand_sigmoid(allocator, node_id, metadata)
  assert len(nodes) > 5

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN Sigmoid ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "MUFU" in opcodes
  assert "FADD" in opcodes


def test_sass_macro_tanh() -> None:
  """Verifies that expand_tanh generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "tanh1"
  metadata = {}

  nodes = expand_tanh(allocator, node_id, metadata)
  assert len(nodes) > 2

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN Tanh ({node_id})" in comments


def test_sass_macro_gelu() -> None:
  """Verifies that expand_gelu generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "gelu1"
  metadata = {}

  nodes = expand_gelu(allocator, node_id, metadata)
  assert len(nodes) > 5

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN GELU ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "MUFU" in opcodes
  assert "FMUL" in opcodes


def test_sass_analyzer_activations():
  """Verifies analyzer handles activations."""
  instructions = []
  assert len(SassAnalyzer.analyze_block("Sigmoid", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("Tanh", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("GELU", instructions)) == 0
