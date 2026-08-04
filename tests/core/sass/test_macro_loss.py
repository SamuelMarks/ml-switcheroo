"""Test suite for the Loss SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_mseloss, expand_crossentropyloss
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassComment, SassRegister, SassImmediate


def test_sass_macro_mseloss() -> None:
  """Verifies that expand_mseloss generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "mse1"
  metadata = {"elements": 32, "reduction": "mean"}

  nodes = expand_mseloss(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN MSELoss ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FMUL" in opcodes
  assert "FADD" in opcodes


def test_sass_macro_crossentropyloss() -> None:
  """Verifies that expand_crossentropyloss generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "ce1"
  metadata = {"elements": 64}

  nodes = expand_crossentropyloss(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN CrossEntropyLoss ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "MUFU" in opcodes
  assert "FMUL" in opcodes


def test_sass_analyzer_loss():
  """Verifies analyzer handles loss limits."""
  instructions = [
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        SassRegister(name="P0"),
        SassRegister(name="PT"),
        SassRegister(name="R0"),
        SassImmediate(value=100),
        SassRegister(name="PT"),
      ],
    )
  ]
  metadata_mse = SassAnalyzer.analyze_block("MSELoss", instructions)
  assert metadata_mse["elements"] == 100

  metadata_ce = SassAnalyzer.analyze_block("CrossEntropyLoss", instructions)
  assert metadata_ce["elements"] == 100
