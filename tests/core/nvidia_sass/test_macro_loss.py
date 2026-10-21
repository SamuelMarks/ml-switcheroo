"""Test suite for macro loss."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_crossentropyloss, expand_mseloss
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode


def test_nvidia_sass_macro_mseloss() -> None:
  """Verifies that expand_mseloss generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  node_id = "mse1"
  metadata = {"elements": 32, "reduction": "mean"}

  nodes: list[NvidiaSassNode] = getattr(expand_mseloss, "__call__")(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments: list[str] = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN MSELoss ({node_id})" in comments

  opcodes: list[str] = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "FMUL" in opcodes
  assert "FADD" in opcodes


def test_nvidia_sass_macro_crossentropyloss() -> None:
  """Verifies that expand_crossentropyloss generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  node_id = "ce1"
  metadata = {"elements": 64}

  nodes: list[NvidiaSassNode] = getattr(expand_crossentropyloss, "__call__")(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments: list[str] = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN CrossEntropyLoss ({node_id})" in comments

  opcodes: list[str] = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "MUFU" in opcodes
  assert "FMUL" in opcodes


def test_nvidia_sass_analyzer_loss() -> None:
  """Verifies analyzer handles loss limits."""
  instructions = [
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        NvidiaSassRegister(name="P0"),
        NvidiaSassRegister(name="PT"),
        NvidiaSassRegister(name="R0"),
        NvidiaSassImmediate(value=100),
        NvidiaSassRegister(name="PT"),
      ],
    )
  ]
  metadata_mse = NvidiaSassAnalyzer.analyze_block("MSELoss", instructions)
  assert metadata_mse["elements"] == 100

  metadata_ce = NvidiaSassAnalyzer.analyze_block("CrossEntropyLoss", instructions)
  assert metadata_ce["elements"] == 100
