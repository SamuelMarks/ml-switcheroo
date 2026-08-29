"""Test suite for the Loss SASS Macros."""

import typing

from ml_switcheroo.core.compiler.backends.sass.macros import expand_crossentropyloss, expand_mseloss
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassImmediate,
  SassInstruction,
  SassNode,
  SassRegister,
)


def test_sass_macro_mseloss() -> None:
  """Verifies that expand_mseloss generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "mse1"
  metadata: dict[str, typing.Any] = {"elements": 32, "reduction": "mean"}

  nodes: list[SassNode] = expand_mseloss(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments: list[str] = [typing.cast(SassComment, n).text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN MSELoss ({node_id})" in comments

  opcodes: list[str] = [typing.cast(SassInstruction, n).opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FMUL" in opcodes
  assert "FADD" in opcodes


def test_sass_macro_crossentropyloss() -> None:
  """Verifies that expand_crossentropyloss generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "ce1"
  metadata: dict[str, typing.Any] = {"elements": 64}

  nodes: list[SassNode] = expand_crossentropyloss(allocator, node_id, metadata)
  assert len(nodes) > 10

  comments: list[str] = [typing.cast(SassComment, n).text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN CrossEntropyLoss ({node_id})" in comments

  opcodes: list[str] = [typing.cast(SassInstruction, n).opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "MUFU" in opcodes
  assert "FMUL" in opcodes


def test_sass_analyzer_loss() -> None:
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
  metadata_mse: dict[str, typing.Any] = SassAnalyzer.analyze_block("MSELoss", instructions)
  assert metadata_mse["elements"] == 100

  metadata_ce: dict[str, typing.Any] = SassAnalyzer.analyze_block("CrossEntropyLoss", instructions)
  assert metadata_ce["elements"] == 100
