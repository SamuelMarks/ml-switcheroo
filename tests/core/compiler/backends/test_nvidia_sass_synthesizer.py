"""Unit tests for `NvidiaSassSynthesizer` and `NvidiaSassBackend` (`ml_switcheroo.core.compiler.backends.nvidia_sass`)."""

import json
import os
import typing
from unittest.mock import MagicMock, mock_open

import libcst as cst
import pytest
from ml_switcheroo_ir import AttributeValue
from pydantic import JsonValue

import ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer as synth_mod
from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassNode,
  NvidiaSassOperand,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.semantics.manager import SemanticsManager


class _RawSassOperand(NvidiaSassOperand):
  """Custom `NvidiaSassOperand` emitting raw operand text."""

  raw: str = ""

  def __init__(self, raw: str) -> None:
    """Store raw text.

    Args:
        raw: Raw operand text.
    """
    super().__init__()
    self.raw = raw

  def to_text(self) -> str:
    """Return `self.raw`.

    Returns:
        Raw operand text.
    """
    return self.raw

  def __str__(self) -> str:
    """Return `self.raw`.

    Returns:
        Raw operand text.
    """
    return self.raw


class _OtherSassNode(NvidiaSassNode):
  """Unhandled `NvidiaSassNode` subclass ignored by `to_python`."""

  def __str__(self) -> str:
    """Return `'; other'`.

    Returns:
        Literal `'; other'`.
    """
    return "; other"


def test_sass_register_allocator_and_backend_compile() -> None:
  """Verify `RegisterAllocator` pooling/liveness/overflow and `NvidiaSassBackend.compile`."""
  allocator: RegisterAllocator[NvidiaSassRegister] = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  assert allocator.get_register("var1").name == "R0"
  assert allocator.get_register("var1").name == "R0"
  assert allocator.allocate_temp().name == "R1"
  allocator.record_usage("untracked_var")
  assert "untracked_var" not in allocator._var_to_reg_id
  allocator.reset()
  assert allocator.get_register("var3").name == "R0"

  allocator._free_pool = []
  with pytest.raises(ValueError, match="overflow"):
    allocator.allocate_temp()

  backend = NvidiaSassBackend()
  assert backend.synthesizer.semantics is not None
  assert "Input in1" in backend.compile(LogicalGraph(nodes={"in1": LogicalNode("in1", "Input")}))


def test_sass_synthesizer_from_graph_macros_prefixes_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `NvidiaSassSynthesizer.from_graph` across inputs/outputs, macros, prefixed ops, and unmapped ops.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  mock_sem = MagicMock(spec=SemanticsManager)

  def _get_def(kind: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Resolve test operations including suffix `v_add_f32`.

    Args:
        kind: Operation kind string.

    Returns:
        `(abstract_id, details)` or `None`.
    """
    if kind == "UnknownNode" or kind.startswith("rdna."):
      return None
    if kind == "v_add_f32":
      return ("add", {})
    if kind in ("DirectMatch", "FallbackOp"):
      return (kind, {})
    return ("abstract.Linear", {})

  def _resolve(aid: str, _target: str) -> dict[str, JsonValue] | None:
    """Resolve NVIDIA SASS variant for `aid`.

    Args:
        aid: Abstract operation identifier.
        _target: Target dialect name.

    Returns:
        Variant dictionary or `None`.
    """
    if aid in ("FallbackOp", "add"):
      return {"api": "FADD", "args": []}
    if aid in ("unmapped", "UnknownNode"):
      return None
    return {"api": "NOP", "args": []}

  mock_sem.get_definition.side_effect = _get_def
  mock_sem.resolve_variant.side_effect = _resolve
  synth = NvidiaSassSynthesizer(semantics=mock_sem)

  def _dummy_expander(
    _alloc: RegisterAllocator[NvidiaSassRegister],
    _nid: str,
    _meta: typing.Mapping[str, AttributeValue],
  ) -> list[NvidiaSassNode]:
    """Emit a single `NOP` instruction.

    Args:
        _alloc: Register allocator.
        _nid: Node identifier.
        _meta: Node attributes.

    Returns:
        Single-instruction list.
    """
    return [NvidiaSassInstruction(opcode="NOP", operands=[])]

  synth.macro_registry = {"Linear": _dummy_expander, "DirectMatch": _dummy_expander}

  graph = LogicalGraph(
    nodes={
      "in1": LogicalNode("in1", "Input", attributes={"name": "x"}),
      "in2": LogicalNode("in2", "Input"),
      "conv1": LogicalNode("conv1", "Conv2d", attributes={"k": "3"}),
      "lin1": LogicalNode("lin1", "Linear"),
      "dir1": LogicalNode("dir1", "DirectMatch"),
      "fall1": LogicalNode("fall1", "FallbackOp"),
      "pref1": LogicalNode("pref1", "rdna.v_add_f32"),
      "unk1": LogicalNode("unk1", "UnknownNode"),
      "out1": LogicalNode("out1", "Output", inputs=["lin1"]),
      "out_disc": LogicalNode("out_disc", "Output"),
    },
    edges=[
      LogicalEdge("in1", "conv1"),
      LogicalEdge("conv1", "lin1"),
      LogicalEdge("in1", "dir1"),
      LogicalEdge("in1", "fall1"),
      LogicalEdge("in2", "fall1"),
      LogicalEdge("lin1", "out1"),
    ],
  )
  nodes = synth.from_graph(graph)
  texts = [str(n) for n in nodes]
  assert any("Input x -> " in t for t in texts)
  assert any("Return:" in t for t in texts)
  assert any("Unmapped Op: UnknownNode" in t for t in texts)
  assert any(isinstance(n, NvidiaSassInstruction) and n.opcode == "FADD" for n in nodes)

  # Unmapped variant when `resolve_variant` returns `None`
  mock_sem.get_definition.side_effect = lambda _k: ("unmapped", {})
  synth.macro_registry = {}
  unmapped_nodes = synth.from_graph(LogicalGraph(nodes={"op1": LogicalNode("op1", "UnknownOp")}))
  assert "Unmapped Op" in str(unmapped_nodes[0])

  # Missing `macros.json` and missing function in `macros.json`
  with monkeypatch.context() as mp:
    mp.setattr(os.path, "exists", lambda _p: False)
    assert synth_mod.NvidiaSassSynthesizer(semantics=None).macro_registry == {}

  with monkeypatch.context() as mp:
    mp.setattr(os.path, "exists", lambda _p: True)
    mp.setattr("builtins.open", mock_open(read_data=json.dumps({"dummy_op": "non_existent_func"})))
    assert "dummy_op" not in synth_mod.NvidiaSassSynthesizer(semantics=None).macro_registry


def test_sass_synthesizer_to_python_and_operand_conversions() -> None:
  """Verify `to_python` and `_convert_operand_to_py` across ALU, stores, branches, predicates, and immediates."""
  synth = NvidiaSassSynthesizer(semantics=None)
  assert synth.to_python([NvidiaSassComment(text="standalone comment"), _OtherSassNode()]).body == []

  nodes: list[NvidiaSassNode] = [
    NvidiaSassComment(text="BEGIN Conv2d"),
    NvidiaSassComment(text="ordinary comment"),
    NvidiaSassLabel(name="L_LOOP"),
    NvidiaSassInstruction(opcode="NOP", operands=[]),
    NvidiaSassInstruction(opcode="BRA", operands=[NvidiaSassLabel(name="L_LOOP")]),
    NvidiaSassInstruction(opcode="ST_E", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")]),
    NvidiaSassInstruction(opcode="STG", operands=[NvidiaSassMemory(base=NvidiaSassRegister(name="R0"), offset=4)]),
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")],
    ),
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R0")],
      predicate=NvidiaSassPredicate(name="P0", is_guard=True),
    ),
    NvidiaSassInstruction(
      opcode="MOV",
      operands=[NvidiaSassRegister(name="R0"), _RawSassOperand("R[0:1]")],
      predicate=NvidiaSassPredicate(name="P0"),
    ),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), _RawSassOperand("#hashtag")]),
    NvidiaSassInstruction(opcode="MOV", operands=[_RawSassOperand("-R0"), NvidiaSassImmediate(value=42)]),
    NvidiaSassInstruction(opcode="MOV", operands=[_RawSassOperand("valid_ident"), NvidiaSassImmediate(value=10)]),
  ]
  code = synth.to_python(nodes).code
  assert "# BEGIN Conv2d" in code
  assert "# NvidiaSassLabel: L_LOOP" in code
  assert "nvidia_sass.NOP()" in code
  assert "nvidia_sass.ST_E(R0, R1)" in code
  assert "R0 = nvidia_sass.FADD(R1, R2)" in code
  assert "predicate = '@P0'" in code
  assert "'R[0:1]'" in code
  assert "'#hashtag'" in code
  assert "valid_ident = " in code

  py_int = synth._convert_operand_to_py(NvidiaSassImmediate(value=10, is_hex=False))
  assert isinstance(py_int, cst.Integer)
  assert py_int.value == "10"

  py_hex = synth._convert_operand_to_py(NvidiaSassImmediate(value=255, is_hex=True))
  assert isinstance(py_hex, cst.Integer)
  assert py_hex.value == "0xff"

  py_flt = synth._convert_operand_to_py(NvidiaSassImmediate(value=1.5, is_hex=False))
  assert isinstance(py_flt, cst.Float)
  assert py_flt.value == "1.5"

  py_str = synth._convert_operand_to_py(_RawSassOperand("[R0]"))
  assert isinstance(py_str, cst.SimpleString)
  assert py_str.value == "'[R0]'"
