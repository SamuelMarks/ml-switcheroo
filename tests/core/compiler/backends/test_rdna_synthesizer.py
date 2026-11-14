"""Unit tests for `RdnaSynthesizer` and `RdnaBackend` (`ml_switcheroo.core.compiler.backends.rdna.synthesizer`)."""

import json
import os
import typing
from unittest.mock import MagicMock, mock_open

import libcst as cst
import pytest
from ml_switcheroo_ir import AttributeValue
from pydantic import JsonValue

import ml_switcheroo.core.compiler.backends.rdna.synthesizer as synth_mod
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend, RdnaSynthesizer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaOperand,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.semantics.manager import SemanticsManager


class _RawRdnaOperand(RdnaOperand):
  """Custom `RdnaOperand` emitting a raw string for `to_python` operand sanitization tests."""

  raw: str = ""

  def __init__(self, raw: str) -> None:
    """Store raw operand string.

    Args:
        raw: Raw operand text.
    """
    super().__init__()
    self.raw = raw

  def __str__(self) -> str:
    """Return `self.raw`.

    Returns:
        Raw operand text.
    """
    return self.raw

  def to_text(self) -> str:
    """Return `self.raw`.

    Returns:
        Raw operand text.
    """
    return self.raw


def test_rdna_register_allocator_and_backend_compile() -> None:
  """Verify `RegisterAllocator` pooling/overflow/reset and `RdnaBackend.compile`."""
  allocator: RegisterAllocator[RdnaVGPR] = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  assert allocator.get_register("var1").index == 0
  assert allocator.get_register("var1").index == 0
  assert allocator.allocate_temp().index == 1
  assert allocator.allocate_temp().index == 2
  allocator.reset()
  assert allocator.get_register("var3").index == 0

  allocator._free_pool = []
  with pytest.raises(ValueError, match="overflow"):
    allocator.allocate_temp()

  backend_default = RdnaBackend()
  assert backend_default.target_arch == "gfx1030"
  assert backend_default.synthesizer.semantics is not None
  in_graph = LogicalGraph(nodes={"in1": LogicalNode("in1", "Input")})
  assert "RDNA Code Generation Initialized" in backend_default.compile(in_graph)

  mock_sem = MagicMock(spec=SemanticsManager)
  mock_sem.get_definition.return_value = None
  mock_sem.resolve_variant.return_value = None
  assert "RDNA Code Generation Initialized" in RdnaBackend(semantics=mock_sem).compile(in_graph)


def test_rdna_synthesizer_from_graph_macros_directives_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `RdnaSynthesizer.from_graph` across inputs/outputs, macros, `; Macro.*`, dotted kinds, and unmapped ops.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  mock_sem = MagicMock(spec=SemanticsManager)

  def _get_def(kind: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Resolve `Add`, `nn.Dotted`, and `op_macro_*` definitions.

    Args:
        kind: Operation kind string.

    Returns:
        `(abstract_id, details)` tuple or `None`.
    """
    if kind in ("Add", "op_macro_semicolon", "op_macro_unknown", "op_semicolon_comment", "ExactMacro"):
      return (kind, {})
    if kind == "nn.Dotted":
      return ("DottedOp", {})
    return None

  def _resolve(aid: str, _fw: str) -> dict[str, JsonValue] | None:
    """Resolve target RDNA variant for `aid`.

    Args:
        aid: Abstract operation identifier.
        _fw: Target framework string.

    Returns:
        Variant dictionary or `None`.
    """
    mapping: dict[str, dict[str, JsonValue]] = {
      "Add": {"api": "v_add_f32"},
      "DottedOp": {"api": "v_nop"},
      "op_macro_semicolon": {"api": "; Macro.LayerNorm"},
      "op_macro_unknown": {"api": "Macro.CustomUnknown"},
      "op_semicolon_comment": {"api": "; unmapped custom comment"},
    }
    return mapping.get(aid)

  mock_sem.get_definition.side_effect = _get_def
  mock_sem.resolve_variant.side_effect = _resolve
  synth = RdnaSynthesizer(mock_sem)

  # Input -> Add -> Output + dotted + macro directives + unmapped + disconnected output
  graph = LogicalGraph(
    nodes={
      "x": LogicalNode("x", "Input", attributes={"name": "x"}),
      "y": LogicalNode("y", "Input"),
      "z": LogicalNode("z", "Add", inputs=["x", "y"]),
      "conv1": LogicalNode("conv1", "Conv2d", attributes={"k": "3"}),
      "dot": LogicalNode("dot", "torch.nn.Dotted"),
      "m_semi": LogicalNode("m_semi", "op_macro_semicolon"),
      "m_unk": LogicalNode("m_unk", "op_macro_unknown"),
      "semi_cmt": LogicalNode("semi_cmt", "op_semicolon_comment"),
      "unk": LogicalNode("unk", "MysteryOp"),
      "out_conn": LogicalNode("out_conn", "Output", inputs=["z"]),
      "out_disc": LogicalNode("out_disc", "Output"),
    },
    edges=[LogicalEdge("x", "z"), LogicalEdge("y", "z"), LogicalEdge("z", "out_conn")],
  )
  nodes = synth.from_graph(graph)
  texts = [str(n) for n in nodes]
  assert any("Input x -> v" in t for t in texts)
  assert any("Return: v" in t for t in texts)
  assert any("LayerNorm" in t for t in texts)
  assert any("CustomUnknown" in t for t in texts)
  assert any("Unmapped Op: ; unmapped custom comment" in t for t in texts)
  assert any("Unmapped Op: MysteryOp" in t for t in texts)

  # Exact macro registry match
  def _exact_expander(
    _alloc: RegisterAllocator[RdnaVGPR],
    _nid: str,
    _meta: typing.Mapping[str, AttributeValue],
  ) -> list[RdnaComment]:
    """Emit a single `RdnaComment('exact_hit')`.

    Args:
        _alloc: Register allocator.
        _nid: Node identifier.
        _meta: Node attributes.

    Returns:
        Single-comment list.
    """
    return [RdnaComment(text="exact_hit")]

  synth.macro_registry["ExactMacro"] = _exact_expander
  exact_out = synth.from_graph(LogicalGraph(nodes={"e1": LogicalNode("e1", "ExactMacro")}))
  assert isinstance(exact_out[0], RdnaComment)
  assert exact_out[0].text == "exact_hit"

  # Missing `macros.json` and non-existent macro function inside `macros.json`
  with monkeypatch.context() as mp:
    mp.setattr(os.path, "exists", lambda _p: False)
    assert synth_mod.RdnaSynthesizer(semantics=None).macro_registry == {}

  with monkeypatch.context() as mp:
    mp.setattr(os.path, "exists", lambda _p: True)
    mp.setattr("builtins.open", mock_open(read_data=json.dumps({"dummy_op": "non_existent_func"})))
    assert "dummy_op" not in synth_mod.RdnaSynthesizer(semantics=None).macro_registry


def test_rdna_synthesizer_to_python_and_operand_conversions() -> None:
  """Verify `to_python` and `_convert_operand_to_py` across ALU, stores, ranges, labels, comments, and immediates."""
  synth = RdnaSynthesizer(semantics=None)
  assert synth.to_python([RdnaComment(text="ignored comment")]).body == []

  nodes = [
    RdnaLabel(name="L_LOOP"),
    RdnaInstruction(opcode="s_nop", operands=[]),
    RdnaInstruction(opcode="buffer_store_dword", operands=[RdnaVGPR(index=0), RdnaSGPR(index=0)]),
    RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)]),
    RdnaInstruction(
      opcode="image_load",
      operands=[RdnaVGPR(index=0, count=4), RdnaVGPR(index=4, count=4), RdnaSGPR(index=0, count=4)],
    ),
    RdnaInstruction(opcode="s_branch", operands=[RdnaLabelRef(name="L_LOOP")]),
    RdnaInstruction(opcode="global_store", operands=[RdnaMemory(base=RdnaVGPR(index=0), offset=4)]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=0), _RawRdnaOperand("s[0:1]")]),
    RdnaInstruction(opcode="s_mov_b32", operands=[RdnaSGPR(index=0), _RawRdnaOperand("#hashtag")]),
  ]
  code = synth.to_python(nodes).code
  assert "# RdnaLabel: L_LOOP" in code
  assert "rdna.s_nop()" in code
  assert "rdna.buffer_store_dword(v0, s0)" in code
  assert "v0 = rdna.v_add_f32(v1, v2)" in code
  assert "v_0_3 = rdna.image_load(v_4_7, s_0_3)" in code
  assert "s_0_1" in code
  assert "'#hashtag'" in code

  py_int = synth._convert_operand_to_py(RdnaImmediate(value=10, is_hex=False))
  assert isinstance(py_int, cst.Integer)
  assert py_int.value == "10"

  py_hex = synth._convert_operand_to_py(RdnaImmediate(value=255, is_hex=True))
  assert isinstance(py_hex, cst.Integer)
  assert py_hex.value == "0xff"

  py_flt = synth._convert_operand_to_py(RdnaImmediate(value=1.5, is_hex=False))
  assert isinstance(py_flt, cst.Float)
  assert py_flt.value == "1.5"

  py_braced = synth._convert_operand_to_py(_RawRdnaOperand("[v0]"))
  assert isinstance(py_braced, cst.Name)
  assert py_braced.value == "_v0"

  py_special = synth._convert_operand_to_py(_RawRdnaOperand("something!"))
  assert isinstance(py_special, cst.SimpleString)
  assert py_special.value == "'something!'"
