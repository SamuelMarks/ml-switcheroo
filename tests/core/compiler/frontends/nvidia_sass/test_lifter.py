"""Unit tests for `NvidiaSassLifter` (`ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter`)."""

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter


def test_nvidia_sass_lifter_chains_markers_and_standalone_instructions() -> None:
  """Verify `NvidiaSassLifter.lift` handles chains, unmapped ops, deduplication, ALU, and non-ALU instructions."""
  lifter = NvidiaSassLifter()

  # Simple mixed instruction + semantic block chain
  chain_nodes: list[NvidiaSassNode] = [
    NvidiaSassComment(text="Input x -> R0"),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="RZ")]),
    NvidiaSassComment(text="BEGIN Conv2d (conv1)"),
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R0")],
    ),
    NvidiaSassComment(text="END Conv2d (conv1)"),
    NvidiaSassComment(text="Return: R1"),
  ]
  chain_g = lifter.lift(chain_nodes)
  assert set(chain_g.nodes.keys()) == {"x", "R1", "conv1", "output"}
  assert {"asm.MOV", "Conv2d"} <= {n.op_type for n in chain_g.nodes.values()}
  assert len(chain_g.edges) == 3

  # Complex multi-layer chain with `torch.flatten` `start_dim = 1` default
  complex_g = lifter.lift(
    [
      NvidiaSassComment(text="Input x -> R0"),
      NvidiaSassComment(text="BEGIN Conv2d (conv)"),
      NvidiaSassComment(text="END Conv2d (conv)"),
      NvidiaSassComment(text="Unmapped Op: torch.flatten (func_flatten)"),
      NvidiaSassComment(text="BEGIN Linear (fc)"),
      NvidiaSassComment(text="END Linear (fc)"),
      NvidiaSassComment(text="Return: R7"),
    ]
  )
  assert list(complex_g.nodes.keys()) == ["x", "conv", "func_flatten", "fc", "output"]
  assert complex_g.nodes["func_flatten"].attributes["arg_1"] == 1
  assert (complex_g.edges[0].source, complex_g.edges[0].target) == ("x", "conv")

  # Custom unmapped op, plain comment, duplicate blocks/inputs/returns, ALU, and non-ALU branch
  unmapped_g = lifter.lift([NvidiaSassComment(text="// Unmapped Op: custom.op (custom_id)")])
  assert unmapped_g.nodes["custom_id"].op_type == "custom.op"

  assert len(lifter.lift([NvidiaSassComment(text="// Just a comment without marker")]).nodes) == 0

  dup_blk = lifter.lift(
    [
      NvidiaSassComment(text="BEGIN Layer (l1)"),
      NvidiaSassComment(text="END Layer (l1)"),
      NvidiaSassComment(text="BEGIN Layer (l1)"),
      NvidiaSassComment(text="END Layer (l1)"),
    ]
  )
  assert list(dup_blk.nodes.keys()) == ["l1"]

  dup_in = lifter.lift([NvidiaSassComment(text="// Input x -> x"), NvidiaSassComment(text="// Input x -> x")])
  assert list(dup_in.nodes.keys()) == ["x"]

  dup_ret = lifter.lift([NvidiaSassComment(text="// Return: output"), NvidiaSassComment(text="// Return: output")])
  assert list(dup_ret.nodes.keys()) == ["output"]

  alu_g = lifter.lift(
    [
      NvidiaSassInstruction(
        opcode="IADD3",
        operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")],
      )
    ]
  )
  assert alu_g.nodes["R0"].op_type == "asm.IADD3"
  assert (alu_g.nodes["R0"].attributes["arg_0"], alu_g.nodes["R0"].attributes["arg_2"]) == ("R0", "R2")

  bra_g = lifter.lift([NvidiaSassInstruction(opcode="BRA", operands=[NvidiaSassImmediate(value=10)])])
  assert bra_g.nodes["inst_0"].op_type == "asm.BRA"
  assert bra_g.nodes["inst_0"].attributes["arg_0"] == "10"
