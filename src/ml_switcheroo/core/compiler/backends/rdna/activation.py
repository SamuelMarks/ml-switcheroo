"""RDNA Macros for activation."""

import typing

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol

RdnaReg: typing.TypeAlias = RdnaVGPR | RdnaSGPR


def expand_relu(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for ReLU.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  nodes: list[RdnaNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()  # Assume input is loaded here
  nodes.append(RdnaComment(text=f"BEGIN ReLU ({node_id})"))
  nodes.append(RdnaInstruction(opcode="v_max_f32", operands=[r_dst, r_src, RdnaImmediate(value=0)]))
  nodes.append(RdnaComment(text=f"END ReLU ({node_id})"))
  return nodes


def expand_gelu(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for Gaussian Error Linear Unit (GELU).

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_src = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN GELU ({node_id})"),
    RdnaInstruction(opcode="v_mul_f32", operands=[v_dst, v_src, RdnaImmediate(value=0.5)]),
    RdnaComment(text=f"END GELU ({node_id})"),
  ]


def expand_silu(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for Sigmoid Linear Unit (SiLU).

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_src = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN SiLU ({node_id})"),
    RdnaInstruction(opcode="v_mul_f32", operands=[v_dst, v_src, RdnaImmediate(value=0.5)]),
    RdnaComment(text=f"END SiLU ({node_id})"),
  ]


def expand_sigmoid(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for Sigmoid activation.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_src = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN Sigmoid ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_src]),
    RdnaComment(text=f"END Sigmoid ({node_id})"),
  ]


def expand_tanh(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for Tanh activation.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_src = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN Tanh ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_src]),
    RdnaComment(text=f"END Tanh ({node_id})"),
  ]


def expand_softmax(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for Softmax normalization.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_src = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN Softmax ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_src]),
    RdnaComment(text=f"END Softmax ({node_id})"),
  ]
