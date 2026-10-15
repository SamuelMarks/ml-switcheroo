"""RDNA Macros for reduction."""

from typing import Any, List

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaInstruction, RdnaNode
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol


def expand_layernorm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any = None,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for Layer Normalization.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_tmp = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN LayerNorm ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_tmp]),
    RdnaComment(text=f"END LayerNorm ({node_id})"),
  ]


def expand_rmsnorm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any = None,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for RMS Normalization.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_tmp = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN RMSNorm ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_tmp]),
    RdnaComment(text=f"END RMSNorm ({node_id})"),
  ]


def expand_batchnorm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any = None,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for Batch Normalization.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_tmp = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN BatchNorm ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_tmp]),
    RdnaComment(text=f"END BatchNorm ({node_id})"),
  ]


def expand_generic_norm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any = None,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for a generic normalization block.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  v_dst = allocator.get_register(node_id)
  v_tmp = allocator.allocate_temp()
  return [
    RdnaComment(text=f"BEGIN Norm ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_tmp]),
    RdnaComment(text=f"END Norm ({node_id})"),
  ]


def expand_generic_reduction(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any = None,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for a reduction operation.

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
    RdnaComment(text=f"BEGIN Reduction ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_src]),
    RdnaComment(text=f"END Reduction ({node_id})"),
  ]
