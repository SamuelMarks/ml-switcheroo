"""RDNA Macros for gemm."""

import typing

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaNode,
  RdnaReg,
)
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol


@typing.runtime_checkable
class _ScalarRegisterAllocatorProtocol(typing.Protocol):
  """Protocol for register allocators providing `allocate_scalar_temp`."""

  def allocate_scalar_temp(self) -> RdnaReg:
    """Allocate a scalar temporary register.

    Returns:
        Allocated scalar register.
    """
    ...


def expand_linear(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue],
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for a Linear Layer.

  Constructs the sequence of CST nodes implementing a matrix-vector or vector-vector
  reduction loop for a linear fully-connected layer.

  Args:
      allocator: The register allocator to use for
          managing temporary and variable registers.
      node_id: A unique identifier for the linear operation node.
      metadata: Metadata containing configuration details
          such as "in_features" (input feature size) and "bias" (boolean flag).

  Returns:
      List[RdnaNode]: A list of RDNA CST nodes representing the compiled linear
      layer logic.
  """
  nodes: list[RdnaNode] = []

  v_acc = allocator.get_register(node_id)
  alloc_scalar = (
    allocator.allocate_scalar_temp if isinstance(allocator, _ScalarRegisterAllocatorProtocol) else allocator.allocate_temp
  )
  s_idx = alloc_scalar()
  v_val_i = allocator.allocate_temp()
  v_val_w = allocator.allocate_temp()
  v_ptr_i = allocator.allocate_temp()
  v_ptr_w = allocator.allocate_temp()

  l_gemm = RdnaLabel(name=f"L_GEMM_{node_id}")
  raw_limit = metadata.get("in_features", metadata.get("input_dims", 128))
  limit = int(raw_limit) if isinstance(raw_limit, (int, float, str)) else 128
  raw_out = metadata.get("out_features", 0)
  out_features = int(raw_out) if isinstance(raw_out, (int, float, str)) else 0

  nodes.append(RdnaComment(text=f"BEGIN Linear ({node_id})"))
  if ("in_features" in metadata or "input_dims" in metadata) and (
    "runtime_comment" in metadata or "num_elements" in metadata
  ):
    nodes.append(RdnaComment(text=f"[on-device shape] in_features={limit}, out_features={out_features}"))
  nodes.append(RdnaInstruction(opcode="v_mov_b32", operands=[v_acc, RdnaImmediate(value=0)]))
  nodes.append(RdnaInstruction(opcode="s_mov_b32", operands=[s_idx, RdnaImmediate(value=0)]))

  nodes.append(l_gemm)
  nodes.append(
    RdnaInstruction(opcode="global_load_dword", operands=[v_val_i, RdnaMemory(base=v_ptr_i), RdnaModifier(name="off")])
  )
  nodes.append(
    RdnaInstruction(opcode="global_load_dword", operands=[v_val_w, RdnaMemory(base=v_ptr_w), RdnaModifier(name="off")])
  )

  nodes.append(RdnaInstruction(opcode="s_waitcnt", operands=[RdnaModifier(name="vmcnt(0)")]))
  nodes.append(RdnaInstruction(opcode="v_fmac_f32", operands=[v_acc, v_val_i, v_val_w]))

  nodes.append(RdnaInstruction(opcode="v_add_u32", operands=[v_ptr_i, v_ptr_i, RdnaImmediate(value=4)]))
  nodes.append(RdnaInstruction(opcode="v_add_u32", operands=[v_ptr_w, v_ptr_w, RdnaImmediate(value=4)]))

  nodes.append(RdnaInstruction(opcode="s_add_i32", operands=[s_idx, s_idx, RdnaImmediate(value=1)]))
  nodes.append(RdnaInstruction(opcode="s_cmp_lt_i32", operands=[s_idx, RdnaImmediate(value=limit)]))
  nodes.append(RdnaInstruction(opcode="s_cbranch_scc1", operands=[RdnaLabelRef(name=l_gemm.name)]))

  if metadata.get("bias"):
    nodes.append(RdnaComment(text="Add Bias"))
    v_bias = allocator.allocate_temp()
    nodes.append(RdnaInstruction(opcode="v_add_f32", operands=[v_acc, v_acc, v_bias]))

  nodes.append(RdnaComment(text=f"END Linear ({node_id})"))
  return nodes


def expand_generic_linalg(
  allocator: RegisterAllocatorProtocol[RdnaReg],
  node_id: str,
  metadata: typing.Mapping[str, AttributeValue] | None = None,
) -> list[RdnaNode]:
  """Generate the RDNA assembly kernel for a linear algebra operation.

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
    RdnaComment(text=f"BEGIN LinAlg ({node_id})"),
    RdnaInstruction(opcode="v_mov_b32", operands=[v_dst, v_src]),
    RdnaComment(text=f"END LinAlg ({node_id})"),
  ]
