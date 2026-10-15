"""RDNA Macros for conv."""

from typing import Any, List

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaNode,
)
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for a 2D Convolution loop.

  Constructs the sequence of CST nodes implementing a nested 2D convolution
  kernel loop with register allocation, memory loads, and accumulation.

  Args:
      allocator: The register allocator to use for
            managing temporary and variable registers.
      node_id: A unique identifier for the convolution operation node.
      metadata: Metadata containing configuration details
            such as "k" (kernel size).

  Returns:
      List[RdnaNode]: A list of RDNA CST nodes representing the compiled 2D
      convolution kernel logic.
  """
  nodes: List[RdnaNode] = []

  v_acc = allocator.get_register(node_id)
  s_ky = allocator.allocate_temp()
  s_kx = allocator.allocate_temp()
  v_val_i = allocator.allocate_temp()
  v_val_w = allocator.allocate_temp()
  v_addr = allocator.allocate_temp()

  v_base_img = allocator.get_register("x")
  v_base_wgt = allocator.allocate_temp()

  l_ky_start = RdnaLabel(name=f"L_KY_{node_id}")
  l_kx_start = RdnaLabel(name=f"L_KX_{node_id}")

  kernel_size = int(metadata.get("k", 3))

  nodes.append(RdnaComment(text=f"BEGIN Conv2d ({node_id})"))
  nodes.append(RdnaComment(text="Zero Accumulator"))
  nodes.append(RdnaInstruction(opcode="v_mov_b32", operands=[v_acc, RdnaImmediate(value=0)]))
  nodes.append(RdnaInstruction(opcode="s_mov_b32", operands=[s_ky, RdnaImmediate(value=0)]))

  nodes.append(l_ky_start)
  nodes.append(RdnaInstruction(opcode="s_mov_b32", operands=[s_kx, RdnaImmediate(value=0)]))
  nodes.append(l_kx_start)

  nodes.append(RdnaComment(text="Calc Address & Load Image"))
  nodes.append(RdnaInstruction(opcode="v_add_f32", operands=[v_addr, v_base_img, v_base_img]))
  nodes.append(
    RdnaInstruction(opcode="global_load_dword", operands=[v_val_i, RdnaMemory(base=v_addr), RdnaModifier(name="off")])
  )

  nodes.append(RdnaComment(text="Load Weight"))
  nodes.append(
    RdnaInstruction(opcode="global_load_dword", operands=[v_val_w, RdnaMemory(base=v_base_wgt), RdnaModifier(name="off")])
  )

  nodes.append(RdnaInstruction(opcode="s_waitcnt", operands=[RdnaModifier(name="vmcnt(0)")]))

  nodes.append(RdnaInstruction(opcode="v_fmac_f32", operands=[v_acc, v_val_i, v_val_w]))

  nodes.append(RdnaInstruction(opcode="s_add_i32", operands=[s_kx, s_kx, RdnaImmediate(value=1)]))
  nodes.append(RdnaInstruction(opcode="s_cmp_lt_i32", operands=[s_kx, RdnaImmediate(value=kernel_size)]))
  nodes.append(RdnaInstruction(opcode="s_cbranch_scc1", operands=[RdnaLabelRef(name=l_kx_start.name)]))

  nodes.append(RdnaInstruction(opcode="s_add_i32", operands=[s_ky, s_ky, RdnaImmediate(value=1)]))
  nodes.append(RdnaInstruction(opcode="s_cmp_lt_i32", operands=[s_ky, RdnaImmediate(value=kernel_size)]))
  nodes.append(RdnaInstruction(opcode="s_cbranch_scc1", operands=[RdnaLabelRef(name=l_ky_start.name)]))

  nodes.append(RdnaComment(text=f"END Conv2d ({node_id})"))
  return nodes


def expand_conv3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for Conv3d.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  nodes: List[RdnaNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()
  nodes.append(RdnaComment(text=f"BEGIN Conv3d ({node_id})"))
  label = RdnaLabel(name=f"BB_{node_id.replace('-', '_')}_1")
  nodes.append(label)
  nodes.append(RdnaInstruction(opcode="v_fmac_f32", operands=[r_dst, r_src, r_src]))
  nodes.append(RdnaInstruction(opcode="s_cbranch_vccnz", operands=[RdnaLabelRef(name=label.name)]))
  nodes.append(RdnaComment(text=f"END Conv3d ({node_id})"))
  return nodes


def expand_conv_general_dilated(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[RdnaNode]:
  """Expand a conv_general_dilated operation into RDNA nodes.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [
    RdnaComment(text=f"BEGIN conv_general_dilated ({node_id})"),
    RdnaComment(text=f"END conv_general_dilated ({node_id})"),
  ]
