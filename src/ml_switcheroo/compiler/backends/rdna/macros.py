"""
RDNA Macro Expansion Logic.

Procedural generators for complex RDNA kernel logic (Conv2d, Linear).
"""

from typing import Any, Dict, List, Protocol

from ml_switcheroo.compiler.frontends.rdna.nodes import (
  Comment,
  Immediate,
  Instruction,
  Label,
  LabelRef,
  Memory,
  Modifier,
  RdnaNode,
  SGPR,
  VGPR,
)


class RegisterAllocatorProtocol(Protocol):
  """Protocol for the Dual-Pool Register Allocator."""

  def get_vector_register(self, var_name: str) -> VGPR:
    """Gets or allocates a VGPR."""
    ...

  def get_scalar_register(self, var_name: str) -> SGPR:
    """Gets or allocates an SGPR."""
    ...

  def allocate_vector_temp(self) -> VGPR:
    """Allocates temp VGPR."""
    ...

  def allocate_scalar_temp(self) -> SGPR:
    """Allocates temp SGPR."""
    ...


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for a 2D Convolution loop."""
  nodes: List[RdnaNode] = []

  v_acc = allocator.get_vector_register(node_id)
  s_ky = allocator.allocate_scalar_temp()
  s_kx = allocator.allocate_scalar_temp()
  v_val_i = allocator.allocate_vector_temp()
  v_val_w = allocator.allocate_vector_temp()
  v_addr = allocator.allocate_vector_temp()

  v_base_img = allocator.get_vector_register("x")
  v_base_wgt = allocator.allocate_vector_temp()

  l_ky_start = Label(f"L_KY_{node_id}")
  l_kx_start = Label(f"L_KX_{node_id}")

  kernel_size = int(metadata.get("k", 3))

  nodes.append(Comment(f"BEGIN Conv2d ({node_id})"))
  nodes.append(Comment("Zero Accumulator"))
  nodes.append(Instruction("v_mov_b32", [v_acc, Immediate(0)]))
  nodes.append(Instruction("s_mov_b32", [s_ky, Immediate(0)]))

  nodes.append(l_ky_start)
  nodes.append(Instruction("s_mov_b32", [s_kx, Immediate(0)]))
  nodes.append(l_kx_start)

  nodes.append(Comment("Calc Address & Load Image"))
  nodes.append(Instruction("v_add_f32", [v_addr, v_base_img, v_base_img]))
  nodes.append(Instruction("global_load_dword", [v_val_i, Memory(v_addr), Modifier("off")]))

  nodes.append(Comment("Load Weight"))
  nodes.append(Instruction("global_load_dword", [v_val_w, Memory(v_base_wgt), Modifier("off")]))

  nodes.append(Instruction("s_waitcnt", [Modifier("vmcnt(0)")]))

  nodes.append(Instruction("v_fmac_f32", [v_acc, v_val_i, v_val_w]))

  nodes.append(Instruction("s_add_i32", [s_kx, s_kx, Immediate(1)]))
  nodes.append(Instruction("s_cmp_lt_i32", [s_kx, Immediate(kernel_size)]))
  nodes.append(Instruction("s_cbranch_scc1", [LabelRef(l_kx_start.name)]))

  nodes.append(Instruction("s_add_i32", [s_ky, s_ky, Immediate(1)]))
  nodes.append(Instruction("s_cmp_lt_i32", [s_ky, Immediate(kernel_size)]))
  nodes.append(Instruction("s_cbranch_scc1", [LabelRef(l_ky_start.name)]))

  nodes.append(Comment(f"END Conv2d ({node_id})"))
  return nodes


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for a Linear Layer."""
  nodes: List[RdnaNode] = []

  v_acc = allocator.get_vector_register(node_id)
  s_idx = allocator.allocate_scalar_temp()
  v_val_i = allocator.allocate_vector_temp()
  v_val_w = allocator.allocate_vector_temp()
  v_ptr_i = allocator.allocate_vector_temp()
  v_ptr_w = allocator.allocate_vector_temp()

  l_gemm = Label(f"L_GEMM_{node_id}")
  limit = int(metadata.get("in_features", 128))

  nodes.append(Comment(f"BEGIN Linear ({node_id})"))
  nodes.append(Instruction("v_mov_b32", [v_acc, Immediate(0)]))
  nodes.append(Instruction("s_mov_b32", [s_idx, Immediate(0)]))

  nodes.append(l_gemm)
  nodes.append(Instruction("global_load_dword", [v_val_i, Memory(v_ptr_i), Modifier("off")]))
  nodes.append(Instruction("global_load_dword", [v_val_w, Memory(v_ptr_w), Modifier("off")]))

  nodes.append(Instruction("s_waitcnt", [Modifier("vmcnt(0)")]))
  nodes.append(Instruction("v_fmac_f32", [v_acc, v_val_i, v_val_w]))

  nodes.append(Instruction("v_add_u32", [v_ptr_i, v_ptr_i, Immediate(4)]))
  nodes.append(Instruction("v_add_u32", [v_ptr_w, v_ptr_w, Immediate(4)]))

  nodes.append(Instruction("s_add_i32", [s_idx, s_idx, Immediate(1)]))
  nodes.append(Instruction("s_cmp_lt_i32", [s_idx, Immediate(limit)]))
  nodes.append(Instruction("s_cbranch_scc1", [LabelRef(l_gemm.name)]))

  if "bias" in metadata and metadata["bias"]:
    nodes.append(Comment("Add Bias"))
    v_bias = allocator.allocate_vector_temp()
    nodes.append(Instruction("v_add_f32", [v_acc, v_acc, v_bias]))

  nodes.append(Comment(f"END Linear ({node_id})"))
  return nodes
