"""
RDNA Macro Expansion Logic.

This module defines procedural generators for complex RDNA kernel logic.
Unlike 1:1 opcode mappings, these macros generate full instruction sequences
including loops, memory loads, wait counters, and math operations required
to implement high-level Neural Network layers like `Conv2d` and `Linear`
directly in assembly.

It adheres to the `RegisterAllocatorProtocol` to manage Scalar (SGPR) and
Vector (VGPR) usage separately.
"""

from typing import Any, Dict, List, Protocol

from ml_switcheroo.core.rdna.nodes import (
  Comment,
  Immediate,
  Instruction,
  Label,
  Memory,
  Modifier,
  RdnaNode,
  SGPR,
  VGPR,
)


class RegisterAllocatorProtocol(Protocol):
  """Protocol for the Dual-Pool Register Allocator used during expansion."""

  def get_vector_register(self, var_name: str) -> VGPR:
    """
    Gets or allocates a VGPR for a symbolic variable.

    Args:
       var_name (str): The logical identifier.

    Returns:
       VGPR: The physical register.
    """
    ...

  def get_scalar_register(self, var_name: str) -> SGPR:
    """
    Gets or allocates an SGPR for a symbolic variable.

    Args:
       var_name (str): The logical identifier.

    Returns:
       SGPR: The physical register.
    """
    ...

  def allocate_vector_temp(self) -> VGPR:
    """
    Allocates an anonymous temporary VGPR.

    Returns:
       VGPR: The physical register.
    """
    ...

  def allocate_scalar_temp(self) -> SGPR:
    """
    Allocates an anonymous temporary SGPR.

    Returns:
       SGPR: The physical register.
    """
    ...


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """
  Generates the RDNA assembly kernel for a 2D Convolution loop.

  Logic flow:
  1.  Initialize Accumulator (VGPR).
  2.  Setup Loop Counters (SGPR: Ky, Kx).
  3.  Enter Y Loop -> Enter X Loop.
  4.  Calculate addresses (v_add_f32 placeholder for indexing).
  5.  Issue Loads (`global_load_dword`).
  6.  Wait for memory (`s_waitcnt vmcnt(0)`).
  7.  Math: Fused Multiply-Accumulate (`v_fmac_f32`).
  8.  Increment counters and conditional branch (`s_cbranch_scc1`).

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node.
      metadata (Dict[str, Any]): Layer configuration (k=kernel_size).

  Returns:
      List[RdnaNode]: Sequence of RDNA AST nodes.
  """
  nodes: List[RdnaNode] = []

  # 1. Register Allocation
  # Output Accumulator (Vector)
  v_acc = allocator.get_vector_register(node_id)
  # Loop Counters (Scalar)
  s_ky = allocator.allocate_scalar_temp()
  s_kx = allocator.allocate_scalar_temp()
  # Data Values (Vector)
  v_val_i = allocator.allocate_vector_temp()
  v_val_w = allocator.allocate_vector_temp()
  # Address Calc Temp (Vector)
  v_addr = allocator.allocate_vector_temp()

  # Inputs (Simulated mapping from implicit previous nodes)
  # In P4 synthesizer, inputs are allocated to v0, v1...
  # We assume v0 is Image Base, v1 is Weight Base
  v_base_img = allocator.get_vector_register("x")  # Input
  # Weights usually loaded from constant or global. Assume v1 for pointer
  v_base_wgt = allocator.allocate_vector_temp()

  # Labels
  l_ky_start = Label(f"L_KY_{node_id}")
  l_kx_start = Label(f"L_KX_{node_id}")

  kernel_size = int(metadata.get("k", 3))

  # 2. Setup
  nodes.append(Comment(f"BEGIN Conv2d ({node_id})"))
  nodes.append(Comment("Zero Accumulator"))
  nodes.append(Instruction("v_mov_b32", [v_acc, Immediate(0)]))
  nodes.append(Instruction("s_mov_b32", [s_ky, Immediate(0)]))

  # 3. Y Loop
  nodes.append(l_ky_start)
  nodes.append(Instruction("s_mov_b32", [s_kx, Immediate(0)]))

  # 4. X Loop
  nodes.append(l_kx_start)

  # Address Calculation (Placeholder logic)
  # RDNA uses specific addressing modes often, but here we simulate pointer math
  nodes.append(Comment("Calc Address & Load Image"))
  # v_addr = v_base + offset (simulated)
  nodes.append(Instruction("v_add_f32", [v_addr, v_base_img, v_base_img]))  # Stub
  nodes.append(Instruction("global_load_dword", [v_val_i, Memory(v_addr), Modifier("off")]))

  nodes.append(Comment("Load Weight"))
  # Weights often Scalar load if uniform, but Per-Pixel weights are Vector.
  # We simulate Global Load as Vector.
  nodes.append(Instruction("global_load_dword", [v_val_w, Memory(v_base_wgt), Modifier("off")]))

  # Barrier
  nodes.append(Instruction("s_waitcnt", [Modifier("vmcnt(0)")]))

  # Math: Accum += Val * Wgt
  nodes.append(Instruction("v_fmac_f32", [v_acc, v_val_i, v_val_w]))

  # 5. Loop Control X
  nodes.append(Instruction("s_add_i32", [s_kx, s_kx, Immediate(1)]))  # Increment scalar counter
  nodes.append(Instruction("s_cmp_lt_i32", [s_kx, Immediate(kernel_size)]))  # Compare sets SCC
  nodes.append(Instruction("s_cbranch_scc1", [l_kx_start]))  # Branch if SCC true

  # 6. Loop Control Y
  nodes.append(Instruction("s_add_i32", [s_ky, s_ky, Immediate(1)]))
  nodes.append(Instruction("s_cmp_lt_i32", [s_ky, Immediate(kernel_size)]))
  nodes.append(Instruction("s_cbranch_scc1", [l_ky_start]))

  nodes.append(Comment(f"END Conv2d ({node_id})"))
  return nodes


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """
  Generates the RDNA assembly kernel for a Linear Layer (Matrix Multiply).

  Structure:
  1. Zero Accumulator.
  2. Loop over input features (Dot Product).
  3. Global Load input/weight.
  4. Wait for memory.
  5. Fused Multiply-Accumulate.
  6. Loop Control.
  7. Bias addition (optional).

  Args:
      allocator (RegisterAllocatorProtocol): Registry.
      node_id (str): Op ID.
      metadata (Dict[str, Any]): Attributes (in_features, bias).

  Returns:
      List[RdnaNode]: Instructions.
  """
  nodes: List[RdnaNode] = []

  # 1. Allocation
  v_acc = allocator.get_vector_register(node_id)
  s_idx = allocator.allocate_scalar_temp()
  v_val_i = allocator.allocate_vector_temp()
  v_val_w = allocator.allocate_vector_temp()

  # Pointers
  v_ptr_i = allocator.allocate_vector_temp()
  v_ptr_w = allocator.allocate_vector_temp()

  l_gemm = Label(f"L_GEMM_{node_id}")

  limit = int(metadata.get("in_features", 128))

  # 2. Setup
  nodes.append(Comment(f"BEGIN Linear ({node_id})"))
  nodes.append(Instruction("v_mov_b32", [v_acc, Immediate(0)]))
  nodes.append(Instruction("s_mov_b32", [s_idx, Immediate(0)]))

  # 3. Loop
  nodes.append(l_gemm)

  # Loads
  nodes.append(Instruction("global_load_dword", [v_val_i, Memory(v_ptr_i), Modifier("off")]))
  nodes.append(Instruction("global_load_dword", [v_val_w, Memory(v_ptr_w), Modifier("off")]))

  # Wait
  nodes.append(Instruction("s_waitcnt", [Modifier("vmcnt(0)")]))

  # Math
  nodes.append(Instruction("v_fmac_f32", [v_acc, v_val_i, v_val_w]))

  # Pointer arithmetic (4 bytes)
  nodes.append(Instruction("v_add_u32", [v_ptr_i, v_ptr_i, Immediate(4)]))
  nodes.append(Instruction("v_add_u32", [v_ptr_w, v_ptr_w, Immediate(4)]))

  # Control
  nodes.append(Instruction("s_add_i32", [s_idx, s_idx, Immediate(1)]))
  nodes.append(Instruction("s_cmp_lt_i32", [s_idx, Immediate(limit)]))
  nodes.append(Instruction("s_cbranch_scc1", [l_gemm]))

  # 4. Optional Bias
  if "bias" in metadata and metadata["bias"]:
    nodes.append(Comment("Add Bias"))
    v_bias = allocator.allocate_vector_temp()
    # Mock load bias
    nodes.append(Instruction("v_add_f32", [v_acc, v_acc, v_bias]))

  nodes.append(Comment(f"END Linear ({node_id})"))
  return nodes
