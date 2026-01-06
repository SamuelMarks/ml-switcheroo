"""
SASS Macro Expansion Logic.

This module defines procedural generators for complex SASS instruction kernels.
Unlike 1:1 mappings (e.g. ``Add`` -> ``FADD``), these macros generate entire
control flow blocks (loops, address calculations, memory loads) required to
implement high-level Neural Network layers like Convolution and Linear layers
directly in assembly.
"""

from typing import List, Protocol, Dict, Any

from ml_switcheroo.core.sass.nodes import (
  Instruction,
  Label,
  Register,
  Immediate,
  Memory,
  Predicate,
  Comment,
  SassNode,
)


class RegisterAllocatorProtocol(Protocol):
  """Protocol for the Register Allocator used during expansion."""

  def get_register(self, var_name: str) -> Register:
    """
    Gets or allocates a register for a symbolic variable.

    Args:
       var_name (str): The logical identifier.

    Returns:
       Register: The physical register.
    """
    ...

  def allocate_temp(self) -> Register:
    """
    Allocates an anonymous temporary register.

    Returns:
       Register: The physical register.
    """
    ...


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """
  Generates the SASS assembly kernel for a 2D Convolution loop.

  Logic flow:
  1.  Initialize Accumulator (R_ACC).
  2.  Setup Loop Counters (Ky, Kx).
  3.  Enter Y Loop -> Enter X Loop.
  4.  Calculate addresses (IMAD) for image and weights.
  5.  Load values (LDG).
  6.  Multiply-Add (FFMA).
  7.  Increment and Branch.
  8.  Store result.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration (k, stride, etc).

  Returns:
      List[SassNode]: Sequence of labels and instructions.
  """
  nodes: List[SassNode] = []

  # 1. Register Allocation
  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  # Assume base pointers are passed in standard input regs (simulated here)
  # In a full compiler, these would come from input edges
  r_base_img = Register("R3")
  r_base_wgt = Register("R4")

  # Predicate for loops
  p_loop = Predicate("P0")

  # Labels
  l_ky_start = Label(f"L_KY_{node_id}")
  l_kx_start = Label(f"L_KX_{node_id}")

  # 2. Setup (Comments and Clear Accumulator)
  nodes.append(Comment(f"BEGIN Conv2d ({node_id})"))
  nodes.append(Instruction("MOV", [r_acc, Register("RZ")]))
  nodes.append(Instruction("MOV", [r_ky, Register("RZ")]))

  # 3. Y Loop
  nodes.append(l_ky_start)
  nodes.append(Instruction("MOV", [r_kx, Register("RZ")]))

  # 4. X Loop
  nodes.append(l_kx_start)

  # Address Calculation (Simplified IMAD: Base + Offset)
  nodes.append(Comment("Calc Address & Load Image Pixel"))
  # R_ADDR = R_BASE + R_KY * STRIDE + R_KX * 4
  # Simplified simulation: just add offsets
  nodes.append(Instruction("IMAD", [r_addr_calc, r_ky, Immediate(32), r_base_img]))
  nodes.append(Instruction("IADD3", [r_addr_calc, r_addr_calc, r_kx, Register("RZ")]))
  nodes.append(Instruction("LDG.E.F32", [r_val_i, Memory(r_addr_calc)]))

  nodes.append(Comment("Calc Address & Load Weight"))
  nodes.append(Instruction("IMAD", [r_addr_calc, r_ky, Immediate(16), r_base_wgt]))
  nodes.append(Instruction("IADD3", [r_addr_calc, r_addr_calc, r_kx, Register("RZ")]))
  nodes.append(Instruction("LDG.E.F32", [r_val_w, Memory(r_addr_calc)]))

  # Math: Accum += Val * Wgt
  nodes.append(Instruction("FFMA", [r_acc, r_val_i, r_val_w, r_acc]))

  # 5. Loop Control X
  nodes.append(Instruction("IADD3", [r_kx, r_kx, Immediate(1), Register("RZ")]))
  # Compare Kx < 3 (Kernel Size)
  kernel_size = int(metadata.get("k", 3))
  nodes.append(Instruction("ISETP.LT.AND", [p_loop, Register("PT"), r_kx, Immediate(kernel_size), Register("PT")]))
  # Branch back
  nodes.append(Instruction("BRA", [l_kx_start], predicate=p_loop))

  # 6. Loop Control Y
  nodes.append(Instruction("IADD3", [r_ky, r_ky, Immediate(1), Register("RZ")]))
  nodes.append(Instruction("ISETP.LT.AND", [p_loop, Register("PT"), r_ky, Immediate(kernel_size), Register("PT")]))
  nodes.append(Instruction("BRA", [l_ky_start], predicate=p_loop))

  nodes.append(Comment(f"END Conv2d ({node_id})"))

  return nodes


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """
  Generates the SASS assembly kernel for a Linear Layer (Matrix Multiply).

  Structure:
  1. Initialize Accumulator.
  2. Loop over input features (Dot Product).
  3. Load Input element and Weight element.
  4. Fused Multiply-Add.
  5. Increment pointers.
  6. Add Bias (if present).

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node.
      metadata (Dict[str, Any]): Attributes (in_features, out_features).

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  nodes: List[SassNode] = []

  # 1. Allocation
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()

  # Assume generic pointer inputs
  r_ptr_i = Register("R2")
  r_ptr_w = Register("R3")

  p_loop = Predicate("P0")
  l_gemm = Label(f"L_GEMM_{node_id}")

  # 2. Setup
  limit = int(metadata.get("in_features", 128))

  nodes.append(Comment(f"BEGIN Linear ({node_id})"))
  nodes.append(Instruction("MOV", [r_acc, Register("RZ")]))
  nodes.append(Instruction("MOV", [r_counter, Register("RZ")]))

  # 3. GEMM Loop
  nodes.append(l_gemm)

  # Load
  nodes.append(Instruction("LDG.E.F32", [r_val_i, Memory(r_ptr_i)]))
  nodes.append(Instruction("LDG.E.F32", [r_val_w, Memory(r_ptr_w)]))

  # Math
  nodes.append(Instruction("FFMA", [r_acc, r_val_i, r_val_w, r_acc]))

  # Increment Pointers (float32 = 4 bytes)
  nodes.append(Instruction("IADD3", [r_ptr_i, r_ptr_i, Immediate(4), Register("RZ")]))
  nodes.append(Instruction("IADD3", [r_ptr_w, r_ptr_w, Immediate(4), Register("RZ")]))

  # Loop Check
  nodes.append(Instruction("IADD3", [r_counter, r_counter, Immediate(1), Register("RZ")]))
  nodes.append(Instruction("ISETP.LT.AND", [p_loop, Register("PT"), r_counter, Immediate(limit), Register("PT")]))
  nodes.append(Instruction("BRA", [l_gemm], predicate=p_loop))

  # 4. Optional Bias
  if "bias" in metadata and metadata["bias"]:
    nodes.append(Comment("Add Bias"))
    r_bias_val = allocator.allocate_temp()
    r_bias_ptr = Register("R5")  # Assumed
    nodes.append(Instruction("LDG.E.F32", [r_bias_val, Memory(r_bias_ptr)]))
    nodes.append(Instruction("FADD", [r_acc, r_acc, r_bias_val]))

  nodes.append(Comment(f"END Linear ({node_id})"))
  return nodes
