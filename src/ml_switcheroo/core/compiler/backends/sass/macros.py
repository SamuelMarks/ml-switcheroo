"""SASS Macro Expansion Logic.

This module defines procedural generators for complex SASS instruction kernels.
Unlike 1:1 mappings (e.g. ``Add`` -> ``FADD``), these macros generate entire
control flow blocks (loops, address calculations, memory loads) required to
implement high-level Neural Network layers like Convolution and Linear layers
directly in assembly.
"""

from typing import List, Protocol, Dict, Any

from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassOperand,
  SassInstruction,
  SassLabel,
  SassRegister,
  SassImmediate,
  SassMemory,
  SassPredicate,
  SassComment,
  SassNode,
)


class RegisterAllocatorProtocol(Protocol):
  """Protocol for the SassRegister Allocator used during expansion."""

  def get_register(self, var_name: str) -> SassRegister:
    """Gets or allocates a register for a symbolic variable.

    Args:
       var_name (str): The logical identifier.

    Returns:
       SassRegister: The physical register.

    """
    ...

  def allocate_temp(self) -> SassRegister:
    """Allocates an anonymous temporary register.

    Returns:
       SassRegister: The physical register.

    """
    ...


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for a 2D Convolution loop.

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

  # 1. SassRegister Allocation
  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  # Assume base pointers are passed in standard input regs (simulated here)
  # In a full compiler, these would come from input edges
  r_base_img = SassRegister(name="R3")
  r_base_wgt = SassRegister(name="R4")

  # SassPredicate for loops
  p_loop = SassPredicate(name="P0")

  # Labels
  l_ky_start: SassOperand = SassLabel(name=f"L_KY_{node_id}")
  l_kx_start: SassOperand = SassLabel(name=f"L_KX_{node_id}")

  # 2. Setup (Comments and Clear Accumulator)
  nodes.append(SassComment(text=f"BEGIN Conv2d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_ky, SassRegister(name="RZ")]))

  # 3. Y Loop
  nodes.append(l_ky_start)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kx, SassRegister(name="RZ")]))

  # 4. X Loop
  nodes.append(l_kx_start)

  # Address Calculation (Simplified IMAD: Base + Offset)
  nodes.append(SassComment(text="Calc Address & Load Image Pixel"))
  # R_ADDR = R_BASE + R_KY * STRIDE + R_KX * 4
  # Simplified simulation: just add offsets
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, SassImmediate(value=32), r_base_img]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val_i, SassMemory(base=r_addr_calc)]))

  nodes.append(SassComment(text="Calc Address & Load Weight"))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, SassImmediate(value=16), r_base_wgt]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val_w, SassMemory(base=r_addr_calc)]))

  # Math: Accum += Val * Wgt
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_acc, r_val_i, r_val_w, r_acc]))

  # 5. Loop Control X
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kx, r_kx, SassImmediate(value=1), SassRegister(name="RZ")]))
  # Compare Kx < 3 (Kernel Size)
  kernel_size = int(metadata.get("k", 3))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kx, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  # Branch back
  nodes.append(SassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  # 6. Loop Control Y
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_ky, r_ky, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_ky, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  nodes.append(SassComment(text=f"END Conv2d ({node_id})"))

  return nodes


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for a Linear Layer (Matrix Multiply).

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
  r_ptr_i = SassRegister(name="R2")
  r_ptr_w = SassRegister(name="R3")

  p_loop = SassPredicate(name="P0")
  l_gemm: SassOperand = SassLabel(name=f"L_GEMM_{node_id}")

  # 2. Setup
  limit = int(metadata.get("in_features", 128))

  nodes.append(SassComment(text=f"BEGIN Linear ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_counter, SassRegister(name="RZ")]))

  # 3. GEMM Loop
  nodes.append(l_gemm)

  # Load
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val_i, SassMemory(base=r_ptr_i)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val_w, SassMemory(base=r_ptr_w)]))

  # Math
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_acc, r_val_i, r_val_w, r_acc]))

  # Increment Pointers (float32 = 4 bytes)
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_ptr_i, r_ptr_i, SassImmediate(value=4), SassRegister(name="RZ")])
  )
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_ptr_w, r_ptr_w, SassImmediate(value=4), SassRegister(name="RZ")])
  )

  # Loop Check
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_counter, r_counter, SassImmediate(value=1), SassRegister(name="RZ")])
  )
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_counter, SassImmediate(value=limit), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_gemm], predicate=p_loop))

  # 4. Optional Bias
  if "bias" in metadata and metadata["bias"]:
    nodes.append(SassComment(text="Add Bias"))
    r_bias_val = allocator.allocate_temp()
    r_bias_ptr = SassRegister(name="R5")  # Assumed
    nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_bias_val, SassMemory(base=r_bias_ptr)]))
    nodes.append(SassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_bias_val]))

  nodes.append(SassComment(text=f"END Linear ({node_id})"))
  return nodes


def expand_mean(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for a Mean reduction loop.

  Calculates the sum over elements, and then multiplies the accumulator by
  the reciprocal of the number of elements to compute the average.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Operation metadata (expects "elements" key).

  Returns:
      List[SassNode]: Sequence of instructions for the mean kernel.

  """
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_val = allocator.allocate_temp()
  r_ptr = SassRegister(name="R2")
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_MEAN_{node_id}")
  limit = int(metadata.get("elements", 128))

  nodes.append(SassComment(text=f"BEGIN Mean ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_counter, SassRegister(name="RZ")]))
  nodes.append(l_loop)
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val, SassMemory(base=r_ptr)]))
  nodes.append(SassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_val]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_ptr, r_ptr, SassImmediate(value=4), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_counter, r_counter, SassImmediate(value=1), SassRegister(name="RZ")])
  )
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_counter, SassImmediate(value=limit), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  # Multiply by inverse of count
  inv_count = 1.0 / limit if limit > 0 else 0.0
  r_inv = allocator.allocate_temp()
  nodes.append(SassInstruction(opcode="MOV", operands=[r_inv, SassImmediate(value=inv_count)]))
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_acc, r_acc, r_inv]))
  nodes.append(SassComment(text=f"END Mean ({node_id})"))
  return nodes


def expand_relu(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for ReLU.

  Performs element-wise maximum comparison against zero using `FMAX`.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata.

  Returns:
      List[SassNode]: Sequence of instructions implementing ReLU.

  """
  nodes: List[SassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()  # Assume input is loaded here
  nodes.append(SassComment(text=f"BEGIN ReLU ({node_id})"))
  nodes.append(SassInstruction(opcode="FMAX", operands=[r_dst, r_src, SassRegister(name="RZ")]))
  nodes.append(SassComment(text=f"END ReLU ({node_id})"))
  return nodes


def expand_flatten(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for Flatten.

  Generates an assignment instruction representing a logical reshape/flatten
  by moving the source pointer value to the destination register.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata.

  Returns:
      List[SassNode]: Sequence of instructions implementing Flatten.

  """
  nodes: List[SassNode] = []
  nodes.append(SassComment(text=f"BEGIN Flatten ({node_id})"))
  # Logical reshape, just pointer assignment
  r_dst = allocator.get_register(node_id)
  r_src = SassRegister(name="R2")
  nodes.append(SassInstruction(opcode="MOV", operands=[r_dst, r_src]))
  nodes.append(SassComment(text=f"END Flatten ({node_id})"))
  return nodes


def expand_reshape(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for Reshape.

  Generates an assignment instruction representing a logical reshape
  by moving the source pointer value to the destination register.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata.

  Returns:
      List[SassNode]: Sequence of instructions implementing Reshape.

  """
  nodes: List[SassNode] = []
  nodes.append(SassComment(text=f"BEGIN Reshape ({node_id})"))
  # Logical reshape, just pointer assignment
  r_dst = allocator.get_register(node_id)
  r_src = SassRegister(name="R2")
  nodes.append(SassInstruction(opcode="MOV", operands=[r_dst, r_src]))
  nodes.append(SassComment(text=f"END Reshape ({node_id})"))
  return nodes


def expand_conv3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for 3D Convolution.

  Logic flow:
  1. Initialize Accumulator (R_ACC) and Z Loop counter (R_KZ).
  2. Outer Loop over Z (depth) -> Middle Loop over Y (height) -> Inner Loop over X (width).
  3. Calculate multidimensional memory addresses (IMAD) for input image and weights.
  4. Load input values (LDG) and weights.
  5. Fused Multiply-Add (FFMA).
  6. Increment loop counters, verify bounds, and conditional branch back.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata (expects "k" for kernel size).

  Returns:
      List[SassNode]: Sequence of labels and instructions implementing 3D Convolution.

  """
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  r_kz = allocator.allocate_temp()
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  r_base_img = SassRegister(name="R3")
  r_base_wgt = SassRegister(name="R4")
  p_loop = SassPredicate(name="P0")

  l_kz_start: SassOperand = SassLabel(name=f"L_KZ_{node_id}")
  l_ky_start: SassOperand = SassLabel(name=f"L_KY_{node_id}")
  l_kx_start: SassOperand = SassLabel(name=f"L_KX_{node_id}")

  nodes.append(SassComment(text=f"BEGIN Conv3d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kz, SassRegister(name="RZ")]))

  nodes.append(l_kz_start)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_ky, SassRegister(name="RZ")]))

  nodes.append(l_ky_start)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kx, SassRegister(name="RZ")]))

  nodes.append(l_kx_start)
  nodes.append(SassComment(text="Calc Address & Load Image Pixel"))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_kz, SassImmediate(value=64), r_base_img]))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, SassImmediate(value=32), r_addr_calc]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val_i, SassMemory(base=r_addr_calc)]))

  nodes.append(SassComment(text="Calc Address & Load Weight"))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_kz, SassImmediate(value=32), r_base_wgt]))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, SassImmediate(value=16), r_addr_calc]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val_w, SassMemory(base=r_addr_calc)]))

  nodes.append(SassInstruction(opcode="FFMA", operands=[r_acc, r_val_i, r_val_w, r_acc]))

  kernel_size = int(metadata.get("k", 3))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kx, r_kx, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kx, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_ky, r_ky, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_ky, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kz, r_kz, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kz, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_kz_start], predicate=p_loop))

  nodes.append(SassComment(text=f"END Conv3d ({node_id})"))
  return nodes


def expand_avgpool2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for AvgPool2d.

  Logic flow:
  1. Initialize Accumulator (R_ACC) to zero.
  2. Nested loops over Kernel Y and Kernel X.
  3. Load values (LDG).
  4. Add to accumulator (FADD).
  5. Multiply accumulator by 1/(Kx*Ky) (FMUL).

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration (k, stride, etc).

  Returns:
      List[SassNode]: Sequence of labels and instructions.
  """
  nodes: List[SassNode] = []

  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  r_base_img = SassRegister(name="R3")
  p_loop = SassPredicate(name="P0")

  l_ky_start: SassOperand = SassLabel(name=f"L_KY_{node_id}")
  l_kx_start: SassOperand = SassLabel(name=f"L_KX_{node_id}")

  nodes.append(SassComment(text=f"BEGIN AvgPool2d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_ky, SassRegister(name="RZ")]))

  nodes.append(l_ky_start)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kx, SassRegister(name="RZ")]))

  nodes.append(l_kx_start)
  nodes.append(SassComment(text="Calc Address & Load Image Pixel"))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, SassImmediate(value=32), r_base_img]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val, SassMemory(base=r_addr_calc)]))

  nodes.append(SassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_val]))

  kernel_size = int(metadata.get("kernel_size", 3))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kx, r_kx, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kx, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_ky, r_ky, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_ky, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  # Multiply by inverse of kernel_size^2
  inv_count = 1.0 / (kernel_size * kernel_size) if kernel_size > 0 else 0.0
  r_inv = allocator.allocate_temp()
  nodes.append(SassInstruction(opcode="MOV", operands=[r_inv, SassImmediate(value=inv_count)]))
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_acc, r_acc, r_inv]))

  nodes.append(SassComment(text=f"END AvgPool2d ({node_id})"))

  return nodes


def expand_maxpool2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for MaxPool2d.

  Logic flow:
  1. Initialize Accumulator (R_ACC) to strongly negative value.
  2. Nested loops over Kernel Y and Kernel X.
  3. Load values (LDG).
  4. Maximize with accumulator (FMAX).

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration (k, stride, etc).

  Returns:
      List[SassNode]: Sequence of labels and instructions.
  """
  nodes: List[SassNode] = []

  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  r_base_img = SassRegister(name="R3")
  p_loop = SassPredicate(name="P0")

  l_ky_start: SassOperand = SassLabel(name=f"L_KY_{node_id}")
  l_kx_start: SassOperand = SassLabel(name=f"L_KX_{node_id}")

  nodes.append(SassComment(text=f"BEGIN MaxPool2d ({node_id})"))
  # Initialize with a very small number (e.g. -inf) - using a large negative literal
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassImmediate(value=-99999.0)]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_ky, SassRegister(name="RZ")]))

  nodes.append(l_ky_start)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kx, SassRegister(name="RZ")]))

  nodes.append(l_kx_start)
  nodes.append(SassComment(text="Calc Address & Load Image Pixel"))
  nodes.append(SassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, SassImmediate(value=32), r_base_img]))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val, SassMemory(base=r_addr_calc)]))

  nodes.append(SassInstruction(opcode="FMAX", operands=[r_acc, r_acc, r_val]))

  kernel_size = int(metadata.get("kernel_size", 3))
  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kx, r_kx, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kx, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_ky, r_ky, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_ky, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  nodes.append(SassComment(text=f"END MaxPool2d ({node_id})"))

  return nodes


def expand_batchnorm2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for BatchNorm2d.

  Logic flow:
  1. Load mean, variance, gamma, beta from memory.
  2. Compute inv_std = 1.0 / sqrt(var + eps).
  3. Load input tensor value.
  4. Compute (x - mean) * inv_std * gamma + beta.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of labels and instructions.
  """
  nodes: List[SassNode] = []

  r_dst = allocator.get_register(node_id)
  r_val = allocator.allocate_temp()
  r_mean = allocator.allocate_temp()
  r_var = allocator.allocate_temp()
  r_gamma = allocator.allocate_temp()
  r_beta = allocator.allocate_temp()
  r_inv_std = allocator.allocate_temp()

  # Base pointers (simulated inputs)
  r_base_x = SassRegister(name="R3")
  r_base_mean = SassRegister(name="R4")
  r_base_var = SassRegister(name="R5")
  r_base_gamma = SassRegister(name="R6")
  r_base_beta = SassRegister(name="R7")

  eps = float(metadata.get("eps", 1e-5))

  nodes.append(SassComment(text=f"BEGIN BatchNorm2d ({node_id})"))

  # Load parameters
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val, SassMemory(base=r_base_x)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_mean, SassMemory(base=r_base_mean)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_var, SassMemory(base=r_base_var)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_gamma, SassMemory(base=r_base_gamma)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_beta, SassMemory(base=r_base_beta)]))

  # Compute inv_std = 1.0 / sqrt(var + eps)
  nodes.append(SassInstruction(opcode="FADD", operands=[r_var, r_var, SassImmediate(value=eps)]))
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_inv_std, r_var]))  # MUFU RSQ typically handles 1/sqrt

  # Compute output = (val - mean) * (gamma * inv_std) + beta
  # 1. val_centered = val - mean (FADD with negated mean ideally, simplified here)
  r_temp1 = allocator.allocate_temp()
  nodes.append(SassInstruction(opcode="FADD", operands=[r_temp1, r_val, r_mean]))  # Note: Should be subtract

  # 2. scale = gamma * inv_std
  r_scale = allocator.allocate_temp()
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_scale, r_gamma, r_inv_std]))

  # 3. out = val_centered * scale + beta
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_dst, r_temp1, r_scale, r_beta]))

  nodes.append(SassComment(text=f"END BatchNorm2d ({node_id})"))

  return nodes


def expand_dropout(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for Dropout.

  Logic flow:
  1. Load input value.
  2. Generate or load a random float in [0, 1).
  3. Compare random value with dropout probability.
  4. Scale output or set to 0.

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of labels and instructions.
  """
  nodes: List[SassNode] = []

  r_dst = allocator.get_register(node_id)
  r_val = allocator.allocate_temp()
  r_rand = allocator.allocate_temp()
  r_scale = allocator.allocate_temp()

  r_base_x = SassRegister(name="R3")
  r_base_rand = SassRegister(name="R4")

  p = float(metadata.get("p", 0.5))
  scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

  nodes.append(SassComment(text=f"BEGIN Dropout ({node_id})"))

  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_val, SassMemory(base=r_base_x)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_rand, SassMemory(base=r_base_rand)]))

  p_keep = SassPredicate(name="P0")
  nodes.append(
    SassInstruction(
      opcode="FSETP.GE.AND",
      operands=[p_keep, SassRegister(name="PT"), r_rand, SassImmediate(value=p), SassRegister(name="PT")],
    )
  )

  nodes.append(SassInstruction(opcode="MOV", operands=[r_scale, SassImmediate(value=scale)]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_dst, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_dst, r_val, r_scale], predicate=p_keep))

  nodes.append(SassComment(text=f"END Dropout ({node_id})"))

  return nodes


def expand_sigmoid(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for Sigmoid.

  1 / (1 + exp(-x)) -> 1 / (1 + exp2(-x * log2(e)))

  Args:
      allocator (RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node.
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  nodes: List[SassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()  # Assume loaded from R2
  r_tmp = allocator.allocate_temp()

  nodes.append(SassComment(text=f"BEGIN Sigmoid ({node_id})"))

  # R_SRC = x (Assume it is passed in R2 for simple ops)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_src, SassRegister(name="R2")]))

  # x * -log2(e) -> R_TMP (approx -1.442695)
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_tmp, r_src, SassImmediate(value=-1.442695)]))

  # MUFU.EX2
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_tmp, r_tmp]))  # EX2 mode implicit

  # 1 + exp2
  nodes.append(SassInstruction(opcode="FADD", operands=[r_tmp, r_tmp, SassImmediate(value=1.0)]))

  # 1 / (1 + exp2) -> MUFU.RCP
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_dst, r_tmp]))  # RCP mode implicit

  nodes.append(SassComment(text=f"END Sigmoid ({node_id})"))
  return nodes


def expand_tanh(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for Tanh."""
  nodes: List[SassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()

  nodes.append(SassComment(text=f"BEGIN Tanh ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_src, SassRegister(name="R2")]))
  # Simplified macro representation for Tanh
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_dst, r_src]))  # Tanh implicit in our IR
  nodes.append(SassComment(text=f"END Tanh ({node_id})"))
  return nodes


def expand_gelu(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for GELU."""
  nodes: List[SassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()
  r_tmp = allocator.allocate_temp()

  nodes.append(SassComment(text=f"BEGIN GELU ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_src, SassRegister(name="R2")]))
  # Fast approx: x * sigmoid(1.702 * x)
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_tmp, r_src, SassImmediate(value=1.702)]))
  # Sigmoid inline
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_tmp, r_tmp, SassImmediate(value=-1.442695)]))
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_tmp, r_tmp]))
  nodes.append(SassInstruction(opcode="FADD", operands=[r_tmp, r_tmp, SassImmediate(value=1.0)]))
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_tmp, r_tmp]))

  nodes.append(SassInstruction(opcode="FMUL", operands=[r_dst, r_src, r_tmp]))
  nodes.append(SassComment(text=f"END GELU ({node_id})"))
  return nodes


def expand_mseloss(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for MSELoss.

  Accumulates (pred - target)^2 over N elements.
  """
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_pred = allocator.allocate_temp()
  r_tgt = allocator.allocate_temp()
  r_diff = allocator.allocate_temp()
  r_sq = allocator.allocate_temp()

  r_ptr_pred = SassRegister(name="R2")
  r_ptr_tgt = SassRegister(name="R3")

  limit = int(metadata.get("elements", 128))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_MSE_{node_id}")

  nodes.append(SassComment(text=f"BEGIN MSELoss ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_counter, SassRegister(name="RZ")]))

  nodes.append(l_loop)
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_pred, SassMemory(base=r_ptr_pred)]))
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_tgt, SassMemory(base=r_ptr_tgt)]))

  # diff = pred - tgt
  nodes.append(SassInstruction(opcode="FADD", operands=[r_diff, r_pred, r_tgt]))  # Needs negation in full implementation

  # sq = diff * diff
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_sq, r_diff, r_diff]))

  # acc += sq
  nodes.append(SassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_sq]))

  # Pointers and loop control
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_ptr_pred, r_ptr_pred, SassImmediate(value=4), SassRegister(name="RZ")])
  )
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_ptr_tgt, r_ptr_tgt, SassImmediate(value=4), SassRegister(name="RZ")])
  )
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_counter, r_counter, SassImmediate(value=1), SassRegister(name="RZ")])
  )

  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_counter, SassImmediate(value=limit), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  if metadata.get("reduction", "mean") == "mean":
    inv_count = 1.0 / limit if limit > 0 else 0.0
    r_inv = allocator.allocate_temp()
    nodes.append(SassInstruction(opcode="MOV", operands=[r_inv, SassImmediate(value=inv_count)]))
    nodes.append(SassInstruction(opcode="FMUL", operands=[r_acc, r_acc, r_inv]))

  nodes.append(SassComment(text=f"END MSELoss ({node_id})"))
  return nodes


def expand_crossentropyloss(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for CrossEntropyLoss."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_prob = allocator.allocate_temp()
  r_log = allocator.allocate_temp()

  r_ptr_prob = SassRegister(name="R2")

  limit = int(metadata.get("elements", 32))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_CE_{node_id}")

  nodes.append(SassComment(text=f"BEGIN CrossEntropyLoss ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_counter, SassRegister(name="RZ")]))

  nodes.append(l_loop)
  # Simplified: Load probability of correct class directly
  nodes.append(SassInstruction(opcode="LDG.E.F32", operands=[r_prob, SassMemory(base=r_ptr_prob)]))

  # log2(prob)
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_log, r_prob]))  # LG2 implicit

  # ln(prob) = log2(prob) * 0.693147
  nodes.append(SassInstruction(opcode="FMUL", operands=[r_log, r_log, SassImmediate(value=0.693147)]))

  # acc += -ln(prob)
  nodes.append(SassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_log]))  # Needs proper negation

  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_ptr_prob, r_ptr_prob, SassImmediate(value=4), SassRegister(name="RZ")])
  )
  nodes.append(
    SassInstruction(opcode="IADD3", operands=[r_counter, r_counter, SassImmediate(value=1), SassRegister(name="RZ")])
  )

  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_counter, SassImmediate(value=limit), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(SassComment(text=f"END CrossEntropyLoss ({node_id})"))
  return nodes


def expand_rnn(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for a basic RNN cell over time.

  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
  """
  nodes: List[SassNode] = []
  r_h = allocator.get_register(node_id)
  r_t = allocator.allocate_temp()

  seq_len = int(metadata.get("seq_len", 10))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_RNN_TIME_{node_id}")

  nodes.append(SassComment(text=f"BEGIN RNN ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_t, SassRegister(name="RZ")]))
  # Initialize hidden state (could be loaded from R5)
  nodes.append(SassInstruction(opcode="MOV", operands=[r_h, SassRegister(name="RZ")]))

  nodes.append(l_loop)

  # Simulated math for Wx + Wh + b
  nodes.append(SassComment(text="Compute RNN gates (Wx + Wh + b)"))
  r_gate = allocator.allocate_temp()
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_gate, r_h, SassRegister(name="R3"), SassRegister(name="R4")]))

  # Tanh activation
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_h, r_gate]))  # Tanh implicit

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_t, r_t, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_t, SassImmediate(value=seq_len), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(SassComment(text=f"END RNN ({node_id})"))
  return nodes


def expand_lstm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for LSTM over time."""
  nodes: List[SassNode] = []
  r_h = allocator.get_register(node_id)
  r_c = allocator.allocate_temp()
  r_t = allocator.allocate_temp()

  seq_len = int(metadata.get("seq_len", 10))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_LSTM_TIME_{node_id}")

  nodes.append(SassComment(text=f"BEGIN LSTM ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_t, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_h, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_c, SassRegister(name="RZ")]))

  nodes.append(l_loop)

  # Simulated gates
  nodes.append(SassComment(text="Compute LSTM gates (i, f, g, o)"))
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_c, r_c, SassRegister(name="R3"), SassRegister(name="R4")]))
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_h, r_c]))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_t, r_t, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_t, SassImmediate(value=seq_len), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(SassComment(text=f"END LSTM ({node_id})"))
  return nodes


def expand_gru(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for GRU over time."""
  nodes: List[SassNode] = []
  r_h = allocator.get_register(node_id)
  r_t = allocator.allocate_temp()

  seq_len = int(metadata.get("seq_len", 10))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_GRU_TIME_{node_id}")

  nodes.append(SassComment(text=f"BEGIN GRU ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_t, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_h, SassRegister(name="RZ")]))

  nodes.append(l_loop)

  # Simulated gates
  nodes.append(SassComment(text="Compute GRU gates (r, z, n)"))
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_h, r_h, SassRegister(name="R3"), SassRegister(name="R4")]))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_t, r_t, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_t, SassImmediate(value=seq_len), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(SassComment(text=f"END GRU ({node_id})"))
  return nodes


def expand_multiheadattention(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for MultiheadAttention."""
  nodes: List[SassNode] = []
  r_out = allocator.get_register(node_id)
  r_q = allocator.allocate_temp()
  r_k = allocator.allocate_temp()
  r_v = allocator.allocate_temp()

  nodes.append(SassComment(text=f"BEGIN MultiheadAttention ({node_id})"))

  # Simulated Q, K, V projections
  nodes.append(SassComment(text="Q, K, V Projections"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_q, SassRegister(name="R2")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_k, SassRegister(name="R3")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_v, SassRegister(name="R4")]))

  # Simulated Attention: softmax(Q*K^T/sqrt(d)) * V
  nodes.append(SassComment(text="Attention = Softmax(Q*K^T) * V"))
  r_attn = allocator.allocate_temp()
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_attn, r_q, r_k, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MUFU", operands=[r_attn, r_attn]))  # Softmax approx
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_out, r_attn, r_v, SassRegister(name="RZ")]))

  nodes.append(SassComment(text=f"END MultiheadAttention ({node_id})"))
  return nodes


def expand_transformer(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for a Transformer block."""
  nodes: List[SassNode] = []
  r_out = allocator.get_register(node_id)
  r_in = allocator.allocate_temp()
  r_tmp = allocator.allocate_temp()

  nodes.append(SassComment(text=f"BEGIN Transformer ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_in, SassRegister(name="R2")]))

  # Self Attention
  nodes.append(SassComment(text="Self Attention"))
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_tmp, r_in, r_in, r_in]))

  # FFN
  nodes.append(SassComment(text="Feed Forward"))
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_out, r_tmp, SassRegister(name="R3"), r_tmp]))

  nodes.append(SassComment(text=f"END Transformer ({node_id})"))
  return nodes


def expand_transformerencoder(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for TransformerEncoder."""
  nodes: List[SassNode] = []
  r_out = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN TransformerEncoder ({node_id})"))
  # Simplified
  nodes.append(SassInstruction(opcode="MOV", operands=[r_out, SassRegister(name="R2")]))
  nodes.append(SassComment(text=f"END TransformerEncoder ({node_id})"))
  return nodes


def expand_transformerdecoder(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for TransformerDecoder."""
  nodes: List[SassNode] = []
  r_out = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN TransformerDecoder ({node_id})"))
  # Simplified
  nodes.append(SassInstruction(opcode="MOV", operands=[r_out, SassRegister(name="R2")]))
  nodes.append(SassComment(text=f"END TransformerDecoder ({node_id})"))
  return nodes


def expand_conv1d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for 1D Convolution."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  r_kx = allocator.allocate_temp()
  kernel_size = int(metadata.get("k", 3))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_KX_{node_id}")

  nodes.append(SassComment(text=f"BEGIN Conv1d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kx, SassRegister(name="RZ")]))

  nodes.append(l_loop)
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_acc, SassRegister(name="R3"), SassRegister(name="R4"), r_acc]))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kx, r_kx, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kx, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(SassComment(text=f"END Conv1d ({node_id})"))
  return nodes


def expand_depthwiseconv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for DepthwiseConv2d."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  r_kx = allocator.allocate_temp()
  kernel_size = int(metadata.get("k", 3))
  p_loop = SassPredicate(name="P0")
  l_loop: SassOperand = SassLabel(name=f"L_DW_KX_{node_id}")

  nodes.append(SassComment(text=f"BEGIN DepthwiseConv2d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="RZ")]))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_kx, SassRegister(name="RZ")]))

  nodes.append(l_loop)
  nodes.append(SassInstruction(opcode="FFMA", operands=[r_acc, SassRegister(name="R3"), SassRegister(name="R4"), r_acc]))

  nodes.append(SassInstruction(opcode="IADD3", operands=[r_kx, r_kx, SassImmediate(value=1), SassRegister(name="RZ")]))
  nodes.append(
    SassInstruction(
      opcode="ISETP.LT.AND",
      operands=[p_loop, SassRegister(name="PT"), r_kx, SassImmediate(value=kernel_size), SassRegister(name="PT")],
    )
  )
  nodes.append(SassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(SassComment(text=f"END DepthwiseConv2d ({node_id})"))
  return nodes


def expand_convtranspose(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for ConvTranspose (generic representation)."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN ConvTranspose ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END ConvTranspose ({node_id})"))
  return nodes


def expand_pool1d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for 1D Pooling."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN Pool1d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END Pool1d ({node_id})"))
  return nodes


def expand_pool3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for 3D Pooling."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN Pool3d ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END Pool3d ({node_id})"))
  return nodes


def expand_adaptivepool(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for Adaptive Pooling."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN AdaptivePool ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END AdaptivePool ({node_id})"))
  return nodes


def expand_generic_norm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for generic Normalization."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN Norm ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END Norm ({node_id})"))
  return nodes


def expand_generic_activation(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for generic Activation."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN Activation ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END Activation ({node_id})"))
  return nodes


def expand_generic_linalg(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for generic Linear Algebra op."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN LinAlg ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END LinAlg ({node_id})"))
  return nodes


def expand_generic_reduction(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for generic Reduction."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN Reduction ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END Reduction ({node_id})"))
  return nodes


def expand_generic_loss(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for generic Loss."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN Loss ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END Loss ({node_id})"))
  return nodes


def expand_generic_dropout(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates SASS kernel for generic Dropout."""
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN DropoutVar ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END DropoutVar ({node_id})"))
  return nodes
