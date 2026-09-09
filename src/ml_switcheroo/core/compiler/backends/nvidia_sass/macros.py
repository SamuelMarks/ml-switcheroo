"""NVIDIA_SASS Macro Expansion Logic.

This module defines procedural generators for complex NVIDIA_SASS instruction kernels.
Unlike 1:1 mappings (e.g. ``Add`` -> ``FADD``), these macros generate entire
control flow blocks (loops, address calculations, memory loads) required to
implement high-level Neural Network layers like Convolution and Linear layers
directly in assembly.
"""

from typing import List, Protocol

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassOperand,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassRegister,
  NvidiaSassImmediate,
  NvidiaSassMemory,
  NvidiaSassPredicate,
  NvidiaSassComment,
  NvidiaSassNode,
)


class RegisterAllocatorProtocol(Protocol):
  """Protocol for the NvidiaSassRegister Allocator used during expansion."""

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Get or allocates a register for a symbolic variable.

    Args:
        var_name: The logical identifier.

    Returns:
       NvidiaSassRegister: The physical register.

    """
    ...

  def allocate_temp(self) -> NvidiaSassRegister:
    """Allocate an anonymous temporary register.

    Returns:
       NvidiaSassRegister: The physical register.

    """
    ...


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a 2D Convolution loop.

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
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration (k, stride, etc).

  Returns:
      List[NvidiaSassNode]: Sequence of labels and instructions.

  """
  nodes: List[NvidiaSassNode] = []

  # 1. NvidiaSassRegister Allocation
  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  # Assume base pointers are passed in standard input regs (simulated here)
  # In a full compiler, these would come from input edges
  r_base_img = NvidiaSassRegister(name="R3")
  r_base_wgt = NvidiaSassRegister(name="R4")

  # NvidiaSassPredicate for loops
  p_loop = NvidiaSassPredicate(name="P0")

  # Labels
  l_ky_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KY_{node_id}")
  l_kx_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KX_{node_id}")

  # 2. Setup (Comments and Clear Accumulator)
  nodes.append(NvidiaSassComment(text=f"BEGIN Conv2d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_ky, NvidiaSassRegister(name="RZ")]))

  # 3. Y Loop
  nodes.append(l_ky_start)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kx, NvidiaSassRegister(name="RZ")]))

  # 4. X Loop
  nodes.append(l_kx_start)

  # Address Calculation (Simplified IMAD: Base + Offset)
  nodes.append(NvidiaSassComment(text="Calc Address & Load Image Pixel"))
  # R_ADDR = R_BASE + R_KY * STRIDE + R_KX * 4
  # Simplified simulation: just add offsets
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, NvidiaSassImmediate(value=32), r_base_img])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, NvidiaSassRegister(name="RZ")])
  )
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val_i, NvidiaSassMemory(base=r_addr_calc)]))

  nodes.append(NvidiaSassComment(text="Calc Address & Load Weight"))
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, NvidiaSassImmediate(value=16), r_base_wgt])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, NvidiaSassRegister(name="RZ")])
  )
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val_w, NvidiaSassMemory(base=r_addr_calc)]))

  # Math: Accum += Val * Wgt
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_acc, r_val_i, r_val_w, r_acc]))

  # 5. Loop Control X
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_kx, r_kx, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  # Compare Kx < 3 (Kernel Size)
  kernel_size = int(metadata.get("k", 3))
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_kx,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  # Branch back
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  # 6. Loop Control Y
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ky, r_ky, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_ky,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END Conv2d ({node_id})"))

  return nodes


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a Linear Layer (Matrix Multiply).

  Structure:
  1. Initialize Accumulator.
  2. Loop over input features (Dot Product).
  3. Load Input element and Weight element.
  4. Fused Multiply-Add.
  5. Increment pointers.
  6. Add Bias (if present).

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node.
      metadata (Dict[str, Any]): Attributes (in_features, out_features).

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.

  """
  nodes: List[NvidiaSassNode] = []

  # 1. Allocation
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()

  # Assume generic pointer inputs
  r_ptr_i = NvidiaSassRegister(name="R2")
  r_ptr_w = NvidiaSassRegister(name="R3")

  p_loop = NvidiaSassPredicate(name="P0")
  l_gemm: NvidiaSassOperand = NvidiaSassLabel(name=f"L_GEMM_{node_id}")

  # 2. Setup
  limit = int(metadata.get("in_features", 128))

  nodes.append(NvidiaSassComment(text=f"BEGIN Linear ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_counter, NvidiaSassRegister(name="RZ")]))

  # 3. GEMM Loop
  nodes.append(l_gemm)

  # Load
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val_i, NvidiaSassMemory(base=r_ptr_i)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val_w, NvidiaSassMemory(base=r_ptr_w)]))

  # Math
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_acc, r_val_i, r_val_w, r_acc]))

  # Increment Pointers (float32 = 4 bytes)
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ptr_i, r_ptr_i, NvidiaSassImmediate(value=4), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ptr_w, r_ptr_w, NvidiaSassImmediate(value=4), NvidiaSassRegister(name="RZ")]
    )
  )

  # Loop Check
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_counter, r_counter, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_counter,
        NvidiaSassImmediate(value=limit),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_gemm], predicate=p_loop))

  # 4. Optional Bias
  if "bias" in metadata and metadata["bias"]:
    nodes.append(NvidiaSassComment(text="Add Bias"))
    r_bias_val = allocator.allocate_temp()
    r_bias_ptr = NvidiaSassRegister(name="R5")  # Assumed
    nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_bias_val, NvidiaSassMemory(base=r_bias_ptr)]))
    nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_bias_val]))

  nodes.append(NvidiaSassComment(text=f"END Linear ({node_id})"))
  return nodes


def expand_mean(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a Mean reduction loop.

  Calculates the sum over elements, and then multiplies the accumulator by
  the reciprocal of the number of elements to compute the average.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Operation metadata (expects "elements" key).

  Returns:
      List[NvidiaSassNode]: Sequence of instructions for the mean kernel.

  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_val = allocator.allocate_temp()
  r_ptr = NvidiaSassRegister(name="R2")
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_MEAN_{node_id}")
  limit = int(metadata.get("elements", 128))

  nodes.append(NvidiaSassComment(text=f"BEGIN Mean ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_counter, NvidiaSassRegister(name="RZ")]))
  nodes.append(l_loop)
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val, NvidiaSassMemory(base=r_ptr)]))
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_val]))
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ptr, r_ptr, NvidiaSassImmediate(value=4), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_counter, r_counter, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_counter,
        NvidiaSassImmediate(value=limit),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  # Multiply by inverse of count
  inv_count = 1.0 / limit if limit > 0 else 0.0
  r_inv = allocator.allocate_temp()
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_inv, NvidiaSassImmediate(value=inv_count)]))
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_acc, r_acc, r_inv]))
  nodes.append(NvidiaSassComment(text=f"END Mean ({node_id})"))
  return nodes


def expand_relu(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for ReLU.

  Performs element-wise maximum comparison against zero using `FMAX`.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions implementing ReLU.

  """
  nodes: List[NvidiaSassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()  # Assume input is loaded here
  nodes.append(NvidiaSassComment(text=f"BEGIN ReLU ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="FMAX", operands=[r_dst, r_src, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassComment(text=f"END ReLU ({node_id})"))
  return nodes


def expand_flatten(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Flatten.

  Generates an assignment instruction representing a logical reshape/flatten
  by moving the source pointer value to the destination register.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions implementing Flatten.

  """
  nodes: List[NvidiaSassNode] = []
  nodes.append(NvidiaSassComment(text=f"BEGIN Flatten ({node_id})"))
  # Logical reshape, just pointer assignment
  r_dst = allocator.get_register(node_id)
  r_src = NvidiaSassRegister(name="R2")
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_dst, r_src]))
  nodes.append(NvidiaSassComment(text=f"END Flatten ({node_id})"))
  return nodes


def expand_reshape(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Reshape.

  Generates an assignment instruction representing a logical reshape
  by moving the source pointer value to the destination register.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions implementing Reshape.

  """
  nodes: List[NvidiaSassNode] = []
  nodes.append(NvidiaSassComment(text=f"BEGIN Reshape ({node_id})"))
  # Logical reshape, just pointer assignment
  r_dst = allocator.get_register(node_id)
  r_src = NvidiaSassRegister(name="R2")
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_dst, r_src]))
  nodes.append(NvidiaSassComment(text=f"END Reshape ({node_id})"))
  return nodes


def expand_conv3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for 3D Convolution.

  Logic flow:
  1. Initialize Accumulator (R_ACC) and Z Loop counter (R_KZ).
  2. Outer Loop over Z (depth) -> Middle Loop over Y (height) -> Inner Loop over X (width).
  3. Calculate multidimensional memory addresses (IMAD) for input image and weights.
  4. Load input values (LDG) and weights.
  5. Fused Multiply-Add (FFMA).
  6. Increment loop counters, verify bounds, and conditional branch back.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer metadata (expects "k" for kernel size).

  Returns:
      List[NvidiaSassNode]: Sequence of labels and instructions implementing 3D Convolution.

  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  r_kz = allocator.allocate_temp()
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val_i = allocator.allocate_temp()
  r_val_w = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  r_base_img = NvidiaSassRegister(name="R3")
  r_base_wgt = NvidiaSassRegister(name="R4")
  p_loop = NvidiaSassPredicate(name="P0")

  l_kz_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KZ_{node_id}")
  l_ky_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KY_{node_id}")
  l_kx_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KX_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN Conv3d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kz, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_kz_start)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_ky, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_ky_start)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kx, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_kx_start)
  nodes.append(NvidiaSassComment(text="Calc Address & Load Image Pixel"))
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_kz, NvidiaSassImmediate(value=64), r_base_img])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, NvidiaSassImmediate(value=32), r_addr_calc])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, NvidiaSassRegister(name="RZ")])
  )
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val_i, NvidiaSassMemory(base=r_addr_calc)]))

  nodes.append(NvidiaSassComment(text="Calc Address & Load Weight"))
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_kz, NvidiaSassImmediate(value=32), r_base_wgt])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, NvidiaSassImmediate(value=16), r_addr_calc])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, NvidiaSassRegister(name="RZ")])
  )
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val_w, NvidiaSassMemory(base=r_addr_calc)]))

  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_acc, r_val_i, r_val_w, r_acc]))

  kernel_size = int(metadata.get("k", 3))
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_kx, r_kx, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_kx,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ky, r_ky, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_ky,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_kz, r_kz, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_kz,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_kz_start], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END Conv3d ({node_id})"))
  return nodes


def expand_avgpool2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for AvgPool2d.

  Logic flow:
  1. Initialize Accumulator (R_ACC) to zero.
  2. Nested loops over Kernel Y and Kernel X.
  3. Load values (LDG).
  4. Add to accumulator (FADD).
  5. Multiply accumulator by 1/(Kx*Ky) (FMUL).

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration (k, stride, etc).

  Returns:
      List[NvidiaSassNode]: Sequence of labels and instructions.
  """
  nodes: List[NvidiaSassNode] = []

  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  r_base_img = NvidiaSassRegister(name="R3")
  p_loop = NvidiaSassPredicate(name="P0")

  l_ky_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KY_{node_id}")
  l_kx_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KX_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN AvgPool2d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_ky, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_ky_start)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kx, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_kx_start)
  nodes.append(NvidiaSassComment(text="Calc Address & Load Image Pixel"))
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, NvidiaSassImmediate(value=32), r_base_img])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, NvidiaSassRegister(name="RZ")])
  )
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val, NvidiaSassMemory(base=r_addr_calc)]))

  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_val]))

  kernel_size = int(metadata.get("kernel_size", 3))
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_kx, r_kx, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_kx,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ky, r_ky, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_ky,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  # Multiply by inverse of kernel_size^2
  inv_count = 1.0 / (kernel_size * kernel_size) if kernel_size > 0 else 0.0
  r_inv = allocator.allocate_temp()
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_inv, NvidiaSassImmediate(value=inv_count)]))
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_acc, r_acc, r_inv]))

  nodes.append(NvidiaSassComment(text=f"END AvgPool2d ({node_id})"))

  return nodes


def expand_maxpool2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for MaxPool2d.

  Logic flow:
  1. Initialize Accumulator (R_ACC) to strongly negative value.
  2. Nested loops over Kernel Y and Kernel X.
  3. Load values (LDG).
  4. Maximize with accumulator (FMAX).

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration (k, stride, etc).

  Returns:
      List[NvidiaSassNode]: Sequence of labels and instructions.
  """
  nodes: List[NvidiaSassNode] = []

  r_acc = allocator.get_register(node_id)
  r_ky = allocator.allocate_temp()
  r_kx = allocator.allocate_temp()
  r_val = allocator.allocate_temp()
  r_addr_calc = allocator.allocate_temp()

  r_base_img = NvidiaSassRegister(name="R3")
  p_loop = NvidiaSassPredicate(name="P0")

  l_ky_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KY_{node_id}")
  l_kx_start: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KX_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN MaxPool2d ({node_id})"))
  # Initialize with a very small number (e.g. -inf) - using a large negative literal
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassImmediate(value=-99999.0)]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_ky, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_ky_start)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kx, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_kx_start)
  nodes.append(NvidiaSassComment(text="Calc Address & Load Image Pixel"))
  nodes.append(
    NvidiaSassInstruction(opcode="IMAD", operands=[r_addr_calc, r_ky, NvidiaSassImmediate(value=32), r_base_img])
  )
  nodes.append(
    NvidiaSassInstruction(opcode="IADD3", operands=[r_addr_calc, r_addr_calc, r_kx, NvidiaSassRegister(name="RZ")])
  )
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val, NvidiaSassMemory(base=r_addr_calc)]))

  nodes.append(NvidiaSassInstruction(opcode="FMAX", operands=[r_acc, r_acc, r_val]))

  kernel_size = int(metadata.get("kernel_size", 3))
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_kx, r_kx, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_kx,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_kx_start], predicate=p_loop))

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ky, r_ky, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_ky,
        NvidiaSassImmediate(value=kernel_size),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_ky_start], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END MaxPool2d ({node_id})"))

  return nodes


def expand_batchnorm2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for BatchNorm2d.

  Logic flow:
  1. Load mean, variance, gamma, beta from memory.
  2. Compute inv_std = 1.0 / sqrt(var + eps).
  3. Load input tensor value.
  4. Compute (x - mean) * inv_std * gamma + beta.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of labels and instructions.
  """
  nodes: List[NvidiaSassNode] = []

  r_dst = allocator.get_register(node_id)
  r_val = allocator.allocate_temp()
  r_mean = allocator.allocate_temp()
  r_var = allocator.allocate_temp()
  r_gamma = allocator.allocate_temp()
  r_beta = allocator.allocate_temp()
  r_inv_std = allocator.allocate_temp()

  # Base pointers (simulated inputs)
  r_base_x = NvidiaSassRegister(name="R3")
  r_base_mean = NvidiaSassRegister(name="R4")
  r_base_var = NvidiaSassRegister(name="R5")
  r_base_gamma = NvidiaSassRegister(name="R6")
  r_base_beta = NvidiaSassRegister(name="R7")

  eps = float(metadata.get("eps", 1e-5))

  nodes.append(NvidiaSassComment(text=f"BEGIN BatchNorm2d ({node_id})"))

  # Load parameters
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val, NvidiaSassMemory(base=r_base_x)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_mean, NvidiaSassMemory(base=r_base_mean)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_var, NvidiaSassMemory(base=r_base_var)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_gamma, NvidiaSassMemory(base=r_base_gamma)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_beta, NvidiaSassMemory(base=r_base_beta)]))

  # Compute inv_std = 1.0 / sqrt(var + eps)
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_var, r_var, NvidiaSassImmediate(value=eps)]))
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_inv_std, r_var]))  # MUFU RSQ typically handles 1/sqrt

  # Compute output = (val - mean) * (gamma * inv_std) + beta
  # 1. val_centered = val - mean (FADD with negated mean ideally, simplified here)
  r_temp1 = allocator.allocate_temp()
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_temp1, r_val, r_mean]))  # Note: Should be subtract

  # 2. scale = gamma * inv_std
  r_scale = allocator.allocate_temp()
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_scale, r_gamma, r_inv_std]))

  # 3. out = val_centered * scale + beta
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_dst, r_temp1, r_scale, r_beta]))

  nodes.append(NvidiaSassComment(text=f"END BatchNorm2d ({node_id})"))

  return nodes


def expand_dropout(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Dropout.

  Logic flow:
  1. Load input value.
  2. Generate or load a random float in [0, 1).
  3. Compare random value with dropout probability.
  4. Scale output or set to 0.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of labels and instructions.
  """
  nodes: List[NvidiaSassNode] = []

  r_dst = allocator.get_register(node_id)
  r_val = allocator.allocate_temp()
  r_rand = allocator.allocate_temp()
  r_scale = allocator.allocate_temp()

  r_base_x = NvidiaSassRegister(name="R3")
  r_base_rand = NvidiaSassRegister(name="R4")

  p = float(metadata.get("p", 0.5))
  scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0

  nodes.append(NvidiaSassComment(text=f"BEGIN Dropout ({node_id})"))

  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_val, NvidiaSassMemory(base=r_base_x)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_rand, NvidiaSassMemory(base=r_base_rand)]))

  p_keep = NvidiaSassPredicate(name="P0")
  nodes.append(
    NvidiaSassInstruction(
      opcode="FSETP.GE.AND",
      operands=[
        p_keep,
        NvidiaSassRegister(name="PT"),
        r_rand,
        NvidiaSassImmediate(value=p),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )

  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_scale, NvidiaSassImmediate(value=scale)]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_dst, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_dst, r_val, r_scale], predicate=p_keep))

  nodes.append(NvidiaSassComment(text=f"END Dropout ({node_id})"))

  return nodes


def expand_sigmoid(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Sigmoid.

  1 / (1 + exp(-x)) -> 1 / (1 + exp2(-x * log2(e)))

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node.
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()  # Assume loaded from R2
  r_tmp = allocator.allocate_temp()

  nodes.append(NvidiaSassComment(text=f"BEGIN Sigmoid ({node_id})"))

  # R_SRC = x (Assume it is passed in R2 for simple ops)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_src, NvidiaSassRegister(name="R2")]))

  # x * -log2(e) -> R_TMP (approx -1.442695)
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_tmp, r_src, NvidiaSassImmediate(value=-1.442695)]))

  # MUFU.EX2
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_tmp, r_tmp]))  # EX2 mode implicit

  # 1 + exp2
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_tmp, r_tmp, NvidiaSassImmediate(value=1.0)]))

  # 1 / (1 + exp2) -> MUFU.RCP
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_dst, r_tmp]))  # RCP mode implicit

  nodes.append(NvidiaSassComment(text=f"END Sigmoid ({node_id})"))
  return nodes


def expand_tanh(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Tanh.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()

  nodes.append(NvidiaSassComment(text=f"BEGIN Tanh ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_src, NvidiaSassRegister(name="R2")]))
  # Simplified macro representation for Tanh
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_dst, r_src]))  # Tanh implicit in our IR
  nodes.append(NvidiaSassComment(text=f"END Tanh ({node_id})"))
  return nodes


def expand_gelu(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for GELU.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_dst = allocator.get_register(node_id)
  r_src = allocator.allocate_temp()
  r_tmp = allocator.allocate_temp()

  nodes.append(NvidiaSassComment(text=f"BEGIN GELU ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_src, NvidiaSassRegister(name="R2")]))
  # Fast approx: x * sigmoid(1.702 * x)
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_tmp, r_src, NvidiaSassImmediate(value=1.702)]))
  # Sigmoid inline
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_tmp, r_tmp, NvidiaSassImmediate(value=-1.442695)]))
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_tmp, r_tmp]))
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_tmp, r_tmp, NvidiaSassImmediate(value=1.0)]))
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_tmp, r_tmp]))

  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_dst, r_src, r_tmp]))
  nodes.append(NvidiaSassComment(text=f"END GELU ({node_id})"))
  return nodes


def expand_mseloss(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for MSELoss.

  Accumulates (pred - target)^2 over N elements.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_pred = allocator.allocate_temp()
  r_tgt = allocator.allocate_temp()
  r_diff = allocator.allocate_temp()
  r_sq = allocator.allocate_temp()

  r_ptr_pred = NvidiaSassRegister(name="R2")
  r_ptr_tgt = NvidiaSassRegister(name="R3")

  limit = int(metadata.get("elements", 128))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_MSE_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN MSELoss ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_counter, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_pred, NvidiaSassMemory(base=r_ptr_pred)]))
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_tgt, NvidiaSassMemory(base=r_ptr_tgt)]))

  # diff = pred - tgt
  nodes.append(
    NvidiaSassInstruction(opcode="FADD", operands=[r_diff, r_pred, r_tgt])
  )  # Needs negation in full implementation

  # sq = diff * diff
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_sq, r_diff, r_diff]))

  # acc += sq
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_sq]))

  # Pointers and loop control
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ptr_pred, r_ptr_pred, NvidiaSassImmediate(value=4), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ptr_tgt, r_ptr_tgt, NvidiaSassImmediate(value=4), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_counter, r_counter, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )

  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_counter,
        NvidiaSassImmediate(value=limit),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  if metadata.get("reduction", "mean") == "mean":
    inv_count = 1.0 / limit if limit > 0 else 0.0
    r_inv = allocator.allocate_temp()
    nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_inv, NvidiaSassImmediate(value=inv_count)]))
    nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_acc, r_acc, r_inv]))

  nodes.append(NvidiaSassComment(text=f"END MSELoss ({node_id})"))
  return nodes


def expand_crossentropyloss(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for CrossEntropyLoss.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  r_counter = allocator.allocate_temp()
  r_prob = allocator.allocate_temp()
  r_log = allocator.allocate_temp()

  r_ptr_prob = NvidiaSassRegister(name="R2")

  limit = int(metadata.get("elements", 32))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_CE_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN CrossEntropyLoss ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_counter, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)
  # Simplified: Load probability of correct class directly
  nodes.append(NvidiaSassInstruction(opcode="LDG.E.F32", operands=[r_prob, NvidiaSassMemory(base=r_ptr_prob)]))

  # log2(prob)
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_log, r_prob]))  # LG2 implicit

  # ln(prob) = log2(prob) * 0.693147
  nodes.append(NvidiaSassInstruction(opcode="FMUL", operands=[r_log, r_log, NvidiaSassImmediate(value=0.693147)]))

  # acc += -ln(prob)
  nodes.append(NvidiaSassInstruction(opcode="FADD", operands=[r_acc, r_acc, r_log]))  # Needs proper negation

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_ptr_prob, r_ptr_prob, NvidiaSassImmediate(value=4), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_counter, r_counter, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )

  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_counter,
        NvidiaSassImmediate(value=limit),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END CrossEntropyLoss ({node_id})"))
  return nodes


from .macros_extra import *  # noqa: E402, F403
