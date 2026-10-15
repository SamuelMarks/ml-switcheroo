"""NVIDIA SASS Macros for conv."""

from typing import Any, List

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassNode,
  NvidiaSassOperand,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
      allocator: The register manager.
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


def expand_conv3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
      allocator: The register manager.
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
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for AvgPool2d.

  Logic flow:
  1. Initialize Accumulator (R_ACC) to zero.
  2. Nested loops over Kernel Y and Kernel X.
  3. Load values (LDG).
  4. Add to accumulator (FADD).
  5. Multiply accumulator by 1/(Kx*Ky) (FMUL).

  Args:
      allocator: The register manager.
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
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for MaxPool2d.

  Logic flow:
  1. Initialize Accumulator (R_ACC) to strongly negative value.
  2. Nested loops over Kernel Y and Kernel X.
  3. Load values (LDG).
  4. Maximize with accumulator (FMAX).

  Args:
      allocator: The register manager.
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


def expand_conv1d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for 1D Convolution.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  r_kx = allocator.allocate_temp()
  kernel_size = int(metadata.get("k", 3))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_KX_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN Conv1d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kx, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)
  nodes.append(
    NvidiaSassInstruction(
      opcode="FFMA", operands=[r_acc, NvidiaSassRegister(name="R3"), NvidiaSassRegister(name="R4"), r_acc]
    )
  )

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
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END Conv1d ({node_id})"))
  return nodes


def expand_depthwiseconv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for DepthwiseConv2d.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  r_kx = allocator.allocate_temp()
  kernel_size = int(metadata.get("k", 3))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_DW_KX_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN DepthwiseConv2d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_kx, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)
  nodes.append(
    NvidiaSassInstruction(
      opcode="FFMA", operands=[r_acc, NvidiaSassRegister(name="R3"), NvidiaSassRegister(name="R4"), r_acc]
    )
  )

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
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END DepthwiseConv2d ({node_id})"))
  return nodes


def expand_convtranspose(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for ConvTranspose (generic representation).

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(NvidiaSassComment(text=f"BEGIN ConvTranspose ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="R3")]))
  nodes.append(NvidiaSassComment(text=f"END ConvTranspose ({node_id})"))
  return nodes


def expand_pool1d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate NVIDIA_SASS kernel for 1D Pooling.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(NvidiaSassComment(text=f"BEGIN Pool1d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="R3")]))
  nodes.append(NvidiaSassComment(text=f"END Pool1d ({node_id})"))
  return nodes


def expand_pool3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate NVIDIA_SASS kernel for 3D Pooling.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(NvidiaSassComment(text=f"BEGIN Pool3d ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="R3")]))
  nodes.append(NvidiaSassComment(text=f"END Pool3d ({node_id})"))
  return nodes


def expand_adaptivepool(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate NVIDIA_SASS kernel for Adaptive Pooling.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(NvidiaSassComment(text=f"BEGIN AdaptivePool ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="R3")]))
  nodes.append(NvidiaSassComment(text=f"END AdaptivePool ({node_id})"))
  return nodes


def expand_conv_general_dilated(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Expand a conv_general_dilated operation into NVIDIA_SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  return [
    NvidiaSassComment(text=f"BEGIN conv_general_dilated ({node_id})"),
    NvidiaSassComment(text=f"END conv_general_dilated ({node_id})"),
  ]
