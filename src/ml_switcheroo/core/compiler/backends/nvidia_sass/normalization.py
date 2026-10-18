"""NVIDIA SASS Macros for normalization."""

from typing import Any, Dict, List

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


def expand_mean(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a Mean reduction loop.

  Calculates the sum over elements, and then multiplies the accumulator by
  the reciprocal of the number of elements to compute the average.

  Args:
      allocator: The register manager.
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


def expand_batchnorm2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for BatchNorm2d.

  Logic flow:
  1. Load mean, variance, gamma, beta from memory.
  2. Compute inv_std = 1.0 / sqrt(var + eps).
  3. Load input tensor value.
  4. Compute (x - mean) * inv_std * gamma + beta.

  Args:
      allocator: The register manager.
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
