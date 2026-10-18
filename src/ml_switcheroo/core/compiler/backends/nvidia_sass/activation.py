"""NVIDIA SASS Macros for activation."""

from typing import Any, Dict, List

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol


def expand_relu(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for ReLU.

  Performs element-wise maximum comparison against zero using `FMAX`.

  Args:
      allocator: The register manager.
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


def expand_sigmoid(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Sigmoid.

  1 / (1 + exp(-x)) -> 1 / (1 + exp2(-x * log2(e)))

  Args:
      allocator: The register manager.
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
  metadata: Dict[str, Any],
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Tanh.

  Args:
      allocator: The register manager.
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
  metadata: Dict[str, Any],
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for GELU.

  Args:
      allocator: The register manager.
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
