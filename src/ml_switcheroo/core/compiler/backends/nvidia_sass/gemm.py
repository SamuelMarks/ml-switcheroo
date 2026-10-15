"""NVIDIA SASS Macros for gemm."""

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


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
      allocator: The register manager.
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


def expand_multiheadattention(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for MultiheadAttention.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_out = allocator.get_register(node_id)
  r_q = allocator.allocate_temp()
  r_k = allocator.allocate_temp()
  r_v = allocator.allocate_temp()

  nodes.append(NvidiaSassComment(text=f"BEGIN MultiheadAttention ({node_id})"))

  # Simulated Q, K, V projections
  nodes.append(NvidiaSassComment(text="Q, K, V Projections"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_q, NvidiaSassRegister(name="R2")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_k, NvidiaSassRegister(name="R3")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_v, NvidiaSassRegister(name="R4")]))

  # Simulated Attention: softmax(Q*K^T/sqrt(d)) * V
  nodes.append(NvidiaSassComment(text="Attention = Softmax(Q*K^T) * V"))
  r_attn = allocator.allocate_temp()
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_attn, r_q, r_k, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_attn, r_attn]))  # Softmax approx
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_out, r_attn, r_v, NvidiaSassRegister(name="RZ")]))

  nodes.append(NvidiaSassComment(text=f"END MultiheadAttention ({node_id})"))
  return nodes


def expand_transformer(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a Transformer block.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_out = allocator.get_register(node_id)
  r_in = allocator.allocate_temp()
  r_tmp = allocator.allocate_temp()

  nodes.append(NvidiaSassComment(text=f"BEGIN Transformer ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_in, NvidiaSassRegister(name="R2")]))

  # Self Attention
  nodes.append(NvidiaSassComment(text="Self Attention"))
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_tmp, r_in, r_in, r_in]))

  # FFN
  nodes.append(NvidiaSassComment(text="Feed Forward"))
  nodes.append(NvidiaSassInstruction(opcode="FFMA", operands=[r_out, r_tmp, NvidiaSassRegister(name="R3"), r_tmp]))

  nodes.append(NvidiaSassComment(text=f"END Transformer ({node_id})"))
  return nodes


def expand_transformerencoder(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for TransformerEncoder.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_out = allocator.get_register(node_id)
  nodes.append(NvidiaSassComment(text=f"BEGIN TransformerEncoder ({node_id})"))
  # Simplified
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_out, NvidiaSassRegister(name="R2")]))
  nodes.append(NvidiaSassComment(text=f"END TransformerEncoder ({node_id})"))
  return nodes


def expand_transformerdecoder(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for TransformerDecoder.

  Args:
      allocator: The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_out = allocator.get_register(node_id)
  nodes.append(NvidiaSassComment(text=f"BEGIN TransformerDecoder ({node_id})"))
  # Simplified
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_out, NvidiaSassRegister(name="R2")]))
  nodes.append(NvidiaSassComment(text=f"END TransformerDecoder ({node_id})"))
  return nodes
