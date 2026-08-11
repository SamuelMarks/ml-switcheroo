"""RDNA Macro Expansion Logic.

Procedural generators for complex RDNA kernel logic (Conv2d, Linear).
"""

from typing import Any, Dict, List, Protocol

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaModifier,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)


class RegisterAllocatorProtocol(Protocol):
  """Protocol for the Dual-Pool Register Allocator.

  Provides interface definitions for allocating and retrieving vector and scalar
  registers needed during RDNA assembly generation.
  """

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Gets or allocates a vector register (VGPR) associated with a variable name.

    Args:
        var_name (str): The identifier of the variable.

    """
    ...

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Gets or allocates a scalar register (SGPR) associated with a variable name.

    Args:
        var_name (str): The identifier of the variable.

    """
    ...

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Allocates a temporary, unmapped vector register (VGPR)."""
    ...

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Allocates a temporary, unmapped scalar register (SGPR)."""
    ...


def expand_conv2d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for a 2D Convolution loop.

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

  v_acc = allocator.get_vector_register(node_id)
  s_ky = allocator.allocate_scalar_temp()
  s_kx = allocator.allocate_scalar_temp()
  v_val_i = allocator.allocate_vector_temp()
  v_val_w = allocator.allocate_vector_temp()
  v_addr = allocator.allocate_vector_temp()

  v_base_img = allocator.get_vector_register("x")
  v_base_wgt = allocator.allocate_vector_temp()

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


def expand_linear(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for a Linear Layer.

  Constructs the sequence of CST nodes implementing a matrix-vector or vector-vector
  reduction loop for a linear fully-connected layer.

  Args:
      allocator: The register allocator to use for
          managing temporary and variable registers.
      node_id: A unique identifier for the linear operation node.
      metadata: Metadata containing configuration details
          such as "in_features" (input feature size) and "bias" (boolean flag).

  Returns:
      List[RdnaNode]: A list of RDNA CST nodes representing the compiled linear
      layer logic.
  """
  nodes: List[RdnaNode] = []

  v_acc = allocator.get_vector_register(node_id)
  s_idx = allocator.allocate_scalar_temp()
  v_val_i = allocator.allocate_vector_temp()
  v_val_w = allocator.allocate_vector_temp()
  v_ptr_i = allocator.allocate_vector_temp()
  v_ptr_w = allocator.allocate_vector_temp()

  l_gemm = RdnaLabel(name=f"L_GEMM_{node_id}")
  limit = int(metadata.get("in_features", 128))

  nodes.append(RdnaComment(text=f"BEGIN Linear ({node_id})"))
  nodes.append(RdnaInstruction(opcode="v_mov_b32", operands=[v_acc, RdnaImmediate(value=0)]))
  nodes.append(RdnaInstruction(opcode="s_mov_b32", operands=[s_idx, RdnaImmediate(value=0)]))

  nodes.append(l_gemm)
  nodes.append(
    RdnaInstruction(opcode="global_load_dword", operands=[v_val_i, RdnaMemory(base=v_ptr_i), RdnaModifier(name="off")])
  )
  nodes.append(
    RdnaInstruction(opcode="global_load_dword", operands=[v_val_w, RdnaMemory(base=v_ptr_w), RdnaModifier(name="off")])
  )

  nodes.append(RdnaInstruction(opcode="s_waitcnt", operands=[RdnaModifier(name="vmcnt(0)")]))
  nodes.append(RdnaInstruction(opcode="v_fmac_f32", operands=[v_acc, v_val_i, v_val_w]))

  nodes.append(RdnaInstruction(opcode="v_add_u32", operands=[v_ptr_i, v_ptr_i, RdnaImmediate(value=4)]))
  nodes.append(RdnaInstruction(opcode="v_add_u32", operands=[v_ptr_w, v_ptr_w, RdnaImmediate(value=4)]))

  nodes.append(RdnaInstruction(opcode="s_add_i32", operands=[s_idx, s_idx, RdnaImmediate(value=1)]))
  nodes.append(RdnaInstruction(opcode="s_cmp_lt_i32", operands=[s_idx, RdnaImmediate(value=limit)]))
  nodes.append(RdnaInstruction(opcode="s_cbranch_scc1", operands=[RdnaLabelRef(name=l_gemm.name)]))

  if "bias" in metadata and metadata["bias"]:
    nodes.append(RdnaComment(text="Add Bias"))
    v_bias = allocator.allocate_vector_temp()
    nodes.append(RdnaInstruction(opcode="v_add_f32", operands=[v_acc, v_acc, v_bias]))

  nodes.append(RdnaComment(text=f"END Linear ({node_id})"))
  return nodes


def expand_relu(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for ReLU.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  nodes: List[RdnaNode] = []
  r_dst = allocator.get_vector_register(node_id)
  r_src = allocator.allocate_vector_temp()  # Assume input is loaded here
  nodes.append(RdnaComment(text=f"BEGIN ReLU ({node_id})"))
  nodes.append(RdnaInstruction(opcode="v_max_f32", operands=[r_dst, r_src, RdnaImmediate(value=0)]))
  nodes.append(RdnaComment(text=f"END ReLU ({node_id})"))
  return nodes


def expand_flatten(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for Flatten.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN Flatten ({node_id})"), RdnaComment(text=f"END Flatten ({node_id})")]


def expand_reshape(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for Reshape.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN Reshape ({node_id})"), RdnaComment(text=f"END Reshape ({node_id})")]


def expand_conv3d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Generates the RDNA assembly kernel for Conv3d.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  nodes: List[RdnaNode] = []
  r_dst = allocator.get_vector_register(node_id)
  r_src = allocator.allocate_vector_temp()
  nodes.append(RdnaComment(text=f"BEGIN Conv3d ({node_id})"))
  label = RdnaLabel(name=f"BB_{node_id.replace('-', '_')}_1")
  nodes.append(label)
  nodes.append(RdnaInstruction(opcode="v_fmac_f32", operands=[r_dst, r_src, r_src]))
  nodes.append(RdnaInstruction(opcode="s_cbranch_vccnz", operands=[RdnaLabelRef(name=label.name)]))
  nodes.append(RdnaComment(text=f"END Conv3d ({node_id})"))
  return nodes


def expand_dropout(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Expand a dropout operation into RDNA nodes.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN Dropout ({node_id})"), RdnaComment(text=f"END Dropout ({node_id})")]


def expand_variable(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Expand a variable operation into RDNA nodes.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN Variable ({node_id})"), RdnaComment(text=f"END Variable ({node_id})")]


def expand_transpose(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Expand a transpose operation into RDNA nodes.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN transpose ({node_id})"), RdnaComment(text=f"END transpose ({node_id})")]


def expand_conv_general_dilated(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
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


def expand_adam(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Expand an adam operation into RDNA nodes.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN adam ({node_id})"), RdnaComment(text=f"END adam ({node_id})")]


def expand_l(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[RdnaNode]:
  """Expand an l operation into RDNA nodes.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN l ({node_id})"), RdnaComment(text=f"END l ({node_id})")]
