"""NVIDIA_SASS Macro Expansion Logic - Extra Macros."""

from typing import List
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassOperand,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassRegister,
  NvidiaSassImmediate,
  NvidiaSassPredicate,
  NvidiaSassComment,
  NvidiaSassNode,
)
from .macros import RegisterAllocatorProtocol


def expand_rnn(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a basic RNN cell over time.

  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_h = allocator.get_register(node_id)
  r_t = allocator.allocate_temp()

  seq_len = int(metadata.get("seq_len", 10))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_RNN_TIME_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN RNN ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_t, NvidiaSassRegister(name="RZ")]))
  # Initialize hidden state (could be loaded from R5)
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_h, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)

  # Simulated math for Wx + Wh + b
  nodes.append(NvidiaSassComment(text="Compute RNN gates (Wx + Wh + b)"))
  r_gate = allocator.allocate_temp()
  nodes.append(
    NvidiaSassInstruction(
      opcode="FFMA", operands=[r_gate, r_h, NvidiaSassRegister(name="R3"), NvidiaSassRegister(name="R4")]
    )
  )

  # Tanh activation
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_h, r_gate]))  # Tanh implicit

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_t, r_t, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_t,
        NvidiaSassImmediate(value=seq_len),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END RNN ({node_id})"))
  return nodes


def expand_lstm(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for LSTM over time.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_h = allocator.get_register(node_id)
  r_c = allocator.allocate_temp()
  r_t = allocator.allocate_temp()

  seq_len = int(metadata.get("seq_len", 10))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_LSTM_TIME_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN LSTM ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_t, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_h, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_c, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)

  # Simulated gates
  nodes.append(NvidiaSassComment(text="Compute LSTM gates (i, f, g, o)"))
  nodes.append(
    NvidiaSassInstruction(
      opcode="FFMA", operands=[r_c, r_c, NvidiaSassRegister(name="R3"), NvidiaSassRegister(name="R4")]
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="MUFU", operands=[r_h, r_c]))

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_t, r_t, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_t,
        NvidiaSassImmediate(value=seq_len),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END LSTM ({node_id})"))
  return nodes


def expand_gru(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for GRU over time.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  nodes: List[NvidiaSassNode] = []
  r_h = allocator.get_register(node_id)
  r_t = allocator.allocate_temp()

  seq_len = int(metadata.get("seq_len", 10))
  p_loop = NvidiaSassPredicate(name="P0")
  l_loop: NvidiaSassOperand = NvidiaSassLabel(name=f"L_GRU_TIME_{node_id}")

  nodes.append(NvidiaSassComment(text=f"BEGIN GRU ({node_id})"))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_t, NvidiaSassRegister(name="RZ")]))
  nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_h, NvidiaSassRegister(name="RZ")]))

  nodes.append(l_loop)

  # Simulated gates
  nodes.append(NvidiaSassComment(text="Compute GRU gates (r, z, n)"))
  nodes.append(
    NvidiaSassInstruction(
      opcode="FFMA", operands=[r_h, r_h, NvidiaSassRegister(name="R3"), NvidiaSassRegister(name="R4")]
    )
  )

  nodes.append(
    NvidiaSassInstruction(
      opcode="IADD3", operands=[r_t, r_t, NvidiaSassImmediate(value=1), NvidiaSassRegister(name="RZ")]
    )
  )
  nodes.append(
    NvidiaSassInstruction(
      opcode="ISETP.LT.AND",
      operands=[
        p_loop,
        NvidiaSassRegister(name="PT"),
        r_t,
        NvidiaSassImmediate(value=seq_len),
        NvidiaSassRegister(name="PT"),
      ],
    )
  )
  nodes.append(NvidiaSassInstruction(opcode="BRA", operands=[l_loop], predicate=p_loop))

  nodes.append(NvidiaSassComment(text=f"END GRU ({node_id})"))
  return nodes


def expand_multiheadattention(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for MultiheadAttention.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a Transformer block.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for TransformerEncoder.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for TransformerDecoder.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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


def expand_conv1d(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for 1D Convolution.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for DepthwiseConv2d.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for ConvTranspose (generic representation).

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate NVIDIA_SASS kernel for 1D Pooling.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate NVIDIA_SASS kernel for 3D Pooling.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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
  metadata,
) -> List[NvidiaSassNode]:
  """Generate NVIDIA_SASS kernel for Adaptive Pooling.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
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


def _make_generic_expand(name: str):
  """Create a generic macro expansion function for NVIDIA_SASS.

  Args:
      name (str): The name of the operation.

  Returns:
      Callable: The generated expansion function.
  """

  def expand(
    allocator: RegisterAllocatorProtocol,
    node_id: str,
    metadata,
  ) -> List[NvidiaSassNode]:
    """Generate a generic NVIDIA_SASS kernel.

    Args:
        allocator (~ml_switcheroo.core.compiler.backends.nvidia_sass.macros.RegisterAllocatorProtocol): The register manager.
        node_id (str): The unique ID of the operation node (used for output reg).
        metadata (Dict[str, Any]): Layer configuration.

    Returns:
        List[NvidiaSassNode]: The list of NVIDIA_SASS nodes.
    """
    nodes: List[NvidiaSassNode] = []

    r_acc = allocator.get_register(node_id)
    nodes.append(NvidiaSassComment(text=f"BEGIN {name} ({node_id})"))
    nodes.append(NvidiaSassInstruction(opcode="MOV", operands=[r_acc, NvidiaSassRegister(name="R3")]))
    nodes.append(NvidiaSassComment(text=f"END {name} ({node_id})"))
    return nodes

  return expand


expand_generic_norm = _make_generic_expand("Norm")
expand_generic_activation = _make_generic_expand("Activation")
expand_generic_linalg = _make_generic_expand("LinAlg")
expand_generic_reduction = _make_generic_expand("Reduction")
expand_generic_loss = _make_generic_expand("Loss")
expand_generic_dropout = _make_generic_expand("DropoutVar")


def expand_variable(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Expand a variable operation into NVIDIA_SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  return [NvidiaSassComment(text=f"BEGIN Variable ({node_id})"), NvidiaSassComment(text=f"END Variable ({node_id})")]


def expand_transpose(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Expand a transpose operation into NVIDIA_SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  return [NvidiaSassComment(text=f"BEGIN transpose ({node_id})"), NvidiaSassComment(text=f"END transpose ({node_id})")]


def expand_conv_general_dilated(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
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


def expand_adam(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Expand an adam operation into NVIDIA_SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  return [NvidiaSassComment(text=f"BEGIN adam ({node_id})"), NvidiaSassComment(text=f"END adam ({node_id})")]


def expand_l(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata,
) -> List[NvidiaSassNode]:
  """Expand an l operation into NVIDIA_SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[NvidiaSassNode]: Sequence of instructions.
  """
  return [NvidiaSassComment(text=f"BEGIN l ({node_id})"), NvidiaSassComment(text=f"END l ({node_id})")]
