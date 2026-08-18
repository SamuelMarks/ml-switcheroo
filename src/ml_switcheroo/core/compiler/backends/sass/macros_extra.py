"""SASS Macro Expansion Logic - Extra Macros."""

from typing import List, Dict, Any, Callable
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassOperand,
  SassInstruction,
  SassLabel,
  SassRegister,
  SassImmediate,
  SassPredicate,
  SassComment,
  SassNode,
)
from .macros import RegisterAllocatorProtocol


def expand_rnn(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Generates the SASS assembly kernel for a basic RNN cell over time.

  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
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
  """Generates the SASS assembly kernel for LSTM over time.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for GRU over time.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for MultiheadAttention.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for a Transformer block.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for TransformerEncoder.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for TransformerDecoder.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for 1D Convolution.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for DepthwiseConv2d.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates the SASS assembly kernel for ConvTranspose (generic representation).

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates SASS kernel for 1D Pooling.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates SASS kernel for 3D Pooling.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
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
  """Generates SASS kernel for Adaptive Pooling.

  Args:
      allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
      node_id (str): The unique ID of the operation node (used for output reg).
      metadata (Dict[str, Any]): Layer configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  nodes: List[SassNode] = []
  r_acc = allocator.get_register(node_id)
  nodes.append(SassComment(text=f"BEGIN AdaptivePool ({node_id})"))
  nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
  nodes.append(SassComment(text=f"END AdaptivePool ({node_id})"))
  return nodes


def _make_generic_expand(name: str) -> Callable[[RegisterAllocatorProtocol, str, Dict[str, Any]], List[SassNode]]:
  """Creates a generic macro expansion function for SASS.

  Args:
      name (str): The name of the operation.

  Returns:
      Callable: The generated expansion function.
  """

  def expand(
    allocator: RegisterAllocatorProtocol,
    node_id: str,
    metadata: Dict[str, Any],
  ) -> List[SassNode]:
    """Generates a generic SASS kernel.

    Args:
        allocator (~ml_switcheroo.core.compiler.backends.sass.macros.RegisterAllocatorProtocol): The register manager.
        node_id (str): The unique ID of the operation node (used for output reg).
        metadata (Dict[str, Any]): Layer configuration.

    Returns:
        List[SassNode]: The list of SASS nodes.
    """
    nodes: List[SassNode] = []

    r_acc = allocator.get_register(node_id)
    nodes.append(SassComment(text=f"BEGIN {name} ({node_id})"))
    nodes.append(SassInstruction(opcode="MOV", operands=[r_acc, SassRegister(name="R3")]))
    nodes.append(SassComment(text=f"END {name} ({node_id})"))
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
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Expand a variable operation into SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  return [SassComment(text=f"BEGIN Variable ({node_id})"), SassComment(text=f"END Variable ({node_id})")]


def expand_transpose(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Expand a transpose operation into SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  return [SassComment(text=f"BEGIN transpose ({node_id})"), SassComment(text=f"END transpose ({node_id})")]


def expand_conv_general_dilated(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Expand a conv_general_dilated operation into SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  return [
    SassComment(text=f"BEGIN conv_general_dilated ({node_id})"),
    SassComment(text=f"END conv_general_dilated ({node_id})"),
  ]


def expand_adam(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Expand an adam operation into SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  return [SassComment(text=f"BEGIN adam ({node_id})"), SassComment(text=f"END adam ({node_id})")]


def expand_l(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Dict[str, Any],
) -> List[SassNode]:
  """Expand an l operation into SASS nodes.

  Args:
      allocator: The register allocator.
      node_id: A unique identifier for the operation node.
      metadata: Metadata configuration.

  Returns:
      List[SassNode]: Sequence of instructions.
  """
  return [SassComment(text=f"BEGIN l ({node_id})"), SassComment(text=f"END l ({node_id})")]
