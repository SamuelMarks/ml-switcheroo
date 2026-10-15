"""NVIDIA SASS Macros for misc."""

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


def expand_flatten(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Flatten.

  Generates an assignment instruction representing a logical reshape/flatten
  by moving the source pointer value to the destination register.

  Args:
      allocator: The register manager.
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
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Reshape.

  Generates an assignment instruction representing a logical reshape
  by moving the source pointer value to the destination register.

  Args:
      allocator: The register manager.
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


def expand_dropout(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for Dropout.

  Logic flow:
  1. Load input value.
  2. Generate or load a random float in [0, 1).
  3. Compare random value with dropout probability.
  4. Scale output or set to 0.

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


def expand_mseloss(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for MSELoss.

  Accumulates (pred - target)^2 over N elements.

  Args:
      allocator: The register manager.
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
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for CrossEntropyLoss.

  Args:
      allocator: The register manager.
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


def expand_rnn(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for a basic RNN cell over time.

  h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)

  Args:
      allocator: The register manager.
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
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for LSTM over time.

  Args:
      allocator: The register manager.
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
  metadata: Any,
) -> List[NvidiaSassNode]:
  """Generate the NVIDIA_SASS assembly kernel for GRU over time.

  Args:
      allocator: The register manager.
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


def _make_generic_expand(name: str) -> Any:
  """Create a generic macro expansion function for NVIDIA_SASS.

  Args:
      name (str): The name of the operation.

  Returns:
      Callable: The generated expansion function.
  """

  def expand(
    allocator: RegisterAllocatorProtocol,
    node_id: str,
    metadata: Any,
  ) -> List[NvidiaSassNode]:
    """Generate a generic NVIDIA_SASS kernel.

    Args:
        allocator: The register manager.
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


def expand_variable(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
  metadata: Any,
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


def expand_adam(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
  metadata: Any,
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
