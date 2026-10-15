"""RDNA Macros for misc."""

from typing import Any, List

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaNode
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol


def expand_flatten(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for Flatten.

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
  metadata: Any,
) -> List[RdnaNode]:
  """Generate the RDNA assembly kernel for Reshape.

  Args:
      allocator: The register allocator to use for managing temporary and variable registers.
      node_id: A unique identifier for the operation node.
      metadata: Metadata containing configuration details.

  Returns:
      A list of RDNA CST nodes representing the compiled logic.
  """
  return [RdnaComment(text=f"BEGIN Reshape ({node_id})"), RdnaComment(text=f"END Reshape ({node_id})")]


def expand_dropout(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
  metadata: Any,
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
  metadata: Any,
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


def expand_adam(
  allocator: RegisterAllocatorProtocol,
  node_id: str,
  metadata: Any,
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
  metadata: Any = None,
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
