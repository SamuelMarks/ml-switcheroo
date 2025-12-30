"""
Switcheroo Dialect Definition ('sw').

This module formally defines the MLIR operations for the 'sw' high-level dialect.
It acts as the schema validation layer for the custom IR, ensuring that
transformations produce semantically valid intermediate representations.

Operations:
- `sw.module`: Represents a Class or Scope container.
- `sw.func`: Represents a Function definition.
- `sw.op`: Represents a generic Operation instantiation (e.g. Layers).
- `sw.call`: Represents a function invocation.
- `sw.getattr`: Represents attribute access (v = self.layer).
- `sw.setattr`: Represents attribute assignment (self.layer = v).
- `sw.constant`: Represents literals.
- `sw.return`: Control flow exit.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# We reuse the CST nodes but wrap them in semantic validators
from ml_switcheroo.core.mlir.nodes import OperationNode, AttributeNode


@dataclass
class OpSchema:
  """
  Validation schema for a MLIR operation.
  """

  name: str
  num_regions: int = 0
  required_attributes: List[str] = field(default_factory=list)
  has_results: bool = False

  def validate(self, node: OperationNode) -> bool:
    """
    Checks if the CST node conforms to the dialect schema.
    """
    if node.name != self.name:
      return False

    if len(node.regions) != self.num_regions:
      return False

    found_attrs = {a.name for a in node.attributes}
    for req in self.required_attributes:
      if req not in found_attrs:
        return False

    if self.has_results and not node.results:
      return False

    # results checking is loose (if has_results=False, it can still have 0 results)
    # but if has_results=True, it MUST have >0.

    return True


# --- Dialect Definitions ---

SW_MODULE = OpSchema(
  name="sw.module",
  num_regions=1,
  # 'bases' is supported but optional to maintain compatibility with basic modules
  required_attributes=["sym_name"],
)

SW_FUNC = OpSchema(name="sw.func", num_regions=1, required_attributes=["sym_name"])

SW_OP = OpSchema(
  name="sw.op",
  required_attributes=["type"],  # e.g. "torch.nn.Linear"
  has_results=True,
)

SW_CALL = OpSchema(
  name="sw.call",
  has_results=True,  # Usually returns something, or Side Effect
  # Operands are checked dynamically
)

SW_GETATTR = OpSchema(name="sw.getattr", required_attributes=["name"], has_results=True)

SW_SETATTR = OpSchema(
  name="sw.setattr",
  required_attributes=["name"],
  has_results=False,  # Assignment is a side effect statement
)

SW_CONSTANT = OpSchema(name="sw.constant", required_attributes=["value"], has_results=True)

SW_RETURN = OpSchema(name="sw.return")


class DialectRegistry:
  """
  Central registry for dialect validation.
  """

  _OPS = {
    "sw.module": SW_MODULE,
    "sw.func": SW_FUNC,
    "sw.op": SW_OP,
    "sw.call": SW_CALL,
    "sw.getattr": SW_GETATTR,
    "sw.setattr": SW_SETATTR,
    "sw.constant": SW_CONSTANT,
    "sw.return": SW_RETURN,
  }

  @classmethod
  def validate_op(cls, node: OperationNode) -> bool:
    """
    Validates a single operation node against the schema.
    Returns False if op is unknown or invalid.
    """
    schema = cls._OPS.get(node.name)
    if not schema:
      # Allow unknown ops for resilience? Or strict?
      # For this core dialect definition, we are strict about 'sw.*' namespace.
      if node.name.startswith("sw."):
        return False
      return True  # Allow other dialects

    return schema.validate(node)

  @classmethod
  def get_abstract_op(cls, op_name: str) -> str:
    """
    Maps a high-level framework op string (e.g. 'Linear') to the canonical dialect op.
    Currently they all map to 'sw.op' with type attributes, but this allows future expansion.
    """
    return "sw.op"
