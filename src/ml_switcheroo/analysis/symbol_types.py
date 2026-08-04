"""Symbol types and scopes used for static analysis and type inference."""

from dataclasses import dataclass
from typing import Dict, List, Optional


class SymbolType:
  """Base class for inferred types."""

  name: str
  """A string representation of the type (e.g., 'Tensor')."""

  def __str__(self) -> str:
    """Returns the type name.

    Returns:
        The string representation of the symbol type.
    """
    return getattr(self, "name", "Unknown")

  def __eq__(self, other: object) -> bool:
    """Check equality with another symbol type based on name.

    Args:
        other: The other object to compare.

    Returns:
        True if other is a SymbolType and names match, False otherwise.
    """
    if not isinstance(other, SymbolType):
      return False
    return getattr(self, "name", "") == getattr(other, "name", "")


@dataclass
class TensorType(SymbolType):
  """Represents a Tensor object from a specific framework."""

  name: str = "Tensor"
  framework: str = "unknown"
  """The framework key (e.g. "torch" or "jax") responsible for this tensor."""

  def __eq__(self, other: object) -> bool:
    """Check equality with another tensor type.

    Args:
        other: The other object to compare.

    Returns:
        True if other is a TensorType with matching name and framework,
        False otherwise.
    """
    if not isinstance(other, TensorType):
      return False
    return self.name == other.name and self.framework == other.framework


@dataclass
class ModuleType(SymbolType):
  """Represents an imported module or sub-module."""

  path: str
  name: str = "Module"
  """Fully qualified path string (e.g. "torch.nn")."""

  def __eq__(self, other: object) -> bool:
    """Check equality with another module type.

    Args:
        other: The other object to compare.

    Returns:
        True if other is a ModuleType with matching name and path, False otherwise.
    """
    if not isinstance(other, ModuleType):
      return False
    return self.name == other.name and self.path == other.path


@dataclass
class UnionType(SymbolType):
  """Represents a union of potential types resulting from control flow divergence."""

  types: List[SymbolType]
  name: str = "Union"

  def __str__(self) -> str:
    """Get string representation of the union type.

    Returns:
        A formatted string of unique types enclosed in Union[...].
    """
    unique_names = sorted(list(set(str(t) for t in self.types)))
    return f"Union[{', '.join(unique_names)}]"

  def __eq__(self, other: object) -> bool:
    """Check equality with another union type.

    Args:
        other: The other object to compare.

    Returns:
        True if other is a UnionType containing equivalent types regardless of
        order, False otherwise.
    """
    if not isinstance(other, UnionType):
      return False
    # Set based comparison for equivalence ignoring order
    return set(str(t) for t in self.types) == set(str(t) for t in other.types)


class Scope:
  """Represents a variable scope (Global, Class, or Function)."""

  def __init__(self, parent: Optional["Scope"] = None, name: str = "<root>"):
    """Initialize the scope.

    Args:
        parent: The enclosing scope (None for global).
        name: Debug name for the scope.

    """
    self.parent = parent
    self.name = name
    self.symbols: Dict[str, SymbolType] = {}

  def set(self, name: str, sym_type: SymbolType) -> None:
    """Register a symbol in the current scope.

    Args:
        name: Variable identifier.
        sym_type: Inferred Type object.

    """
    self.symbols[name] = sym_type

  def get(self, name: str) -> Optional[SymbolType]:
    """Resolve a symbol, traversing parent scopes.

    Args:
        name: Variable identifier to lookup.

    Returns:
        The SymbolType if found, else None.

    """
    if name in self.symbols:
      return self.symbols[name]
    if self.parent:
      return self.parent.get(name)
    return None

  def snapshot(self) -> Dict[str, SymbolType]:
    """Returns a shallow copy of the current symbol table for branching.

    Returns:
        A dictionary mapping symbol names to their types in the current scope.
    """
    return self.symbols.copy()
