"""Symbol types and scopes used for static analysis and type inference."""

import pydantic


class SymbolType(pydantic.BaseModel):
  """Base class for inferred types."""

  model_config = pydantic.ConfigDict(extra="forbid")

  name: str = "Unknown"
  """A string representation of the type (e.g., 'Tensor')."""

  def __str__(self) -> str:
    """Return the type name.

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


class TensorType(SymbolType):
  """Represent an N-dimensional array or tensor associated with a specific framework."""

  name: str = "Tensor"
  framework: str = "unknown"
  """Execute framework key (e.g. "torch" or "jax") responsible for this tensor."""

  def __init__(self, name: str = "Tensor", framework: str = "unknown") -> None:
    """Initialize TensorType.

    Args:
        name: Type name.
        framework: Framework key.
    """
    pydantic.BaseModel.__init__(self, name=name, framework=framework)

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


class ModuleType(SymbolType):
  """Represent an imported module or sub-module."""

  path: str
  name: str = "Module"
  """Fully qualified path string (e.g. "torch.nn")."""

  def __init__(self, name: str = "Module", path: str = "") -> None:
    """Initialize ModuleType.

    Args:
        name: Module type name or path.
        path: Module path string.
    """
    pydantic.BaseModel.__init__(self, name=name, path=path)

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


class UnionType(SymbolType):
  """Represent a union of potential types resulting from control flow divergence."""

  types: list[SymbolType]
  name: str = "Union"

  def __init__(self, types: list[SymbolType], name: str = "Union") -> None:
    """Initialize UnionType.

    Args:
        types: List of symbol types in the union.
        name: Name of the union type.
    """
    pydantic.BaseModel.__init__(self, types=types, name=name)

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


class Scope(pydantic.BaseModel):
  """Represent a variable scope (Global, Class, or Function)."""

  model_config = pydantic.ConfigDict(extra="forbid")

  parent: "Scope | None" = None
  name: str = "<root>"
  symbols: dict[str, SymbolType] = pydantic.Field(default_factory=dict)

  def __init__(self, parent: "Scope | None" = None, name: str = "<root>") -> None:
    """Initialize the scope.

    Args:
        parent: The enclosing scope (None for global).
        name: Debug name for the scope.
    """
    super().__init__(parent=parent, name=name)

  def set(self, name: str, sym_type: SymbolType) -> None:
    """Register a symbol in the current scope.

    Args:
        name: Variable identifier.
        sym_type: Inferred Type object.

    """
    self.symbols[name] = sym_type

  def get(self, name: str) -> SymbolType | None:
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

  def snapshot(self) -> dict[str, SymbolType]:
    """Return a shallow copy of the current symbol table for branching.

    Returns:
        A dictionary mapping symbol names to their types in the current scope.
    """
    return self.symbols.copy()
