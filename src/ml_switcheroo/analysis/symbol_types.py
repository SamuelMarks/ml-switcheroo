"""Symbol types and scopes used for static analysis and type inference."""

import dataclasses


@dataclasses.dataclass
class SymbolType:
  """Base class for inferred types.

  Attributes:
      name: A string representation of the type (e.g., `'Tensor'`).
  """

  name: str = "Unknown"

  def __str__(self) -> str:
    """Return the type name.

    Returns:
        The string representation of the symbol type.
    """
    return self.name


@dataclasses.dataclass
class TensorType(SymbolType):
  """Represent an N-dimensional array or tensor associated with a specific framework.

  Attributes:
      name: Type name (`str`).
      framework: Framework key (e.g., `'torch'` or `'jax'`) responsible for this tensor.
  """

  name: str = "Tensor"
  framework: str = "unknown"


@dataclasses.dataclass
class ModuleType(SymbolType):
  """Represent an imported module or sub-module.

  Attributes:
      name: Module type name (`str`).
      path: Fully qualified module path string (e.g., `'torch.nn'`).
  """

  name: str = "Module"
  path: str = ""


@dataclasses.dataclass
class UnionType(SymbolType):
  """Represent a union of potential types resulting from control flow divergence.

  Attributes:
      types: List of `SymbolType` instances in the union.
      name: Display label for the union type (`str`).
  """

  types: list[SymbolType] = dataclasses.field(default_factory=list, compare=False)
  name: str = "Union"
  _canonical_members: frozenset[str] = dataclasses.field(default_factory=frozenset, init=False, repr=False)

  def __init__(self, types: list[SymbolType], name: str = "Union") -> None:
    """Initialize `UnionType` and compute its order-independent canonical key.

    Args:
        types: List of symbol types in the union.
        name: Name of the union type.
    """
    self.name = name
    self.types = types
    self._canonical_members = frozenset(str(t) for t in types)

  def __str__(self) -> str:
    """Get string representation of the union type.

    Returns:
        A formatted string of unique types enclosed in `Union[...]`.
    """
    unique_names = sorted(self._canonical_members)
    return f"Union[{', '.join(unique_names)}]"


@dataclasses.dataclass
class Scope:
  """Represent a variable scope (Global, Class, or Function).

  Attributes:
      parent: Enclosing `Scope` (`None` for global scope).
      name: Debug label for the scope (`str`).
      symbols: Mapping from variable identifier to inferred `SymbolType`.
  """

  parent: "Scope | None" = None
  name: str = "<root>"
  symbols: dict[str, SymbolType] = dataclasses.field(default_factory=dict)

  def set(self, name: str, sym_type: SymbolType) -> None:
    """Register a symbol in the current scope.

    Args:
        name: Variable identifier.
        sym_type: Inferred `SymbolType` instance.
    """
    self.symbols[name] = sym_type

  def get(self, name: str) -> SymbolType | None:
    """Resolve a symbol, traversing parent scopes when not defined locally.

    Args:
        name: Variable identifier to look up.

    Returns:
        The `SymbolType` if found, otherwise `None`.
    """
    if name in self.symbols:
      return self.symbols[name]
    if self.parent is not None:
      return self.parent.get(name)
    return None

  def snapshot(self) -> dict[str, SymbolType]:
    """Return a shallow copy of the current local symbol table for branching.

    Returns:
        A dictionary mapping symbol names to their types in the current scope.
    """
    return self.symbols.copy()
