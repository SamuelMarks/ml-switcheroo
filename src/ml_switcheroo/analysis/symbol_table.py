"""
Symbol Table and Type Inference Analysis.

This module provides a static analysis pass to infer variable types and scopes
before rewriting occurs. It builds a mapping of AST nodes to inferred type objects,
allowing the rewriter to make decisions based on the semantic type of a variable
(e.g., "is this a Tensor?") rather than just its lexical name.

The `SymbolTableAnalyzer` visitor populates a `SymbolTable` by tracking:
1.  **Imports**: Mapping module aliases to `ModuleType`.
2.  **Assignments**: Propagating types from RHS to LHS.
3.  **Scopes**: Handling nested function/class definitions.
"""

import libcst as cst
from typing import Dict, Optional
from dataclasses import dataclass

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.scanners import get_full_name


@dataclass
class SymbolType:
  """
  Base class for inferred types.
  """

  name: str
  """A string representation of the type (e.g., 'Tensor')."""

  def __str__(self) -> str:
    """Returns the type name."""
    return self.name


@dataclass
class TensorType(SymbolType):
  """
  Represents a Tensor object from a specific framework.
  """

  framework: str
  """The framework key (e.g. "torch" or "jax") responsible for this tensor."""


@dataclass
class ModuleType(SymbolType):
  """
  Represents an imported module or sub-module.
  """

  path: str
  """Fully qualified path string (e.g. "torch.nn")."""


class Scope:
  """
  Represents a variable scope (Global, Class, or Function).
  """

  def __init__(self, parent: Optional["Scope"] = None, name: str = "<root>"):
    """
    Initialize the scope.

    Args:
        parent: The enclosing scope (None for global).
        name: Debug name for the scope.
    """
    self.parent = parent
    self.name = name
    self.symbols: Dict[str, SymbolType] = {}

  def set(self, name: str, sym_type: SymbolType) -> None:
    """
    Register a symbol in the current scope.

    Args:
        name: Variable identifier.
        sym_type: Inferred Type object.
    """
    self.symbols[name] = sym_type

  def get(self, name: str) -> Optional[SymbolType]:
    """
    Resolve a symbol, traversing parent scopes.

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


class SymbolTable:
  """
  Container for analysis results. Maps CST Nodes (by identity) to inferred Types.
  """

  def __init__(self):
    """Initializes an empty node map."""
    self._node_types: Dict[cst.CSTNode, SymbolType] = {}

  def record_type(self, node: cst.CSTNode, sym_type: SymbolType) -> None:
    """
    Associates a CST node with a type.

    Args:
        node: The CST node.
        sym_type: The determined type.
    """
    self._node_types[node] = sym_type

  def get_type(self, node: cst.CSTNode) -> Optional[SymbolType]:
    """
    Retrieves the inferred type for a CST node.

    Args:
        node: The CST node to inspect.

    Returns:
        The stored SymbolType or None.
    """
    return self._node_types.get(node)


class SymbolTableAnalyzer(cst.CSTVisitor):
  """
  Static Analysis pass to populate the SymbolTable.
  Runs post-order traversal logic (via leave methods) to propagate types bottom-up.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the analyzer.

    Args:
        semantics: Reference to semantic knowledge base for type inference rules.
    """
    self.semantics = semantics
    self.table = SymbolTable()
    self.root_scope = Scope(name="global")
    self.current_scope = self.root_scope

  # --- Scoping ---

  def visit_ClassDef(self, node: cst.ClassDef) -> None:
    """
    Enter class scope.

    Args:
        node: Class definition node.
    """
    self.current_scope = Scope(parent=self.current_scope, name=f"class_{node.name.value}")

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    """
    Exit class scope.

    Args:
        node: Class definition node.
    """
    if self.current_scope.parent:
      self.current_scope = self.current_scope.parent

  def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
    """
    Enter function scope.

    Args:
        node: Function definition node.
    """
    self.current_scope = Scope(parent=self.current_scope, name=f"func_{node.name.value}")

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """
    Exit function scope.

    Args:
        node: Function definition node.
    """
    if self.current_scope.parent:
      self.current_scope = self.current_scope.parent

  # --- Definition Tracking ---

  def leave_Import(self, node: cst.Import) -> None:
    """
    Track imports.
    e.g. `import torch` -> symbols['torch'] = ModuleType(name='Module', path='torch')

    Args:
        node: Import statement node.
    """
    for alias in node.names:
      full_path = get_full_name(alias.name)
      bind_name = alias.asname.name.value if alias.asname else full_path.split(".")[0]
      self.current_scope.set(bind_name, ModuleType(name="Module", path=full_path))

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """
    Track from-imports.
    e.g. `from torch import nn` -> symbols['nn'] = ModuleType(name='Module', path='torch.nn')

    Args:
        node: ImportFrom statement.
    """
    if not node.module:
      return
    base_mod = get_full_name(node.module)

    for alias in node.names:
      if isinstance(alias, cst.ImportAlias):
        import_name = alias.name.value
        bind_name = alias.asname.name.value if alias.asname else import_name
        full_path = f"{base_mod}.{import_name}"
        self.current_scope.set(bind_name, ModuleType(name="Module", path=full_path))

  def leave_Assign(self, node: cst.Assign) -> None:
    """
    Propagate type from RHS to LHS.
    x = torch.randn() -> x is Tensor.

    Args:
        node: Assignment node.
    """
    rhs_type = self.table.get_type(node.value)
    if not rhs_type:
      return

    for target in node.targets:
      # Handle simple name assignment: x = ...
      if isinstance(target.target, cst.Name):
        name = target.target.value
        self.current_scope.set(name, rhs_type)
        self.table.record_type(target.target, rhs_type)
      # Handle attributes: self.x = ...
      elif isinstance(target.target, cst.Attribute):
        # We record the type on the attribute node for usage lookup,
        # but tracking object property scope is complex statically without full class analysis.
        # For now we just tag the node.
        self.table.record_type(target.target, rhs_type)

  # --- Usage Resolution ---

  def leave_Name(self, node: cst.Name) -> None:
    """
    Look up variable in scope.

    Args:
        node: Name node usage.
    """
    sym_type = self.current_scope.get(node.value)
    if sym_type:
      self.table.record_type(node, sym_type)

  def leave_Attribute(self, node: cst.Attribute) -> None:
    """
    Resolve attributes.
    If `x` is Module('torch'), `x.nn` is Module('torch.nn').

    Args:
        node: Attribute access node.
    """
    base_type = self.table.get_type(node.value)
    if isinstance(base_type, ModuleType):
      new_path = f"{base_type.path}.{node.attr.value}"
      self.table.record_type(node, ModuleType(name="Module", path=new_path))

  def leave_Call(self, node: cst.Call) -> None:
    """
    Infer return type of a call.
    1. Resolve function fully qualified name.
    2. Check SemanticsManager for return type.

    Args:
        node: Function call node.
    """
    api_path = None

    # Case A: Called on a Module (e.g. torch.randn)
    func_type = self.table.get_type(node.func)
    if isinstance(func_type, ModuleType):
      # The attribute chain resolution logic in leave_Attribute constructs paths
      api_path = func_type.path

    # Case B: Called on a Tensor (e.g. x.view()) -> Implicit API 'torch.Tensor.view'
    elif isinstance(node.func, cst.Attribute):
      receiver_type = self.table.get_type(node.func.value)
      # If valid tensor type detected
      if isinstance(receiver_type, TensorType):
        method = node.func.attr.value
        # Construct hypothetical API path for lookup: {framework}.Tensor.{method}
        if hasattr(receiver_type, "framework"):
          api_path = f"{receiver_type.framework}.Tensor.{method}"

    if api_path:
      # Query Semantics (Hub)
      # We first try the exact API path
      definition = self.semantics.get_definition(api_path)

      # Fallback: Loose lookup if strict match fails but matches naming convention
      if not definition and "Tensor" in api_path:
        leaf_method = api_path.split(".")[-1]
        # Try finding if 'leaf' is a known op that targets this framework
        # This is heuristic but useful for things like 'view' -> 'Reshape'
        definition = self.semantics.get_definition(leaf_method)

      if definition:
        op_id, details = definition
        # Check return type definition
        # Default for Array/Neural ops is Tensor/Array
        ret_type = details.get("return_type", "Any")

        # Check tier origin to guess if it returns a tensor
        # We use internal access safely
        key_origins = getattr(self.semantics, "_key_origins", {})
        tier = key_origins.get(op_id, "")

        # If return type is Tensor-like or op is in Math/Neural tier, assume Tensor return
        if ret_type in ["Tensor", "Array"] or "array" in str(tier).lower() or "neural" in str(tier).lower():
          # Infer framework from the API path prefix (e.g. 'torch')
          fw_hint = api_path.split(".")[0]
          self.table.record_type(node, TensorType(name="Tensor", framework=fw_hint))
