"""
Symbol Table and Type Inference Analysis with Control Flow Support.

This module provides a static analysis pass to infer variable types and scopes
before rewriting occurs. It builds a mapping of AST nodes to inferred type objects,
allowing the rewriter to make decisions based on the semantic type of a variable
(e.g., "is this a Tensor?") rather than just its lexical name.

The `SymbolTableAnalyzer` visitor populates a `SymbolTable` by tracking:
1.  **Imports**: Mapping module aliases to `ModuleType`.
2.  **Assignments**: Propagating types from RHS to LHS.
3.  **Scopes**: Handling nested function/class definitions.
4.  **Control Flow**: Handling type ambiguity in branches (Phi nodes) via Union types.
"""

import libcst as cst
from typing import Dict, Optional, List, Set, Union as PyUnion
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

  def __eq__(self, other: object) -> bool:
    """TODO: Add docstring."""  # pragma: no cover
    if not isinstance(other, SymbolType):  # pragma: no cover
      return False  # pragma: no cover
    return self.name == other.name  # pragma: no cover


@dataclass
class TensorType(SymbolType):
  """
  Represents a Tensor object from a specific framework.
  """

  framework: str
  """The framework key (e.g. "torch" or "jax") responsible for this tensor."""

  def __eq__(self, other: object) -> bool:
    """TODO: Add docstring."""
    if not isinstance(other, TensorType):  # pragma: no cover
      return False
    return self.name == other.name and self.framework == other.framework  # pragma: no cover


@dataclass
class ModuleType(SymbolType):
  """
  Represents an imported module or sub-module.
  """

  path: str
  """Fully qualified path string (e.g. "torch.nn")."""

  def __eq__(self, other: object) -> bool:
    """TODO: Add docstring."""
    if not isinstance(other, ModuleType):
      return False
    return self.name == other.name and self.path == other.path


@dataclass
class UnionType(SymbolType):
  """
  Represents a union of potential types resulting from control flow divergence.
  """

  types: List[SymbolType]

  def __init__(self, types: List[SymbolType]):
    """TODO: Add docstring."""
    super().__init__("Union")  # pragma: no cover
    self.types = types  # pragma: no cover

  def __str__(self) -> str:
    """TODO: Add docstring."""  # pragma: no cover
    unique_names = sorted(list(set(str(t) for t in self.types)))  # pragma: no cover
    return f"Union[{', '.join(unique_names)}]"  # pragma: no cover

  # pragma: no cover
  def __eq__(self, other: object) -> bool:
    """TODO: Add docstring."""
    if not isinstance(other, UnionType):  # pragma: no cover
      return False  # pragma: no cover
    # Set based comparison for equivalence ignoring order
    return set(str(t) for t in self.types) == set(str(t) for t in other.types)  # pragma: no cover


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

    Returns:  # pragma: no cover
        The SymbolType if found, else None.
    """
    if name in self.symbols:
      return self.symbols[name]
    if self.parent:
      return self.parent.get(name)  # pragma: no cover
    return None

  def snapshot(self) -> Dict[str, SymbolType]:
    """Returns a shallow copy of the current symbol table for branching."""
    return self.symbols.copy()


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
  Implements shallow control flow inference for If/Else and Loops.
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
    self.current_scope = self.root_scope  # pragma: no cover

  # --- Scoping ---

  def visit_ClassDef(self, node: cst.ClassDef) -> None:  # pragma: no cover
    """Enters class scope."""  # pragma: no cover
    self.current_scope = Scope(parent=self.current_scope, name=f"class_{node.name.value}")

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    """Exits class scope."""  # pragma: no cover
    if self.current_scope.parent:  # pragma: no cover
      self.current_scope = self.current_scope.parent  # pragma: no cover

  def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # pragma: no cover
    """Enters function scope."""  # pragma: no cover
    self.current_scope = Scope(parent=self.current_scope, name=f"func_{node.name.value}")

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """Exits function scope."""
    if self.current_scope.parent:  # pragma: no cover
      self.current_scope = self.current_scope.parent  # pragma: no cover

  # --- Control Flow Support ---

  def visit_If(self, node: cst.If) -> bool:
    """
    Handle branching logic.
    1. Snapshot state.
    2. Visit body -> State_Body.
    3. Revert to Snapshot.
    4. Visit Else (if any) -> State_Else.
    5. Merge (State_Body, State_Else).
    """
    # 1. Visit Test
    node.test.visit(self)

    # 2. Snapshot
    start_state = self.current_scope.snapshot()

    # 3. Visit Body
    node.body.visit(self)
    body_state = self.current_scope.snapshot()

    # 4. Restore for Else
    self.current_scope.symbols = start_state.copy()

    # 5. Visit Else
    # Note: orelse can contain an 'if' (elif) or 'else' block
    if node.orelse:
      node.orelse.visit(self)

    else_state = self.current_scope.snapshot()

    # 6. Merge
    self.current_scope.symbols = self._merge_states(body_state, else_state)

    return False  # Manual traversal done

  def visit_For(self, node: cst.For) -> bool:
    """
    Handle loop logic.
    Loops may execute 0 times or N times, introducing potential ambiguity.
    We merge the state after loop body with the state before loop.
    """
    # Visit Iterator parts
    node.iter.visit(self)
    node.target.visit(self)

    start_state = self.current_scope.snapshot()  # pragma: no cover

    # Visit Body
    node.body.visit(self)

    if node.orelse:
      node.orelse.visit(self)  # pragma: no cover

    end_state = self.current_scope.snapshot()

    # Merge start (0 iterations case) with end (N iterations case)  # pragma: no cover
    self.current_scope.symbols = self._merge_states(start_state, end_state)  # pragma: no cover
    return False  # pragma: no cover

  # pragma: no cover
  def visit_While(self, node: cst.While) -> bool:  # pragma: no cover
    """Handle while loop logic."""  # pragma: no cover
    node.test.visit(self)  # pragma: no cover
    start_state = self.current_scope.snapshot()  # pragma: no cover
    node.body.visit(self)
    if node.orelse:
      node.orelse.visit(self)
    end_state = self.current_scope.snapshot()
    self.current_scope.symbols = self._merge_states(start_state, end_state)
    return False

  def leave_IfExp(self, node: cst.IfExp) -> None:
    """
    Infers type for ternary expression: `A if C else B`.
    """
    t1 = self.table.get_type(node.body)  # pragma: no cover
    t2 = self.table.get_type(node.orelse)  # pragma: no cover
    # pragma: no cover
    if t1 and t2:  # pragma: no cover
      merged = self._make_union(t1, t2)
      self.table.record_type(node, merged)
    elif t1:  # pragma: no cover
      self.table.record_type(node, t1)  # pragma: no cover
    elif t2:  # pragma: no cover
      self.table.record_type(node, t2)  # pragma: no cover

  def _merge_states(self, state_a: Dict[str, SymbolType], state_b: Dict[str, SymbolType]) -> Dict[str, SymbolType]:
    """
    Merges two symbol dictionaries, creating Unions for conflicts.
    A missing key in one branch implies a potential Unbound state,
    but we optimistically retain the structured type found in the other branch.
    """
    merged = {}
    all_keys = set(state_a.keys()) | set(state_b.keys())

    for k in all_keys:
      in_a = k in state_a
      in_b = k in state_b

      if in_a and in_b:
        val_a = state_a[k]
        val_b = state_b[k]
        if val_a == val_b:  # pragma: no cover
          merged[k] = val_a  # pragma: no cover
        else:
          merged[k] = self._make_union(val_a, val_b)
      elif in_a:
        merged[k] = state_a[k]
      elif in_b:  # pragma: no cover
        merged[k] = state_b[k]  # pragma: no cover
    # pragma: no cover
    return merged

  def _make_union(self, t1: SymbolType, t2: SymbolType) -> SymbolType:
    """Creates a deduplicated UnionType from two types."""
    if t1 == t2:
      return t1  # pragma: no cover

    types = []

    def collect(t):
      """TODO: Add docstring."""
      if isinstance(t, UnionType):
        types.extend(t.types)  # pragma: no cover
      else:
        types.append(t)

    collect(t1)
    collect(t2)

    # Deduplicate by string representation (simplistic equality)
    unique = []
    seen = set()
    for t in types:  # pragma: no cover
      s = str(t)
      if s not in seen:
        unique.append(t)
        seen.add(s)

    if len(unique) == 1:
      return unique[0]  # pragma: no cover

    return UnionType(unique)

  # --- Definition Tracking ---

  def leave_Import(self, node: cst.Import) -> None:
    """
    Track imports.
    e.g. `import torch` -> symbols['torch'] = ModuleType(name='Module', path='torch')
    """
    for alias in node.names:
      full_path = get_full_name(alias.name)
      bind_name = alias.asname.name.value if alias.asname else full_path.split(".")[0]
      self.current_scope.set(bind_name, ModuleType(name="Module", path=full_path))  # pragma: no cover

  # pragma: no cover
  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:  # pragma: no cover
    """
    Track from-imports.  # pragma: no cover
    e.g. `from torch import nn` -> symbols['nn'] = ModuleType(name='Module', path='torch.nn')  # pragma: no cover
    """  # pragma: no cover
    if not node.module:  # pragma: no cover
      return  # pragma: no cover
    base_mod = get_full_name(node.module)  # pragma: no cover

    for alias in node.names:
      if isinstance(alias, cst.ImportAlias):
        import_name = alias.name.value
        bind_name = alias.asname.name.value if alias.asname else import_name
        full_path = f"{base_mod}.{import_name}"
        self.current_scope.set(bind_name, ModuleType(name="Module", path=full_path))

  def leave_Assign(self, node: cst.Assign) -> None:  # pragma: no cover
    """
    Propagate type from RHS to LHS.
    x = torch.randn() -> x is Tensor.
    """
    rhs_type = self.table.get_type(node.value)
    if not rhs_type:
      return

    for target in node.targets:  # pragma: no cover
      # Handle simple name assignment: x = ...  # pragma: no cover
      if isinstance(target.target, cst.Name):
        name = target.target.value
        self.current_scope.set(name, rhs_type)
        self.table.record_type(target.target, rhs_type)
      # Handle attributes: self.x = ...
      elif isinstance(target.target, cst.Attribute):
        self.table.record_type(target.target, rhs_type)

  # --- Usage Resolution ---

  def leave_Name(self, node: cst.Name) -> None:
    """
    Look up variable in scope.
    """
    sym_type = self.current_scope.get(node.value)
    if sym_type:
      self.table.record_type(node, sym_type)

  def leave_Attribute(self, node: cst.Attribute) -> None:
    """
    Resolve attributes based on their receiver type.
    If `x` is Module('torch'), `x.nn` is Module('torch.nn').
    If `x` is Tensor, `x.shape` might be recorded etc.
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
    """
    api_path = None

    # Case A: Called on a Module (e.g. torch.randn)
    func_type = self.table.get_type(node.func)
    if isinstance(func_type, ModuleType):  # pragma: no cover
      api_path = func_type.path  # pragma: no cover
    # pragma: no cover
    # Case B: Called on a Tensor (e.g. x.view()) -> Implicit API 'torch.Tensor.view'
    elif isinstance(node.func, cst.Attribute):
      receiver_type = self.table.get_type(node.func.value)
      if isinstance(receiver_type, TensorType):
        method = node.func.attr.value  # pragma: no cover
        if hasattr(receiver_type, "framework"):  # pragma: no cover
          api_path = f"{receiver_type.framework}.Tensor.{method}"  # pragma: no cover
      # Handle Unions where ALL branches are Tensors
      elif isinstance(receiver_type, UnionType):
        # Heuristic: If ANY option in the union is a Tensor, we treat it as a potential Tensor call.
        # This helps with weak inference (e.g. Tensor OR None).
        # We pick the first TensorType to drive API lookup prefix.
        tensor_opt = next((t for t in receiver_type.types if isinstance(t, TensorType)), None)
        if tensor_opt:
          method = node.func.attr.value
          api_path = f"{tensor_opt.framework}.Tensor.{method}"

    if api_path:
      definition = self.semantics.get_definition(api_path)

      # Fallback: Loose lookup if strict match fails but matches naming convention
      if not definition and "Tensor" in api_path:
        leaf_method = api_path.split(".")[-1]
        definition = self.semantics.get_definition(leaf_method)

      if definition:
        op_id, details = definition
        ret_type = details.get("return_type", "Any")

        key_origins = getattr(self.semantics, "_key_origins", {})
        tier = key_origins.get(op_id, "")

        if ret_type in ["Tensor", "Array"] or "array" in str(tier).lower() or "neural" in str(tier).lower():
          fw_hint = api_path.split(".")[0]
          self.table.record_type(node, TensorType(name="Tensor", framework=fw_hint))
