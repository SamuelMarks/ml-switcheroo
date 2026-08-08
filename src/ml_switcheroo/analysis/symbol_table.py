"""Symbol Table and Type Inference Analysis with Control Flow Support.

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

from typing import Any

import libcst as cst
from typing import Dict, Optional

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.scanners import get_full_name


from ml_switcheroo.analysis.symbol_types import SymbolType, TensorType, ModuleType, UnionType, Scope


class SymbolTable:
  """Container for static analysis results that maps CST nodes to inferred symbol types."""

  def __init__(self) -> None:
    """Initializes an empty symbol table mapping."""
    self._node_types: Dict[cst.CSTNode, SymbolType] = {}

  def record_type(self, node: cst.CSTNode, sym_type: SymbolType) -> None:
    """Associates a libcst CSTNode with its inferred SymbolType.

    Args:
        node: The CST node for which to record type information.
        sym_type: The determined type of the given AST node.

    """
    self._node_types[node] = sym_type

  def get_type(self, node: cst.CSTNode) -> Optional[SymbolType]:
    """Retrieves the inferred SymbolType for a specific CSTNode.

    Args:
        node: The CST node whose type needs to be looked up.

    Returns:
        The stored SymbolType if found, or None if no type has been recorded for the node.
    """
    return self._node_types.get(node)


class SymbolTableAnalyzer(cst.CSTVisitor):
  """Static analysis pass that populates a SymbolTable by traversing a libcst AST.

  This analyzer uses a bottom-up, post-order traversal to propagate types and
  track module imports, variable assignments, scoped entities, and control flow branching.
  """

  def __init__(self, semantics: SemanticsManager):
    """Initializes the symbol table analyzer with a semantics manager and root scope.

    Args:
        semantics: Reference to semantic knowledge base for type inference rules.

    """
    self.semantics = semantics
    self.table = SymbolTable()
    self.root_scope = Scope(name="global")
    self.current_scope = self.root_scope

  # --- Scoping ---

  def visit_ClassDef(self, node: cst.ClassDef) -> None:
    """Enters class scope, creating and pushing a new nested scope.

    Args:
        node: The ClassDef CST node representing the class definition.

    """
    self.current_scope = Scope(parent=self.current_scope, name=f"class_{node.name.value}")

  def leave_ClassDef(self, node: cst.ClassDef) -> None:
    """Exits the class scope, restoring the parent scope.

    Args:
        node: The ClassDef CST node representing the class definition.

    """
    assert self.current_scope.parent is not None
    self.current_scope = self.current_scope.parent

  def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
    """Enters a function scope, creating and pushing a new nested scope.

    Args:
        node: The FunctionDef CST node representing the function definition.

    """
    self.current_scope = Scope(parent=self.current_scope, name=f"func_{node.name.value}")

  def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
    """Exits the function scope, restoring the parent scope.

    Args:
        node: The FunctionDef CST node representing the function definition.

    """
    assert self.current_scope.parent is not None
    self.current_scope = self.current_scope.parent

  # --- Control Flow Support ---

  def visit_If(self, node: cst.If) -> bool:
    """Handles branching logic for an If node.

    This method takes snapshots of the active symbol table state prior to visiting the
    then and else branches, traverses both branches manually, and merges the resulting symbol
    mappings into the current scope via Union types.

    Args:
        node: The If CST node representing the conditional statement.

    Returns:
        False to indicate that manual AST traversal has been performed and the default
        visitor traversal should be bypassed for this node.
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
    """Handles looping logic for a For node, managing potential type ambiguity.

    Since loops can execute zero or many times, this method merges the symbols before the loop with the
    symbols after visiting the loop's body and orelse blocks, using Union types where necessary.

    Args:
        node: The For CST node representing the loop.

    Returns:
        False to indicate that manual AST traversal has been performed and the default
        visitor traversal should be bypassed for this node.
    """
    # Visit Iterator parts
    node.iter.visit(self)
    node.target.visit(self)

    start_state = self.current_scope.snapshot()

    # Visit Body
    node.body.visit(self)

    if node.orelse:
      node.orelse.visit(self)

    end_state = self.current_scope.snapshot()

    # Merge start (0 iterations case) with end (N iterations case)
    self.current_scope.symbols = self._merge_states(start_state, end_state)
    return False

  def visit_While(self, node: cst.While) -> bool:
    """Handles loop logic for a While node, managing potential type ambiguity.

    This method merges the symbols in the scope before executing the loop with the symbols in the scope after
    traversing the loop's body and orelse blocks.

    Args:
        node: The While CST node representing the while loop.

    Returns:
        False to indicate that manual AST traversal has been performed and the default
        visitor traversal should be bypassed for this node.
    """
    node.test.visit(self)
    start_state = self.current_scope.snapshot()
    node.body.visit(self)
    if node.orelse:
      node.orelse.visit(self)
    end_state = self.current_scope.snapshot()
    self.current_scope.symbols = self._merge_states(start_state, end_state)
    return False

  def leave_IfExp(self, node: cst.IfExp) -> None:
    """Infers and records the symbol type for a ternary conditional expression (A if C else B).

    Args:
        node: The IfExp CST node representing the ternary expression.

    """
    t1 = self.table.get_type(node.body)
    t2 = self.table.get_type(node.orelse)

    if t1 and t2:
      merged = self._make_union(t1, t2)
      self.table.record_type(node, merged)
    elif t1:
      self.table.record_type(node, t1)
    elif t2:
      self.table.record_type(node, t2)

  def _merge_states(self, state_a: Dict[str, SymbolType], state_b: Dict[str, SymbolType]) -> Dict[str, SymbolType]:
    """Merges two symbol dictionary states, generating UnionType for conflicting variables.

    When a symbol exists in both states but with different types, this method creates a UnionType
    containing both types. If a symbol is present in only one branch, we optimistically retain its type.

    Args:
        state_a: The first symbol state dictionary mapping names to SymbolType.
        state_b: The second symbol state dictionary mapping names to SymbolType.

    Returns:
        A merged symbol dictionary representing the unified state.
    """
    merged = {}
    all_keys = set(state_a.keys()) | set(state_b.keys())

    for k in all_keys:
      in_a = k in state_a
      in_b = k in state_b

      if in_a and in_b:
        val_a = state_a[k]
        val_b = state_b[k]
        if val_a == val_b:
          merged[k] = val_a
        else:
          merged[k] = self._make_union(val_a, val_b)
      elif in_a:
        merged[k] = state_a[k]
      else:
        merged[k] = state_b[k]

    return merged

  def _make_union(self, t1: SymbolType, t2: SymbolType) -> SymbolType:
    """Creates a deduplicated, flattened UnionType from two given SymbolTypes.

    Args:
        t1: The first SymbolType.
        t2: The second SymbolType.

    Returns:
        A UnionType containing the deduplicated and flattened types of both inputs,
        or a single SymbolType if the types are equivalent.
    """
    if t1 == t2:
      return t1

    types = []

    def collect(t: Any) -> Any:
      """Recursively extracts and flattens types from nested UnionType instances into a list.

      Args:
          t: The SymbolType to inspect and collect components from.

      """
      if isinstance(t, UnionType):
        types.extend(t.types)
      else:
        types.append(t)

    collect(t1)
    collect(t2)

    # Deduplicate by string representation (simplistic equality)
    unique = []
    seen = set()
    for t in types:
      s = str(t)
      if s not in seen:
        unique.append(t)
        seen.add(s)

    if len(unique) == 1:
      return unique[0]

    return UnionType(unique)

  # --- Definition Tracking ---

  def leave_Import(self, node: cst.Import) -> None:
    """Tracks imported modules and binds their alias names to ModuleType instances in the current scope.

    Args:
        node: The Import CST node representing the import statement.

    """
    for alias in node.names:
      full_path = get_full_name(alias.name)
      bind_name = (
        (alias.asname.name.value if isinstance(alias.asname.name, cst.Name) else "")
        if alias.asname
        else full_path.split(".")[0]
      )
      self.current_scope.set(bind_name, ModuleType(name="Module", path=full_path))

  def leave_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Tracks relative/from imports and binds imported names in the current scope.

    Args:
        node: The ImportFrom CST node representing the from-import statement.

    """
    if not node.module:
      return
    base_mod = get_full_name(node.module)

    if isinstance(node.names, cst.ImportStar):
      return

    for alias in node.names:
      import_name = alias.name.value if isinstance(alias.name, cst.Name) else ""
      bind_name = (
        (alias.asname.name.value if isinstance(alias.asname.name, cst.Name) else "") if alias.asname else import_name
      )
      full_path = f"{base_mod}.{import_name}"
      self.current_scope.set(bind_name, ModuleType(name="Module", path=full_path))

  def leave_Assign(self, node: cst.Assign) -> None:
    """Propagates and binds the inferred type from the right-hand side of an assignment to the targets.

    Args:
        node: The Assign CST node representing the assignment statement.

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
        self.table.record_type(target.target, rhs_type)

  # --- Usage Resolution ---

  def leave_Name(self, node: cst.Name) -> None:
    """Looks up a variable's type by name in the active scopes and records it on the CST node.

    Args:
        node: The Name CST node representing the variable usage.

    """
    sym_type = self.current_scope.get(node.value)
    if sym_type:
      self.table.record_type(node, sym_type)

  def leave_Attribute(self, node: cst.Attribute) -> None:
    """Resolves and records attribute types based on the receiver's inferred type.

    For module receivers, resolves sub-module or property paths.

    Args:
        node: The Attribute CST node representing the attribute access.

    """
    base_type = self.table.get_type(node.value)
    if isinstance(base_type, ModuleType):
      new_path = f"{base_type.path}.{node.attr.value}"
      self.table.record_type(node, ModuleType(name="Module", path=new_path))

  def leave_Call(self, node: cst.Call) -> None:
    """Infers and records the return type of a call expression using semantics definitions.

    It handles function calls on modules as well as methods invoked on tensors or union of tensors.

    Args:
        node: The Call CST node representing the function or method call.

    """
    api_path = None

    # Case A: Called on a Module (e.g. torch.randn)
    func_type = self.table.get_type(node.func)
    if isinstance(func_type, ModuleType):
      api_path = func_type.path

    # Case B: Called on a Tensor (e.g. x.view()) -> Implicit API 'torch.Tensor.view'
    elif isinstance(node.func, cst.Attribute):
      receiver_type = self.table.get_type(node.func.value)
      if isinstance(receiver_type, TensorType):
        method = node.func.attr.value
        api_path = f"{receiver_type.framework}.Tensor.{method}"
      # Handle Unions where ALL branches are Tensors
      elif isinstance(receiver_type, UnionType):
        # Heuristic: If ANY option in the union is a Tensor, we treat it as a potential Tensor call.
        # This helps with weak inference (e.g. Tensor OR None).
        # We pick the first TensorType to drive API lookup prefix.
        tensor_opt = next((t for t in receiver_type.types if isinstance(t, TensorType)), None)
        if tensor_opt is not None:
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
