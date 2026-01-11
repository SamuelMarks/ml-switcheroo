"""
Graph Patcher for AST Surgery.

This module implements the execution phase of the Graph-Guided Rewriting pipeline.
It applies a set of topological mutations (Deletions, Replacements) generated
by the ``GraphDiffer`` to the concrete syntax tree (CST) of the source code.

It bridges the gap between the Abstract Logical Graph and the physical source code
by using **Provenance Tracking** to locate the exact CST nodes corresponding
to graph nodes, and the **Snippet Emitter** to generate valid replacement code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union, Any

import libcst as cst
from ml_switcheroo.compiler.ir import LogicalNode
from ml_switcheroo.compiler.backends.python_snippet import PythonSnippetEmitter


@dataclass
class PatchAction:
  """Base class for patch instructions."""

  node_id: str


@dataclass
class DeleteAction(PatchAction):
  """
  Instruction to remove a node from the AST.
  Used for nodes that have been fused into others or pruned.
  """

  pass


@dataclass
class ReplaceAction(PatchAction):
  """
  Instruction to replace an existing node with a new code snippet.

  Commonly used for:
  1. Replacing a sequence anchor (e.g. Conv2d) with a Fused Block.
  2. Swapping layer definitions in `__init__`.

  Attributes:
      new_node: The logical definition of the new component.
      input_vars: List of variable names to pass as arguments.
      output_var: The variable name to assign result to.
      is_init: If True, uses `emit_init` logic (assignments).
               If False, uses `emit_call` logic (executions).
  """

  new_node: LogicalNode
  input_vars: List[str] = field(default_factory=list)
  output_var: str = "x"
  is_init: bool = False


class GraphPatcher(cst.CSTTransformer):
  """
  LibCST Transformer that applies graph optimizations to source code.

  It builds a reverse-lookup map of the Provenance Registry to identify
  CST nodes by object identity, then checks against the Transformation Plan
  during traversal to apply Deletes or Replacements.
  """

  def __init__(
    self,
    plan: List[PatchAction],
    provenance: Dict[str, cst.CSTNode],
    emitter: PythonSnippetEmitter,
  ) -> None:
    """
    Initialize the patcher.

    Args:
        plan: Ordered list of actions to perform.
        provenance: Mapping of LogicalNode ID -> Original CST Node.
        emitter: Logic for synthesizing new Python code.
    """
    self.plan = plan
    self.provenance = provenance
    self.emitter = emitter

    # Index actions for O(1) lookup during traversal
    # Map: id(CSTNode) -> List[PatchAction]
    # One node might trigger multiple actions (e.g. if one ID maps to Init and Call)
    # But wait, provenance maps ID -> Node. One ID maps to one Node?
    # Actually GraphExtractor maps ID -> Node.
    # If 'conv1' is used in init and forward, GraphExtractor usually picks one primary definition?
    # GraphExtractor `node_map` currently maps ID -> Definition (Init) OR Call (Forward) depending on phase.
    # It overwrites. This is a limitation of simple provenance.
    # Ideally, we need ID -> {InitNode, CallNode}.
    # For this implementation, we assume we map by ID. If we encounter a node, we check if there's an action.

    # Improvement: Provenance map should support multiple contexts or keys?
    # Or Actions distinguish init/call.
    # Actions distinguish is_init.
    # We need to map `id(node)` to the specific action.
    # But provenance only gives `node_id` -> `cst_node`.
    # If `node_id` is used for both init and call, provenance only stores ONE.
    # GraphExtractor in phase 1 implementation overwrote node_map entries?
    # "self.node_map[attr_name] = node" (Init)
    # "self.node_map[layer_name] = context_node" (Call - function)
    # "self.node_map[ext_id] = arg" (Input)

    # If a layer has both Init and Call, GraphExtractor only tracked the Init?
    # No, `_analyze_layer_def` tracks init. `_analyze_call_expression` tracks functional/ephemeral nodes.
    # For Stateful layers, `provenance` in GraphExtractor `layer_registry` stores the *logical* node.
    # The `node_map` stores CST anchor.
    # Currently the extractor implementation only stored ONE cst anchor per ID.
    # So deleting a stateful layer deletes its Init, but not its Call?
    # We need to support patching based on the ID available.
    # For now, let's assume the Patcher acts on what is in Provenance.

    self._action_map: Dict[int, PatchAction] = {}
    self._build_action_index()

  def _build_action_index(self) -> None:
    """
    Correlates Plan IDs with CST Nodes via Provenance.
    Populates `_action_map` keying by `id(node)`.
    """
    # Create map of node_id -> Action
    # Note: A plan might contain multiple actions for one ID (e.g. replace INIT and replace CALL?)
    # But plan list order matters.
    # Given simple provenance (1:1 ID->Node), we map 1 action.
    id_to_action = {a.node_id: a for a in self.plan}

    for node_id, cst_node in self.provenance.items():
      if node_id in id_to_action:
        self._action_map[id(cst_node)] = id_to_action[node_id]

  # --- Statement Level Hooks ---

  def leave_Assign(
    self, original_node: cst.Assign, updated_node: cst.Assign
  ) -> Union[cst.Assign, cst.SimpleStatementLine, cst.RemovalSentinel]:
    """
    Intercepts Assignment statements (e.g. `self.conv = ...`, `y = func(x)`).
    """
    return self._handle_node(original_node, updated_node)

  def leave_Expr(
    self, original_node: cst.Expr, updated_node: cst.Expr
  ) -> Union[cst.Expr, cst.SimpleStatementLine, cst.RemovalSentinel]:
    """
    Intercepts Expression statements (e.g. `func(x)` without assignment).
    """
    return self._handle_node(original_node, updated_node)

  def leave_Call(
    self, original_node: cst.Call, updated_node: cst.Call
  ) -> Union[cst.Call, cst.BaseExpression, cst.RemovalSentinel]:
    return self._handle_node(original_node, updated_node)

  def leave_SimpleStatementLine(
    self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
  ) -> Union[cst.SimpleStatementLine, cst.RemovalSentinel]:
    """
    Cleans up statements that became empty due to children deletion.
    """
    # If logic removed the inner assign/expr, `updated_node.body` is empty list
    if not updated_node.body:
      return cst.RemoveFromParent()
    return updated_node

  def _handle_node(self, original: cst.CSTNode, updated: cst.CSTNode) -> Any:
    """
    Core dispatch logic.
    """
    oid = id(original)
    if oid not in self._action_map:
      return updated

    action = self._action_map[oid]

    if isinstance(action, DeleteAction):
      return cst.RemoveFromParent()

    if isinstance(action, ReplaceAction):
      if action.is_init:
        # Initialization (Statement) replacement
        new_stmt = self.emitter.emit_init(action.new_node)
        return self._unwrap_stmt_if_nested(original, new_stmt)
      else:
        # Execution logic
        # Replacing Call/Expr/Assign

        # Check if context expects Expression or Statement
        is_expr_context = isinstance(original, (cst.Call, cst.BaseExpression)) and not isinstance(
          original, (cst.Expr, cst.Assign)
        )

        if is_expr_context:
          return self.emitter.emit_expression(action.new_node, action.input_vars)
        else:
          # Statement-like replacement
          new_stmt = self.emitter.emit_call(action.new_node, action.input_vars, action.output_var)
          return self._unwrap_stmt_if_nested(original, new_stmt)

    return updated

  def _unwrap_stmt_if_nested(self, context_node: cst.CSTNode, new_stmt: cst.SimpleStatementLine) -> Any:
    """
    Helper: If we are replacing a node that is already inside a SimpleStatementLine body list
    (like Assign or Expr), we should return the inner component to avoid double wrapping.
    """
    if isinstance(context_node, (cst.Assign, cst.Expr)):
      if new_stmt.body and len(new_stmt.body) > 0:
        # Return FlattenSentinel of the inner nodes
        # This injects the new assignment/expr into the parent statement line
        # Wait, if we return FlattenSentinel here, does LibCST support splicing into a body list?
        # Assign/Expr are in SimpleStatementLine.body.
        # Yes, return FlattenSentinel([node]) splits it.
        return cst.FlattenSentinel(new_stmt.body)

    return new_stmt
