"""
Topological Diff Engine.

Analyzes differences between two LogicalGraphs to produce a patch plan.
It supports:
1. Identifying deleted nodes (nodes in source but not in target).
2. Identifying replacements (new fused nodes).
3. Mapping new nodes to anchors in the original graph for placement.
"""

from typing import List, Dict, Set, Any
from dataclasses import dataclass
from ml_switcheroo.compiler.ir import LogicalGraph


@dataclass
class PatchAction:
  """Base patch operation."""

  node_id: str


@dataclass
class DeleteAction(PatchAction):
  """Instruction to remove a node."""

  pass


@dataclass
class ReplaceAction(PatchAction):
  """
  Instruction to replace an anchor node with new logic.
  For the Differ context, 'new_node' is the LogicalNode from the target graph.
  """

  new_node: Any  # LogicalNode
  input_vars: List[str]
  output_var: str
  is_init: bool = False


# Type hint workaround for circular ref
from ml_switcheroo.compiler.ir import LogicalNode as _LogicalNode  # noqa: F401


class GraphDiffer:
  """
  Calculates transformation steps to migrate from Source Graph to Target Graph.

  Assumption:
  - Optimization is mostly fusion/deletion.
  - New nodes (fused) are distinct by ID or metadata.
  - We anchor rewrites to the *last* node of a fused chain (the sink).
  """

  def diff(self, source: LogicalGraph, target: LogicalGraph) -> List[PatchAction]:
    """
    Compare graphs and return list of actions.

    Algorithm:
    1. Identify Removed IDs: `source_ids - target_ids`.
    2. Identify Added/Changed Nodes.
    3. Match Added Nodes to Removed Nodes (Find Anchor).
       - Heuristic: If FusedNode depends on input X, and RemovedNode N depends on input X,
         they might be related.
       - Current Heuristic: Use provenance or manual mapping supplied by Optimizer?
       - Simpler Heuristic: Fused nodes usually REPLACE a subgraph. The "output" variable
         usually stays consistent flow-wise, or we replace the Sink of the subgraph.

    For this implementation, we rely on the Optimizer naming convention or graph
    structure. Since graph optimization logic isn't fully inspectable here,
    we implement a Diff strategy based on ID presence.

    Strategies:
    - If ID is missing in Target -> DELETE.
    - If ID is new/fused in Target -> REPLACE.
      - We need an Anchor in Source to attach the Replacement.
      - We attach the Replacement to the *first available* deleted node that matches topology?
      - Better: If Optimizer produced FusedNode `fused_c1`, and `c1` is deleted, `c1` is anchor?
    """
    actions: List[PatchAction] = []

    src_ids = {n.id for n in source.nodes}
    tgt_ids = {n.id for n in target.nodes}

    # 1. Deletions
    # All nodes present in Source but not Target are candidates for Deletion
    deleted_ids = src_ids - tgt_ids

    # 2. Additions (Replacements)
    new_nodes = [n for n in target.nodes if n.id not in src_ids]

    # We need to map New Nodes to an Anchor (one of the deleted nodes).
    # We use a greedy mapping strategy based on graph position or name heuristic.
    # For this version, we assume metadata 'fused_from' or similar tracking could help,
    # but for now we'll mark all deleted nodes as DELETE.
    # AND we need to insert the new nodes. We insert them by replacing one of the deletes.

    # Heuristic: Map new node 'fused_X' to 'X'.
    matched_anchors = set()

    for new_node in new_nodes:
      anchor = None

      # Metadata hint (Populated by GraphOptimizer in ideal implementation)
      if "anchor" in new_node.metadata:
        anchor = new_node.metadata["anchor"]
      # Naming heuristic (fused_c1 -> c1)
      elif new_node.id.startswith("fused_"):
        candidate = new_node.id.replace("fused_", "")
        if candidate in deleted_ids:
          anchor = candidate

      if anchor and anchor in deleted_ids:
        # Determine inputs/outputs for the replacement snippet
        # Inputs: edges pointing to new_node in Target Graph
        in_edges = [e for e in target.edges if e.target == new_node.id]
        input_vars = [e.source for e in in_edges]

        # Output: new_node.id usually unless mapped
        out_var = new_node.id

        # Create Replacement Action
        # Check execution context (Init vs Call).
        # Optimizer usually fuses layers (Init) or Ops (Call).
        # We generate TWO actions: One for Init, One for Call.

        # 1. Init Replacement (If stateful)
        if _is_likely_stateful(new_node):
          actions.append(
            ReplaceAction(
              node_id=anchor,
              new_node=new_node,
              input_vars=[],  # Init takes generic config args handled by emitter
              output_var="",  # Not used for init
              is_init=True,
            )
          )

        # 2. Call Replacement
        # We need to find the call-site of the anchor.
        # In Source, 'anchor' is used.
        # We mark the anchor as REPLACED.
        # Since we issued a replacement on the ID 'anchor', the Patcher will
        # handle both Init and Call sites for that ID index.
        actions.append(
          ReplaceAction(node_id=anchor, new_node=new_node, input_vars=input_vars, output_var=out_var, is_init=False)
        )

        matched_anchors.add(anchor)

    # Mark remaining deleted nodes as pure Delete
    for did in deleted_ids:
      if did not in matched_anchors:
        actions.append(DeleteAction(node_id=did))

    return actions


def _is_likely_stateful(node: Any) -> bool:
  """Heuristic for statefulness based on kind (naming convention)."""
  return node.kind and node.kind[0].isupper() or "Fused" in node.kind
