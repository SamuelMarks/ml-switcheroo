"""Perform plugin for in_top_k.

Checks if target indices are present in top_k(predictions).
"""

import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("in_top_k_plugin")
def in_top_k_plugin(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """Plugin Transform Generic plugin for in_top_k.

  Checks if target indices are present in top_k(predictions).
  Typically translates in_top_k(predictions, targets, k) to
  something like (targets.unsqueeze(-1) == top_k(predictions, k).indices).any(dim=-1).

  Args:
      node: The CST Call node representing the in_top_k operation to be
          processed.
      ctx: The hook context containing metadata, configuration, and translation
          environment details.

  Returns:
      The transformed CST node, or the original node if no transformation was
          applied.
  """
  config = getattr(ctx, "_runtime_config", getattr(ctx, "config", None))
  target_fw = getattr(config, "target_framework", getattr(config, "target", None)) if config else None

  if target_fw != "torch":
    return node

  if len(node.args) < 3:
    return node

  predictions = node.args[0].value
  targets = node.args[1].value
  k = node.args[2].value

  template = cst.parse_expression("(TARGETS.unsqueeze(-1) == torch.topk(PREDICTIONS, K_VAL).indices).any(dim=-1)")

  class TemplateReplacer(cst.CSTTransformer):
    """Replaces placeholders in the AST template."""

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
      """Replaces the specific name placeholders with correct nodes.

      Args:
          original_node: Original name node.
          updated_node: Updated name node.

      Returns:
          cst.BaseExpression: The replacement node.
      """
      if original_node.value == "PREDICTIONS":
        return predictions
      if original_node.value == "TARGETS":
        return targets
      if original_node.value == "K_VAL":
        return k
      return updated_node

  # We know this returns an expression node
  return template.visit(TemplateReplacer())  # type: ignore
