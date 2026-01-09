"""
Attribute Rewriting Logic.

Handles transformation of attributes (e.g. `torch.float32`),
and tracks assignment logic to detect stateful variables (e.g. `self.layer = ...`).
"""

from typing import TYPE_CHECKING
import libcst as cst
from ml_switcheroo.enums import SemanticTier

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.calls.mixer import ApiStage


class AttributeMixin:
  """
  Mixin for transforming attributes and tracking assignments.

  Expects host `ApiStage` to provide `context` via `RewriterProxy`.
  """

  def leave_Assign(self: "ApiStage", original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Track stateful assignments (e.g. self.layer = Linear(...)).
    """
    if isinstance(original_node.value, cst.Call):
      func_name = self._get_qualified_name(original_node.value.func)
      if func_name:
        definition = self.context.semantics.get_definition(func_name)
        if definition:
          abstract_id, _ = definition
          origins = getattr(self.context.semantics, "_key_origins", {})
          tier = origins.get(abstract_id)

          if tier == SemanticTier.NEURAL.value:
            for target in original_node.targets:
              target_name = self._get_qualified_name(target.target)
              if target_name:
                if target_name.startswith("self.") and len(self.context.scope_stack) > 1:
                  # Add to class level scope (index -2 if inside __init__)
                  self.context.scope_stack[-2].add(target_name)
                else:
                  self._mark_stateful(target_name)

    return updated_node

  def leave_Attribute(self: "ApiStage", original: cst.Attribute, updated: cst.Attribute) -> cst.BaseExpression:
    """
    Visits attributes (e.g. torch.float32).
    """
    name = self._get_qualified_name(original)
    if not name:
      return updated

    lookup = self.context.semantics.get_definition(name)
    if lookup:
      _, details = lookup
      # CRITICAL CHECK for plugins + Safe access for None
      target_var = details.get("variants", {}).get(self.context.target_fw)

      # If target_var is None (explicit failure) or dict with plugin, skip rewrite
      if target_var and "requires_plugin" in target_var:
        return updated

      # Standard check: If it looks like a function (has args), keep it for leave_Call
      if "std_args" in details and details["std_args"]:
        return updated

    target_impl = self._get_mapping(name)
    if target_impl and "api" in target_impl:
      # --- Feature 13: Dynamic Import Injection ---
      self._handle_variant_imports(target_impl)
      return self._create_name_node(target_impl["api"])

    return updated
