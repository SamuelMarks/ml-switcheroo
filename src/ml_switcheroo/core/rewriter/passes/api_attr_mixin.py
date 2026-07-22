"""Mixin for ApiTransformer Attribute and Assign rewriting."""

import libcst as cst
from libcst import Attribute, CSTNode, Name
from typing import Union


from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.utils.node_diff import capture_node_source
from ml_switcheroo_ir.schema.ghost import SemanticTier


class ApiTransformerAttrMixin:
  """Docstring."""

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Intercepts assignments to track state or apply unwrapping rules."""
    # 1. Track Variable Initialization
    if isinstance(original_node.value, cst.Call):  # pragma: no cover
      func_name = self._get_qualified_name(original_node.value.func)  # type: ignore
      if func_name:  # pragma: no cover
        definition = self.semantics.get_definition(func_name)  # type: ignore
        if definition:
          abstract_id, _ = definition
          origins = getattr(self.semantics, "_key_origins", {})  # type: ignore
          tier = origins.get(abstract_id)
          if tier == SemanticTier.NEURAL.value:  # pragma: no cover
            for target in original_node.targets:
              target_name = self._get_qualified_name(target.target)  # type: ignore
              if target_name:  # pragma: no cover
                if target_name.startswith("self.") and len(self.context.scope_stack) > 1:  # type: ignore
                  # Track stateful variable in the class scope (parent of init scope)
                  self.context.scope_stack[-2].add(target_name)  # type: ignore
                else:
                  self._mark_stateful(target_name)  # type: ignore

    # 2. Assignment Unwrapping (Functional -> OOP)
    if isinstance(original_node.value, cst.Call):  # pragma: no cover
      # Fix: Check property existence before access
      if hasattr(self, "source_traits"):
        traits = self.source_traits
      else:
        traits = StructuralTraits()

      unwrap_method = traits.functional_execution_method
      if is_functional_apply(original_node.value, unwrap_method):
        if len(updated_node.targets) == 1:  # pragma: no cover
          target = updated_node.targets[0].target  # type: ignore
          if isinstance(target, (cst.Tuple, cst.List)):  # pragma: no cover
            elements = target.elements
            if len(elements) > 0:  # pragma: no cover
              primary_target = elements[0].value
              new_target = cst.AssignTarget(target=primary_target)
              new_node = updated_node.with_changes(targets=[new_target])
              get_tracer().log_mutation(
                "Assignment Unwrapping",
                capture_node_source(original_node),
                capture_node_source(new_node),
              )
              return new_node

    return updated_node

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> Union[Attribute, Name, CSTNode]:  # type: ignore
    """Rewrites attributes and constants (e.g. torch.float32)."""
    name = self._get_qualified_name(original_node)  # type: ignore
    if not name:
      return updated_node  # pragma: no cover

    lookup = self.semantics.get_definition(name)  # type: ignore
    if lookup:
      _, details = lookup
      target_var = details.get("variants", {}).get(self.target_fw)  # type: ignore

      # Plugin guard
      if target_var and "requires_plugin" in target_var:
        return updated_node

      # Check Op Type
      op_type = details.get("op_type", "function")

      # Function guard: If it is a function, let leave_Call handle it.
      # attributes should be processed here.
      if op_type == "function":
        if "std_args" in details and details["std_args"]:
          return updated_node

    # Perform mapping logic for constant/enum/attribute
    target_impl = self._get_mapping(name, silent=True)  # type: ignore

    if target_impl:
      # If semantic definition says it's an attribute/context, we rewrite aliases
      if "api" in target_impl:
        self._handle_variant_imports(target_impl)  # type: ignore
        return self._create_dotted_name(target_impl["api"])  # type: ignore

      # Support macros for constants (e.g. inf -> float('inf'))
      if "macro_template" in target_impl:  # pragma: no cover
        try:
          from ml_switcheroo.core.rewriter.calls.transformers import rewrite_as_macro

          # Constants have no args, pass empty lists
          return rewrite_as_macro(target_impl["macro_template"], [], [])
        except Exception:
          pass

    return updated_node
