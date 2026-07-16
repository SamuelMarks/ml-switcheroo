"""Mixin for ApiTransformer Attribute and Assign rewriting."""

import libcst as cst  # pragma: no cover
from libcst import Attribute, CSTNode, Name  # pragma: no cover
from typing import Union  # pragma: no cover

# pragma: no cover
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply  # pragma: no cover
from ml_switcheroo.core.tracer import get_tracer  # pragma: no cover
from ml_switcheroo.semantics.schema import StructuralTraits  # pragma: no cover
from ml_switcheroo.utils.node_diff import capture_node_source  # pragma: no cover
from ml_switcheroo_ir.schema.ghost import SemanticTier  # pragma: no cover


# pragma: no cover
# pragma: no cover
class ApiTransformerAttrMixin:  # pragma: no cover
  """Docstring."""  # pragma: no cover

  # pragma: no cover
  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:  # pragma: no cover
    """Intercepts assignments to track state or apply unwrapping rules."""  # pragma: no cover
    # 1. Track Variable Initialization  # pragma: no cover
    if isinstance(original_node.value, cst.Call):  # pragma: no cover
      func_name = self._get_qualified_name(original_node.value.func)  # type: ignore  # pragma: no cover
      if func_name:  # pragma: no cover
        definition = self.semantics.get_definition(func_name)  # type: ignore  # pragma: no cover
        if definition:  # pragma: no cover
          abstract_id, _ = definition  # pragma: no cover
          origins = getattr(self.semantics, "_key_origins", {})  # type: ignore  # pragma: no cover
          tier = origins.get(abstract_id)  # pragma: no cover
          if tier == SemanticTier.NEURAL.value:  # pragma: no cover
            for target in original_node.targets:  # pragma: no cover
              target_name = self._get_qualified_name(target.target)  # type: ignore  # pragma: no cover
              if target_name:  # pragma: no cover
                if target_name.startswith("self.") and len(self.context.scope_stack) > 1:  # type: ignore  # pragma: no cover
                  # Track stateful variable in the class scope (parent of init scope)  # pragma: no cover
                  self.context.scope_stack[-2].add(target_name)  # type: ignore  # pragma: no cover
                else:  # pragma: no cover
                  self._mark_stateful(target_name)  # type: ignore  # pragma: no cover
    # pragma: no cover
    # 2. Assignment Unwrapping (Functional -> OOP)  # pragma: no cover
    if isinstance(original_node.value, cst.Call):  # pragma: no cover
      # Fix: Check property existence before access  # pragma: no cover
      if hasattr(self, "source_traits"):  # pragma: no cover
        traits = self.source_traits  # pragma: no cover
      else:  # pragma: no cover
        traits = StructuralTraits()  # pragma: no cover
      # pragma: no cover
      unwrap_method = traits.functional_execution_method  # pragma: no cover
      if is_functional_apply(original_node.value, unwrap_method):  # pragma: no cover
        if len(updated_node.targets) == 1:  # pragma: no cover
          target = updated_node.targets[0].target  # type: ignore  # pragma: no cover
          if isinstance(target, (cst.Tuple, cst.List)):  # pragma: no cover
            elements = target.elements  # pragma: no cover
            if len(elements) > 0:  # pragma: no cover
              primary_target = elements[0].value  # pragma: no cover
              new_target = cst.AssignTarget(target=primary_target)  # pragma: no cover
              new_node = updated_node.with_changes(targets=[new_target])  # pragma: no cover
              get_tracer().log_mutation(  # pragma: no cover
                "Assignment Unwrapping",
                capture_node_source(original_node),
                capture_node_source(new_node),  # pragma: no cover
              )  # pragma: no cover
              return new_node  # pragma: no cover
    # pragma: no cover
    return updated_node  # pragma: no cover

  # pragma: no cover
  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> Union[Attribute, Name, CSTNode]:  # type: ignore  # pragma: no cover
    """Rewrites attributes and constants (e.g. torch.float32)."""  # pragma: no cover
    name = self._get_qualified_name(original_node)  # type: ignore  # pragma: no cover
    if not name:  # pragma: no cover
      return updated_node  # pragma: no cover
    # pragma: no cover
    lookup = self.semantics.get_definition(name)  # type: ignore  # pragma: no cover
    if lookup:  # pragma: no cover
      _, details = lookup  # pragma: no cover
      target_var = details.get("variants", {}).get(self.target_fw)  # type: ignore  # pragma: no cover
      # pragma: no cover
      # Plugin guard  # pragma: no cover
      if target_var and "requires_plugin" in target_var:  # pragma: no cover
        return updated_node  # pragma: no cover
      # pragma: no cover
      # Check Op Type  # pragma: no cover
      op_type = details.get("op_type", "function")  # pragma: no cover
      # pragma: no cover
      # Function guard: If it is a function, let leave_Call handle it.  # pragma: no cover
      # attributes should be processed here.  # pragma: no cover
      if op_type == "function":  # pragma: no cover
        if "std_args" in details and details["std_args"]:  # pragma: no cover
          return updated_node  # pragma: no cover
    # pragma: no cover
    # Perform mapping logic for constant/enum/attribute  # pragma: no cover
    target_impl = self._get_mapping(name, silent=True)  # type: ignore  # pragma: no cover
    # pragma: no cover
    if target_impl:  # pragma: no cover
      # If semantic definition says it's an attribute/context, we rewrite aliases  # pragma: no cover
      if "api" in target_impl:  # pragma: no cover
        self._handle_variant_imports(target_impl)  # type: ignore  # pragma: no cover
        return self._create_dotted_name(target_impl["api"])  # type: ignore  # pragma: no cover
      # pragma: no cover
      # Support macros for constants (e.g. inf -> float('inf'))  # pragma: no cover
      if "macro_template" in target_impl:  # pragma: no cover
        try:  # pragma: no cover
          from ml_switcheroo.core.rewriter.calls.transformers import rewrite_as_macro  # pragma: no cover

          # pragma: no cover
          # Constants have no args, pass empty lists  # pragma: no cover
          return rewrite_as_macro(target_impl["macro_template"], [], [])  # pragma: no cover
        except Exception:  # pragma: no cover
          pass  # pragma: no cover
    # pragma: no cover
    return updated_node  # pragma: no cover
