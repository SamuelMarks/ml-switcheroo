"""Mixin for ApiTransformer Attribute and Assign rewriting.

This module provides `ApiTransformerAttrMixin`, a mixin class intended to be used with
LibCST CSTTransformers (specifically the main ApiTransformer). It implements the AST
traversal hooks for `Assign` and `Attribute` nodes. This allows tracking assignments
of stateful neural network variables, performing functional-to-OOP assignment unwrapping
based on framework-specific structural traits, and mapping attributes, constants, enums,
or macros from a source machine learning framework to a target framework.
"""

import libcst as cst
from libcst import Attribute, CSTNode, Name
from typing import Union


from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.utils.node_diff import capture_node_source
from ml_switcheroo_ir.schema.ghost import SemanticTier


class ApiTransformerAttrMixin:
  """Mixin class that handles rewriting and tracking for assignment and attribute CST nodes.

  This mixin is designed to be combined with a LibCST `CSTTransformer` that provides
  semantic definitions (`self.semantics`), target framework settings (`self.target_fw`),
  and context/scope tracking (`self.context`).

  It implements:
  1. Assignment interception to track neural network state variable instantiation
     and scope tracking.
  2. Assignment unwrapping from functional style execution (e.g., returning state and output)
     to OOP-style assignment where only the output is captured.
  3. Attribute rewriting for constants, enums, and other properties, supporting macro
     substitutions and framework-specific API mappings.
  """

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Intercepts and post-processes CST assignment nodes during traversal.

    This method performs two distinct transformations:
    1. Stateful Variable Tracking: It checks if the right-hand side of the assignment
       is a functional call to a neural network component (based on the framework's semantic tier).
       If so, it tracks the variable name inside the corresponding context scope.
    2. Assignment Unwrapping: It handles situations where functional execution returns a tuple
       of (output, updated_state) but the target OOP style expects only the primary output.
       If the assignment matches the functional execution method pattern, it unwraps the targets
       to assign only to the first element (the primary output).

    Args:
        original_node: The original LibCST assignment node before children traversal.
        updated_node: The updated LibCST assignment node with child-level updates.

    Returns:
        The mutated or original updated LibCST assignment node.
    """
    # 1. Track Variable Initialization
    if isinstance(original_node.value, cst.Call):
      func_name = self._get_qualified_name(original_node.value.func)  # type: ignore
      if func_name:
        definition = self.semantics.get_definition(func_name)  # type: ignore
        if definition:
          abstract_id, _ = definition
          origins = getattr(self.semantics, "_key_origins", {})  # type: ignore
          tier = origins.get(abstract_id)
          if tier == SemanticTier.NEURAL.value:
            for target in original_node.targets:
              target_name = self._get_qualified_name(target.target)  # type: ignore
              if target_name:
                if target_name.startswith("self.") and len(self.context.scope_stack) > 1:  # type: ignore
                  # Track stateful variable in the class scope (parent of init scope)
                  self.context.scope_stack[-2].add(target_name)  # type: ignore
                else:
                  self._mark_stateful(target_name)  # type: ignore

    # 2. Assignment Unwrapping (Functional -> OOP)
    if isinstance(original_node.value, cst.Call):
      # Fix: Check property existence before access
      if hasattr(self, "source_traits"):
        traits = self.source_traits
      else:
        traits = StructuralTraits()

      unwrap_method = traits.functional_execution_method
      if is_functional_apply(original_node.value, unwrap_method):
        if len(updated_node.targets) == 1:
          target = updated_node.targets[0].target  # type: ignore
          if isinstance(target, (cst.Tuple, cst.List)):
            elements = target.elements
            if len(elements) > 0:
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

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> Union[Attribute, Name, CSTNode]:
    """Intercepts and rewrites CST attribute nodes during traversal.

    This method resolves the qualified name of an attribute (e.g., `torch.float32`) and
    maps it to the equivalent representation in the target framework (e.g., `jax.numpy.float32`
    or `mlx.core.float32`). It supports standard API replacements, plugin-guarded attributes,
    and constant macro templates (e.g., rewriting infinite values using custom macro patterns).

    Args:
        original_node: The original LibCST attribute node before children traversal.
        updated_node: The updated LibCST attribute node with child-level updates.

    Returns:
        The rewritten CST node (which could be an Attribute, a Name, or a general CSTNode)
        corresponding to the target framework's equivalent, or the original updated node
        if no rewriting rule was matched.
    """
    name = self._get_qualified_name(original_node)  # type: ignore
    if not name:
      return updated_node

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
      if "macro_template" in target_impl:
        try:
          from ml_switcheroo.core.rewriter.calls.transformers import rewrite_as_macro

          # Constants have no args, pass empty lists
          return rewrite_as_macro(target_impl["macro_template"], [], [])
        except Exception:
          pass

    return updated_node
