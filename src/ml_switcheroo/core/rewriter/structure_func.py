"""
Function Structure Rewriting Logic.

Handles transformations relative to function definitions, specifically:
1.  Method Renaming (forward <-> call).
2.  Signature Modification.
3.  Body Injection.
4.  Docstring Updating.
"""

from typing import Optional, Set, TYPE_CHECKING
import libcst as cst

from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.core.rewriter.func_signature import FuncSignatureMixin
from ml_switcheroo.core.rewriter.func_body import FuncBodyMixin
from ml_switcheroo.core.rewriter.func_docstring import FuncDocstringMixin

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.structure import StructureStage


class FuncStructureMixin(FuncSignatureMixin, FuncBodyMixin, FuncDocstringMixin):
  """
  Mixin for transforming FunctionDef nodes via Traits.

  Requires host to provide `context` and `semantics`.
  """

  def _get_traits(self: "StructureStage") -> StructuralTraits:
    """
    Lazily retrieves the Structural Traits for the configured Target Framework.
    """
    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "traits" in config_dict:
          return StructuralTraits.model_validate(config_dict["traits"])
    except Exception:
      pass
    return StructuralTraits()

  def _get_source_inference_methods(self: "StructureStage") -> Set[str]:
    """
    Lazily loads the set of known inference methods for the Source Framework.
    """
    default_methods = {"forward", "__call__", "call"}
    try:
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.source_fw)
        if config_dict and "traits" in config_dict:
          traits = StructuralTraits.model_validate(config_dict["traits"])
          if traits.known_inference_methods:
            return traits.known_inference_methods
    except Exception:
      pass
    return default_methods

  def visit_FunctionDef(self: "StructureStage", node: cst.FunctionDef) -> Optional[bool]:
    """
    Enters a function definition scope.
    """
    self._enter_scope()

    existing_args = set()
    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        existing_args.add(param.name.value)

    is_init = node.name.value == "__init__"

    self.context.signature_stack.append(
      SignatureContext(
        existing_args=existing_args,
        is_init=is_init,
        is_module_method=self.context.in_module_class,
      )
    )
    return True

  def leave_FunctionDef(
    self: "StructureStage", _original_node: cst.FunctionDef, updated_node: cst.FunctionDef
  ) -> cst.CSTNode:
    """
    Exits a function definition scope and applies structural transformations.
    """
    self._exit_scope()

    if not self.context.signature_stack:
      return updated_node

    sig_ctx = self.context.signature_stack.pop()
    target_traits = self._get_traits()

    # Handle Method Renaming
    if sig_ctx.is_module_method:
      curr_name = updated_node.name.value
      target_name = target_traits.forward_method
      known_methods = self._get_source_inference_methods()

      if target_name and curr_name in known_methods and curr_name != target_name:
        updated_node = updated_node.with_changes(name=cst.Name(target_name))

      if sig_ctx.is_init and target_traits.init_method_name and target_traits.init_method_name != "__init__":
        updated_node = updated_node.with_changes(name=cst.Name(target_traits.init_method_name))

    # Handle Constructor Attributes
    if sig_ctx.is_init and sig_ctx.is_module_method:
      # A. Inject Magic Arguments
      for arg_name, arg_type in target_traits.inject_magic_args:
        if arg_name not in sig_ctx.existing_args:
          # Avoid duplicates
          found = any(n == arg_name for n, _ in sig_ctx.injected_args)
          if not found:
            sig_ctx.injected_args.append((arg_name, arg_type))

      # B. Strip Magic Arguments
      args_to_strip = set(target_traits.strip_magic_args)
      if target_traits.auto_strip_magic_args and hasattr(self.semantics, "known_magic_args"):
        args_to_strip.update(self.semantics.known_magic_args)
        native = {a[0] for a in target_traits.inject_magic_args}
        args_to_strip = args_to_strip - native

      for arg_name in args_to_strip:
        updated_node = self._strip_argument_from_signature(updated_node, arg_name)

      # C. Super Init Logic
      if target_traits.requires_super_init:
        updated_node = self._ensure_super_init(updated_node)
      else:
        updated_node = self._strip_super_init(updated_node)

    # Apply injections
    for name, annotation in sig_ctx.injected_args:
      updated_node = self._inject_argument_to_signature(updated_node, name, annotation)

    if sig_ctx.preamble_stmts:
      updated_node = self._apply_preamble(updated_node, sig_ctx.preamble_stmts)

    if sig_ctx.injected_args:
      updated_node = self._update_docstring(updated_node, sig_ctx.injected_args)

    return updated_node
