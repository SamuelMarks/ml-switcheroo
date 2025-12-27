"""
Function Structure Rewriting Logic.

Handles transformations relative to function definitions, specifically:
1.  **Logic 5: Method Renaming**: Mapping ``forward`` <-> ``__call__`` <-> ``call`` using Configuration Traits.
    *Decoupling Update*: Uses ``known_inference_methods`` from traits to detect candidate methods.
2.  Signature Modification: Injecting hooks or state arguments (Logic 2).
3.  Body Injection: Preamble handling (super init, rng splitting).
4.  Docstring Updating.

This class acts as an aggregator mixin, pulling specialized logic from:
- :class:`FuncSignatureMixin`
- :class:`FuncBodyMixin`
- :class:`FuncDocstringMixin`
"""

from typing import Optional, Set
import libcst as cst

from ml_switcheroo.core.rewriter.base import BaseRewriter
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.core.rewriter.func_signature import FuncSignatureMixin
from ml_switcheroo.core.rewriter.func_body import FuncBodyMixin
from ml_switcheroo.core.rewriter.func_docstring import FuncDocstringMixin


class FuncStructureMixin(FuncSignatureMixin, FuncBodyMixin, FuncDocstringMixin, BaseRewriter):
  """
  Mixin for transforming FunctionDef nodes via Traits.

  Orchestrates the granular rewriting components (Signature, Body, Docstring)
  based on the analysis of the Target Framework's traits.
  """

  def _get_traits(self) -> StructuralTraits:
    """
    Lazily retrieves the Structural Traits for the configured Target Framework.

    Queries the SemanticsManager configuration. If missing or invalid,
    returns a default empty Traits object.

    Returns:
        StructuralTraits: Configuration object governing structural rewrites.
    """
    try:
      # Access traits for the TARGET framework
      if hasattr(self.semantics, "get_framework_config"):
        config_dict = self.semantics.get_framework_config(self.target_fw)
        if config_dict and "traits" in config_dict:
          return StructuralTraits.model_validate(config_dict["traits"])
    except Exception:
      pass
    return StructuralTraits()

  def _get_source_inference_methods(self) -> Set[str]:
    """
    Lazily loads the set of known inference methods for the Source Framework.

    This allows flexible detection (e.g. 'predict' vs 'forward') without hardcoding
    specific method names, enabling support for custom frameworks defined purely in JSON.

    Returns:
        Set[str]: A set of method names (e.g. {'forward', '__call__'}).
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

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """
    Enters a function definition scope.

    Initializes a ``SignatureContext`` to track argument names and injection requirements.
    Determines if the function is a method of a recognized Module Class (tracked by ``ClassStructureMixin``).

    Args:
        node (cst.FunctionDef): The function definition node.

    Returns:
        Optional[bool]: True to continue visiting children.
    """
    self._enter_scope()

    existing_args = set()
    for param in node.params.params:
      if isinstance(param.name, cst.Name):
        existing_args.add(param.name.value)

    is_init = node.name.value == "__init__"

    self._signature_stack.append(
      SignatureContext(
        existing_args=existing_args,
        is_init=is_init,
        is_module_method=self._in_module_class,
      )
    )
    return True

  def leave_FunctionDef(self, _original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
    """
    Exits a function definition scope and applies structural transformations.

    Applies logic based on Target Traits:
    1.  **Method Renaming**: Renames inference methods (e.g. ``forward`` -> ``__call__``).
    2.  **Constructor Ops**: Renames ``__init__`` if required (e.g. ``setup`` for Pax).
    3.  **State Injection**: Injects magic arguments (e.g. ``rngs``) if targeting Flax/JAX.
    4.  **Arg Stripping**: Removes magic arguments (e.g. ``rngs``) if targeting Torch.
    5.  **Super Init**: Injects or removes ``super().__init__()`` calls.
    6.  **Preamble**: Injects code requested by inner plugins (e.g. RNG splitting).
    7.  **Docstrings**: Updates docstrings to document injected arguments.

    Args:
        _original_node: The original CST node (unused).
        updated_node: The CST node with children already visited/transformed.

    Returns:
        cst.CSTNode: The transformed function definition.
    """
    self._exit_scope()

    if not self._signature_stack:
      return updated_node

    sig_ctx = self._signature_stack.pop()
    target_traits = self._get_traits()

    # Handle Method Renaming (Inference & Init)
    if sig_ctx.is_module_method:
      curr_name = updated_node.name.value
      target_name = target_traits.forward_method

      # DECOUPLING: Use traits from Source Framework (or shared config) to detect candidates
      known_methods = self._get_source_inference_methods()

      # If current method is a known inference method (e.g. 'forward'), rename it to target (e.g. '__call__')
      if target_name and curr_name in known_methods and curr_name != target_name:
        updated_node = updated_node.with_changes(name=cst.Name(target_name))

      if sig_ctx.is_init and target_traits.init_method_name and target_traits.init_method_name != "__init__":
        updated_node = updated_node.with_changes(name=cst.Name(target_traits.init_method_name))

    # Handle Constructor Attributes
    if sig_ctx.is_init and sig_ctx.is_module_method:
      # A. Inject Magic Arguments (Target Requirement)
      for arg_name, arg_type in target_traits.inject_magic_args:
        if arg_name not in sig_ctx.existing_args:
          found = any(n == arg_name for n, _ in sig_ctx.injected_args)
          if not found:
            sig_ctx.injected_args.append((arg_name, arg_type))

      # B. Strip Magic Arguments (Source Artifact Removal)
      # Decoupling Feature: Auto-Strip Magic Args
      args_to_strip = set(target_traits.strip_magic_args)
      if target_traits.auto_strip_magic_args and hasattr(self.semantics, "known_magic_args"):
        # Add all known magic args
        args_to_strip.update(self.semantics.known_magic_args)
        # But DO NOT strip args that we just injected or are native to this framework
        native_args = {a[0] for a in target_traits.inject_magic_args}
        args_to_strip = args_to_strip - native_args

      for arg_name in args_to_strip:
        updated_node = self._strip_argument_from_signature(updated_node, arg_name)

      # C. Super Init Logic
      if target_traits.requires_super_init:
        updated_node = self._ensure_super_init(updated_node)
      else:
        # Logic 3: Strip super() if not required
        updated_node = self._strip_super_init(updated_node)

    # Apply accumulated injections (from Traits or Plugins)
    for name, annotation in sig_ctx.injected_args:
      updated_node = self._inject_argument_to_signature(updated_node, name, annotation)

    if sig_ctx.preamble_stmts:
      updated_node = self._apply_preamble(updated_node, sig_ctx.preamble_stmts)

    if sig_ctx.injected_args:
      updated_node = self._update_docstring(updated_node, sig_ctx.injected_args)

    return updated_node
