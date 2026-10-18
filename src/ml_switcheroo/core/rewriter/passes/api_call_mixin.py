"""Mixin for ApiTransformer Call rewriting.

This module provides the `ApiTransformerCallMixin` class, which implements
the logic for intercepting and rewriting function call nodes (`cst.Call`) during the
CST transformation pass. It coordinates pre-checks, mapping resolution, version constraints,
deprecation checks, and strategy execution to convert source-framework APIs into their
target-framework equivalents.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Tuple, Union

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import (
  is_builtin,
  is_neural_on_math_backend,
  is_super_call,
  log_diff,
)
from ml_switcheroo.core.tracer import get_tracer


class ApiTransformerCallMixin:
  """Mixin class that handles rewriting function call nodes during CST traversal.

  This class contains the `leave_Call` visitor method that intercepts call expressions.
  It is designed to be mixed into `ApiTransformer` or mock transformers for testing.
  It relies on duck typing and expects the inheriting class to provide attributes and
  methods such as:

  - `strict_mode` (bool): Whether to fail on unmapped source APIs.
  - `source_fw` (str): Name of the source framework (e.g., 'torch').
  - `target_fw` (str): Name of the target framework (e.g., 'jax').
  - `semantics` (SemanticsManager): Semantic lookup dictionary/object.
  - `_get_qualified_name(node)`: Resolves fully-qualified names of functions.
  - `_get_mapping(name)`: Retrieves API translation details/mappings.
  - `check_version_constraints(min_v, max_v)`: Validates version constraints.
  - `_report_warning(msg)`: Handles issuing warnings.
  - `_report_failure(msg)`: Handles throwing or logging failures.
  """

  # Added for type checking
  context: Any
  config: Any
  semantics: Any
  source_fw: Any
  target_fw: Any
  strict_mode: Any
  source_traits: Any

  if TYPE_CHECKING:
    _is_module_alias: Callable[[cst.CSTNode], bool]

    def _get_target_traits(self) -> Any:
      """Return structural traits for the target framework.

      Returns:
          StructuralTraits: Structural traits for the target framework.
      """
      ...

    def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
      """Return strip and warning lifecycle method sets for the source framework.

      Returns:
          Tuple[Set[str], Set[str]]: Tuple of (strip_methods, warn_methods).
      """
      ...

    def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
      """Resolve a CST expression node to its fully qualified name string.

      Args:
          node: CST expression node to resolve.

      Returns:
          Optional[str]: Resolved qualified name string or None.
      """
      ...

    def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
      """Retrieve the target framework mapping for a qualified API name.

      Args:
          name: Qualified API name string.
          silent: Whether to suppress failure reporting on missing mappings.

      Returns:
          Optional[Dict[str, Any]]: Target mapping dictionary or None.
      """
      ...

    def _report_failure(self, reason: str) -> None:
      """Record a failure message in the current statement error buffer.

      Args:
          reason: Failure description string.
      """
      ...

    def _report_warning(self, msg: str) -> None:
      """Record a warning message in the current statement warning buffer.

      Args:
          msg: Warning description string.
      """
      ...

    def check_version_constraints(self, min_v: Optional[str], max_v: Optional[str]) -> Optional[str]:
      """Validate target framework version constraints for an operation mapping.

      Args:
          min_v: Minimum version string.
          max_v: Maximum version string.

      Returns:
          Optional[str]: Warning message if out of bounds, otherwise None.
      """
      ...

    def _handle_variant_imports(self, variant: Dict[str, Any]) -> None:
      """Inject required imports specified by a target variant mapping.

      Args:
          variant: Target variant dictionary.
      """
      ...

    def _create_name_node(self, api_path: str) -> Union[cst.Name, cst.Attribute]:
      """Construct a CST Name or Attribute chain from a dotted API path.

      Args:
          api_path: Dotted API path string.

      Returns:
          Union[cst.Name, cst.Attribute]: Constructed CST Name or Attribute node.
      """
      ...

    def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
      """Construct a CST Name or Attribute chain from a dotted name string.

      Args:
          name_str: Dotted name string.

      Returns:
          Union[cst.Name, cst.Attribute]: Constructed CST Name or Attribute node.
      """
      ...

    def _mark_stateful(self, var_name: str) -> None:
      """Mark a variable name as stateful in the current scope.

      Args:
          var_name: Variable identifier to mark as stateful.
      """
      ...

    def _is_stateful(self, var_name: str) -> bool:
      """Check whether a variable name is marked stateful in any active scope.

      Args:
          var_name: Variable identifier to check.

      Returns:
          bool: True if the variable is marked stateful.
      """
      ...

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Intercept and rewrites a function call node during CST traversal.

    The rewriting process consists of the following phases:

    1. Resolve the qualified name of the original function callable.
    2. Execute pre-checks and optional functional unwrappings (e.g. `layer.apply`).
    3. Retrieve the target API mapping from the semantics registry, falling back to implicit
       method resolution if needed.
    4. Validate version compatibility constraints and log deprecation warnings.
    5. Execute the designated transformation strategy (e.g. inline lambda, macro templates).
    6. Run post-processing hooks to finalized the rewritten node structure.

    Args:
        original_node (cst.Call): The original LibCST Call node before traversal.
        updated_node (cst.Call): The updated LibCST Call node with children visited.

    Returns:
        cst.BaseExpression: The fully rewritten AST node (e.g., cst.Call, cst.BinaryOperation,
        cst.UnaryOperation) representing the target-framework equivalent,
        or the original/updated node if no rewrite is performed.
    """
    # 1. Identify Function
    func_name = self._get_qualified_name(original_node.func)

    # 2. Pre-Checks
    handled, result_node = handle_pre_checks(self, original_node, updated_node, func_name)
    if handled:
      return result_node

    # 3. Resolve Mapping
    mapping = self._get_mapping(func_name) if func_name else None

    # Fallback: Implicit Method
    if not mapping:
      guessed_name = resolve_implicit_method(self, original_node, func_name)
      if guessed_name:
        mapping = self._get_mapping(guessed_name, silent=True)
        if mapping:
          func_name = guessed_name

    if not mapping:
      if is_super_call(original_node):
        return updated_node

      if func_name and not is_builtin(func_name):
        get_tracer().log_inspection(node_str=func_name, outcome="Skipped", detail="No Entry in Semantics Knowledge Base")

      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):
        abstract_id = (
          self.semantics.resolve_op_id(self.source_fw, func_name) if hasattr(self.semantics, "resolve_op_id") else None
        )

        # If we can't map a neural layer to NumPy/jax, fail with a clear decomposition error
        origins = getattr(self.semantics, "_key_origins", {})

        # In case we can't find abstract_id, try to infer it from func_name
        lookup_id = abstract_id
        if not lookup_id and func_name:
          parts = func_name.split(".")
          lookup_id = parts[-1]

        tier = origins.get(lookup_id)

        if is_neural_on_math_backend(tier, self.target_fw):
          self._report_failure(
            f"Cannot map neural network abstraction '{func_name}' directly to pure math backend '{self.target_fw}'. Use a framework like Flax or Keras."
          )
        else:
          self._report_failure(f"API '{func_name}' not found in semantics.")

      return updated_node

    # 4. Version Check
    min_v = mapping.get("min_version")
    max_v = mapping.get("max_version")
    v_warn = self.check_version_constraints(min_v, max_v)
    if v_warn:
      self._report_warning(v_warn)

    lookup = self.semantics.get_definition(str(func_name or ""))
    if not lookup:
      return updated_node

    abstract_id, details = lookup

    if details.get("deprecated", False):
      msg = f"Usage of deprecated operation '{abstract_id}'."
      if details.get("replaced_by"):
        msg += f" Consider using '{details['replaced_by']}' instead."
      self._report_warning(msg)

    # 5. Execute Strategy
    result_node = execute_strategy(self, original_node, updated_node, mapping, details, abstract_id)

    # 6. Post Processing
    result_node = handle_post_processing(self, result_node, mapping, abstract_id)

    log_diff(f"Operation ({abstract_id})", original_node, result_node)
    return result_node
