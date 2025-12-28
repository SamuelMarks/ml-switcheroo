"""
Function Invocation Rewriter.

This module provides the :class:`InvocationMixin`, a component of the
:class:`PivotRewriter` responsible for transforming function call nodes within
the Abstract Syntax Tree (AST).

It orchestrates three phases of processing:
1.  **Pre-Processing**: Checks for functional unwrapping, plugin claims, and lifecycle methods.
2.  **Strategy Execution**: Switches API calls based on Semantic Definitions, handling
    infix operators, macros, and standard function renaming. This phase includes compatibility checks
    and deprecation warnings.
3.  **Post-Processing**: Applies output adapters, casting, and state threading logic.
"""

from typing import Union

import libcst as cst

from ml_switcheroo.core.rewriter.calls.traits_cache import TraitsCachingMixin
from ml_switcheroo.core.rewriter.normalization import NormalizationMixin
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.core.rewriter.calls.utils import is_super_call, is_builtin, log_diff

# Import sub-logic modules
from ml_switcheroo.core.rewriter.calls.pre import (
  handle_pre_checks,
  resolve_implicit_method,
)
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class InvocationMixin(NormalizationMixin, TraitsCachingMixin):
  """
  Mixin for transforming :class:`libcst.Call` nodes.
  Integrates normalization, trait lookup, and transformation strategies.
  """

  def leave_Call(
    self, original: cst.Call, updated: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """
    Visits and rewrites a function call node.
    Delegates logic to Pre/Strategy/Post handler modules.
    """
    # 1. Identify Function Name
    func_name = self._get_qualified_name(original.func)

    # 2. RUN PRE-CHECKS (Safe transformations & short-circuits)
    # Handles: Functional Unwrap, Plugin Claims, Lifecycle Strip/Warn, Stateful Calls
    handled, result_node = handle_pre_checks(self, original, updated, func_name)
    if handled:
      return result_node

    # 3. Resolve Mapping
    mapping = self._get_mapping(func_name) if func_name else None

    # Fallback: Implicit Method Resolution (e.g. x.float())
    if not mapping:
      guessed_name = resolve_implicit_method(self, original, func_name)
      if guessed_name:
        mapping = self._get_mapping(guessed_name, silent=True)
        if mapping:
          func_name = guessed_name

    # 4. Final Validation
    if not mapping:
      if is_super_call(original):
        return updated

      if func_name and not is_builtin(func_name):
        get_tracer().log_inspection(
          node_str=func_name,
          outcome="Skipped",
          detail="No Entry in Semantics Knowledge Base",
        )

      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):
        self._report_failure(f"API '{func_name}' not found in semantics.")

      return updated

    # 5a. Version Verification
    min_v = mapping.get("min_version")
    max_v = mapping.get("max_version")
    version_warning = self.check_version_constraints(min_v, max_v)

    if version_warning:
      self._report_warning(version_warning)

    # 5b. Retrieve Definition Details
    lookup = self.semantics.get_definition(func_name)
    if not lookup:
      return updated

    abstract_id, details = lookup

    # 5c. Deprecation Check (Feature)
    if details.get("deprecated", False):
      msg = f"Usage of deprecated operation '{abstract_id}'."
      if details.get("replaced_by"):
        msg += f" Consider using '{details['replaced_by']}' instead."
      self._report_warning(msg)

    # 6. EXECUTE REWRITE STRATEGY
    # Handles: Imports, Dispatch, Infix, Lambda, Plugin, Macro, Standard
    result_node = execute_strategy(self, original, updated, mapping, details, abstract_id)

    # 7. EXECUTE POST-PROCESSING
    # Handles: Output Index/Adapter, Casting, State Threading
    result_node = handle_post_processing(self, result_node, mapping, abstract_id)

    log_diff(f"Operation ({abstract_id})", original, result_node)
    return result_node
