"""
Function Invocation Rewriter.

Orchestrates the rewrite of a Call node (`func(...)`) via:
1.  Pre-checks (Unwrapping, Plugins, Lifecycle).
2.  Strategy execution (Infix, Lambda, Standard).
3.  Post-processing (Casting, Adapters).
"""

from typing import Union, TYPE_CHECKING

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

if TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.calls.mixer import ApiStage


class InvocationMixin(NormalizationMixin, TraitsCachingMixin):
  """
  Mixin for transforming :class:`libcst.Call` nodes.

  Compatible with `ApiStage`.
  """

  def leave_Call(
    self: "ApiStage", original: cst.Call, updated: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """
    Visits and rewrites a function call node.
    """
    # 1. Identify Function Name
    func_name = self._get_qualified_name(original.func)

    # 2. RUN PRE-CHECKS (Safe transformations & short-circuits)
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
    lookup = self.context.semantics.get_definition(func_name)
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
    result_node = execute_strategy(self, original, updated, mapping, details, abstract_id)

    # 7. EXECUTE POST-PROCESSING
    result_node = handle_post_processing(self, result_node, mapping, abstract_id)

    log_diff(f"Operation ({abstract_id})", original, result_node)
    return result_node
