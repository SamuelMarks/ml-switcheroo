"""Mixin for ApiTransformer Call rewriting."""

from typing import Union
import libcst as cst


from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import is_builtin, is_super_call, log_diff
from ml_switcheroo.core.tracer import get_tracer


class ApiTransformerCallMixin:
  """Docstring."""

  def leave_Call(  # type: ignore
    self,
    original_node: cst.Call,
    updated_node: cst.Call,
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """Main entry point for function call rewriting."""
    # 1. Identify Function
    func_name = self._get_qualified_name(original_node.func)  # type: ignore

    # 2. Pre-Checks
    # Pass 'self' as rewriter interface (duck typing via properties)
    handled, result_node = handle_pre_checks(self, original_node, updated_node, func_name)
    if handled:
      return result_node

    # 3. Resolve Mapping
    mapping = self._get_mapping(func_name) if func_name else None  # type: ignore

    # Fallback: Implicit Method
    if not mapping:
      guessed_name = resolve_implicit_method(self, original_node, func_name)
      if guessed_name:
        mapping = self._get_mapping(guessed_name, silent=True)  # type: ignore
        if mapping:  # pragma: no cover
          func_name = guessed_name

    if not mapping:
      if is_super_call(original_node):
        return updated_node

      if func_name and not is_builtin(func_name):  # pragma: no cover
        get_tracer().log_inspection(node_str=func_name, outcome="Skipped", detail="No Entry in Semantics Knowledge Base")

      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):  # type: ignore  # pragma: no cover
        self._report_failure(f"API '{func_name}' not found in semantics.")  # type: ignore

      return updated_node

    # 4. Version Check
    min_v = mapping.get("min_version")
    max_v = mapping.get("max_version")
    v_warn = self.check_version_constraints(min_v, max_v)  # type: ignore
    if v_warn:
      self._report_warning(v_warn)  # type: ignore

    lookup = self.semantics.get_definition(func_name)  # type: ignore
    if not lookup:
      return updated_node  # pragma: no cover

    abstract_id, details = lookup

    if details.get("deprecated", False):
      msg = f"Usage of deprecated operation '{abstract_id}'."
      if details.get("replaced_by"):  # pragma: no cover
        msg += f" Consider using '{details['replaced_by']}' instead."
      self._report_warning(msg)  # type: ignore

    # 5. Execute Strategy
    result_node = execute_strategy(self, original_node, updated_node, mapping, details, abstract_id)

    # 6. Post Processing
    result_node = handle_post_processing(self, result_node, mapping, abstract_id)

    log_diff(f"Operation ({abstract_id})", original_node, result_node)
    return result_node
