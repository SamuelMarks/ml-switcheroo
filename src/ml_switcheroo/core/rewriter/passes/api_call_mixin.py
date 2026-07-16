"""Mixin for ApiTransformer Call rewriting."""

from typing import Union  # pragma: no cover
import libcst as cst  # pragma: no cover

# pragma: no cover
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing  # pragma: no cover
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method  # pragma: no cover
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy  # pragma: no cover
from ml_switcheroo.core.rewriter.calls.utils import is_builtin, is_super_call, log_diff  # pragma: no cover
from ml_switcheroo.core.tracer import get_tracer  # pragma: no cover


# pragma: no cover
# pragma: no cover
class ApiTransformerCallMixin:  # pragma: no cover
  """Docstring."""  # pragma: no cover

  # pragma: no cover
  def leave_Call(  # type: ignore  # pragma: no cover
    self,
    original_node: cst.Call,
    updated_node: cst.Call,  # pragma: no cover
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:  # pragma: no cover
    """Main entry point for function call rewriting."""  # pragma: no cover
    # 1. Identify Function  # pragma: no cover
    func_name = self._get_qualified_name(original_node.func)  # type: ignore  # pragma: no cover
    # pragma: no cover
    # 2. Pre-Checks  # pragma: no cover
    # Pass 'self' as rewriter interface (duck typing via properties)  # pragma: no cover
    handled, result_node = handle_pre_checks(self, original_node, updated_node, func_name)  # pragma: no cover
    if handled:  # pragma: no cover
      return result_node  # pragma: no cover
    # pragma: no cover
    # 3. Resolve Mapping  # pragma: no cover
    mapping = self._get_mapping(func_name) if func_name else None  # type: ignore  # pragma: no cover
    # pragma: no cover
    # Fallback: Implicit Method  # pragma: no cover
    if not mapping:  # pragma: no cover
      guessed_name = resolve_implicit_method(self, original_node, func_name)  # pragma: no cover
      if guessed_name:  # pragma: no cover
        mapping = self._get_mapping(guessed_name, silent=True)  # type: ignore  # pragma: no cover
        if mapping:  # pragma: no cover
          func_name = guessed_name  # pragma: no cover
    # pragma: no cover
    if not mapping:  # pragma: no cover
      if is_super_call(original_node):  # pragma: no cover
        return updated_node  # pragma: no cover
      # pragma: no cover
      if func_name and not is_builtin(func_name):  # pragma: no cover
        get_tracer().log_inspection(
          node_str=func_name, outcome="Skipped", detail="No Entry in Semantics Knowledge Base"
        )  # pragma: no cover
      # pragma: no cover
      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):  # type: ignore  # pragma: no cover
        self._report_failure(f"API '{func_name}' not found in semantics.")  # type: ignore  # pragma: no cover
      # pragma: no cover
      return updated_node  # pragma: no cover
    # pragma: no cover
    # 4. Version Check  # pragma: no cover
    min_v = mapping.get("min_version")  # pragma: no cover
    max_v = mapping.get("max_version")  # pragma: no cover
    v_warn = self.check_version_constraints(min_v, max_v)  # type: ignore  # pragma: no cover
    if v_warn:  # pragma: no cover
      self._report_warning(v_warn)  # type: ignore  # pragma: no cover
    # pragma: no cover
    lookup = self.semantics.get_definition(func_name)  # type: ignore  # pragma: no cover
    if not lookup:  # pragma: no cover
      return updated_node  # pragma: no cover
    # pragma: no cover
    abstract_id, details = lookup  # pragma: no cover
    # pragma: no cover
    if details.get("deprecated", False):  # pragma: no cover
      msg = f"Usage of deprecated operation '{abstract_id}'."  # pragma: no cover
      if details.get("replaced_by"):  # pragma: no cover
        msg += f" Consider using '{details['replaced_by']}' instead."  # pragma: no cover
      self._report_warning(msg)  # type: ignore  # pragma: no cover
    # pragma: no cover
    # 5. Execute Strategy  # pragma: no cover
    result_node = execute_strategy(self, original_node, updated_node, mapping, details, abstract_id)  # pragma: no cover
    # pragma: no cover
    # 6. Post Processing  # pragma: no cover
    result_node = handle_post_processing(self, result_node, mapping, abstract_id)  # pragma: no cover
    # pragma: no cover
    log_diff(f"Operation ({abstract_id})", original_node, result_node)  # pragma: no cover
    return result_node  # pragma: no cover
