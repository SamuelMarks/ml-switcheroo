"""Runtime Annotation & Constructor Concretization Rewriter Pass.

This module implements ``RuntimeAnnotationPass``, ``RuntimeAnnotationTransformer``,
and ``SourceLazyConcretizer``, which enrich transpiled target framework code with
metadata discovered during on-device forward and backward execution.
"""

from __future__ import annotations

import libcst as cst

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.passes.runtime_annotation_helpers import (
  _attach_comment_to_block_header,
  _attach_comment_to_line,
  _concretize_creation_call,
  _concretize_layer_call,
  _concretize_range_iter,
  _concretize_reshape_view_call,
  _concretize_symbolic_ctor_expr,
  _format_telemetry_comment,
  _is_symbolic_dim_expr,
  _out_kwarg_for_in_kwarg,
  _replace_call_leaf,
)
from ml_switcheroo.core.runtime_models import (
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
)
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.schema import StructuralTraits


def _resolve_target_traits(context: RewriterContext) -> StructuralTraits:
  """Resolve ``StructuralTraits`` for ``context.target_framework``.

  Args:
      context: Active ``RewriterContext``.

  Returns:
      Resolved ``StructuralTraits`` for the target framework.
  """
  adapter = get_adapter(context.target_framework)
  if adapter is not None and isinstance(adapter.structural_traits, StructuralTraits):
    return adapter.structural_traits
  from ml_switcheroo.semantics.resolution import query_framework_traits

  fw_traits = query_framework_traits(context.semantics, context.target_framework)
  return fw_traits.traits if fw_traits.traits is not None else StructuralTraits()


class SourceLazyConcretizer(cst.CSTTransformer):
  """Pre-pipeline CST transformer that normalizes ``Lazy*`` constructors and symbolic ``for`` bounds before ``ApiPass``."""

  def __init__(self, runtime_trace: RuntimeTraceReport) -> None:
    """Initialize with the on-device ``RuntimeTraceReport``.

    Args:
        runtime_trace: Populated ``RuntimeTraceReport``.
    """
    self.runtime_trace = runtime_trace
    self._for_cursor: int = 0

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Concretize ``self.<layer> = ...LazyLinear(out)`` assignments before ``ApiPass``.

    Args:
        original_node: Original Assign node.
        updated_node: Updated Assign node.

    Returns:
        Concretized Assign node.
    """
    if not isinstance(updated_node.value, cst.Call):
      return updated_node
    for target in updated_node.targets:
      if (
        isinstance(target.target, cst.Attribute)
        and isinstance(target.target.value, cst.Name)
        and target.target.value.value == "self"
      ):
        res = self.runtime_trace.resolved_constructors.get(target.target.attr.value)
        if res is not None:
          call = updated_node.value
          leaf = (
            call.func.value
            if isinstance(call.func, cst.Name)
            else (call.func.attr.value if isinstance(call.func, cst.Attribute) else "")
          )
          if leaf.startswith("Lazy"):
            new_func = _replace_call_leaf(call.func, leaf.removeprefix("Lazy"))
            in_arg = cst.Arg(value=cst.Integer(str(res.inferred_in_features)))
            return updated_node.with_changes(value=call.with_changes(func=new_func, args=[in_arg, *call.args]))
    return updated_node

  def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
    """Concretize symbolic ``range(expr)`` bounds on ``for`` loops using ``loop_trip_counts``.

    Args:
        original_node: Original For node.
        updated_node: Updated For node.

    Returns:
        For node with concretized ``range(<trip_count>)`` iterator.
    """
    counts = list(self.runtime_trace.loop_trip_counts.values())
    if self._for_cursor < len(counts):
      trips = counts[self._for_cursor]
      self._for_cursor += 1
      new_iter = _concretize_range_iter(updated_node.iter, trips)
      if new_iter is not updated_node.iter:
        return updated_node.with_changes(iter=new_iter)
    return updated_node


class RuntimeAnnotationTransformer(cst.CSTTransformer):
  """CST transformer that concretizes constructors and injects inline telemetry & assertions."""

  def __init__(
    self,
    context: RewriterContext,
    target_traits: StructuralTraits,
  ) -> None:
    """Initialize the runtime annotation transformer.

    Args:
        context: Active ``RewriterContext`` containing ``runtime_trace`` and ``config``.
        target_traits: Resolved ``StructuralTraits`` for the target framework.
    """
    self.context = context
    self.target_traits = target_traits
    self.assertions_injected: bool = False
    self._in_forward: bool = False
    self._telemetry_cursor: int = 0
    self._branch_cursor: int = 0
    self._loop_cursor: int = 0
    self._if_depth: int = 0

  def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
    """Track entry into forward inference functions/methods.

    Args:
        node: FunctionDef node being visited.

    Returns:
        True to traverse the function body.
    """
    if node.name.value not in ("__init__", "setup"):
      self._in_forward = True
    return True

  def leave_FunctionDef(
    self,
    original_node: cst.FunctionDef,
    updated_node: cst.FunctionDef,
  ) -> cst.FunctionDef:
    """Reset forward tracking state when leaving a function definition.

    Args:
        original_node: Original FunctionDef node.
        updated_node: Updated FunctionDef node.

    Returns:
        Updated FunctionDef node.
    """
    if original_node.name.value not in ("__init__", "setup"):
      self._in_forward = False
    return updated_node

  def _try_concretize_ctor_line(
    self,
    updated_node: cst.SimpleStatementLine,
    small: cst.BaseSmallStatement,
    trace: RuntimeTraceReport,
  ) -> cst.SimpleStatementLine | None:
    """Concretize ``self.<layer> = <Call>`` or ``self.create_child('<layer>', <Call>)`` lines.

    Args:
        updated_node: Statement line being visited.
        small: Single inner small statement.
        trace: Active ``RuntimeTraceReport``.

    Returns:
        Concretized ``SimpleStatementLine`` if matched, otherwise ``None``.
    """
    if isinstance(small, cst.Assign) and isinstance(small.value, cst.Call) and small.targets:
      tgt = small.targets[0].target
      if isinstance(tgt, cst.Attribute) and isinstance(tgt.value, cst.Name) and tgt.value.value == "self":
        res = trace.resolved_constructors.get(tgt.attr.value)
        if res is not None:
          new_call = _concretize_layer_call(small.value, res, self.target_traits)
          out_line = updated_node.with_changes(body=[small.with_changes(value=new_call)])
          if self.context.config.inject_runtime_annotations:
            out_line = _attach_comment_to_line(
              out_line, f"# [on-device ctor] resolved in_features={res.inferred_in_features}"
            )
          return out_line

    if (
      isinstance(small, cst.Expr)
      and isinstance(small.value, cst.Call)
      and isinstance(small.value.func, cst.Attribute)
      and isinstance(small.value.func.value, cst.Name)
      and small.value.func.value.value == "self"
      and small.value.func.attr.value == "create_child"
      and len(small.value.args) >= 2
      and isinstance(small.value.args[0].value, cst.SimpleString)
      and isinstance(small.value.args[1].value, cst.Call)
    ):
      layer_name = str(small.value.args[0].value.evaluated_value)
      res = trace.resolved_constructors.get(layer_name)
      if res is not None:
        inner_call = _concretize_layer_call(small.value.args[1].value, res, self.target_traits)
        new_args = [small.value.args[0], small.value.args[1].with_changes(value=inner_call), *small.value.args[2:]]
        out_line = updated_node.with_changes(body=[small.with_changes(value=small.value.with_changes(args=new_args))])
        if self.context.config.inject_runtime_annotations:
          out_line = _attach_comment_to_line(
            out_line, f"# [on-device ctor] resolved in_features={res.inferred_in_features}"
          )
        return out_line

    return None

  @staticmethod
  def _concretize_forward_expr(expr: cst.BaseExpression, tel: NodeRuntimeTelemetry) -> cst.BaseExpression:
    """Apply reshape/view, creation-call, and inline-constructor dimension concretization to ``expr``.

    Args:
        expr: Forward assignment or return value expression.
        tel: Matched ``NodeRuntimeTelemetry`` record.

    Returns:
        Concretized ``cst.BaseExpression``.
    """
    val = _concretize_reshape_view_call(expr, tel.output_shape)
    val = _concretize_creation_call(val, tel.output_shape)
    return _concretize_symbolic_ctor_expr(val, tel)

  def _annotate_forward_statement(
    self,
    updated_node: cst.SimpleStatementLine,
    small: cst.BaseSmallStatement,
    trace: RuntimeTraceReport,
  ) -> cst.SimpleStatementLine | cst.FlattenSentinel[cst.BaseStatement]:
    """Attach forward/backward telemetry comments and shape assertions to a forward statement.

    Args:
        updated_node: Statement line being visited.
        small: Single inner small statement.
        trace: Active ``RuntimeTraceReport``.

    Returns:
        Annotated ``SimpleStatementLine`` or ``FlattenSentinel``.
    """
    seed = self.context.config.random_seed
    inject_ann = self.context.config.inject_runtime_annotations

    if isinstance(small, cst.Assign) and small.targets:
      var_name = self._extract_var_name(small.targets[0].target)
      tel = self._match_telemetry(var_name, trace)
      if tel is not None:
        new_val = self._concretize_forward_expr(small.value, tel)
        if new_val is not small.value:
          updated_node = updated_node.with_changes(body=[small.with_changes(value=new_val)])
        if inject_ann:
          updated_node = _attach_comment_to_line(updated_node, _format_telemetry_comment(tel, seed))
        if (
          self.context.config.inject_shape_assertions
          and self.target_traits.shape_assertion_template
          and var_name != "tmp"
        ):
          stmt_code = self.target_traits.shape_assertion_template.format(var=var_name, shape=tuple(tel.output_shape))
          self.assertions_injected = True
          return cst.FlattenSentinel([updated_node, cst.parse_statement(stmt_code)])
    elif isinstance(small, cst.AugAssign):
      tel = self._match_telemetry(self._extract_var_name(small.target), trace)
      if tel is not None and inject_ann:
        updated_node = _attach_comment_to_line(updated_node, _format_telemetry_comment(tel, seed))
    elif isinstance(small, cst.Return) and small.value is not None:
      tel = self._match_telemetry("return", trace)
      if tel is not None:
        new_ret_val = self._concretize_forward_expr(small.value, tel)
        updated_node = updated_node.with_changes(body=[small.with_changes(value=new_ret_val)])
        if inject_ann:
          updated_node = _attach_comment_to_line(updated_node, _format_telemetry_comment(tel, seed))

    return updated_node

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> cst.SimpleStatementLine | cst.FlattenSentinel[cst.BaseStatement]:
    """Concretize layer init lines and annotate forward assignment/return lines.

    Args:
        original_node: Original SimpleStatementLine node.
        updated_node: Updated SimpleStatementLine node.

    Returns:
        Annotated ``SimpleStatementLine`` or ``FlattenSentinel`` including shape assertions.
    """
    trace = self.context.runtime_trace
    if trace is None or len(updated_node.body) != 1:
      return updated_node

    small = updated_node.body[0]
    ctor_result = self._try_concretize_ctor_line(updated_node, small, trace)
    if ctor_result is not None:
      return ctor_result

    if self._in_forward:
      return self._annotate_forward_statement(updated_node, small, trace)
    return updated_node

  def visit_If(self, node: cst.If) -> bool:
    """Track nesting depth of ``if`` / ``elif`` statements.

    Args:
        node: If node being visited.

    Returns:
        True to traverse child nodes.
    """
    self._if_depth += 1
    return True

  def leave_If(
    self,
    original_node: cst.If,
    updated_node: cst.If,
  ) -> cst.If | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
    """Attach ``# [on-device branch] taken=...`` comments or specialize taken branches when configured.

    Args:
        original_node: Original If node.
        updated_node: Updated If node.

    Returns:
        Annotated ``If`` node, specialized ``FlattenSentinel``, or ``RemovalSentinel``.
    """
    self._if_depth -= 1
    trace = self.context.runtime_trace
    if not self._in_forward or trace is None:
      return updated_node

    if self.context.config.specialize_dynamic_branches:
      if self._if_depth == 0 and self._branch_cursor < len(trace.resolved_branches):
        return self._specialize_if_chain(updated_node, trace)
      return updated_node

    if self.context.config.inject_runtime_annotations and self._branch_cursor < len(trace.resolved_branches):
      rec = trace.resolved_branches[self._branch_cursor]
      self._branch_cursor += 1
      taken_title = "True" if rec.taken_branch == "true" else "False"
      new_body = _attach_comment_to_block_header(updated_node.body, f"# [on-device branch] taken={taken_title}")
      return updated_node.with_changes(body=new_body)
    return updated_node

  def _specialize_if_chain(
    self,
    node: cst.If,
    trace: RuntimeTraceReport,
  ) -> cst.If | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
    """Specialize a top-level ``if`` / ``elif`` / ``else`` statement to the branch taken on-device.

    Args:
        node: Top-level ``cst.If`` node.
        trace: Active ``RuntimeTraceReport``.

    Returns:
        Specialized statement sequence or ``RemovalSentinel``.
    """
    rec = trace.resolved_branches[self._branch_cursor]
    self._branch_cursor += 1

    header_lines: list[cst.EmptyLine] = [
      *node.leading_lines,
      cst.EmptyLine(
        comment=cst.Comment(
          f"# [on-device branch specialized: cond={rec.condition_src} evaluated to {rec.taken_branch}]"
        ),
        newline=cst.Newline(),
      ),
    ]
    if isinstance(node.body, cst.IndentedBlock) and node.body.header.comment is not None:
      header_lines.append(cst.EmptyLine(comment=node.body.header.comment, newline=cst.Newline()))

    taken_stmts: list[cst.BaseStatement] = []
    untaken_comment_lines: list[cst.EmptyLine] = []
    body_stmts = list(node.body.body) if isinstance(node.body, (cst.IndentedBlock, cst.SimpleStatementSuite)) else []

    if rec.taken_branch == "true":
      taken_stmts = [s for s in body_stmts if isinstance(s, cst.BaseStatement)]
      if node.orelse is not None:
        untaken_src = cst.Module([]).code_for_node(node.orelse).strip()
        untaken_comment_lines.extend(
          cst.EmptyLine(comment=cst.Comment(f"# [untaken] {raw_line.strip()}"), newline=cst.Newline())
          for raw_line in untaken_src.splitlines()
        )
    else:
      for s in body_stmts:
        raw_if_body = cst.Module([]).code_for_node(s).strip()
        untaken_comment_lines.extend(
          cst.EmptyLine(comment=cst.Comment(f"# [untaken] {raw_line.strip()}"), newline=cst.Newline())
          for raw_line in raw_if_body.splitlines()
        )
      if isinstance(node.orelse, cst.Else):
        if isinstance(node.orelse.body, cst.IndentedBlock) and node.orelse.body.header.comment is not None:
          header_lines.append(cst.EmptyLine(comment=node.orelse.body.header.comment, newline=cst.Newline()))
        else_stmts = (
          list(node.orelse.body.body)
          if isinstance(node.orelse.body, (cst.IndentedBlock, cst.SimpleStatementSuite))
          else []
        )
        taken_stmts = [s for s in else_stmts if isinstance(s, cst.BaseStatement)]
      elif isinstance(node.orelse, cst.If):
        nested = self._specialize_if_chain(node.orelse, trace)
        taken_stmts = (
          list(nested.nodes)
          if isinstance(nested, cst.FlattenSentinel)
          else ([nested] if isinstance(nested, cst.BaseStatement) else [])
        )

    all_leading = [*header_lines, *untaken_comment_lines]
    if not taken_stmts:
      return cst.FlattenSentinel([cst.SimpleStatementLine(body=[cst.Pass()], leading_lines=all_leading)])

    first_stmt = taken_stmts[0]
    existing_leading = (
      list(first_stmt.leading_lines)
      if isinstance(first_stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement))
      else []
    )
    taken_stmts[0] = first_stmt.with_changes(leading_lines=[*all_leading, *existing_leading])
    return cst.FlattenSentinel(taken_stmts)

  def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
    """Concretize symbolic ``range(expr)`` bounds on remaining ``for`` loops and attach trip-count comments.

    Args:
        original_node: Original For node.
        updated_node: Updated For node.

    Returns:
        Concretized and annotated For node.
    """
    trace = self.context.runtime_trace
    if self._in_forward and trace is not None:
      counts = list(trace.loop_trip_counts.values())
      if self._loop_cursor < len(counts):
        trips = counts[self._loop_cursor]
        self._loop_cursor += 1
        updated_node = updated_node.with_changes(iter=_concretize_range_iter(updated_node.iter, trips))
        if self.context.config.inject_runtime_annotations:
          new_body = _attach_comment_to_block_header(updated_node.body, f"# [on-device loop] iterations={trips}")
          updated_node = updated_node.with_changes(body=new_body)
    return updated_node

  def leave_While(self, original_node: cst.While, updated_node: cst.While) -> cst.While:
    """Attach ``# [on-device loop] iterations=...`` comments to while loops.

    Args:
        original_node: Original While node.
        updated_node: Updated While node.

    Returns:
        Annotated While node.
    """
    trace = self.context.runtime_trace
    if self._in_forward and trace is not None and self.context.config.inject_runtime_annotations:
      counts = list(trace.loop_trip_counts.values())
      if self._loop_cursor < len(counts):
        trips = counts[self._loop_cursor]
        self._loop_cursor += 1
        new_body = _attach_comment_to_block_header(updated_node.body, f"# [on-device loop] iterations={trips}")
        return updated_node.with_changes(body=new_body)
    return updated_node

  def _match_telemetry(
    self,
    var_name: str,
    trace: RuntimeTraceReport,
  ) -> NodeRuntimeTelemetry | None:
    """Look up the next matching ``NodeRuntimeTelemetry`` entry for ``var_name``.

    Args:
        var_name: Target variable name or ``'return'``.
        trace: Active ``RuntimeTraceReport``.

    Returns:
        Matching ``NodeRuntimeTelemetry`` or None.
    """
    entries = list(trace.node_telemetry.values())
    for idx in range(self._telemetry_cursor, len(entries)):
      if entries[idx].var_name == var_name:
        self._telemetry_cursor = idx + 1
        return entries[idx]
    for entry in reversed(entries):
      if entry.var_name == var_name:
        return entry
    return None

  @staticmethod
  def _extract_var_name(target: cst.BaseExpression) -> str:
    """Extract variable name from an assignment target node.

    Args:
        target: Assignment target expression.

    Returns:
        Variable identifier string.
    """
    if isinstance(target, cst.Name):
      return target.value
    if isinstance(target, cst.Attribute):
      return target.attr.value
    return "tmp"


class RuntimeAnnotationPass(RewriterPass):
  """Rewriter pass that applies on-device constructor concretization, annotations, and assertions."""

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Transform ``module`` using ``context.runtime_trace``.

    Args:
        module: Input LibCST module.
        context: Shared ``RewriterContext``.

    Returns:
        Transformed LibCST module.
    """
    if context.runtime_trace is None or not context.runtime_trace.executed:
      return module

    target_traits = _resolve_target_traits(context)
    transformer = RuntimeAnnotationTransformer(context=context, target_traits=target_traits)
    updated_module = module.visit(transformer)

    if transformer.assertions_injected and target_traits.shape_assertion_import:
      import_code = target_traits.shape_assertion_import
      context._default_preamble_injector(import_code)
      if import_code not in updated_module.code:
        import_stmt = cst.parse_statement(import_code)
        updated_module = updated_module.with_changes(body=[import_stmt, *updated_module.body])

    if context.config.emit_backward_code and context.runtime_trace.synthesized_backward_code:
      bwd_mod = cst.parse_module(context.runtime_trace.synthesized_backward_code)
      updated_module = updated_module.with_changes(body=[*updated_module.body, *bwd_mod.body])

    return updated_module


__all__ = [
  "RuntimeAnnotationPass",
  "RuntimeAnnotationTransformer",
  "SourceLazyConcretizer",
  "_attach_comment_to_block_header",
  "_attach_comment_to_line",
  "_concretize_creation_call",
  "_concretize_layer_call",
  "_concretize_range_iter",
  "_concretize_reshape_view_call",
  "_concretize_symbolic_ctor_expr",
  "_format_telemetry_comment",
  "_is_symbolic_dim_expr",
  "_out_kwarg_for_in_kwarg",
  "_replace_call_leaf",
  "_resolve_target_traits",
]
