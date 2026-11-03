"""Runtime Annotation & Constructor Concretization Rewriter Pass.

This module implements `RuntimeAnnotationPass`, `RuntimeAnnotationTransformer`,
and `SourceLazyConcretizer`, which enrich transpiled target framework code with
metadata discovered during on-device forward and backward execution:
1. Concretizing lazy layer constructors (`LazyLinear(10)` -> `Linear(in_features=16, out_features=10)`).
2. Attaching inline forward shape, dynamic masking, branch, loop, and backward
   gradient L2 norm comments (`# [on-device fwd] ... | [bwd] ...`).
3. Emitting executable runtime shape assertions (`chex.assert_shape` or `assert tuple(x.shape) == ...`).
"""

from __future__ import annotations


import libcst as cst

from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.runtime_models import (
  ConstructorDimResolution,
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
)
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.schema import StructuralTraits


def _resolve_target_traits(context: RewriterContext) -> StructuralTraits:
  """Resolve `StructuralTraits` for `context.target_framework`.

  Args:
      context: Active `RewriterContext`.

  Returns:
      Resolved `StructuralTraits` for the target framework.
  """
  adapter = get_adapter(context.target_framework)
  if adapter is not None and isinstance(adapter.structural_traits, StructuralTraits):
    return adapter.structural_traits
  fw_cfg = context.semantics.get_framework_config(context.target_framework)
  raw_traits = fw_cfg.get("traits")
  if isinstance(raw_traits, dict):
    return StructuralTraits.model_validate(raw_traits)
  return StructuralTraits()


def _attach_comment_to_line(node: cst.SimpleStatementLine, comment_text: str) -> cst.SimpleStatementLine:
  """Attach or append a trailing comment to a `SimpleStatementLine`.

  Args:
      node: Statement line node to annotate.
      comment_text: Comment string starting with `# `.

  Returns:
      Updated `SimpleStatementLine` with trailing comment.
  """
  existing = node.trailing_whitespace.comment
  if existing is not None:
    combined = f"{existing.value}  {comment_text}"
  else:
    combined = comment_text
  return node.with_changes(
    trailing_whitespace=cst.TrailingWhitespace(
      whitespace=cst.SimpleWhitespace("  "),
      comment=cst.Comment(combined),
      newline=node.trailing_whitespace.newline,
    )
  )


def _attach_comment_to_block_header(block: cst.BaseSuite, comment_text: str) -> cst.BaseSuite:
  """Attach a trailing comment to the header of an `IndentedBlock` (for `if` / `while`).

  Args:
      block: Suite node (`IndentedBlock`) of the `If` or `While` statement.
      comment_text: Comment string starting with `# `.

  Returns:
      Updated suite node with header comment.
  """
  if isinstance(block, cst.IndentedBlock):
    return block.with_changes(
      header=cst.TrailingWhitespace(
        whitespace=cst.SimpleWhitespace("  "),
        comment=cst.Comment(comment_text),
        newline=block.header.newline,
      )
    )
  return block


def _format_telemetry_comment(tel: NodeRuntimeTelemetry, seed: int) -> str:
  """Format the inline forward and backward telemetry comment for a statement.

  Args:
      tel: `NodeRuntimeTelemetry` recorded for the statement.
      seed: Deterministic seed used during on-device execution.

  Returns:
      Formatted comment string beginning with `# `.
  """
  fwd_shape = tuple(tel.output_shape)
  parts = [f"# [on-device fwd] shape={fwd_shape} dtype={tel.dtype}"]
  if tel.grad_shape is not None and tel.grad_norm is not None:
    parts.append(f"[bwd] grad={tuple(tel.grad_shape)} norm={tel.grad_norm:.3e}")
  if tel.is_dynamic:
    parts.append(f"[dynamic-shape: resolved on-device (seed={seed})] shape={fwd_shape}")
  return " | ".join(parts)


def _replace_call_leaf(func: cst.BaseExpression, new_leaf: str, prefix: str | None = None) -> cst.BaseExpression:
  """Replace the leaf identifier of a Call's `func` expression, optionally updating its prefix.

  Args:
      func: Original callable expression (`Name` or `Attribute`).
      new_leaf: New leaf class/function name (e.g. `'Linear'`).
      prefix: Optional namespace prefix (e.g. `'nnx'`).

  Returns:
      Updated callable expression.
  """
  if prefix:
    expr = cst.parse_expression(f"{prefix}.{new_leaf}")
    return expr
  if isinstance(func, cst.Name):
    return cst.Name(new_leaf)
  if isinstance(func, cst.Attribute):
    return func.with_changes(attr=cst.Name(new_leaf))
  return func


def _out_kwarg_for_in_kwarg(in_kw: str) -> str:
  """Map a target input-dimension keyword argument name to its corresponding output keyword name.

  Args:
      in_kw: Input dimension keyword argument name (e.g. `'in_features'`, `'input_dims'`, `'in_channels'`).

  Returns:
      Corresponding output dimension keyword argument name.
  """
  if in_kw == "input_dims":
    return "output_dims"
  if in_kw == "in_channels":
    return "out_channels"
  return "out_features"


def _is_symbolic_dim_expr(expr: cst.BaseExpression) -> bool:
  """Return True if ``expr`` is a dynamic/symbolic dimension expression (``-1``, ``x.shape[i]``, ``x.size(i)``, or Name).

  Args:
      expr: LibCST expression node to inspect.

  Returns:
      True if ``expr`` represents a non-literal or negative-one symbolic dimension.
  """
  if (
    isinstance(expr, cst.UnaryOperation)
    and isinstance(expr.operator, cst.Minus)
    and isinstance(expr.expression, cst.Integer)
  ):
    return True
  if isinstance(expr, cst.Subscript) and isinstance(expr.value, cst.Attribute) and expr.value.attr.value == "shape":
    return True
  if isinstance(expr, cst.Call) and isinstance(expr.func, cst.Attribute) and expr.func.attr.value == "size":
    return True
  if isinstance(expr, cst.Name):
    return True
  return False


def _concretize_reshape_view_call(
  expr: cst.BaseExpression,
  output_shape: list[int] | tuple[int, ...],
) -> cst.BaseExpression:
  """Concretize symbolic dimension arguments (``-1``, ``x.shape[i]``, ``x.size(i)``, ``SymInt``) in ``reshape`` / ``view`` calls.

  Preserves surrounding CST argument formatting and trivia.

  Args:
      expr: Call expression node (e.g. ``x.view(x.size(0), -1)`` or ``jnp.reshape(x, (B, -1))``).
      output_shape: Concrete runtime output shape for the statement.

  Returns:
      Updated ``cst.BaseExpression`` with symbolic dimension arguments replaced by concrete integer literals.
  """
  if not isinstance(expr, cst.Call) or not output_shape:
    return expr

  leaf = ""
  is_method_on_tensor = False
  if isinstance(expr.func, cst.Name):
    leaf = expr.func.value
  elif isinstance(expr.func, cst.Attribute):
    leaf = expr.func.attr.value
    # Distinguish tensor method `x.reshape(d0, d1)` / `x.view(d0, d1)` from module function `jnp.reshape(x, ...)`
    if leaf == "view" or (
      leaf == "reshape"
      and len(expr.args) == len(output_shape)
      and expr.args
      and not isinstance(expr.args[0].value, (cst.Tuple, cst.List))
      and (isinstance(expr.args[0].value, cst.Integer) or _is_symbolic_dim_expr(expr.args[0].value))
    ):
      is_method_on_tensor = True

  if leaf not in ("reshape", "view"):
    return expr

  # Case A: Tuple/List shape argument, e.g. `jnp.reshape(x, (x.shape[0], -1))` or `x.reshape((-1, 10))`
  for arg_idx, arg in enumerate(expr.args):
    if isinstance(arg.value, (cst.Tuple, cst.List)):
      if len(arg.value.elements) == len(output_shape) and any(
        _is_symbolic_dim_expr(el.value) for el in arg.value.elements
      ):
        new_elements = [
          el.with_changes(value=cst.Integer(str(int(output_shape[i])))) for i, el in enumerate(arg.value.elements)
        ]
        new_seq = arg.value.with_changes(elements=new_elements)
        new_args = list(expr.args)
        new_args[arg_idx] = arg.with_changes(value=new_seq)
        return expr.with_changes(args=new_args)
      return expr

  # Case B: Variadic dimension arguments, e.g. `x.view(x.size(0), -1)` or `torch.reshape(x, B, -1)`
  start_idx = 0 if is_method_on_tensor else 1
  dim_args = expr.args[start_idx:]
  if len(dim_args) == len(output_shape) and any(_is_symbolic_dim_expr(a.value) for a in dim_args):
    new_dim_args = [a.with_changes(value=cst.Integer(str(int(output_shape[i])))) for i, a in enumerate(dim_args)]
    return expr.with_changes(args=[*expr.args[:start_idx], *new_dim_args])

  return expr


def _concretize_symbolic_ctor_expr(
  expr: cst.BaseExpression,
  tel: NodeRuntimeTelemetry,
) -> cst.BaseExpression:
  """Concretize symbolic input dimension expressions in inline constructor calls (e.g. ``nn.Linear(x.shape[-1], 10)(x)``).

  Args:
      expr: Expression node to inspect.
      tel: ``NodeRuntimeTelemetry`` for the statement.

  Returns:
      Updated expression with symbolic constructor dimension arguments replaced by concrete integers.
  """
  if not isinstance(expr, cst.Call):
    return expr

  in_dim: int | None = None
  if tel.input_shapes and tel.input_shapes[0]:
    in_dim = int(tel.input_shapes[0][-1])
  elif tel.output_shape:
    in_dim = int(tel.output_shape[-1])

  if in_dim is None:
    return expr

  # Inline chained call: `nn.Linear(x.shape[-1], 10)(x)` or `nn.LayerNorm(x.shape[-1])(x)`
  if isinstance(expr.func, cst.Call) and expr.func.args:
    ctor_call = expr.func
    first_arg = ctor_call.args[0]
    if _is_symbolic_dim_expr(first_arg.value):
      updated_first = first_arg.with_changes(value=cst.Integer(str(in_dim)))
      new_ctor = ctor_call.with_changes(args=[updated_first, *ctor_call.args[1:]])
      return expr.with_changes(func=new_ctor)

  # Direct constructor call assigned in forward: `layer = nn.Linear(in_dim, 10)`
  leaf = ""
  if isinstance(expr.func, cst.Name):
    leaf = expr.func.value
  elif isinstance(expr.func, cst.Attribute):
    leaf = expr.func.attr.value
  if leaf in ("Linear", "LayerNorm", "Dense") and expr.args and _is_symbolic_dim_expr(expr.args[0].value):
    updated_first = expr.args[0].with_changes(value=cst.Integer(str(in_dim)))
    return expr.with_changes(args=[updated_first, *expr.args[1:]])

  return expr


_CREATION_OPS = frozenset(
  {
    "zeros",
    "ones",
    "full",
    "empty",
    "randn",
    "rand",
    "arange",
    "linspace",
  }
)


def _concretize_creation_call(
  expr: cst.BaseExpression,
  output_shape: list[int] | tuple[int, ...],
) -> cst.BaseExpression:
  """Concretize symbolic dimension arguments in creation calls (`zeros`, `ones`, `full`, `empty`, `randn`, `rand`, `arange`, `linspace`).

  Preserves keyword arguments (`dtype=...`, `device=...`) and CST trivia.

  Args:
      expr: Call expression node.
      output_shape: Concrete runtime output shape for the statement.

  Returns:
      Updated ``cst.BaseExpression`` with symbolic dimension arguments replaced by concrete integer literals.
  """
  if not isinstance(expr, cst.Call) or not output_shape:
    return expr

  leaf = ""
  if isinstance(expr.func, cst.Name):
    leaf = expr.func.value
  elif isinstance(expr.func, cst.Attribute):
    leaf = expr.func.attr.value

  lower = leaf.lower()
  if lower not in _CREATION_OPS:
    return expr

  if lower == "arange":
    pos_indices = [i for i, a in enumerate(expr.args) if a.keyword is None]
    if len(pos_indices) == 1:
      idx0 = pos_indices[0]
      if _is_symbolic_dim_expr(expr.args[idx0].value):
        new_args = list(expr.args)
        new_args[idx0] = new_args[idx0].with_changes(value=cst.Integer(str(int(output_shape[0]))))
        return expr.with_changes(args=new_args)
    elif len(pos_indices) >= 2:
      idx1 = pos_indices[1]
      if _is_symbolic_dim_expr(expr.args[idx1].value):
        start_expr = expr.args[pos_indices[0]].value
        start_val = int(start_expr.value) if isinstance(start_expr, cst.Integer) else 0
        new_args = list(expr.args)
        new_args[idx1] = new_args[idx1].with_changes(value=cst.Integer(str(start_val + int(output_shape[0]))))
        return expr.with_changes(args=new_args)
    return expr

  if lower == "linspace":
    new_args = list(expr.args)
    for i, a in enumerate(new_args):
      if a.keyword is not None and a.keyword.value in ("steps", "num") and _is_symbolic_dim_expr(a.value):
        new_args[i] = a.with_changes(value=cst.Integer(str(int(output_shape[0]))))
        return expr.with_changes(args=new_args)
    pos_indices = [i for i, a in enumerate(new_args) if a.keyword is None]
    if len(pos_indices) >= 3:
      idx2 = pos_indices[2]
      if _is_symbolic_dim_expr(new_args[idx2].value):
        new_args[idx2] = new_args[idx2].with_changes(value=cst.Integer(str(int(output_shape[0]))))
        return expr.with_changes(args=new_args)
    return expr

  # Case A: Tuple/List shape argument (positional or `shape=` / `size=` keyword)
  for arg_idx, arg in enumerate(expr.args):
    if isinstance(arg.value, (cst.Tuple, cst.List)):
      if len(arg.value.elements) == len(output_shape) and any(
        _is_symbolic_dim_expr(el.value) for el in arg.value.elements
      ):
        new_elements = [
          el.with_changes(value=cst.Integer(str(int(output_shape[i])))) for i, el in enumerate(arg.value.elements)
        ]
        new_seq = arg.value.with_changes(elements=new_elements)
        new_args = list(expr.args)
        new_args[arg_idx] = arg.with_changes(value=new_seq)
        return expr.with_changes(args=new_args)
      return expr
    if arg.keyword is not None and arg.keyword.value in ("shape", "size") and len(output_shape) == 1:
      if _is_symbolic_dim_expr(arg.value):
        new_args = list(expr.args)
        new_args[arg_idx] = arg.with_changes(value=cst.Integer(str(int(output_shape[0]))))
        return expr.with_changes(args=new_args)

  # Case B: Variadic positional dimension arguments (`torch.zeros(x.size(0), x.size(1), dtype=...)`)
  pos_indices = [i for i, a in enumerate(expr.args) if a.keyword is None]
  if lower == "full" and len(pos_indices) >= 2:
    pos_indices = pos_indices[:1]
  if len(pos_indices) == len(output_shape) and any(_is_symbolic_dim_expr(expr.args[i].value) for i in pos_indices):
    new_args = list(expr.args)
    for dim_i, arg_i in enumerate(pos_indices):
      new_args[arg_i] = new_args[arg_i].with_changes(value=cst.Integer(str(int(output_shape[dim_i]))))
    return expr.with_changes(args=new_args)

  return expr


def _concretize_range_iter(
  iter_expr: cst.BaseExpression,
  trip_count: int,
) -> cst.BaseExpression:
  """Rewrite a symbolic ``range(expr)`` iterator expression to ``range(<trip_count>)``.

  Args:
      iter_expr: Iterator CST expression on a ``cst.For`` node.
      trip_count: Concrete runtime trip count recorded for the loop.

  Returns:
      Updated ``cst.BaseExpression`` with concrete ``cst.Integer(str(trip_count))`` argument.
  """
  if isinstance(iter_expr, cst.Call) and isinstance(iter_expr.func, cst.Name) and iter_expr.func.value == "range":
    if len(iter_expr.args) == 1 and _is_symbolic_dim_expr(iter_expr.args[0].value):
      new_arg = iter_expr.args[0].with_changes(value=cst.Integer(str(int(trip_count))))
      return iter_expr.with_changes(args=[new_arg])
    if len(iter_expr.args) >= 2 and any(_is_symbolic_dim_expr(a.value) for a in iter_expr.args):
      return iter_expr.with_changes(args=[cst.Arg(value=cst.Integer(str(int(trip_count))))])
  return iter_expr


def _concretize_layer_call(
  call: cst.Call,
  resolution: ConstructorDimResolution,
  target_traits: StructuralTraits,
) -> cst.Call:
  """Rewrite a layer constructor Call with runtime-inferred input dimensions.

  Args:
      call: Original layer constructor `cst.Call`.
      resolution: `ConstructorDimResolution` recorded during the forward pass.
      target_traits: Target framework `StructuralTraits`.

  Returns:
      Concretized `cst.Call` node.
  """
  leaf_name = ""
  if isinstance(call.func, cst.Name):
    leaf_name = call.func.value
  elif isinstance(call.func, cst.Attribute):
    leaf_name = call.func.attr.value

  was_lazy = leaf_name.startswith("Lazy")
  clean_leaf = leaf_name.removeprefix("Lazy") if was_lazy else leaf_name
  if clean_leaf == "Dense" and "Linear" in target_traits.lazy_layer_dim_arg_map:
    clean_leaf = "Linear"

  new_func = call.func
  if was_lazy:
    new_func = _replace_call_leaf(call.func, clean_leaf, target_traits.layer_namespace_prefix)

  in_kw = target_traits.lazy_layer_dim_arg_map.get(clean_leaf)
  if not in_kw:
    return call.with_changes(func=new_func)

  out_kw = _out_kwarg_for_in_kwarg(in_kw)
  in_val_node = cst.Integer(str(resolution.inferred_in_features))

  existing_kw_names = {arg.keyword.value for arg in call.args if arg.keyword is not None}
  pos_args = [arg for arg in call.args if arg.keyword is None]
  kw_args = [arg for arg in call.args if arg.keyword is not None]

  new_args: list[cst.Arg] = []
  if in_kw in existing_kw_names:
    for arg in call.args:
      if arg.keyword is not None and arg.keyword.value == in_kw:
        new_args.append(arg.with_changes(value=in_val_node))
      else:
        new_args.append(arg)
    return call.with_changes(func=new_func, args=new_args)

  if len(pos_args) == 1 and out_kw not in existing_kw_names:
    new_args.append(
      cst.Arg(
        keyword=cst.Name(in_kw),
        value=in_val_node,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      )
    )
    new_args.append(
      cst.Arg(
        keyword=cst.Name(out_kw),
        value=pos_args[0].value,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      )
    )
    new_args.extend(kw_args)
    return call.with_changes(func=new_func, args=new_args)

  if len(pos_args) == 2 and out_kw not in existing_kw_names:
    new_args.append(
      cst.Arg(
        keyword=cst.Name(in_kw),
        value=in_val_node,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      )
    )
    new_args.append(
      cst.Arg(
        keyword=cst.Name(out_kw),
        value=pos_args[1].value,
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      )
    )
    new_args.extend(kw_args)
    return call.with_changes(func=new_func, args=new_args)

  if len(pos_args) > 2:
    new_args.append(pos_args[0].with_changes(value=in_val_node))
    new_args.extend(pos_args[1:])
    new_args.extend(kw_args)
    return call.with_changes(func=new_func, args=new_args)

  new_args.append(
    cst.Arg(
      keyword=cst.Name(in_kw),
      value=in_val_node,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
  )
  new_args.extend(kw_args)
  return call.with_changes(func=new_func, args=new_args)


class SourceLazyConcretizer(cst.CSTTransformer):
  """Pre-pipeline CST transformer that normalizes `Lazy*` constructors and symbolic `for` bounds before `ApiPass`.

  Rewrites `self.<layer> = nn.LazyLinear(out)` to `self.<layer> = nn.Linear(in_features, out)`
  using `runtime_trace.resolved_constructors`, and rewrites symbolic `for i in range(expr):`
  bounds using `runtime_trace.loop_trip_counts` so `AuxiliaryPass` (`unroll_static_loops`)
  can unroll them cleanly.
  """

  def __init__(self, runtime_trace: RuntimeTraceReport) -> None:
    """Initialize with the on-device `RuntimeTraceReport`.

    Args:
        runtime_trace: Populated `RuntimeTraceReport`.
    """
    self.runtime_trace = runtime_trace
    self._for_cursor: int = 0

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Concretize `self.<layer> = ...LazyLinear(out)` assignments before `ApiPass`.

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
        layer_name = target.target.attr.value
        res = self.runtime_trace.resolved_constructors.get(layer_name)
        if res is not None:
          call = updated_node.value
          leaf = ""
          if isinstance(call.func, cst.Name):
            leaf = call.func.value
          elif isinstance(call.func, cst.Attribute):
            leaf = call.func.attr.value
          if leaf.startswith("Lazy"):
            clean_leaf = leaf.removeprefix("Lazy")
            new_func = _replace_call_leaf(call.func, clean_leaf)
            in_arg = cst.Arg(value=cst.Integer(str(res.inferred_in_features)))
            return updated_node.with_changes(value=call.with_changes(func=new_func, args=[in_arg, *call.args]))
    return updated_node

  def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
    """Concretize symbolic `range(expr)` bounds on `for` loops using `loop_trip_counts`.

    Args:
        original_node: Original For node.
        updated_node: Updated For node.

    Returns:
        For node with concretized `range(<trip_count>)` iterator.
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
        context: Active `RewriterContext` containing `runtime_trace` and `config`.
        target_traits: Resolved `StructuralTraits` for the target framework.
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
        Annotated `SimpleStatementLine` or `FlattenSentinel` including shape assertions.
    """
    trace = self.context.runtime_trace
    assert trace is not None

    if len(updated_node.body) == 1:
      small = updated_node.body[0]

      # 1. Layer constructor assignment: self.<layer> = <Call>
      if isinstance(small, cst.Assign) and isinstance(small.value, cst.Call) and small.targets:
        tgt = small.targets[0].target
        if isinstance(tgt, cst.Attribute) and isinstance(tgt.value, cst.Name) and tgt.value.value == "self":
          layer_name = tgt.attr.value
          if layer_name in trace.resolved_constructors:
            res = trace.resolved_constructors[layer_name]
            new_call = _concretize_layer_call(small.value, res, self.target_traits)
            updated_node = updated_node.with_changes(body=[small.with_changes(value=new_call)])
            if self.context.config.inject_runtime_annotations:
              updated_node = _attach_comment_to_line(
                updated_node,
                f"# [on-device ctor] resolved in_features={res.inferred_in_features}",
              )
            return updated_node

      # 2. Praxis create_child('<layer>', <Call>) statement
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
        if layer_name in trace.resolved_constructors:
          res = trace.resolved_constructors[layer_name]
          inner_call = _concretize_layer_call(small.value.args[1].value, res, self.target_traits)
          new_args = [small.value.args[0], small.value.args[1].with_changes(value=inner_call), *small.value.args[2:]]
          new_outer = small.value.with_changes(args=new_args)
          updated_node = updated_node.with_changes(body=[small.with_changes(value=new_outer)])
          if self.context.config.inject_runtime_annotations:
            updated_node = _attach_comment_to_line(
              updated_node,
              f"# [on-device ctor] resolved in_features={res.inferred_in_features}",
            )
          return updated_node

      # 3. Forward statement telemetry & executable shape assertions
      if self._in_forward:
        if isinstance(small, cst.Assign) and small.targets:
          var_name = self._extract_var_name(small.targets[0].target)
          tel = self._match_telemetry(var_name, trace)
          if tel is not None:
            new_val = _concretize_reshape_view_call(small.value, tel.output_shape)
            new_val = _concretize_creation_call(new_val, tel.output_shape)
            new_val = _concretize_symbolic_ctor_expr(new_val, tel)
            if new_val is not small.value:
              small = small.with_changes(value=new_val)
              updated_node = updated_node.with_changes(body=[small])
            if self.context.config.inject_runtime_annotations:
              comment = _format_telemetry_comment(tel, self.context.config.random_seed)
              updated_node = _attach_comment_to_line(updated_node, comment)
            if (
              self.context.config.inject_shape_assertions
              and self.target_traits.shape_assertion_template
              and var_name != "tmp"
            ):
              stmt_code = self.target_traits.shape_assertion_template.format(
                var=var_name,
                shape=tuple(tel.output_shape),
              )
              assert_stmt = cst.parse_statement(stmt_code)
              self.assertions_injected = True
              return cst.FlattenSentinel([updated_node, assert_stmt])
        elif isinstance(small, cst.AugAssign):
          var_name = self._extract_var_name(small.target)
          tel = self._match_telemetry(var_name, trace)
          if tel is not None and self.context.config.inject_runtime_annotations:
            comment = _format_telemetry_comment(tel, self.context.config.random_seed)
            updated_node = _attach_comment_to_line(updated_node, comment)
        elif isinstance(small, cst.Return) and small.value is not None:
          tel = self._match_telemetry("return", trace)
          if tel is not None:
            new_ret_val = _concretize_reshape_view_call(small.value, tel.output_shape)
            new_ret_val = _concretize_creation_call(new_ret_val, tel.output_shape)
            new_ret_val = _concretize_symbolic_ctor_expr(new_ret_val, tel)
            small = small.with_changes(value=new_ret_val)
            updated_node = updated_node.with_changes(body=[small])
            if self.context.config.inject_runtime_annotations:
              comment = _format_telemetry_comment(tel, self.context.config.random_seed)
              updated_node = _attach_comment_to_line(updated_node, comment)

    return updated_node

  def visit_If(self, node: cst.If) -> bool:
    """Track nesting depth of `if` / `elif` statements.

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
    """Attach `# [on-device branch] taken=...` comments or specialize taken branches when configured.

    Args:
        original_node: Original If node.
        updated_node: Updated If node.

    Returns:
        Annotated `If` node, specialized `FlattenSentinel`, or `RemovalSentinel`.
    """
    self._if_depth -= 1
    trace = self.context.runtime_trace
    assert trace is not None
    if not self._in_forward:
      return updated_node

    if self.context.config.specialize_dynamic_branches:
      if self._if_depth == 0 and self._branch_cursor < len(trace.resolved_branches):
        return self._specialize_if_chain(updated_node, trace)
      return updated_node

    if self.context.config.inject_runtime_annotations:
      if self._branch_cursor < len(trace.resolved_branches):
        rec = trace.resolved_branches[self._branch_cursor]
        self._branch_cursor += 1
        taken_title = "True" if rec.taken_branch == "true" else "False"
        new_body = _attach_comment_to_block_header(
          updated_node.body,
          f"# [on-device branch] taken={taken_title}",
        )
        return updated_node.with_changes(body=new_body)
    return updated_node

  def _specialize_if_chain(
    self,
    node: cst.If,
    trace: RuntimeTraceReport,
  ) -> cst.If | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
    """Specialize a top-level `if` / `elif` / `else` statement to the branch taken on-device.

    Preserves leading trivia, comments inside the taken branch, and untaken branches as comments.

    Args:
        node: Top-level `cst.If` node.
        trace: Active `RuntimeTraceReport`.

    Returns:
        Specialized statement sequence or `RemovalSentinel`.
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
        for raw_line in untaken_src.splitlines():
          untaken_comment_lines.append(
            cst.EmptyLine(comment=cst.Comment(f"# [untaken] {raw_line.strip()}"), newline=cst.Newline())
          )
    else:
      for s in body_stmts:
        raw_if_body = cst.Module([]).code_for_node(s).strip()
        for raw_line in raw_if_body.splitlines():
          untaken_comment_lines.append(
            cst.EmptyLine(comment=cst.Comment(f"# [untaken] {raw_line.strip()}"), newline=cst.Newline())
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
        assert isinstance(nested, cst.FlattenSentinel)
        taken_stmts = list(nested.nodes)

    all_leading = [*header_lines, *untaken_comment_lines]
    if not taken_stmts:
      pass_stmt = cst.SimpleStatementLine(body=[cst.Pass()], leading_lines=all_leading)
      return cst.FlattenSentinel([pass_stmt])

    first_stmt = taken_stmts[0]
    existing_leading = (
      list(first_stmt.leading_lines)
      if isinstance(first_stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement))
      else []
    )
    taken_stmts[0] = first_stmt.with_changes(leading_lines=[*all_leading, *existing_leading])
    return cst.FlattenSentinel(taken_stmts)

  def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
    """Concretize symbolic `range(expr)` bounds on remaining `for` loops and attach trip-count comments.

    Args:
        original_node: Original For node.
        updated_node: Updated For node.

    Returns:
        Concretized and annotated For node.
    """
    trace = self.context.runtime_trace
    assert trace is not None
    if self._in_forward:
      counts = list(trace.loop_trip_counts.values())
      if self._loop_cursor < len(counts):
        trips = counts[self._loop_cursor]
        self._loop_cursor += 1
        new_iter = _concretize_range_iter(updated_node.iter, trips)
        updated_node = updated_node.with_changes(iter=new_iter)
        if self.context.config.inject_runtime_annotations:
          new_body = _attach_comment_to_block_header(
            updated_node.body,
            f"# [on-device loop] iterations={trips}",
          )
          updated_node = updated_node.with_changes(body=new_body)
    return updated_node

  def leave_While(self, original_node: cst.While, updated_node: cst.While) -> cst.While:
    """Attach `# [on-device loop] iterations=...` comments to while loops.

    Args:
        original_node: Original While node.
        updated_node: Updated While node.

    Returns:
        Annotated While node.
    """
    trace = self.context.runtime_trace
    assert trace is not None
    if self._in_forward and self.context.config.inject_runtime_annotations:
      counts = list(trace.loop_trip_counts.values())
      if self._loop_cursor < len(counts):
        trips = counts[self._loop_cursor]
        self._loop_cursor += 1
        new_body = _attach_comment_to_block_header(
          updated_node.body,
          f"# [on-device loop] iterations={trips}",
        )
        return updated_node.with_changes(body=new_body)
    return updated_node

  def _match_telemetry(
    self,
    var_name: str,
    trace: RuntimeTraceReport,
  ) -> NodeRuntimeTelemetry | None:
    """Look up the next matching `NodeRuntimeTelemetry` entry for `var_name`.

    Args:
        var_name: Target variable name or `'return'`.
        trace: Active `RuntimeTraceReport`.

    Returns:
        Matching `NodeRuntimeTelemetry` or None.
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
    """Transform `module` using `context.runtime_trace`.

    Args:
        module: Input LibCST module.
        context: Shared `RewriterContext`.

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
