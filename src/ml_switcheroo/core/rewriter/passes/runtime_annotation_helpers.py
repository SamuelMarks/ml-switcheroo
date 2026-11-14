"""CST Concretization and Telemetry Comment Helpers for Runtime Annotation Passes.

Provides pure LibCST transformation helpers used by ``SourceLazyConcretizer`` and
``RuntimeAnnotationTransformer`` to materialize concrete layer dimensions, reshape
targets, tensor creation shapes, loop bounds, and inline telemetry comments.
"""

from __future__ import annotations

import libcst as cst

from ml_switcheroo.core.runtime_models import (
  ConstructorDimResolution,
  NodeRuntimeTelemetry,
)
from ml_switcheroo.semantics.schema import StructuralTraits

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


def _attach_comment_to_line(node: cst.SimpleStatementLine, comment_text: str) -> cst.SimpleStatementLine:
  """Attach or append a trailing comment to a ``SimpleStatementLine``.

  Args:
      node: Statement line node to annotate.
      comment_text: Comment string starting with ``# ``.

  Returns:
      Updated ``SimpleStatementLine`` with trailing comment.
  """
  existing = node.trailing_whitespace.comment
  combined = f"{existing.value}  {comment_text}" if existing is not None else comment_text
  return node.with_changes(
    trailing_whitespace=cst.TrailingWhitespace(
      whitespace=cst.SimpleWhitespace("  "),
      comment=cst.Comment(combined),
      newline=node.trailing_whitespace.newline,
    )
  )


def _attach_comment_to_block_header(block: cst.BaseSuite, comment_text: str) -> cst.BaseSuite:
  """Attach a trailing comment to the header of an ``IndentedBlock`` (for ``if`` / ``while``).

  Args:
      block: Suite node (``IndentedBlock``) of the ``If`` or ``While`` statement.
      comment_text: Comment string starting with ``# ``.

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
      tel: ``NodeRuntimeTelemetry`` recorded for the statement.
      seed: Deterministic seed used during on-device execution.

  Returns:
      Formatted comment string beginning with ``# ``.
  """
  fwd_shape = tuple(tel.output_shape)
  parts = [f"# [on-device fwd] shape={fwd_shape} dtype={tel.dtype}"]
  if tel.grad_shape is not None and tel.grad_norm is not None:
    parts.append(f"[bwd] grad={tuple(tel.grad_shape)} norm={tel.grad_norm:.3e}")
  if tel.is_dynamic:
    parts.append(f"[dynamic-shape: resolved on-device (seed={seed})] shape={fwd_shape}")
  return " | ".join(parts)


def _replace_call_leaf(func: cst.BaseExpression, new_leaf: str, prefix: str | None = None) -> cst.BaseExpression:
  """Replace the leaf identifier of a Call's ``func`` expression, optionally updating its prefix.

  Args:
      func: Original callable expression (``Name`` or ``Attribute``).
      new_leaf: New leaf class/function name (e.g. ``'Linear'``).
      prefix: Optional namespace prefix.

  Returns:
      Updated callable expression.
  """
  if prefix:
    return cst.parse_expression(f"{prefix}.{new_leaf}")
  if isinstance(func, cst.Name):
    return cst.Name(new_leaf)
  if isinstance(func, cst.Attribute):
    return func.with_changes(attr=cst.Name(new_leaf))
  return func


def _out_kwarg_for_in_kwarg(in_kw: str) -> str:
  """Map a target input-dimension keyword argument name to its corresponding output keyword name.

  Args:
      in_kw: Input dimension keyword argument name (``'in_features'``, ``'input_dims'``, ``'in_channels'``).

  Returns:
      Corresponding output dimension keyword argument name.
  """
  if in_kw == "input_dims":
    return "output_dims"
  if in_kw == "in_channels":
    return "out_channels"
  return "out_features"


def _is_symbolic_dim_expr(expr: cst.BaseExpression) -> bool:
  """Return True if ``expr`` is a dynamic/symbolic dimension expression (``-1``, ``x.shape[i]``, ``x.size(i)``, or ``Name``).

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
  return isinstance(expr, cst.Name)


def _call_leaf_and_method_flag(
  expr: cst.Call,
  output_shape: list[int] | tuple[int, ...],
) -> tuple[str, bool]:
  """Identify the callable leaf name and whether ``expr`` is a tensor method call.

  Args:
      expr: Candidate ``cst.Call`` node.
      output_shape: Concrete runtime output shape.

  Returns:
      Tuple of ``(leaf_name, is_method_on_tensor)``.
  """
  if isinstance(expr.func, cst.Name):
    return expr.func.value, False
  if isinstance(expr.func, cst.Attribute):
    leaf = expr.func.attr.value
    is_method = leaf == "view" or (
      leaf == "reshape"
      and len(expr.args) == len(output_shape)
      and bool(expr.args)
      and not isinstance(expr.args[0].value, (cst.Tuple, cst.List))
      and (isinstance(expr.args[0].value, cst.Integer) or _is_symbolic_dim_expr(expr.args[0].value))
    )
    return leaf, is_method
  return "", False


def _concretize_reshape_view_call(
  expr: cst.BaseExpression,
  output_shape: list[int] | tuple[int, ...],
) -> cst.BaseExpression:
  """Concretize symbolic dimension arguments in ``reshape`` / ``view`` calls.

  Args:
      expr: Call expression node.
      output_shape: Concrete runtime output shape for the statement.

  Returns:
      Updated ``cst.BaseExpression`` with symbolic dimension arguments replaced by concrete integer literals.
  """
  if not isinstance(expr, cst.Call) or not output_shape:
    return expr

  leaf, is_method_on_tensor = _call_leaf_and_method_flag(expr, output_shape)
  if leaf not in ("reshape", "view"):
    return expr

  for arg_idx, arg in enumerate(expr.args):
    if isinstance(arg.value, (cst.Tuple, cst.List)):
      if len(arg.value.elements) == len(output_shape) and any(
        _is_symbolic_dim_expr(el.value) for el in arg.value.elements
      ):
        new_elements = [
          el.with_changes(value=cst.Integer(str(int(output_shape[i])))) for i, el in enumerate(arg.value.elements)
        ]
        new_args = list(expr.args)
        new_args[arg_idx] = arg.with_changes(value=arg.value.with_changes(elements=new_elements))
        return expr.with_changes(args=new_args)
      return expr

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
  """Concretize symbolic input dimension expressions in inline constructor calls.

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

  if isinstance(expr.func, cst.Call) and expr.func.args:
    ctor_call = expr.func
    first_arg = ctor_call.args[0]
    if _is_symbolic_dim_expr(first_arg.value):
      updated_first = first_arg.with_changes(value=cst.Integer(str(in_dim)))
      return expr.with_changes(func=ctor_call.with_changes(args=[updated_first, *ctor_call.args[1:]]))

  leaf = (
    expr.func.value
    if isinstance(expr.func, cst.Name)
    else (expr.func.attr.value if isinstance(expr.func, cst.Attribute) else "")
  )
  if leaf in ("Linear", "LayerNorm", "Dense") and expr.args and _is_symbolic_dim_expr(expr.args[0].value):
    updated_first = expr.args[0].with_changes(value=cst.Integer(str(in_dim)))
    return expr.with_changes(args=[updated_first, *expr.args[1:]])

  return expr


def _concretize_arange_call(expr: cst.Call, output_len: int) -> cst.Call:
  """Concretize symbolic stop bounds in an ``arange(...)`` call.

  Args:
      expr: ``arange`` ``cst.Call`` node.
      output_len: Concrete 1-D length of the resulting sequence.

  Returns:
      Updated ``cst.Call`` node.
  """
  pos_indices = [i for i, a in enumerate(expr.args) if a.keyword is None]
  if len(pos_indices) == 1:
    idx0 = pos_indices[0]
    if _is_symbolic_dim_expr(expr.args[idx0].value):
      new_args = list(expr.args)
      new_args[idx0] = new_args[idx0].with_changes(value=cst.Integer(str(output_len)))
      return expr.with_changes(args=new_args)
  elif len(pos_indices) >= 2:
    idx1 = pos_indices[1]
    if _is_symbolic_dim_expr(expr.args[idx1].value):
      start_expr = expr.args[pos_indices[0]].value
      start_val = int(start_expr.value) if isinstance(start_expr, cst.Integer) else 0
      new_args = list(expr.args)
      new_args[idx1] = new_args[idx1].with_changes(value=cst.Integer(str(start_val + output_len)))
      return expr.with_changes(args=new_args)
  return expr


def _concretize_linspace_call(expr: cst.Call, output_len: int) -> cst.Call:
  """Concretize symbolic ``steps``/``num`` arguments in a ``linspace(...)`` call.

  Args:
      expr: ``linspace`` ``cst.Call`` node.
      output_len: Concrete 1-D step count of the resulting sequence.

  Returns:
      Updated ``cst.Call`` node.
  """
  new_args = list(expr.args)
  for i, a in enumerate(new_args):
    if a.keyword is not None and a.keyword.value in ("steps", "num") and _is_symbolic_dim_expr(a.value):
      new_args[i] = a.with_changes(value=cst.Integer(str(output_len)))
      return expr.with_changes(args=new_args)
  pos_indices = [i for i, a in enumerate(new_args) if a.keyword is None]
  if len(pos_indices) >= 3:
    idx2 = pos_indices[2]
    if _is_symbolic_dim_expr(new_args[idx2].value):
      new_args[idx2] = new_args[idx2].with_changes(value=cst.Integer(str(output_len)))
      return expr.with_changes(args=new_args)
  return expr


def _concretize_shape_creation_args(
  expr: cst.Call,
  lower: str,
  output_shape: list[int] | tuple[int, ...],
) -> cst.Call:
  """Concretize tuple/list, keyword ``shape``/``size``, or variadic positional dims on creation calls.

  Args:
      expr: Tensor creation ``cst.Call`` node.
      lower: Lowercased creation function name.
      output_shape: Concrete runtime output shape.

  Returns:
      Updated ``cst.Call`` node.
  """
  for arg_idx, arg in enumerate(expr.args):
    if isinstance(arg.value, (cst.Tuple, cst.List)):
      if len(arg.value.elements) == len(output_shape) and any(
        _is_symbolic_dim_expr(el.value) for el in arg.value.elements
      ):
        new_elements = [
          el.with_changes(value=cst.Integer(str(int(output_shape[i])))) for i, el in enumerate(arg.value.elements)
        ]
        new_args = list(expr.args)
        new_args[arg_idx] = arg.with_changes(value=arg.value.with_changes(elements=new_elements))
        return expr.with_changes(args=new_args)
      return expr
    if (
      arg.keyword is not None
      and arg.keyword.value in ("shape", "size")
      and len(output_shape) == 1
      and _is_symbolic_dim_expr(arg.value)
    ):
      new_args = list(expr.args)
      new_args[arg_idx] = arg.with_changes(value=cst.Integer(str(int(output_shape[0]))))
      return expr.with_changes(args=new_args)

  pos_indices = [i for i, a in enumerate(expr.args) if a.keyword is None]
  if lower == "full" and len(pos_indices) >= 2:
    pos_indices = pos_indices[:1]
  if len(pos_indices) == len(output_shape) and any(_is_symbolic_dim_expr(expr.args[i].value) for i in pos_indices):
    new_args = list(expr.args)
    for dim_i, arg_i in enumerate(pos_indices):
      new_args[arg_i] = new_args[arg_i].with_changes(value=cst.Integer(str(int(output_shape[dim_i]))))
    return expr.with_changes(args=new_args)

  return expr


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

  leaf = (
    expr.func.value
    if isinstance(expr.func, cst.Name)
    else (expr.func.attr.value if isinstance(expr.func, cst.Attribute) else "")
  )
  lower = leaf.lower()
  if lower not in _CREATION_OPS:
    return expr
  if lower == "arange":
    return _concretize_arange_call(expr, int(output_shape[0]))
  if lower == "linspace":
    return _concretize_linspace_call(expr, int(output_shape[0]))
  return _concretize_shape_creation_args(expr, lower, output_shape)


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


def _make_kw_arg(name: str, val: cst.BaseExpression) -> cst.Arg:
  """Construct a compact ``name=val`` ``cst.Arg`` node without extra spacing around ``=``.

  Args:
      name: Keyword argument identifier.
      val: Argument value CST expression.

  Returns:
      Constructed ``cst.Arg`` node.
  """
  return cst.Arg(
    keyword=cst.Name(name),
    value=val,
    equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
  )


def _concretize_layer_call(
  call: cst.Call,
  resolution: ConstructorDimResolution,
  target_traits: StructuralTraits,
) -> cst.Call:
  """Rewrite a layer constructor Call with runtime-inferred input dimensions.

  Args:
      call: Original layer constructor ``cst.Call``.
      resolution: ``ConstructorDimResolution`` recorded during the forward pass.
      target_traits: Target framework ``StructuralTraits``.

  Returns:
      Concretized ``cst.Call`` node.
  """
  leaf_name = (
    call.func.value
    if isinstance(call.func, cst.Name)
    else (call.func.attr.value if isinstance(call.func, cst.Attribute) else "")
  )
  was_lazy = leaf_name.startswith("Lazy")
  clean_leaf = leaf_name.removeprefix("Lazy") if was_lazy else leaf_name
  if clean_leaf == "Dense" and "Linear" in target_traits.lazy_layer_dim_arg_map:
    clean_leaf = "Linear"

  new_func = _replace_call_leaf(call.func, clean_leaf, target_traits.layer_namespace_prefix) if was_lazy else call.func
  in_kw = target_traits.lazy_layer_dim_arg_map.get(clean_leaf)
  if not in_kw:
    return call.with_changes(func=new_func)

  out_kw = _out_kwarg_for_in_kwarg(in_kw)
  in_val_node = cst.Integer(str(resolution.inferred_in_features))
  existing_kw_names = {arg.keyword.value for arg in call.args if arg.keyword is not None}
  pos_args = [arg for arg in call.args if arg.keyword is None]
  kw_args = [arg for arg in call.args if arg.keyword is not None]

  if in_kw in existing_kw_names:
    replaced = [
      arg.with_changes(value=in_val_node) if (arg.keyword is not None and arg.keyword.value == in_kw) else arg
      for arg in call.args
    ]
    return call.with_changes(func=new_func, args=replaced)

  if len(pos_args) in (1, 2) and out_kw not in existing_kw_names:
    out_val = pos_args[0].value if len(pos_args) == 1 else pos_args[1].value
    return call.with_changes(
      func=new_func, args=[_make_kw_arg(in_kw, in_val_node), _make_kw_arg(out_kw, out_val), *kw_args]
    )

  if len(pos_args) > 2:
    return call.with_changes(func=new_func, args=[pos_args[0].with_changes(value=in_val_node), *pos_args[1:], *kw_args])

  return call.with_changes(func=new_func, args=[_make_kw_arg(in_kw, in_val_node), *kw_args])


__all__ = [
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
]
