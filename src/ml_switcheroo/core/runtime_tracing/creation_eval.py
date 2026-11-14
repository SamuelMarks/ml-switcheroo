"""Tensor Creation and Inline Constructor Evaluation Helpers for ``_ForwardSymbolicExecutor``.

Provides pure helper functions delegated to by ``_ForwardSymbolicExecutor`` for
``arange``, ``linspace``, ``zeros``/``ones``/``full``/``empty``/``randn``/``rand``,
and inline neural constructor invocations.
"""

from __future__ import annotations

import typing

import libcst as cst

from ml_switcheroo.core.runtime_models import (
  ConstructorDimResolution,
  DeclaredLayerSpec,
)
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _generate_seeded_values,
  _prod,
)
from ml_switcheroo.core.runtime_tracing.sandbox import _TracedLayerInstance

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.runtime_tracing.symbolic_executor import _ForwardSymbolicExecutor


def eval_inline_ctor_call(
  executor: _ForwardSymbolicExecutor,
  call: cst.Call,
  ctor_call: cst.Call,
  ctor_leaf: str,
) -> tuple[OnDeviceTraceTensor, str]:
  """Evaluate an inline constructor invocation such as ``nn.Linear(x.shape[-1], 10)(x)``.

  Args:
      executor: Active ``_ForwardSymbolicExecutor`` state.
      call: Outer invocation ``cst.Call`` node.
      ctor_call: Inner constructor ``cst.Call`` node.
      ctor_leaf: Constructor leaf name.

  Returns:
      Tuple of ``(evaluated_tensor, op_type)``.
  """
  arg_tensor = executor._eval_expr(call.args[0].value)[0] if call.args else executor.output_tensor
  in_dim = arg_tensor.shape[-1] if arg_tensor.shape else 4
  out_dim = in_dim
  if len(ctor_call.args) >= 2:
    parsed_out = executor._eval_dim_expr(ctor_call.args[1].value)
    if parsed_out is not None:
      out_dim = parsed_out
  elif len(ctor_call.args) == 1 and ctor_leaf in ("LazyLinear", "Dense"):
    parsed_out = executor._eval_dim_expr(ctor_call.args[0].value)
    if parsed_out is not None:
      out_dim = parsed_out

  spec_id = f"inline_{ctor_leaf.lower()}_{len(executor.resolved_constructors)}"
  executor.resolved_constructors[spec_id] = ConstructorDimResolution(
    layer_name=spec_id,
    op_type=ctor_leaf,
    inferred_in_features=in_dim,
    out_features=out_dim,
  )
  if ctor_leaf == "LayerNorm":
    return arg_tensor, "LayerNorm"
  spec = DeclaredLayerSpec(spec_id, ctor_leaf, in_dim, out_dim, True)
  inst = _TracedLayerInstance(spec, executor.seed + 77, executor.resolved_constructors, executor.parameters)
  return inst(arg_tensor), ctor_leaf


def eval_arange_creation(
  executor: _ForwardSymbolicExecutor,
  leaf: str,
  pos_args: list[cst.Arg],
) -> tuple[OnDeviceTraceTensor, str]:
  """Evaluate ``arange(...)`` tensor creation calls.

  Args:
      executor: Active ``_ForwardSymbolicExecutor`` state.
      leaf: Original function leaf name.
      pos_args: Positional ``cst.Arg`` list.

  Returns:
      Tuple of ``(created_tensor, leaf)``.
  """
  if len(pos_args) <= 1:
    stop_v = executor._eval_dim_expr(pos_args[0].value) if pos_args else None
    count = (
      max(1, stop_v) if stop_v is not None else (executor.output_tensor.shape[-1] if executor.output_tensor.shape else 4)
    )
    return OnDeviceTraceTensor(shape=(count,), data=[float(i) for i in range(count)], dtype="float32"), leaf

  start_v = executor._eval_dim_expr(pos_args[0].value) or 0
  stop_v = executor._eval_dim_expr(pos_args[1].value) or (start_v + 4)
  step_v = executor._eval_dim_expr(pos_args[2].value) if len(pos_args) >= 3 else 1
  step_clean = step_v if (step_v is not None and step_v != 0) else 1
  vals = [float(v) for v in range(start_v, stop_v, step_clean)] or [0.0]
  return OnDeviceTraceTensor(shape=(len(vals),), data=vals, dtype="float32"), leaf


def eval_linspace_creation(
  executor: _ForwardSymbolicExecutor,
  leaf: str,
  pos_args: list[cst.Arg],
  kw_map: dict[str, cst.BaseExpression],
) -> tuple[OnDeviceTraceTensor, str]:
  """Evaluate ``linspace(...)`` tensor creation calls.

  Args:
      executor: Active ``_ForwardSymbolicExecutor`` state.
      leaf: Original function leaf name.
      pos_args: Positional ``cst.Arg`` list.
      kw_map: Keyword argument expression map.

  Returns:
      Tuple of ``(created_tensor, leaf)``.
  """
  steps_expr = kw_map.get("steps") or kw_map.get("num") or (pos_args[2].value if len(pos_args) >= 3 else None)
  steps_val = executor._eval_dim_expr(steps_expr) if steps_expr is not None else None
  n_steps = (
    max(1, steps_val)
    if steps_val is not None
    else (executor.output_tensor.shape[-1] if executor.output_tensor.shape else 4)
  )
  vals = [float(i) / float(max(1, n_steps - 1)) for i in range(n_steps)]
  return OnDeviceTraceTensor(shape=(n_steps,), data=vals, dtype="float32"), leaf


def eval_creation_call(
  executor: _ForwardSymbolicExecutor,
  lower: str,
  leaf: str,
  call: cst.Call,
) -> tuple[OnDeviceTraceTensor, str]:
  """Evaluate a tensor creation call (`zeros`, `ones`, `full`, `empty`, `randn`, `rand`, `arange`, `linspace`).

  Args:
      executor: Active ``_ForwardSymbolicExecutor`` state.
      lower: Lowercase leaf name of the creation function.
      leaf: Original leaf name of the creation function.
      call: LibCST Call expression.

  Returns:
      Tuple of ``(created_tensor, operation_name)``.
  """
  pos_args = [a for a in call.args if a.keyword is None]
  kw_map = {a.keyword.value: a.value for a in call.args if a.keyword is not None}

  if lower == "arange":
    return eval_arange_creation(executor, leaf, pos_args)
  if lower == "linspace":
    return eval_linspace_creation(executor, leaf, pos_args, kw_map)

  shape_args = pos_args[:1] if (lower == "full" and len(pos_args) >= 2) else pos_args
  kw_shape_expr = kw_map.get("shape") or kw_map.get("size")
  if not shape_args and kw_shape_expr is not None:
    shape_args = [cst.Arg(value=kw_shape_expr)]

  raw_dims = executor._parse_call_int_args(shape_args)
  ref_shape = executor.output_tensor.shape if executor.output_tensor.shape else (2, 4)
  resolved_dims = [
    max(1, int((ref_shape[idx] if idx < len(ref_shape) else ref_shape[-1]) if d < 0 else d))
    for idx, d in enumerate(raw_dims)
  ]
  final_shape = tuple(resolved_dims) if resolved_dims else ref_shape
  numel = _prod(final_shape)

  if lower == "zeros":
    data = [0.0] * numel
  elif lower in ("ones", "empty"):
    data = [1.0] * numel
  elif lower == "full":
    fill_expr = kw_map.get("fill_value") or (pos_args[1].value if len(pos_args) >= 2 else None)
    fill_val = float(executor._eval_expr(fill_expr)[0].data[0]) if fill_expr is not None else 1.0
    data = [fill_val] * numel
  else:
    data = _generate_seeded_values(numel, executor.seed + executor._node_counter + 5, "float32", 0.5)

  return OnDeviceTraceTensor(shape=final_shape, data=data, dtype="float32"), leaf


__all__ = [
  "eval_arange_creation",
  "eval_creation_call",
  "eval_inline_ctor_call",
  "eval_linspace_creation",
]
