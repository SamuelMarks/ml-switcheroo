"""
Plugin for Learning Rate Scheduler Rewiring.

Addresses the architectural difference between:

1. PyTorch: `scheduler = StepLR(optimizer, step_size=30)` (Stateful object wrapping optimizer).
2. JAX/Optax: `schedule_fn = optax.piecewise_constant(...)` (Pure function passed to optimizer).

**Transformation**

1.  **Instantiation**:
    -   Detects specific Scheduler constructors via `scheduler_rewire`.
    -   Removes the `optimizer` argument (Arg 0).
    -   Maps hyperparameters to target equivalents using keys defined in the semantic map
        (e.g. `step_size` -> `transition_steps` for Optax or `decay_steps` for Keras).
    -   Changes the API call to the target factory declared in the Knowledge Base.
    -   Injects `init_value=1.0` (as partial schedule) or tries to preserve semantics.

2.  **Stepping**:
    -   Detects `scheduler.step()` via `scheduler_step_noop`.
    -   Replaces `scheduler.step()` with a no-op placeholder `None`.
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext
from ml_switcheroo.core.escape_hatch import EscapeHatch


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Creates a CST Attribute/Name node from a dotted string."""
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


def _get_target_arg_name(ctx: HookContext, std_name: str, default: str) -> str:
  """
  Resolves the target keyword argument name.
  Checks the Semantic Knowledge Base (Variant Args) first, falls back to default.
  """
  if ctx.current_variant and ctx.current_variant.args:
    return ctx.current_variant.args.get(std_name, default)
  return default


@register_hook("scheduler_rewire")
def transform_scheduler_init(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Transforms Scheduler instantiation.

  Logic routes based on detected Operation ID in context (StepLR vs Cosine).
  Now fully decoupled: reads target API and argument names from `ctx`.
  """
  op_id = ctx.current_op_id or ""
  target_api = ctx.lookup_api(op_id)

  # Safety: If API mapping is missing, we cannot rewrite.
  if not target_api:
    return node

  if "StepLR" in op_id:
    return _transform_step_lr(node, ctx, target_api)
  elif "CosineAnnealingLR" in op_id:
    return _transform_cosine_lr(node, ctx, target_api)

  return node


def _transform_step_lr(node: cst.Call, ctx: HookContext, target_api: str) -> cst.Call:
  """
  Transform StepLR.

  Source: StepLR(optim, step_size, gamma)
  Target: target_api(init_value=1.0, transition_steps=step_size, decay_rate=gamma, staircase=True)
  """
  # Parse Args
  args = list(node.args)
  # Remove Optimizer (Arg 0)
  if args:
    args.pop(0)

  new_args = []

  # 1. init_value (Inject 1.0 placeholder, user must adjust)
  # Note: Required because functional schedules usually return a callable f(step) -> lr
  # that multiplies a base LR, or defines the LR path itself.
  init_val_kw = _get_target_arg_name(ctx, "initial_learning_rate", "init_value")
  new_args.append(
    cst.Arg(
      value=cst.Float("1.0"),
      keyword=cst.Name(init_val_kw),
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
  )

  # 2. transition_steps (was step_size)
  step_size_arg = None
  gamma_arg = None

  # Find existing args
  for arg in args:
    kw = arg.keyword.value if arg.keyword else None
    if kw == "step_size":
      step_size_arg = arg
    elif kw == "gamma":
      gamma_arg = arg
    elif not kw:
      # Positional mapping assumes step_size is 1st remaining, gamma is 2nd
      if step_size_arg is None:
        step_size_arg = arg
      elif gamma_arg is None:
        gamma_arg = arg

  if step_size_arg:
    # Optax defaults to 'transition_steps'. Keras uses 'decay_steps'.
    # We map from standard arg 'step_size'.
    target_kw = _get_target_arg_name(ctx, "step_size", "transition_steps")
    new_args.append(
      step_size_arg.with_changes(
        keyword=cst.Name(target_kw),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

  if gamma_arg:
    # Optax: 'decay_rate', Keras: 'decay_rate'
    target_kw = _get_target_arg_name(ctx, "gamma", "decay_rate")
    new_args.append(
      gamma_arg.with_changes(
        keyword=cst.Name(target_kw),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

  # 3. Inject staircase=True (Common for StepLR behavior)
  target_stair_kw = _get_target_arg_name(ctx, "staircase", "staircase")
  new_args.append(
    cst.Arg(
      value=cst.Name("True"),
      keyword=cst.Name(target_stair_kw),
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
  )

  # Construct Call
  return node.with_changes(func=_create_dotted_name(target_api), args=new_args)


def _transform_cosine_lr(node: cst.Call, ctx: HookContext, target_api: str) -> cst.Call:
  """
  Transform CosineAnnealingLR.

  Source: CosineAnnealingLR(optim, T_max, eta_min)
  Target: target_api(init_value=1.0, decay_steps=T_max, alpha=eta_min/1.0)
  """
  args = list(node.args)
  if args:
    args.pop(0)  # Remove optimizer

  new_args = []
  # 1. init_value
  init_val_kw = _get_target_arg_name(ctx, "initial_learning_rate", "init_value")
  new_args.append(
    cst.Arg(
      value=cst.Float("1.0"),
      keyword=cst.Name(init_val_kw),
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
  )

  # 2. decay_steps (was T_max)
  t_max_arg = None
  eta_min_arg = None

  for arg in args:
    kw = arg.keyword.value if arg.keyword else None
    if kw == "T_max":
      t_max_arg = arg
    elif kw == "eta_min":
      eta_min_arg = arg
    elif not kw:
      if t_max_arg is None:
        t_max_arg = arg
      elif eta_min_arg is None:
        eta_min_arg = arg

  if t_max_arg:
    # Optax: 'decay_steps'
    # Map from standard 'T_max'
    target_kw = _get_target_arg_name(ctx, "T_max", "decay_steps")

    # Ensure trailing comma consistency
    comma_val = cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")) if eta_min_arg else cst.MaybeSentinel.DEFAULT

    new_args.append(
      t_max_arg.with_changes(
        keyword=cst.Name(target_kw),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=comma_val,
      )
    )

  if eta_min_arg:
    # Optax: 'alpha' (end learning rate ratio)
    target_kw = _get_target_arg_name(ctx, "eta_min", "alpha")
    new_args.append(
      eta_min_arg.with_changes(
        keyword=cst.Name(target_kw),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.MaybeSentinel.DEFAULT,
      )
    )

  return node.with_changes(func=_create_dotted_name(target_api), args=new_args)


@register_hook("scheduler_step_noop")
def transform_scheduler_step(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Replaces ``scheduler.step()`` with a no-op value (None).
  Triggered if the scheduler step operation is wired to `scheduler_step_noop`.
  """
  # Return explicit None. In an expression statement, `None` does nothing.
  return cst.Name("None")
