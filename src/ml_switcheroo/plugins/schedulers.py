"""
Plugin for Learning Rate Scheduler Rewiring.

Addresses the architectural difference between:

1. PyTorch: `scheduler = StepLR(optimizer, step_size=30)` (Stateful object wrapping optimizer).
2. JAX/Optax: `schedule_fn = optax.piecewise_constant(...)` (Pure function passed to optimizer).

**Transformation**

1.  **Instantiation**:

    -   Detects specific Scheduler constructors.
    -   Removes the `optimizer` argument (Arg 0).
    -   Maps hyperparameters to Optax equivalents.
    -   Changes the API call to an Optax schedule factory.
    -   Injects `init_value=1.0` (as partial schedule) or tries to preserve semantics.

2.  **Stepping**:

    -   Detects `scheduler.step()`.
    -   Since JAX schedulers are integrated into the gradient transform chain and stepped automatically via state,
        manual stepping is redundant.
    -   Replaces `scheduler.step()` with a no-op placeholder `None`.

Supported Mappings:

-   `StepLR` -> `optax.exponential_decay(staircase=True)`
-   `CosineAnnealingLR` -> `optax.cosine_decay_schedule`
"""

import libcst as cst
from typing import Union

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))
  return node


@register_hook("scheduler_rewire")
def transform_scheduler_init(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Transforms Scheduler instantiation.
  """
  if ctx.target_fw.lower() not in ["jax", "flax", "flax_nnx"]:
    return node

  # We rely on the operation ID to know which scheduler specific logic to apply
  op_id = ctx.current_op_id or ""

  if "StepLR" in op_id:
    return _transform_step_lr(node)
  elif "CosineAnnealingLR" in op_id:
    return _transform_cosine_lr(node)

  return node


def _transform_step_lr(node: cst.Call) -> cst.Call:
  """
  StepLR(optim, step_size, gamma) -> exponential_decay(1.0, step_size, gamma, staircase=True)
  """
  # Parse Args
  args = list(node.args)
  # Remove Optimizer (Arg 0)
  if args:
    args.pop(0)

  new_args = []

  # 1. init_value (Inject 1.0 placeholder, user must adjust)
  new_args.append(
    cst.Arg(
      value=cst.Float("1.0"),
      keyword=cst.Name("init_value"),
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
    new_args.append(
      step_size_arg.with_changes(
        keyword=cst.Name("transition_steps"),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

  if gamma_arg:
    new_args.append(
      gamma_arg.with_changes(
        keyword=cst.Name("decay_rate"),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

  # 3. Inject staircase=True
  new_args.append(
    cst.Arg(
      value=cst.Name("True"),
      keyword=cst.Name("staircase"),
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
  )

  # Construct Call
  return node.with_changes(func=_create_dotted_name("optax.exponential_decay"), args=new_args)


def _transform_cosine_lr(node: cst.Call) -> cst.Call:
  """
  CosineAnnealingLR(optim, T_max, eta_min) -> cosine_decay_schedule(1.0, T_max, alpha=eta_min/1.0)
  """
  args = list(node.args)
  if args:
    args.pop(0)  # Remove optimizer

  new_args = []
  # 1. init_value
  new_args.append(
    cst.Arg(
      value=cst.Float("1.0"),
      keyword=cst.Name("init_value"),
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
    # Ensure trailing comma consistency
    comma_val = cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")) if eta_min_arg else cst.MaybeSentinel.DEFAULT

    new_args.append(
      t_max_arg.with_changes(
        keyword=cst.Name("decay_steps"),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=comma_val,
      )
    )

  if eta_min_arg:
    new_args.append(
      eta_min_arg.with_changes(
        keyword=cst.Name("alpha"),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.MaybeSentinel.DEFAULT,
      )
    )

  return node.with_changes(func=_create_dotted_name("optax.cosine_decay_schedule"), args=new_args)


@register_hook("scheduler_step_noop")
def transform_scheduler_step(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Replaces ``scheduler.step()`` with a no-op value (None).
  """
  if ctx.target_fw.lower() not in ["jax", "flax", "flax_nnx"]:
    return node

  # Return explicit None. In an expression statement, `None` does nothing.
  return cst.Name("None")
