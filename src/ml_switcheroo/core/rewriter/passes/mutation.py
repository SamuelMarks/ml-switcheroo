"""Functional State and Mutation Desugaring Pass.

Consolidates transformation logic for:
1. In-place mutation replacement:
   - Augmented assignments: `x += y` -> `x = x + y` for functional targets.
   - Indexed assignments: `x[indices] = values` -> `x = x.at[indices].set(values)` for JAX.
   - Reverse desugaring: `x = x.at[indices].set(values)` -> `x[indices] = values` for eager frameworks.
2. PRNG key management and threading:
   - Injecting `rng = jax.random.PRNGKey(seed)` and key threading for JAX.
   - Extracting/stripping seed and key arguments when translating from JAX to eager frameworks.
"""

from typing import Dict, List, Optional, Tuple, Union
import libcst as cst

from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.context import RewriterContext


AUG_OP_MAP: Dict[type, type] = {
  cst.AddAssign: cst.Add,
  cst.SubtractAssign: cst.Subtract,
  cst.MultiplyAssign: cst.Multiply,
  cst.DivideAssign: cst.Divide,
  cst.MatrixMultiplyAssign: cst.MatrixMultiply,
  cst.FloorDivideAssign: cst.FloorDivide,
  cst.ModuloAssign: cst.Modulo,
  cst.PowerAssign: cst.Power,
  cst.BitAndAssign: cst.BitAnd,
  cst.BitOrAssign: cst.BitOr,
  cst.BitXorAssign: cst.BitXor,
  cst.LeftShiftAssign: cst.LeftShift,
  cst.RightShiftAssign: cst.RightShift,
}

STOCHASTIC_CALL_NAMES: Tuple[str, ...] = (
  "torch.randn",
  "torch.rand",
  "torch.randn_like",
  "torch.rand_like",
  "torch.randint",
  "torch.normal",
  "torch.bernoulli",
  "torch.dropout",
  "torch.nn.functional.dropout",
  "F.dropout",
  "mx.random.normal",
  "mx.random.uniform",
  "mx.random.bernoulli",
  "mlx.core.random.normal",
  "mlx.core.random.uniform",
  "keras.random.normal",
  "keras.random.uniform",
)


class FunctionalMutationPass(RewriterPass):
  """Pass responsible for desugaring mutations and managing functional PRNG state."""

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Execute functional mutation and PRNG desugaring on the module.

    Args:
        module: The source CST Module.
        context: Shared rewriter execution state.

    Returns:
        The transformed CST Module.
    """
    transformer = FunctionalMutationTransformer(context)
    return module.visit(transformer)


class FunctionalMutationTransformer(cst.CSTTransformer):
  """LibCST Transformer for mutation desugaring and PRNG state threading."""

  def __init__(self, context: RewriterContext) -> None:
    """Initialize the transformer with rewriter context.

    Args:
        context: The shared rewriter context.
    """
    self.context = context
    self._stochastic_call_count = 0
    self._has_rng_param = False

  def _is_functional_target(self) -> bool:
    """Check if the target framework requires purely functional state/mutations.

    Returns:
        True if the target framework enforces functional state.
    """
    return self.context.plugin_traits.requires_functional_state or self.context.target_fw in (
      "jax",
      "flax",
      "flax_nnx",
      "paxml",
    )

  def _requires_explicit_rng(self) -> bool:
    """Check if the target framework requires explicit PRNG keys.

    Returns:
        True if the target framework requires explicit RNG passing.
    """
    return self.context.plugin_traits.requires_explicit_rng or self.context.target_fw in (
      "jax",
      "flax",
      "flax_nnx",
      "paxml",
    )

  def _match_at_call(
    self, call: cst.Call
  ) -> Optional[Tuple[cst.BaseExpression, List[cst.SubscriptElement], str, cst.BaseExpression]]:
    """Match a `target.at[indices].method(values)` JAX call structure.

    Args:
        call: The candidate Call node.

    Returns:
        A tuple of (base_expr, slice_elements, method_name, val_expr) if matched, else None.
    """
    if not isinstance(call.func, cst.Attribute):
      return None

    method_name = call.func.attr.value
    if method_name not in ("set", "add", "multiply", "divide"):
      return None

    if not call.args:
      return None
    val_expr = call.args[0].value

    subscript = call.func.value
    if not isinstance(subscript, cst.Subscript):
      return None

    at_attr = subscript.value
    if not isinstance(at_attr, cst.Attribute) or at_attr.attr.value != "at":
      return None

    base_expr = at_attr.value
    slice_elements = list(subscript.slice)
    return base_expr, slice_elements, method_name, val_expr

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Inspect function definition for existing RNG parameters and reset counts.

    Args:
        node: The CST FunctionDef node being visited.

    Returns:
        True to continue visiting child nodes.
    """
    self._stochastic_call_count = 0
    self._has_rng_param = False

    for param in node.params.params:
      if isinstance(param.name, cst.Name) and param.name.value in ("rng", "key", "rngs"):
        self._has_rng_param = True
        break

    return True

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
    """Inject PRNGKey initialization if stochastic calls were made without RNG param.

    Args:
        original_node: The original CST FunctionDef node.
        updated_node: The transformed CST FunctionDef node.

    Returns:
        The updated CST FunctionDef node.
    """
    if self._requires_explicit_rng() and self._stochastic_call_count > 0 and not self._has_rng_param:
      stmt = cst.SimpleStatementLine(
        body=[
          cst.Assign(
            targets=[cst.AssignTarget(target=cst.Name("rng"))],
            value=cst.Call(
              func=cst.Attribute(
                value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("random")),
                attr=cst.Name("PRNGKey"),
              ),
              args=[cst.Arg(value=cst.Integer("0"))],
            ),
          )
        ]
      )

      if isinstance(updated_node.body, cst.IndentedBlock):  # pragma: no branch
        existing_stmts = list(updated_node.body.body)
        idx = 0
        if existing_stmts and isinstance(existing_stmts[0], cst.SimpleStatementLine):
          first_expr = existing_stmts[0].body[0]
          if isinstance(first_expr, cst.Expr) and isinstance(
            first_expr.value, (cst.SimpleString, cst.ConcatenatedString)
          ):
            idx = 1
        new_body = existing_stmts[:idx] + [stmt] + existing_stmts[idx:]
        updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))

    return updated_node

  def leave_AugAssign(
    self, original_node: cst.AugAssign, updated_node: cst.AugAssign
  ) -> Union[cst.AugAssign, cst.Assign]:
    """Desugar augmented assignments to functional assignments for JAX targets.

    Args:
        original_node: Original AugAssign node.
        updated_node: Updated AugAssign node.

    Returns:
        A functional cst.Assign node if target is functional, otherwise the updated node.
    """
    if not self._is_functional_target():
      return updated_node

    op_type = type(updated_node.operator)
    if op_type not in AUG_OP_MAP:
      return updated_node

    bin_op_cls = AUG_OP_MAP[op_type]

    # Special case: x[indices] += values -> x = x.at[indices].add(values)
    if isinstance(updated_node.target, cst.Subscript):
      base_var = updated_node.target.value
      if isinstance(base_var, cst.BaseAssignTargetExpression):  # pragma: no branch
        slice_elements = list(updated_node.target.slice)
        call_node = cst.Call(
          func=cst.Attribute(
            value=cst.Subscript(
              value=cst.Attribute(value=base_var, attr=cst.Name("at")),
              slice=slice_elements,
            ),
            attr=cst.Name("add"),
          ),
          args=[cst.Arg(value=updated_node.value)],
        )
        return cst.Assign(
          targets=[cst.AssignTarget(target=base_var)],
          value=call_node,
        )

    # General case: x += y -> x = x + y
    bin_op = cst.BinaryOperation(
      left=updated_node.target,
      operator=bin_op_cls(),
      right=updated_node.value,
    )
    return cst.Assign(
      targets=[cst.AssignTarget(target=updated_node.target)],
      value=bin_op,
    )

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> Union[cst.Assign, cst.AugAssign]:
    """Handle indexed assignment desugaring for JAX and reverse desugaring for eager targets.

    Args:
        original_node: Original Assign node.
        updated_node: Updated Assign node.

    Returns:
        The desugared or resugared assignment node.
    """
    # 1. Target is Functional (JAX): x[indices] = values -> x = x.at[indices].set(values)
    if self._is_functional_target():
      if len(updated_node.targets) == 1:
        target = updated_node.targets[0].target
        if isinstance(target, cst.Subscript):
          base_var = target.value
          if isinstance(base_var, cst.BaseAssignTargetExpression):  # pragma: no branch
            slice_elements = list(target.slice)
            call_node = cst.Call(
              func=cst.Attribute(
                value=cst.Subscript(
                  value=cst.Attribute(value=base_var, attr=cst.Name("at")),
                  slice=slice_elements,
                ),
                attr=cst.Name("set"),
              ),
              args=[cst.Arg(value=updated_node.value)],
            )
            return updated_node.with_changes(
              targets=[cst.AssignTarget(target=base_var)],
              value=call_node,
            )

    # 2. Target is Eager (PyTorch, MLX, Keras): Reverse desugaring
    # x = x.at[indices].set(values) -> x[indices] = values
    if not self._is_functional_target():
      if isinstance(updated_node.value, cst.Call):
        match = self._match_at_call(updated_node.value)
        if match:
          base_expr, slice_elements, method_name, val_expr = match
          if isinstance(base_expr, cst.BaseAssignTargetExpression):  # pragma: no branch
            subscript_target = cst.Subscript(value=base_expr, slice=slice_elements)
            if method_name == "set":
              return updated_node.with_changes(
                targets=[cst.AssignTarget(target=subscript_target)],
                value=val_expr,
              )
            else:
              return cst.AugAssign(
                target=subscript_target,
                operator=cst.AddAssign(),
                value=val_expr,
              )

    return updated_node

  def leave_Expr(self, original_node: cst.Expr, updated_node: cst.Expr) -> Union[cst.Expr, cst.Assign]:
    """Handle standalone expression reverse desugaring: x.at[indices].set(values) -> x[indices] = values.

    Args:
        original_node: Original Expr node.
        updated_node: Updated Expr node.

    Returns:
        The resugared assignment or the updated expression node.
    """
    if not self._is_functional_target():
      if isinstance(updated_node.value, cst.Call):
        match = self._match_at_call(updated_node.value)
        if match:
          base_expr, slice_elements, method_name, val_expr = match
          if isinstance(base_expr, cst.BaseAssignTargetExpression):  # pragma: no branch
            subscript_target = cst.Subscript(value=base_expr, slice=slice_elements)
            return cst.Assign(
              targets=[cst.AssignTarget(target=subscript_target)],
              value=val_expr,
            )
    return updated_node

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Manage stochastic calls and seed generation across functional and eager targets.

    Args:
        original_node: Original Call node.
        updated_node: Updated Call node.

    Returns:
        The transformed Call node.
    """
    # Track stochastic calls
    fn_name = ""
    if isinstance(original_node.func, cst.Attribute):
      if isinstance(original_node.func.value, cst.Name):
        fn_name = f"{original_node.func.value.value}.{original_node.func.attr.value}"
      elif isinstance(original_node.func.value, cst.Attribute) and isinstance(original_node.func.value.value, cst.Name):
        fn_name = (
          f"{original_node.func.value.value.value}.{original_node.func.value.attr.value}.{original_node.func.attr.value}"
        )
    elif isinstance(original_node.func, cst.Name):
      fn_name = original_node.func.value

    if any(fn_name.endswith(s.split(".")[-1]) for s in STOCHASTIC_CALL_NAMES):
      self._stochastic_call_count += 1

    # 1. Target is Eager (PyTorch, MLX, Keras): Convert PRNGKey to framework seed
    if not self._requires_explicit_rng():
      if fn_name in ("jax.random.PRNGKey", "PRNGKey"):
        if updated_node.args:
          seed_arg = updated_node.args[0].value
        else:
          seed_arg = cst.Integer("0")
        if self.context.target_fw == "torch":
          return cst.Call(
            func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("manual_seed")),
            args=[cst.Arg(value=seed_arg)],
          )
        elif self.context.target_fw == "mlx":
          return cst.Call(
            func=cst.Attribute(
              value=cst.Attribute(value=cst.Name("mx"), attr=cst.Name("random")),
              attr=cst.Name("seed"),
            ),
            args=[cst.Arg(value=seed_arg)],
          )
        else:
          return cst.Call(
            func=cst.Attribute(
              value=cst.Attribute(value=cst.Name("keras"), attr=cst.Name("utils")),
              attr=cst.Name("set_random_seed"),
            ),
            args=[cst.Arg(value=seed_arg)],
          )

      # Strip 'key' argument from calls when converting from JAX to eager
      cleaned_args = [
        arg for arg in updated_node.args if not (arg.keyword and arg.keyword.value in ("key", "rng", "subkey"))
      ]
      # If first argument is named 'key' or 'subkey' as positional in jax.random.*
      if (
        cleaned_args
        and fn_name.startswith("jax.random.")
        and isinstance(cleaned_args[0].value, cst.Name)
        and cleaned_args[0].value.value in ("key", "rng", "subkey")
      ):
        cleaned_args = cleaned_args[1:]

      if cleaned_args and cleaned_args[-1].comma != cst.MaybeSentinel.DEFAULT:
        cleaned_args[-1] = cleaned_args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

      updated_node = updated_node.with_changes(args=cleaned_args)

    return updated_node
