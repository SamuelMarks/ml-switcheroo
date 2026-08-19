"""Plugin for Dimension-Range Flattening.

PyTorch's `flatten(start_dim, end_dim)` collapses a range of dimensions.
Mapping strategies:
1. JAX: `jax.lax.collapse(x, start, stop)` - Most robust for dynamic shapes.
2. NumPy/Default: `x.reshape(...)` or `x.ravel()`.
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext


def _create_dotted_name(name_str: str) -> cst.BaseExpression:
  """Helper to create a CST Attribute chain from string.

  Args:
      name_str: A dot-separated string representing the target API name,
          such as "jax.lax.collapse".

  Returns:
      A LibCST expression node representing the dotted attribute chain.
  """
  parts = name_str.split(".")
  node = cst.Name(parts[0])
  for part in parts[1:]:
    node = cst.Attribute(value=node, attr=cst.Name(part))  # type: ignore
  return node


def _create_integer(val: int) -> cst.BaseExpression:
  """Creates a CST node for an integer, handling negative values."""
  if val < 0:
    return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer(str(-val)))
  return cst.Integer(str(val))


@register_hook("flatten_range")
def transform_flatten(node: cst.Call, ctx: HookContext) -> cst.Call:
  """Hook: Transforms `flatten(x, start, end)` into target-specific logic.

  This function identifies flattening operations, parses their arguments (including
  start and end dimension specifications), looks up the configured target API,
  and transforms the LibCST call into target-specific patterns (e.g., JAX collapse,
  Ravel, or Reshape with shape calculations).

  Args:
      node: The original LibCST Call node representing the flatten operation.
      ctx: The translation HookContext holding API configurations, state,
          and lookup methods.

  Returns:
      The transformed LibCST Call node or the original node if no target
      API was found or transformation is not applicable.
  """
  args = list(node.args)
  if not args:
    return node

  input_arg = args[0]
  input_val = input_arg.value

  # Default values for Torch flatten semantics
  start_dim = 0
  end_dim = -1

  # Extract positional args
  if len(args) > 1:  # pragma: no branch
    try:
      if isinstance(args[1].value, cst.Integer):
        start_dim = int(args[1].value.value)
      elif (
        isinstance(args[1].value, cst.UnaryOperation)
        and isinstance(args[1].value.operator, cst.Minus)
        and isinstance(args[1].value.expression, cst.Integer)
      ):
        start_dim = -int(args[1].value.expression.value)
    except ValueError:
      pass

  if len(args) > 2:
    try:
      if isinstance(args[2].value, cst.Integer):
        end_dim = int(args[2].value.value)
      elif (
        isinstance(args[2].value, cst.UnaryOperation)
        and isinstance(args[2].value.operator, cst.Minus)
        and isinstance(args[2].value.expression, cst.Integer)
      ):
        end_dim = -int(args[2].value.expression.value)
    except ValueError:
      pass

  # Extract keyword args
  for arg in args:
    if arg.keyword:
      if arg.keyword.value == "start_dim":
        if isinstance(arg.value, cst.Integer):
          start_dim = int(arg.value.value)
        elif (
          isinstance(arg.value, cst.UnaryOperation)
          and isinstance(arg.value.operator, cst.Minus)
          and isinstance(arg.value.expression, cst.Integer)
        ):
          start_dim = -int(arg.value.expression.value)
      if arg.keyword.value == "end_dim":
        if isinstance(arg.value, cst.Integer):
          end_dim = int(arg.value.value)
        elif (
          isinstance(arg.value, cst.UnaryOperation)
          and isinstance(arg.value.operator, cst.Minus)
          and isinstance(arg.value.expression, cst.Integer)
        ):
          end_dim = -int(arg.value.expression.value)

  # Lookup the API configured in ODL/Semantics
  target_api = None
  if ctx.current_op_id:
    target_api = ctx.lookup_api(ctx.current_op_id)

  if not target_api:
    target_api = ctx.lookup_api("flatten") or ctx.lookup_api("Flatten")

  if not target_api:
    # Fallback to internal/legacy keys just in case
    target_api = ctx.lookup_api("flatten_range") or ctx.lookup_api("flatten_full")

  if not target_api:
    return node

  # --- STRATEGY: JAX collapse ---
  # flatten(x, 1) -> collapse(x, 1, x.ndim)
  # flatten(x, 1, 2) -> collapse(x, 1, 3) (Exclusive stop)
  if "collapse" in target_api:
    new_func = _create_dotted_name(target_api)

    # Arg 1: Input (x)
    arg0 = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Arg 2: start_dim
    arg1_val = _create_integer(start_dim)
    arg1 = cst.Arg(value=arg1_val, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Arg 3: stop_dimension
    # PyTorch 'end_dim' is inclusive. JAX 'stop_dimension' is exclusive.
    # If end_dim == -1, it means "until the end", which corresponds to x.ndim in JAX.
    arg2_val: cst.BaseExpression
    if end_dim == -1:
      # Generate: x.ndim
      arg2_val = cst.Attribute(value=input_val, attr=cst.Name("ndim"))
    else:
      # Generate: end_dim + 1
      arg2_val = _create_integer(end_dim + 1)

    arg2 = cst.Arg(value=arg2_val)

    return node.with_changes(func=new_func, args=[arg0, arg1, arg2])

  # --- STRATEGY: MLX core flatten ---
  # mlx.core.flatten(x, start_axis, end_axis)
  if target_api == "mlx.core.flatten":
    new_func = _create_dotted_name(target_api)
    arg0 = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    arg1 = cst.Arg(value=_create_integer(start_dim), comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
    arg2 = cst.Arg(value=_create_integer(end_dim))
    return node.with_changes(func=new_func, args=[arg0, arg1, arg2])

  # --- STRATEGY: Callable Class (Keras / TensorFlow / Paxml) ---
  # keras.layers.Flatten()(x)
  target_variant = ctx.current_variant
  is_class = False
  if target_variant and hasattr(target_variant, "op_type") and target_variant.op_type:
    is_class = target_variant.op_type.value == "class"
  if is_class or target_api.endswith(".Flatten"):
    # Create the class instantiation: tf.keras.layers.Flatten()
    class_name = _create_dotted_name(target_api)
    instantiation = cst.Call(func=class_name, args=[])
    # Call it with the input: tf.keras.layers.Flatten()(x)
    clean_input_arg = input_arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)
    return cst.Call(func=instantiation, args=[clean_input_arg])

  # --- STRATEGY: Ravel (Full Flatten) ---
  # flatten(x) or flatten(x, 0, -1) -> ravel(x)
  if start_dim == 0 and end_dim == -1:
    if "ravel" in target_api.lower() or "flatten" in target_api.lower():
      new_func = _create_dotted_name(target_api)
      return node.with_changes(func=new_func, args=[input_arg])

  # --- STRATEGY: Reshape (Batch Preserving) ---
  # flatten(x, 1) -> reshape(x, (x.shape[0], -1))
  if start_dim == 1 and end_dim == -1:
    new_func = _create_dotted_name(target_api)

    # Construct shape tuple: (x.shape[0], -1)
    shape_attr = cst.Attribute(value=input_val, attr=cst.Name("shape"))
    batch_dim = cst.Subscript(value=shape_attr, slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))])
    neg_one = cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer("1"))

    shape_tuple = cst.Tuple(
      elements=[
        cst.Element(value=batch_dim, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))),
        cst.Element(value=neg_one),
      ]
    )

    if input_arg.comma == cst.MaybeSentinel.DEFAULT:
      input_arg = input_arg.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    new_args = [input_arg, cst.Arg(value=shape_tuple)]
    return node.with_changes(func=new_func, args=new_args)

  return node
