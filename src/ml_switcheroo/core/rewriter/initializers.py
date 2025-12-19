"""
Initializer Rewriter Mixin.

Handles the translation of PyTorch's imperative initialization calls
(e.g., `torch.nn.init.kaiming_normal_(tensor)`) into JAX/Flax style
initializer factories (e.g., `jax.nn.initializers.he_normal()`).

Key Differences:
1. **In-place vs Factory**: Torch modifies tensors in-place. JAX inits are factories returning functions.
2. **Signature**: Torch takes `(tensor, ...)` as the first arg. JAX factories take config args `(...)`
   and return a function `f(key, shape)`.
3. **Naming**: standard variations (Kaiming->He, Xavier->Glorot).

Transformation Logic:
    Input:  `nn.init.kaiming_uniform_(self.weight, a=0)`
    Output: `jax.nn.initializers.he_uniform(a=0)`

    *Note*: This usually leaves a "dangling" factory call in the AST if it was a standalone statement.
    A subsequent pass (Parameter Decl Refactor) is responsible for moving this factory
    into the `self.param(..., kernel_init=HERE)` definition. This Mixin focuses solely on
    correctly translating the API and arguments.
"""

import libcst as cst
from typing import Optional, Union

# Static map for common renames that might not be in generic JSON maps
# or require specific handling (e.g., stripping the underscore).
INIT_NAME_MAP = {
  "kaiming_uniform_": "he_uniform",
  "kaiming_normal_": "he_normal",
  "xavier_uniform_": "glorot_uniform",
  "xavier_normal_": "glorot_normal",
  "constant_": "constant",
  "zeros_": "zeros",
  "ones_": "ones",
  "normal_": "normal",
  "uniform_": "uniform",
  "eye_": "eye",
  "orthogonal_": "orthogonal",
}


class InitializerMixin:
  """
  Mixin for PivotRewriter to handle `torch.nn.init` calls.
  """

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
    """
    Detects initialization calls and standardizes them to JAX factories.
    """
    # 1. Identify if this is a torch.nn.init call
    # We rely on string heuristics or context.
    # Since this mixin runs as part of the main rewriter, we can check basic names.

    func = updated_node.func
    is_init = False
    method_name = ""

    # Check `nn.init.foo` pattern
    if isinstance(func, cst.Attribute):
      if isinstance(func.value, cst.Attribute) and func.value.attr.value == "init":
        # e.g. torch.nn.init.method
        method_name = func.attr.value
        is_init = True
      elif isinstance(func.value, cst.Name) and func.value.value == "init":
        # e.g. init.method (from torch.nn import init)
        method_name = func.attr.value
        is_init = True

    if not is_init:
      # Pass through to other visitors/mixins
      # Note: In LibCST, if we inherit CSTTransformer, we usually call super().
      # But strictly as a Mixin generic, we just return updated_node if no match.
      return updated_node

    if method_name not in INIT_NAME_MAP:
      return updated_node

    # 2. Transform Arguments
    # Torch inits always take the tensor as the first argument.
    # JAX factories DO NOT take the tensor. They take hyperparameters.
    # We strip the first positional argument.

    args = list(updated_node.args)
    if args:
      # Heuristic: The first arg is the tensor 'tensor', 'w', etc.
      # Remove it.
      args.pop(0)

    # 3. Rename API
    target_name = INIT_NAME_MAP[method_name]

    # Build new dotted path: jax.nn.initializers.{target_name}
    # or flax.linen.initializers.{target_name} (often aliased to jax.nn.init)

    # We assume the user imports `jax.nn.initializers` or `nn` in JAX.
    # Let's generate `nn.initializers.target_name` assuming `from flax import linen as nn`.
    root = cst.Attribute(value=cst.Name("nn"), attr=cst.Name("initializers"))
    new_func = cst.Attribute(value=root, attr=cst.Name(target_name))

    return updated_node.with_changes(func=new_func, args=args)
