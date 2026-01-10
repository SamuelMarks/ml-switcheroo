"""
Plugin for translating Device Availability Checks.

This module maps framework-specific availability checks (e.g., ``torch.cuda.is_available()``)
to the target framework's equivalent by querying the active ``FrameworkAdapter``.

Decoupling:
Instead of hardcoding JAX or TensorFlow logic, this plugin delegates syntax generation
to ``adapter.get_device_check_syntax()``.
"""

import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext

# Fix: Import directly from base
from ml_switcheroo.frameworks.base import get_adapter


@register_hook("cuda_is_available")
def transform_cuda_check(node: cst.Call, ctx: HookContext) -> cst.BaseExpression:
  """
  Plugin Hook: Transforms CUDA availability check.

  Triggers:
      ``torch.cuda.is_available()`` via 'cuda_is_available' plugin key.

  Transformation:
      Input:  ``torch.cuda.is_available()``
       Output (JAX):   ``len(jax.devices('gpu')) > 0``
       Output (Keras): ``len(keras.config.list_logical_devices('GPU')) > 0``
       Output (NumPy): ``False``

  Args:
      node: The original CST Call node.
      ctx: HookContext for target framework access.

  Returns:
      A CST Expression representing the boolean check.
  """
  # 1. Retrieve Target Adapter
  target_fw = ctx.target_fw
  adapter = get_adapter(target_fw)

  if not adapter:
    # Fallback to original if adapter not found
    return node

  # 2. Get Syntax String from Adapter
  try:
    check_code = adapter.get_device_check_syntax()
  except NotImplementedError:
    return node
  except Exception:
    # Safety catch for adapter logic errors
    return node

  if not check_code:
    return node

  # 3. Parse into CST
  try:
    new_expression = cst.parse_expression(check_code)
    return new_expression
  except cst.ParserSyntaxError:
    return node
