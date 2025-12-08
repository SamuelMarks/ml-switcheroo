"""
Plugin for translating Device Allocation logic.

This module provides AST transformations to map PyTorch device object construction
to target framework semantics (JAX, MLX, TensorFlow).

Supported Mappings:
- JAX: `jax.devices('gpu')[0]`
- MLX: `mx.Device(mx.gpu, 0)`
- TF: `tf.device('GPU:0')`
"""

import libcst as cst
from typing import Optional, Union, Tuple

from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("device_allocator")
def transform_device_allocator(node: cst.Call, ctx: HookContext) -> cst.BaseExpression:
  """
  Plugin Hook: Transforms device construction calls.

  Triggers:
      Operations marked with `requires_plugin: "device_allocator"` (e.g., `torch.device`).

  Args:
      node: The original CST Call node.
      ctx: HookContext for config access.

  Returns:
      A CST Expression representing the target device access.
  """
  # 1. Parse Arguments to extract Type and Index
  dev_type, dev_index = _parse_device_args(node)

  # 2. Dispatch based on Target Framework
  target = ctx.target_fw

  if target == "jax":
    return _generate_jax(dev_type, dev_index)
  elif target == "mlx":
    return _generate_mlx(dev_type, dev_index)
  elif target == "tensorflow":
    return _generate_tf(dev_type, dev_index)

  return node


def _generate_jax(dev_type, dev_index):
  """Maps to jax.devices('gpu')[i]."""
  jax_backend = _map_backend_str(dev_type)

  new_func = cst.Attribute(value=cst.Name("jax"), attr=cst.Name("devices"))

  args = []
  if jax_backend:
    val_node = cst.SimpleString(f"'{jax_backend}'") if isinstance(jax_backend, str) else jax_backend
    args.append(cst.Arg(value=val_node))

  devices_call = cst.Call(func=new_func, args=args)

  # Indexing
  idx_val = dev_index if dev_index is not None else cst.Integer("0")

  return cst.Subscript(
    value=devices_call,
    slice=[cst.SubscriptElement(slice=cst.Index(value=idx_val))],
  )


def _generate_mlx(dev_type, dev_index):
  """Maps to mx.Device(mx.gpu, i)."""
  if isinstance(dev_type, str):
    attr_name = "cpu"
    s = dev_type.lower()
    if s in ["cuda", "gpu", "mps"]:
      attr_name = "gpu"
    type_node = cst.Attribute(value=cst.Name("mx"), attr=cst.Name(attr_name))
  else:
    # Fallback for dynamic variable: default to mx.cpu if unknown, or pass valid expr
    type_node = dev_type if dev_type else cst.Attribute(value=cst.Name("mx"), attr=cst.Name("cpu"))

  args = [cst.Arg(value=type_node)]
  if dev_index:
    args.append(cst.Arg(value=dev_index))

  return cst.Call(func=cst.Attribute(value=cst.Name("mx"), attr=cst.Name("Device")), args=args)


def _generate_tf(dev_type, dev_index):
  """Maps to tf.device('GPU:0')."""
  type_str = "CPU"
  idx_str = "0"

  if isinstance(dev_type, str):
    s = dev_type.lower()
    if s in ["cuda", "gpu", "mps"]:
      type_str = "GPU"

  if dev_index and isinstance(dev_index, cst.Integer):
    idx_str = dev_index.value

  # Note: Does not currently handle dynamic expression index concatenation for brevity
  device_string = f"{type_str}:{idx_str}"

  return cst.Call(
    func=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("device")),
    args=[cst.Arg(value=cst.SimpleString(f"'{device_string}'"))],
  )


def _parse_device_args(node: cst.Call) -> Tuple[Union[str, cst.BaseExpression, None], Optional[cst.BaseExpression]]:
  """
  Extracts (device_type, index) from arguments.
  Handles string parsing for 'type:index' syntax.
  """
  if not node.args:
    return None, None

  # Heuristic: Argument 0 is the device specification
  arg0 = node.args[0].value

  dev_type = arg0
  dev_index = None

  # Handle "cuda:0" string literal case
  if isinstance(arg0, cst.SimpleString):
    # Strip quotes
    raw_str = arg0.value[1:-1]
    if ":" in raw_str:
      parts = raw_str.split(":")
      raw_type = parts[0]
      try:
        raw_idx = int(parts[1])
        dev_index = cst.Integer(str(raw_idx))
      except ValueError:
        pass
      dev_type = raw_type  # Return raw string
    else:
      dev_type = raw_str

  # Handle explicit index argument (torch.device('cuda', 1))
  if len(node.args) > 1:
    dev_index = node.args[1].value

  return dev_type, dev_index


def _map_backend_str(source_type: Union[str, cst.BaseExpression, None]) -> Union[str, cst.BaseExpression, None]:
  if isinstance(source_type, str) and source_type.lower() in ("cuda", "mps"):
    return "gpu"
  return source_type
