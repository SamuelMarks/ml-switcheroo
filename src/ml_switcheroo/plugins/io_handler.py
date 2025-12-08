"""
Plugin for handling Input/Output (IO) Serialization.

This module maps PyTorch serialization (`torch.save`, `torch.load`) to
target-specific I/O routines.

Supported Targets:
- JAX: `orbax.checkpoint.PyTreeCheckpointer().save/restore`
- Numpy: `np.save(file, arr)` / `np.load(file)`
- TensorFlow (basic): `tf.io.write_file(filename, contents)` / `tf.io.read_file`
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext


def _get_func_name(node: cst.Call) -> Optional[str]:
  """Helper to get function name from Call node (Attribute or Name)."""
  if isinstance(node.func, cst.Name):
    return node.func.value
  if isinstance(node.func, cst.Attribute):
    return node.func.attr.value
  return None


def _get_arg(args: List[cst.Arg], index: int, name: str) -> Optional[cst.Arg]:
  """Retrieves argument by position or keyword."""
  for arg in args:
    if arg.keyword and arg.keyword.value == name:
      return arg
  if index < len(args):
    candidate = args[index]
    if candidate.keyword is None:
      return candidate
  return None


# --- JAX / Orbax Logic ---


def _transform_save_orbax(node: cst.Call, checkpointer: cst.Call) -> cst.Call:
  args = list(node.args)
  obj_arg = _get_arg(args, 0, "obj")
  f_arg = _get_arg(args, 1, "f")

  if not obj_arg or not f_arg:
    return node

  new_args = [
    cst.Arg(
      keyword=cst.Name("directory"),
      value=f_arg.value,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    ),
    cst.Arg(
      keyword=cst.Name("item"),
      value=obj_arg.value,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(" "), whitespace_after=cst.SimpleWhitespace("")),
    ),
  ]
  return cst.Call(func=cst.Attribute(value=checkpointer, attr=cst.Name("save")), args=new_args)


def _transform_load_orbax(node: cst.Call, checkpointer: cst.Call) -> cst.Call:
  args = list(node.args)
  f_arg = _get_arg(args, 0, "f")
  if not f_arg:
    return node
  return cst.Call(func=cst.Attribute(value=checkpointer, attr=cst.Name("restore")), args=[cst.Arg(value=f_arg.value)])


# --- Numpy Logic ---


def _transform_save_numpy(node: cst.Call) -> cst.Call:
  # torch.save(obj, f) -> np.save(file=f, arr=obj)
  args = list(node.args)
  obj_arg = _get_arg(args, 0, "obj")
  f_arg = _get_arg(args, 1, "f")

  if not obj_arg or not f_arg:
    return node

  new_args = [
    cst.Arg(value=f_arg.value),  # Positional 'file'
    cst.Arg(value=obj_arg.value),  # Positional 'arr'
  ]

  return cst.Call(func=cst.Attribute(value=cst.Name("np"), attr=cst.Name("save")), args=new_args)


def _transform_load_numpy(node: cst.Call) -> cst.Call:
  # torch.load(f) -> np.load(f)
  args = list(node.args)
  f_arg = _get_arg(args, 0, "f")
  if not f_arg:
    return node

  return cst.Call(func=cst.Attribute(value=cst.Name("np"), attr=cst.Name("load")), args=[cst.Arg(value=f_arg.value)])


# --- TensorFlow Logic ---


def _transform_save_tf(node: cst.Call) -> cst.Call:
  # Tries to map to tf.io.write_file(filename, contents)
  # Assumes user serializes bytes/string, which isn't always true for `torch.save` (objects).
  # A safer fallback for heavy objects is `tf.saved_model.save`.
  # For this implementation scope, we'll map to `tf.io.write_file` assuming string/bytes content.
  args = list(node.args)
  obj_arg = _get_arg(args, 0, "obj")
  f_arg = _get_arg(args, 1, "f")

  if not obj_arg or not f_arg:
    return node

  new_args = [
    cst.Arg(value=f_arg.value),  # filename
    cst.Arg(value=obj_arg.value),  # contents
  ]
  return cst.Call(
    func=cst.Attribute(value=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("io")), attr=cst.Name("write_file")),
    args=new_args,
  )


def _transform_load_tf(node: cst.Call) -> cst.Call:
  args = list(node.args)
  f_arg = _get_arg(args, 0, "f")
  if not f_arg:
    return node

  return cst.Call(
    func=cst.Attribute(value=cst.Attribute(value=cst.Name("tf"), attr=cst.Name("io")), attr=cst.Name("read_file")),
    args=[cst.Arg(value=f_arg.value)],
  )


@register_hook("io_handler")
def transform_io_calls(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook to rewrite save/load calls to framework-specific IO routines.

  Triggers:
      `torch.save` and `torch.load`.
  """
  func_name = _get_func_name(node)
  if not func_name:
    return node

  target = ctx.target_fw

  # --- JAX Strategy ---
  if target == "jax":
    ctx.inject_preamble("import orbax.checkpoint")
    checkpointer = cst.Call(
      func=cst.Attribute(
        value=cst.Attribute(value=cst.Name("orbax"), attr=cst.Name("checkpoint")), attr=cst.Name("PyTreeCheckpointer")
      ),
      args=[],
    )

    if "save" in func_name:
      return _transform_save_orbax(node, checkpointer)
    elif "load" in func_name:
      return _transform_load_orbax(node, checkpointer)

  # --- Numpy Strategy ---
  elif target == "numpy":
    # Preamble not strictly needed if import fixer works, but good practice
    ctx.inject_preamble("import numpy as np")
    if "save" in func_name:
      return _transform_save_numpy(node)
    elif "load" in func_name:
      return _transform_load_numpy(node)

  # --- TensorFlow Strategy ---
  elif target == "tensorflow":
    ctx.inject_preamble("import tensorflow as tf")
    if "save" in func_name:
      return _transform_save_tf(node)
    elif "load" in func_name:
      return _transform_load_tf(node)

  return node
