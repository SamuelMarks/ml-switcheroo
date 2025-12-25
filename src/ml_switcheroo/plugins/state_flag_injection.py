"""
Plugin for injecting state flags (e.g., training=True/False) based on context.

This module handles the impedance mismatch between Object-Oriented state management
(PyTorch's `model.eval()`, `model.train()`) and Functional statelessness (JAX, Keras functional).

The plugin consists of two cooperating hooks:

1.  `capture_eval_state`: Intercepts `model.eval()`/`train()` calls, records the state
    change in the `HookContext`, and removes the imperative call from the AST.
2.  `inject_training_flag`: Intercepts calls to the model (e.g. `model(x)`), checks if
    state was recorded, and injects the generic `training=...` keyword argument.

State is tracked via a metadata dictionary in `HookContext` keyed by the object name.
"""

from typing import Dict, Optional, Any
import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext

# unique key for storing state in context metadata
_PLUGIN_KEY = "state_flag_injection"


def _get_func_name(node: cst.BaseExpression) -> Optional[str]:
  """
  Refines a CST expression node into a string identifier.

  Args:
      node: CST node (Name or Attribute).

  Returns:
      String representation (e.g. "model" or "self.layer") or None.
  """
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    # We recursively flatten attributes to support 'self.layer.sublayer'
    base = _get_func_name(node.value)
    if base:
      return f"{base}.{node.attr.value}"
  return None


@register_hook("inject_training_flag")
def inject_training_flag_call(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook: Injects `training=True/False` kwargs into function calls.

  This hook is triggered on function calls (like `model(x)` or `model.forward(x)`) if
  they map to an abstract operation configured with `requires_plugin="inject_training_flag"`.

  Logic:

  1. Resolve the name of the object being called (the Receiver).
     It robustly checks both the full callable name (e.g. `self.layer` in `self.layer(x)`)
     and the parent object (e.g. `model` in `model.forward(x)`).
  2. Check `ctx.metadata` to see if `capture_eval_state` previously recorded a state.
  3. If state exists, execute the injection of the `training` argument.

  Args:
      node: The original CST Call node.
      ctx: HookContext containing global metadata state.

  Returns:
      The modified Call node with injected arguments, or the original if no state found.
  """
  store = ctx.metadata.get(_PLUGIN_KEY, {})
  if not store:
    return node

  # 1. Identify the potential state-holding object keys
  candidate_keys = []

  # Case A: Implicit Call `obj(x)` -> func is Name('obj') or Attribute('self.obj')
  full_name = _get_func_name(node.func)
  if full_name:
    candidate_keys.append(full_name)

  # Case B: Explicit Method `obj.method(x)` -> func is Attribute('obj', 'method')
  # We want to check 'obj'
  if isinstance(node.func, cst.Attribute):
    parent_name = _get_func_name(node.func.value)
    if parent_name:
      candidate_keys.append(parent_name)

  # 2. Check Store for any match (Priority to full name match)
  flags = None
  for key in candidate_keys:
    if key in store:
      flags = store[key]
      break

  if not flags:
    return node

  # 3. Inject Arguments
  new_args = list(node.args)

  for arg_name, val_node in flags.items():
    # Avoid duplication if user manually passed the argument
    if any(a.keyword and a.keyword.value == arg_name for a in new_args):
      continue

    # Prepare formatting: Ensure previous arg has a comma
    if new_args and new_args[-1].comma == cst.MaybeSentinel.DEFAULT:
      new_args[-1] = new_args[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    # Create new keyword argument
    arg = cst.Arg(
      keyword=cst.Name(arg_name),
      value=val_node,
      equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
    )
    new_args.append(arg)

  return node.with_changes(args=new_args)


@register_hook("capture_eval_state")
def capture_eval_state(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
  """
  Hook: Intercepts `eval()`/`train()` calls to track state removal.

  Action:

  1. Identifies the receiver object (`model`).
  2. Determines mode (`training=True` for .train(), `False` for .eval()).
  3. Updates `ctx.metadata` with this knowledge.
  4. Returns a No-Op node to strip the imperative call from the output code.

  Args:
      node: The call node (e.g. `model.eval()`).
      ctx: Hook Context.

  Returns:
      cst.Name("None") effectively replacing the statement with a no-op `None`,
      which is valid Python expression statement (does nothing).
  """
  # 1. Validation: Must be an attribute call (obj.method())
  if not isinstance(node.func, cst.Attribute):
    return node

  method_name = node.func.attr.value
  receiver_node = node.func.value
  obj_name = _get_func_name(receiver_node)

  if not obj_name:
    return node

  # 2. Determine State Value
  state_updates: Dict[str, Any] = {}

  if method_name == "eval":
    state_updates["training"] = cst.Name("False")

  elif method_name == "train":
    # Check args for train(mode=bool). Default is True.
    val = cst.Name("True")
    if node.args:
      # Simple heuristic: grab first arg.
      # If it's a literal 'False' or 'True', we use it.
      # For variables, we just passthrough the variable name node.
      val = node.args[0].value
    state_updates["training"] = val

  # 3. persist State in Context
  store = ctx.metadata.setdefault(_PLUGIN_KEY, {})
  if obj_name not in store:
    store[obj_name] = {}

  store[obj_name].update(state_updates)

  # 4. Strip the call (Return None)
  # This transforms `model.eval()` into `None`.
  return node.with_changes(func=cst.Name("None"), args=[])
