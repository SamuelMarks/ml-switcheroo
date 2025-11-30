"""
Plugin for injecting state flags (e.g., training=False).

This module handles the translation of imperative state mutations (like `model.eval()`)
into functional arguments passed to subsequent calls.

Logic:
1. Detects `model.eval()` or `model.train()` statements in the current scope.
2. Tracks the state of specific variables (e.g. `model` is in evaluation mode)
   inside the `HookContext` metadata.
3. Intercepts calls to those variables (e.g. `model(x)`).
4. Injects a keyword argument (e.g. `model(x, training=False)`).
5. Strips the original mutation statement to avoid runtime errors in stateless frameworks.
"""

from typing import Dict, Optional, Any
import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext

_PLUGIN_KEY = "state_flag_injection"


@register_hook("inject_training_flag")
def inject_training_flag_call(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Rewrites a call site to inject state flags if the object was previously mutated.

  Triggers:
      Operations marked with `requires_plugin: "inject_training_flag"`.
      Typically attached to the `__call__` or `forward` definition of a neural module.

  Args:
      node: The function call (e.g. `model(x)`).
      ctx: Hook Context containing metadata.

  Returns:
      Modified call with injected kwargs (e.g. `model(x, training=False)`).
  """
  # 1. Identify the object being called
  obj_name = _get_func_name(node.func)
  if not obj_name:
    return node

  # 2. Check if this object has accumulated state in the current scope metadata
  store = ctx.metadata.setdefault(_PLUGIN_KEY, {})
  if obj_name not in store:
    return node

  flags = store[obj_name]  # e.g. {'training': cst.Name("False")}
  new_args = list(node.args)

  # 3. Inject Flags
  for arg_name, val_node in flags.items():
    # Avoid duplication if user manually passed it
    if any(a.keyword and a.keyword.value == arg_name for a in new_args):
      continue

    # Add `training=False`
    arg = cst.Arg(
      keyword=cst.Name(arg_name),
      value=val_node,
      equal=cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
      ),
    )
    new_args.append(arg)

  return node.with_changes(args=new_args)


@register_hook("capture_eval_state")
def capture_eval_state(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Intercepts `model.eval()` or `model.train()` to track state in Context.

  Triggers:
      Operations mapping `torch.nn.Module.eval` etc.

  Action:
      1. Identifies the receiver object (`model`).
      2. Updates `ctx.metadata` with `training=False/True`.
      3. Returns a no-op/null call to strip the operation from output code.

  Args:
      node: The call node (e.g. `model.eval()`).
      ctx: Hook Context providing metadata storage.

  Returns:
      A no-op CST node (so the statement effectively disappears or becomes harmless).
  """
  # 1. Identify Receiver
  if not isinstance(node.func, cst.Attribute):
    return node

  receiver_node = node.func.value
  method_name = node.func.attr.value
  obj_name = _extract_name(receiver_node)

  if not obj_name:
    return node

  # 2. Determine State
  state_args: Dict[str, Any] = {}
  if method_name == "eval":
    state_args["training"] = cst.Name("False")
  elif method_name == "train":
    # Check args for train(mode=bool)
    val = cst.Name("True")
    if node.args:
      val = node.args[0].value
    state_args["training"] = val

  # 3. Store State in Context Metadata
  store = ctx.metadata.setdefault(_PLUGIN_KEY, {})
  if obj_name not in store:
    store[obj_name] = {}

  # Typed ignore logic: store is Dict[str, Any], sub-dict is Dict[str, CSTNode]
  store[obj_name].update(state_args)

  # 4. Return No-Op
  # We replace `model.eval()` with `None` so the line becomes `None`,
  # which is eliminated by Python compilers or harmless.
  return node.with_changes(func=cst.Name("None"), args=[])


def _get_func_name(node: cst.BaseExpression) -> Optional[str]:
  """Helper to get string name of a function call target."""
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    # Only support simple attributes for state tracking (self.layer)
    base = _extract_name(node.value)
    if base:
      return f"{base}.{node.attr.value}"
  return None


def _extract_name(node: cst.BaseExpression) -> Optional[str]:
  """Recursively resolves name."""
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    base = _extract_name(node.value)
    if base:
      return f"{base}.{node.attr.value}"
  return None
