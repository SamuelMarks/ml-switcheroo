"""
Plugin for handling Input/Output (IO) Serialization.

Delegates serialization logic to the target FrameworkAdapter to ensure logic
remains decoupled from the core transpiler.

Supported:
- `torch.save(obj, f)` rewriting.
- `torch.load(f)` rewriting.
- Arg extraction from keyword or positional parameters.
- Automatic preamble injection via Adapter request.
"""

import libcst as cst
from typing import List, Optional

from ml_switcheroo.core.hooks import register_hook, HookContext

# Fix: Import directly from base
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.utils.node_diff import capture_node_source


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


@register_hook("io_handler")
def transform_io_calls(node: cst.Call, ctx: HookContext) -> cst.Call:
  """
  Hook to rewrite save/load calls using Adapter-specific logic.

  Triggers:
      `torch.save` and `torch.load`.
  """
  target_fw = ctx.target_fw
  adapter = get_adapter(target_fw)

  # Safety: If no adapter logic is available, return original
  if not adapter:
    return node

  func_name = _get_func_name(node)
  if not func_name:
    return node

  # Determine Operation Type
  op = "save" if "save" in func_name else ("load" if "load" in func_name else None)
  if not op:
    return node

  # Extract Arguments based on 'torch' convention (source)
  # save(obj, f) / load(f)
  args = list(node.args)
  obj_arg = None
  file_arg = None

  if op == "save":
    obj_arg = _get_arg(args, 0, "obj")  # Arg 0: obj
    file_arg = _get_arg(args, 1, "f")  # Arg 1: f

    # Must have both
    if not obj_arg or not file_arg:
      return node

  elif op == "load":
    file_arg = _get_arg(args, 0, "f")  # Arg 0: f
    if not file_arg:
      return node

  # Serialize args to Python strings for the adapter
  f_str = capture_node_source(file_arg.value)
  obj_str = capture_node_source(obj_arg.value) if obj_arg else None

  # 1. Preamble Injection
  imports = adapter.get_serialization_imports()
  for imp in imports:
    ctx.inject_preamble(imp)

  # 2. Syntax Generation
  try:
    new_code = adapter.get_serialization_syntax(op, f_str, obj_str)
    if not new_code:
      return node

    # 3. Parse back to CST
    return cst.parse_expression(new_code)
  except Exception:
    return node
