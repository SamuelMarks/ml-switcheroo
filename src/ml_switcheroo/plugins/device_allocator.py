"""
Plugin for translating Device Allocation logic.

Delegates syntax generation to the target FrameworkAdapter to ensure logic
remains decoupled from the core transpiler.

Supported:
- Extracts device type and index from `torch.device(...)`.
- Handles `cuda:0` string parsing.
- Calls `adapter.get_device_syntax()` for target code generation.
"""

import libcst as cst
from typing import Optional, Tuple, Union

from ml_switcheroo.core.hooks import register_hook, HookContext

# Fix: Import directly from base, avoiding frameworks.__init__ trigger
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.utils.node_diff import capture_node_source


@register_hook("device_allocator")
def transform_device_allocator(node: cst.Call, ctx: HookContext) -> cst.BaseExpression:
  """
  Plugin Hook: Transforms device construction calls via Adapter delegation.

  Triggers:
      Operations marked with `requires_plugin: "device_allocator"` (e.g., `torch.device`).

  Args:
      node: The original CST Call node.
      ctx: HookContext for target framework access.

  Returns:
      A CST Expression representing the target device access.
  """
  # 1. Parse Arguments to extract Type and Index CST Nodes
  dev_type_node, dev_index_node = _parse_device_args(node)

  # 2. Convert Nodes to Source Strings
  # We pass python source code strings to the adapter to keep adapter unaware of LibCST
  s_type = capture_node_source(dev_type_node) if dev_type_node else "'cpu'"
  s_index = capture_node_source(dev_index_node) if dev_index_node else None

  # 3. Retrieve Target Adapter
  target_fw = ctx.target_fw
  adapter = get_adapter(target_fw)

  if not adapter:
    # Fallback to original node if adapter not found (safety)
    return node

  # 4. Generate Target Syntax
  try:
    new_code = adapter.get_device_syntax(s_type, s_index)
  except Exception:
    # If adapter doesn't implement or errors, return original
    return node

  # 5. Parse back to CST
  try:
    return cst.parse_expression(new_code)
  except cst.ParserSyntaxError:
    return node


def _parse_device_args(node: cst.Call) -> Tuple[Optional[cst.BaseExpression], Optional[cst.BaseExpression]]:
  """
  Extracts (device_type_node, index_node) from `torch.device` call arguments.

  Handles:
  - `torch.device('cuda')`
  - `torch.device('cuda', 0)`
  - `torch.device('cuda:0')` -> splits literal string into type 'cuda' and index '0' nodes.
  """
  if not node.args:
    return None, None

  # Heuristic: Argument 0 is the device specification
  arg0 = node.args[0].value

  dev_type_node = arg0
  dev_index_node = None

  # Handle "cuda:0" string literal case decomposition
  if isinstance(arg0, cst.SimpleString):
    # Strip quotes for inspection, but reconstruct valid nodes
    raw_quote = arg0.value[0]
    raw_str = arg0.value[1:-1]

    if ":" in raw_str:
      parts = raw_str.split(":")
      raw_type = parts[0]

      try:
        raw_idx = parts[1]
        # Verify it's an int index before splitting
        int(raw_idx)

        # Reconstruct nodes
        dev_type_node = cst.SimpleString(f"{raw_quote}{raw_type}{raw_quote}")
        dev_index_node = cst.Integer(raw_idx)
      except ValueError:
        pass  # Not a simple int index, keep original string

  # Handle explicit index argument (torch.device('cuda', 1))
  if len(node.args) > 1:
    dev_index_node = node.args[1].value

  return dev_type_node, dev_index_node
