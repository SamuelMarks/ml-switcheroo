"""
Logic for Conditional API Dispatch.

Handles evaluation of ODL Rules at runtime to switch APIs based on argument values.
"""

from typing import Any, List, Optional, Dict
import libcst as cst
from ml_switcheroo.enums import LogicOp


def evaluate_dispatch_rules(rewriter, node: cst.Call, rules: List[Any], details: Dict[str, Any]) -> Optional[str]:
  """
  Evaluates conditional dispatch rules against the current call arguments.

  Args:
      rewriter: The calling Rewriter (CallMixin) instance.
      node: The CST Call node.
      rules: List of Rule objects.
      details: Semantic definition details.

  Returns:
      Optional[str]: targeted API string if a rule matches, else None.
  """
  source_variant = details["variants"].get(rewriter.source_fw, {})
  source_arg_map = source_variant.get("args", {})
  # Note: source_arg_map usually maps std_name -> source_name? Or reverse?
  # In ODL: "args": {"axis": "dim"}. Key=Std, Val=Source.
  # So `source_arg_map.get(std)` gives us the source kwarg name.

  std_args_raw = details.get("std_args", [])
  std_args_order = []
  for item in std_args_raw:
    if isinstance(item, (list, tuple)):
      std_args_order.append(item[0])
    else:
      std_args_order.append(item)

  for rule in rules:
    # Determine Source Name for the rule's IF arg (which is standard name)
    src_arg_name = source_arg_map.get(rule.if_arg, rule.if_arg)

    # Extract Value
    val = _extract_value_from_call(rewriter, node, src_arg_name, rule.if_arg, std_args_order)

    if val is None:
      continue

    # Compare
    if _check_rule_condition(val, rule):
      return rule.use_api

  return None


def _extract_value_from_call(
  rewriter,
  node: cst.Call,
  src_name: str,
  std_name: str,
  std_order: List[str],
) -> Optional[Any]:
  """
  Extracts a literal value from a call node's arguments.
  """
  # 1. Keyword Check
  for arg in node.args:
    if arg.keyword and arg.keyword.value == src_name:
      return _node_to_literal(arg.value)

  # 2. Positional Check
  try:
    idx = std_order.index(std_name)
    is_method = isinstance(node.func, cst.Attribute) and not rewriter._is_module_alias(node.func.value)
    call_idx = idx
    if is_method and std_order[0] == "x":
      call_idx = idx - 1

    if call_idx >= 0 and call_idx < len(node.args):
      arg = node.args[call_idx]
      if not arg.keyword:
        return _node_to_literal(arg.value)
  except ValueError:
    pass

  return None


def _node_to_literal(node: cst.CSTNode) -> Any:
  """Converts CST node to Python primitive if possible."""
  if isinstance(node, cst.Integer):
    return int(node.value)
  if isinstance(node, cst.Float):
    return float(node.value)
  if isinstance(node, cst.SimpleString):
    return node.value.strip("'").strip('"')
  if isinstance(node, cst.Name):
    if node.value == "True":
      return True
    if node.value == "False":
      return False
    if node.value == "None":
      return None
  return None


def _check_rule_condition(val: Any, rule: Any) -> bool:
  """Evaluates the logic operator."""
  op = rule.op
  target = rule.is_val

  if op == LogicOp.EQ:
    return val == target
  elif op == LogicOp.NEQ:
    return val != target
  elif op == LogicOp.GT:
    return val > target
  elif op == LogicOp.LT:
    return val < target
  elif op == LogicOp.GTE:
    return val >= target
  elif op == LogicOp.LTE:
    return val <= target
  elif op == LogicOp.IN:
    return val in target
  elif op == LogicOp.NOT_IN:
    return val not in target
  else:
    return False
