"""
Logic for Conditional API Dispatch.
"""

from typing import Any, List, Optional, Dict
import libcst as cst
from ml_switcheroo.enums import LogicOp


def evaluate_dispatch_rules(rewriter, node: cst.Call, rules: List[Any], details: Dict[str, Any]) -> Optional[str]:
  """
  Evaluates conditional dispatch rules against the current call arguments.
  """
  source_variant = details["variants"].get(rewriter.source_fw, {})
  source_arg_map = source_variant.get("args", {})

  std_args_raw = details.get("std_args", [])
  std_args_order = []
  for item in std_args_raw:
    if isinstance(item, (list, tuple)):
      std_args_order.append(item[0])
    elif isinstance(item, dict):
      name = item.get("name")
      if name:
        std_args_order.append(name)
    else:
      std_args_order.append(item)

  for rule in rules:
    src_arg_name = source_arg_map.get(rule.if_arg, rule.if_arg)
    arg_node = _extract_argument_node(rewriter, node, src_arg_name, rule.if_arg, std_args_order)

    if arg_node is None:
      continue

    if _check_rule_condition(arg_node, rule):
      return rule.use_api

  return None


def _extract_argument_node(
  rewriter,
  node: cst.Call,
  src_name: str,
  std_name: str,
  std_order: List[str],
) -> Optional[cst.CSTNode]:
  for arg in node.args:
    if arg.keyword and arg.keyword.value == src_name:
      return arg.value

  try:
    idx = std_order.index(std_name)
    is_module = False
    if hasattr(rewriter, "_is_module_alias"):
      is_module = rewriter._is_module_alias(node.func.value) if isinstance(node.func, cst.Attribute) else False

    is_method = isinstance(node.func, cst.Attribute) and not is_module
    call_idx = idx
    if is_method and std_order and std_order[0] == "x":
      call_idx = idx - 1

    if call_idx >= 0 and call_idx < len(node.args):
      arg = node.args[call_idx]
      if not arg.keyword:
        return arg.value
  except ValueError:
    pass

  return None


def _node_to_literal(node: cst.CSTNode) -> Any:
  if isinstance(node, cst.Integer):
    try:
      return int(node.value)
    except ValueError:
      return None
  if isinstance(node, cst.Float):
    try:
      return float(node.value)
    except ValueError:
      return None
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


def _check_rule_condition(node: cst.CSTNode, rule: Any) -> bool:
  op = rule.op

  if op == LogicOp.IS_TYPE:
    expected_type = str(rule.is_val).lower()
    if isinstance(node, cst.Integer):
      return expected_type == "int"
    if isinstance(node, cst.Float):
      return expected_type == "float"
    if isinstance(node, cst.SimpleString):
      return expected_type == "str"
    if isinstance(node, (cst.List, cst.Tuple)):
      return expected_type in ["list", "tuple", "sequence"]
    if isinstance(node, cst.Dict):
      return expected_type == "dict"
    if isinstance(node, cst.Name) and node.value in ["True", "False"]:
      return expected_type == "bool"
    return False

  val = _node_to_literal(node)
  if val is None:
    return False

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

  return False
