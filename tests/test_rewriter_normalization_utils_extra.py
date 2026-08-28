"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments
from typing import Dict, Any, List


def test_normalization_utils_extra() -> None:
  """Docstring."""
  code: str = "f(a=1)"
  original_node: cst.Call = getattr(cst.parse_statement(code).body[0], "value")
  updated_node: cst.Call = original_node

  op_details: Dict[str, Any] = {"std_args": ["a"]}
  target_impl: Dict[str, Any] = {"arg_values": {"a": "2", "extra": 10}}

  def is_module_alias_fn(x: Any) -> bool:
    """Function doc."""
    return False

  res: List[cst.Arg] = normalize_arguments(
    original_node, updated_node, op_details, target_impl, "torch", is_module_alias_fn
  )
  assert len(res) == 2


def test_convert_value_to_cst_empty_collections() -> None:
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import convert_value_to_cst

  res_list: cst.CSTNode = convert_value_to_cst([])
  assert isinstance(res_list, cst.List) and len(res_list.elements) == 0

  res_tuple: cst.CSTNode = convert_value_to_cst(())
  assert isinstance(res_tuple, cst.Tuple) and len(res_tuple.elements) == 0

  res_dict: cst.CSTNode = convert_value_to_cst({})
  assert isinstance(res_dict, cst.Dict) and len(res_dict.elements) == 0


def test_normalize_args_implicit_receiver() -> None:
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  # receiver_injected = True
  # This happens when implicit_receiver=True and the first standard arg isn't provided.
  # We need an Attribute call.
  node: cst.Call = getattr(getattr(cst.parse_statement("obj.func(a=1)"), "body")[0], "value")
  mapping: Dict[str, Any] = {"std_args": ["x", "a"], "lib_to_std": {"a": "a"}, "implicit_receiver": True}
  # obj should become x
  target_impl: Dict[str, Any] = {"std_args": ["x", "a"]}
  res: List[cst.Arg] = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args['x'] = obj
  assert len(res) == 2

  # Now let's test where it IS provided
  node2: cst.Call = getattr(getattr(cst.parse_statement("obj.func(x=2, a=1)"), "body")[0], "value")
  res2: List[cst.Arg] = normalize_arguments(node2, node2, mapping, target_impl, "torch", lambda x: False)
  # First std arg provided, so no implicit receiver injection
  assert len(res2) == 2

  # Now implicit_receiver = False and it is an Attribute
  mapping3: Dict[str, Any] = {"std_args": ["x"], "implicit_receiver": False}
  node3: cst.Call = getattr(getattr(cst.parse_statement("obj.func(x=1)"), "body")[0], "value")
  res3: List[cst.Arg] = normalize_arguments(node3, node3, mapping3, target_impl, "torch", lambda x: False)
  assert len(res3) == 1


def test_normalize_args_positional_duplicate() -> None:
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node: cst.Call = getattr(getattr(cst.parse_statement("func(1, 2)"), "body")[0], "value")
  mapping: Dict[str, Any] = {"std_args": ["x", "x"], "implicit_receiver": False}
  target_impl: Dict[str, Any] = {"std_args": ["x"]}
  res: List[cst.Arg] = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will have 'x': 1, and the second 'x' will be ignored but pos_idx increments
  assert len(res) == 2


def test_normalize_args_empty_variadic() -> None:
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node: cst.Call = getattr(getattr(cst.parse_statement("func()"), "body")[0], "value")
  mapping: Dict[str, Any] = {"std_args": [{"name": "*args", "is_variadic": True}], "implicit_receiver": False}
  target_impl: Dict[str, Any] = {"std_args": [{"name": "args"}], "pack_to_tuple": "args", "pack_as": "Tuple"}
  res: List[cst.Arg] = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will NOT have '*args' because packing_mode was not entered
  assert len(res) == 0
