"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments


def test_normalization_utils_extra():
  """Docstring."""
  code = "f(a=1)"
  original_node = cst.parse_statement(code).body[0].value
  updated_node = original_node

  op_details = {"std_args": ["a"]}
  target_impl = {"arg_values": {"a": "2", "extra": 10}}

  def is_module_alias_fn(x):
    """Function doc."""
    return False

  res = normalize_arguments(original_node, updated_node, op_details, target_impl, "torch", is_module_alias_fn)
  assert len(res) == 2


def test_convert_value_to_cst_empty_collections():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import convert_value_to_cst

  res_list = convert_value_to_cst([])
  assert isinstance(res_list, cst.List) and len(res_list.elements) == 0

  res_tuple = convert_value_to_cst(())
  assert isinstance(res_tuple, cst.Tuple) and len(res_tuple.elements) == 0

  res_dict = convert_value_to_cst({})
  assert isinstance(res_dict, cst.Dict) and len(res_dict.elements) == 0


def test_normalize_args_implicit_receiver():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  # receiver_injected = True
  # This happens when implicit_receiver=True and the first standard arg isn't provided.
  # We need an Attribute call.
  node = cst.parse_expression("obj.func(a=1)")
  mapping = {"std_args": ["x", "a"], "lib_to_std": {"a": "a"}, "implicit_receiver": True}
  # obj should become x
  target_impl = {"std_args": ["x", "a"]}
  res = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args['x'] = obj
  assert len(res) == 2

  # Now let's test where it IS provided
  node2 = cst.parse_expression("obj.func(x=2, a=1)")
  res2 = normalize_arguments(node2, node2, mapping, target_impl, "torch", lambda x: False)
  # First std arg provided, so no implicit receiver injection
  assert len(res2) == 2

  # Now implicit_receiver = False and it is an Attribute
  mapping3 = {"std_args": ["x"], "implicit_receiver": False}
  node3 = cst.parse_expression("obj.func(x=1)")
  res3 = normalize_arguments(node3, node3, mapping3, target_impl, "torch", lambda x: False)
  assert len(res3) == 1


def test_normalize_args_positional_duplicate():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node = cst.parse_expression("func(1, 2)")
  mapping = {"std_args": ["x", "x"], "implicit_receiver": False}
  target_impl = {"std_args": ["x"]}
  res = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will have 'x': 1, and the second 'x' will be ignored but pos_idx increments
  assert len(res) == 2


def test_normalize_args_empty_variadic():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments

  node = cst.parse_expression("func()")
  mapping = {"std_args": [{"name": "*args", "is_variadic": True}], "implicit_receiver": False}
  target_impl = {"std_args": [{"name": "args"}], "pack_to_tuple": "args", "pack_as": "Tuple"}
  res = normalize_arguments(node, node, mapping, target_impl, "torch", lambda x: False)
  # found_args will NOT have '*args' because packing_mode was not entered
  assert len(res) == 0
