"""Test suite for rewriter calls utils coverage."""

import libcst as cst

from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply


def test_is_functional_apply_branches() -> None:
  """Verify is functional apply branches behavior."""
  assert not is_functional_apply(getattr(getattr(cst.parse_statement("f()"), "body")[0], "value"), None)

  assert is_functional_apply(getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value"), "apply")

  assert not is_functional_apply(getattr(getattr(cst.parse_statement("obj.other()"), "body")[0], "value"), "apply")

  assert not is_functional_apply(getattr(getattr(cst.parse_statement("apply()"), "body")[0], "value"), "apply")


def test_is_super_call_other_call() -> None:
  """Verify is super call other call behavior."""
  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.utils import is_super_call

  node = cst.parse_expression("foo().method()")
  assert isinstance(node, cst.Call)
  assert not is_super_call(node)
  node_name = cst.parse_expression("other()")
  assert isinstance(node_name, cst.Call)
  assert not is_super_call(node_name)
  node_subscript = cst.parse_expression("funcs[0]()")
  assert isinstance(node_subscript, cst.Call)
  assert not is_super_call(node_subscript)


def test_inject_permute_call_empty_indices() -> None:
  """Verify inject permute call empty indices behavior."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.utils import inject_permute_call

  node: cst.BaseExpression = cst.parse_expression("x")
  semantics: MagicMock = MagicMock()
  fw_config = {"transpose_api": "torch.transpose", "transpose_pack_kw": "dim"}
  semantics.get_framework_config.return_value = fw_config
  semantics.resolve_variant.return_value = {"api": "torch.transpose", "pack_to_tuple": "dim"}

  indices: tuple[int, ...] = ()
  res = inject_permute_call(node, indices, semantics, "torch")
  assert isinstance(res, cst.Call)
  # should have an empty tuple
  assert isinstance(res.args[1].value, cst.Tuple)
  assert len(res.args[1].value.elements) == 0
