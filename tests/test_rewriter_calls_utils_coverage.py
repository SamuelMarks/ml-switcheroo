"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply


def test_is_functional_apply_branches():
  """Docstring."""
  # 41 -> 42
  assert not is_functional_apply(cst.parse_statement("f()").body[0].value, None)

  # 44 -> 45 -> 46
  assert is_functional_apply(cst.parse_statement("obj.apply()").body[0].value, "apply")

  # 44 -> 45 -> 47
  assert not is_functional_apply(cst.parse_statement("obj.other()").body[0].value, "apply")

  # 44 -> 47
  assert not is_functional_apply(cst.parse_statement("apply()").body[0].value, "apply")


def test_is_super_call_other_call():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.calls.utils import is_super_call

  node = cst.parse_statement("foo().method()").body[0].value
  assert not is_super_call(node)


def test_inject_permute_call_empty_indices():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.calls.utils import inject_permute_call
  from unittest.mock import MagicMock

  node = cst.parse_expression("x")
  semantics = MagicMock()
  fw_config = {"transpose_api": "torch.transpose", "transpose_pack_kw": "dim"}
  semantics.get_framework_config.return_value = fw_config
  semantics.resolve_variant.return_value = {"api": "torch.transpose", "pack_to_tuple": "dim"}

  res = inject_permute_call(node, [], semantics, "torch")
  # should have an empty tuple
  assert isinstance(res.args[1].value, cst.Tuple)
  assert len(res.args[1].value.elements) == 0
