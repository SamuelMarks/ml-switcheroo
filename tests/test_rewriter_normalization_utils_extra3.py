"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments
from typing import Dict, Any, List


def test_normalize_arguments_full() -> None:
  """Test element."""
  original: cst.Call = cst.Call(
    func=cst.Name("foo"),
    args=[
      cst.Arg(value=cst.Name("val1")),
      cst.Arg(keyword=cst.Name("k1"), value=cst.Name("v1")),
      cst.Arg(keyword=cst.Name("k_extra"), value=cst.Name("v_extra")),
    ],
  )

  config: Dict[str, Any] = {
    "signature": {"args": [{"name": "arg1", "default": "def1"}, {"name": "arg2"}, ("arg3", "int"), "arg4"]},
    "library_to_std_args": {"k1": "arg2"},
    "target": {
      "arg_values": {"arg1": "new_def1", "arg2": {"v1": "target_v1"}},
      "kwargs_map": {"k_extra": None, "arg3": "new_arg3"},
      "inject_args": {"injected1": "inj_val"},
    },
  }

  normalized: List[cst.Arg] = normalize_arguments(
    original, original, config, target_impl=config["target"], source_fw="torch", is_module_alias_fn=lambda x: False
  )
  assert len(normalized) > 0
