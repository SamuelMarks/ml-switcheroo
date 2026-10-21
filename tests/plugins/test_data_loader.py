"""Test suite for the Data Loader module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.data_loader import transform_dataloader
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Execute rewriter factory helper logic.

  Returns:
      Computed test result or fixture value.
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "convert_dataloader", getattr({"v": transform_dataloader}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  dl_def = {
    "variants": {
      "torch": {"api": "TorchShim", "requires_plugin": "convert_dataloader"},
      "jax": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
      "custom": {"api": "CustomShim", "requires_plugin": "convert_dataloader"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: dl_def["variants"].get(fw)
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  @register_framework("custom")
  class CustomAdapter:
    """Test helper class CustomAdapter."""

    pass

  def create(target: str) -> PivotRewriter:
    """Creates .

    Args:
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_blind_execution(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of blind execution.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("torch")
  code: str = "\ndef load_data():\n    loader = torch.utils.data.DataLoader(dataset)\n"
  res: str = rewrite_code(rw, code)
  assert "GenericDataLoader" in res
  assert "class GenericDataLoader" in res


def test_jax_shim_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of JAX shim injection.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "\ndef load_data():\n    loader = torch.utils.data.DataLoader(dataset, batch_size=32)\n"
  res: str = rewrite_code(rw, code)
  assert "class GenericDataLoader" in res
  assert "def __iter__(self):" in res
  clean_res: str = res.replace(" = ", "=")
  assert "GenericDataLoader(dataset" in clean_res
  assert "batch_size=32" in clean_res


def test_dataloader_arg_extraction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify dataloader arg extraction behavior.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "\ndef train():\n    dl = DataLoader(my_ds, batch_size=64, shuffle=True)\n"
  res: str = rewrite_code(rw, code)
  clean: str = res.replace(" ", "")
  assert "GenericDataLoader(my_ds" in clean
  assert "batch_size=64" in clean
  assert "shuffle=True" in clean


def test_dataloader_idempotent_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of dataloader idempotent injection.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "\ndef run():\n    dl1 = DataLoader(d1)\n    dl2 = DataLoader(d2)\n"
  res: str = rewrite_code(rw, code)
  assert res.count("class GenericDataLoader") == 1
  assert res.count("GenericDataLoader(") == 2


def test_custom_target_execution(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of custom target execution.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rw: PivotRewriter = rewriter_factory("custom")
  code: str = "dl = DataLoader(ds)"
  res: str = rewrite_code(rw, code)
  assert "GenericDataLoader" in res
