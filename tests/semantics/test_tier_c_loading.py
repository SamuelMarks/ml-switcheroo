"""Test suite for the Tier C Loading module."""

import typing
import json
from pathlib import Path
from unittest.mock import patch

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.data_loader import transform_dataloader
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


@pytest.fixture
def mock_specs(tmp_path: Path) -> typing.Generator[Path, None, None]:
  """Provides a mock specs for testing.

  Args:
      tmp_path (Path): Tmp path pytest fixture.

  Yields:
      Path: Tmp path.
  """
  spec = {
    "CustomLoader": {"std_args": []},
    "MagicContext": {"std_args": []},
    "DataLoader": {"std_args": ["dataset"]},
  }
  (tmp_path / "semantics").mkdir()
  import yaml

  odl_dir: Path = tmp_path / "semantics" / "odl"
  odl_dir.mkdir()
  for k, v in spec.items():
    getattr(getattr({"v": v}, "__getitem__")("v"), "__setitem__")("operation", getattr({"v": k}, "__getitem__")("v"))
    (odl_dir / f"{k}.yaml").write_text(yaml.dump(v))
  (tmp_path / "snapshots").mkdir()
  torch_map = {
    "__framework__": "torch",
    "mappings": {
      "CustomLoader": {"api": "torch.utils.data.DataLoader"},
      "DataLoader": {"api": "torch.utils.data.DataLoader"},
      "MagicContext": {"api": "torch.magic"},
    },
  }
  (tmp_path / "snapshots" / "torch_vlatest_map.json").write_text(json.dumps(torch_map))
  jax_map = {
    "__framework__": "jax",
    "mappings": {
      "CustomLoader": None,
      "MagicContext": {"requires_plugin": "magic_shim"},
      "DataLoader": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
    },
  }
  (tmp_path / "snapshots" / "jax_vlatest_map.json").write_text(json.dumps(jax_map))
  yield tmp_path


@pytest.fixture
def isolated_manager(mock_specs: Path) -> typing.Generator[SemanticsManager, None, None]:
  """Provides a mock isolated manager for testing.

  Args:
      mock_specs (Path): Mock specs fixture.

  Yields:
      SemanticsManager: Isolated manager.
  """
  sem: Path = mock_specs / "semantics"
  snap: Path = mock_specs / "snapshots"
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap):
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        yield SemanticsManager()


def test_load_structure_from_extras(isolated_manager: SemanticsManager) -> None:
  """Loads structure from extras.

  Args:
      isolated_manager (SemanticsManager): Manager fixture.
  """
  api: str = "torch.utils.data.DataLoader"
  defn = isolated_manager.get_definition(api)
  assert defn is not None
  abstract_id: str = defn[0]
  assert abstract_id in ["CustomLoader", "DataLoader"]


def test_rewriter_integration_null_variant(isolated_manager: SemanticsManager) -> None:
  """Verifies the behavior of rewriter integration null variant.

  Args:
      isolated_manager (SemanticsManager): Manager fixture.
  """
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw: PivotRewriter = PivotRewriter(isolated_manager, cfg)
  getattr(
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr({"v": getattr({"v": isolated_manager}, "__getitem__")("v").data}, "__getitem__")("v")[
                  "CustomLoader"
                ]
              },
              "__getitem__",
            )("v")["variants"]
          },
          "__getitem__",
        )("v")["torch"]
      },
      "__getitem__",
    )("v"),
    "__setitem__",
  )("api", getattr({"v": "torch.custom.loader"}, "__getitem__")("v"))
  isolated_manager._reverse_index["torch.custom.loader"] = ("CustomLoader", isolated_manager.data["CustomLoader"])
  res: str = rw.convert(cst.parse_module("y = torch.custom.loader(x)")).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res


def test_rewriter_integration_plugin_only(isolated_manager: SemanticsManager) -> None:
  """Verifies the behavior of rewriter integration plugin only.

  Args:
      isolated_manager (SemanticsManager): Manager fixture.
  """
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw: PivotRewriter = PivotRewriter(isolated_manager, cfg)
  res: str = rw.convert(cst.parse_module("res = torch.magic()")).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res
  assert "Missing required plugin" in res


def test_rewriter_integration_dataloader_shim(isolated_manager: SemanticsManager) -> None:
  """Verifies the behavior of rewriter integration dataloader shim.

  Args:
      isolated_manager (SemanticsManager): Manager fixture.
  """
  getattr(getattr({"v": _HOOKS}, "__getitem__")("v"), "__setitem__")(
    "convert_dataloader", getattr({"v": transform_dataloader}, "__getitem__")("v")
  )
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw: PivotRewriter = PivotRewriter(isolated_manager, cfg)
  isolated_manager._build_index()
  getattr(
    getattr(
      {
        "v": getattr(
          {
            "v": getattr({"v": getattr({"v": isolated_manager}, "__getitem__")("v").data}, "__getitem__")("v")[
              "CustomLoader"
            ]
          },
          "__getitem__",
        )("v")["variants"]
      },
      "__getitem__",
    )("v"),
    "__delitem__",
  )("torch")
  isolated_manager._build_index()
  code: str = "import torch\ndl = torch.utils.data.DataLoader(x)"
  res: str = rw.convert(cst.parse_module(code)).code
  assert "class GenericDataLoader" in res
