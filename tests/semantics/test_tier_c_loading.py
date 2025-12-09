import json
import pytest
from unittest.mock import patch
import libcst as cst
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig


@pytest.fixture
def mock_extras_content():
  return {
    "__imports__": {},
    "__frameworks__": {},
    "CustomLoader": {
      "variants": {"torch": {"api": "torch.utils.data.DataLoader"}, "jax": None},
    },
    "MagicContext": {
      "variants": {"torch": {"api": "torch.magic"}, "jax": {"requires_plugin": "magic_shim"}},
    },
  }


@pytest.fixture
def isolated_manager(tmp_path, mock_extras_content):
  f = tmp_path / "k_framework_extras.json"
  f.write_text(json.dumps(mock_extras_content), encoding="utf-8")
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    return SemanticsManager()


def test_load_structure_from_extras(isolated_manager):
  assert isolated_manager.get_definition("torch.utils.data.DataLoader")[0] == "CustomLoader"


def test_rewriter_integration_null_variant(isolated_manager):
  # Strict mode forces explicit error on NULL mapping
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)

  res = cst.parse_module("y = torch.utils.data.DataLoader(x)").visit(rw).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res


def test_rewriter_integration_plugin_only(isolated_manager):
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  rw = PivotRewriter(isolated_manager, cfg)

  res = cst.parse_module("res = torch.magic()").visit(rw).code
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in res
  assert "Missing required plugin" in res
