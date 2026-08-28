"""Test suite for the Tf Data Pipeline module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.tf_data_loader import transform_tf_dataloader

SOURCE: str = "\nimport torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\ndef create_pipeline(x, y):\n    ds = TensorDataset(x, y)\n    loader = DataLoader(ds, batch_size=64, shuffle=True)\n    return loader\n"


@pytest.fixture
def tf_semantics() -> SemanticsManager:
  """Provides a mock tf semantics for testing."""
  _HOOKS["tf_data_loader"] = transform_tf_dataloader
  mgr = SemanticsManager()
  mapping: dict[str, str] = {"api": "tf.data.Dataset", "requires_plugin": "tf_data_loader"}
  mgr.data["DataLoader"] = {"std_args": ["dataset"], "variants": {"tensorflow": mapping, "torch": {"api": "DataLoader"}}}
  mgr._reverse_index["DataLoader"] = ("DataLoader", mgr.data["DataLoader"])
  mgr._reverse_index["torch.utils.data.DataLoader"] = ("DataLoader", mgr.data["DataLoader"])
  return mgr


def test_tf_data_pipeline_conversion(tf_semantics: SemanticsManager) -> None:
  """Verifies the behavior of tf data pipeline conversion."""
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  engine = ASTEngine(semantics=tf_semantics, config=config)
  result: ConversionResult = engine.run(SOURCE)
  assert result.success
  code: str = result.code
  assert "tf.data.Dataset.from_tensor_slices" in code
  assert "shuffle" in code
  assert "1024" in code
  assert "batch(64)" in code
  assert "prefetch(tf.data.AUTOTUNE)" in code
  assert "from_tensor_slices(ds)" in code


def test_tf_data_pipeline_inline_construction(tf_semantics: SemanticsManager) -> None:
  """Verifies the behavior of tf data pipeline inline construction."""
  src_inline: str = "loader = DataLoader(TensorDataset(a, b), batch_size=32)"
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow")
  engine = ASTEngine(semantics=tf_semantics, config=config)
  result: ConversionResult = engine.run(src_inline)
  code: str = result.code
  clean_code: str = code.replace(" ", "")
  assert "from_tensor_slices((a,b))" in clean_code or "from_tensor_slices((a,b))" in clean_code
