"""
Integration Tests for TensorFlow Native Data Pipeline.
"""

import pytest
import ast
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS
from ml_switcheroo.plugins.tf_data_loader import transform_tf_dataloader

SOURCE = """
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_pipeline(x, y):
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    return loader
"""


@pytest.fixture
def tf_semantics():
  _HOOKS["tf_data_loader"] = transform_tf_dataloader
  mgr = SemanticsManager()

  mapping = {"api": "tf.data.Dataset", "requires_plugin": "tf_data_loader"}
  mgr.data["DataLoader"] = {"std_args": ["dataset"], "variants": {"tensorflow": mapping, "torch": {"api": "DataLoader"}}}

  mgr._reverse_index["DataLoader"] = ("DataLoader", mgr.data["DataLoader"])
  mgr._reverse_index["torch.utils.data.DataLoader"] = ("DataLoader", mgr.data["DataLoader"])

  return mgr


def test_tf_data_pipeline_conversion(tf_semantics):
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  engine = ASTEngine(semantics=tf_semantics, config=config)
  result = engine.run(SOURCE)

  assert result.success
  code = result.code

  assert "tf.data.Dataset.from_tensor_slices" in code
  # Relaxed whitespace check for shuffle
  assert "shuffle" in code
  assert "1024" in code
  assert "batch(64)" in code
  assert "prefetch(tf.data.AUTOTUNE)" in code
  assert "from_tensor_slices(ds)" in code


def test_tf_data_pipeline_inline_construction(tf_semantics):
  src_inline = "loader = DataLoader(TensorDataset(a, b), batch_size=32)"
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow")
  engine = ASTEngine(semantics=tf_semantics, config=config)
  result = engine.run(src_inline)
  code = result.code

  clean_code = code.replace(" ", "")
  # Robust check against spacing in (a,b) vs (a, b)
  assert "from_tensor_slices((a,b))" in clean_code or "from_tensor_slices((a,b))" in clean_code
