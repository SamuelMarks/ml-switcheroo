"""Tests for official MLIR dialects."""

import importlib.resources as resources
import json

from ml_switcheroo.core.mlir.official_dialects import OFFICIAL_OPS


def test_official_ops_exist() -> None:
  """Docstring."""
  assert len(OFFICIAL_OPS) > 0

  data_path = resources.files("ml_switcheroo.core.mlir.data").joinpath("mlir_official_ops.json")
  with data_path.open("r", encoding="utf-8") as f:
    expected_ops = json.load(f)

  for dialect, ops in expected_ops.items():
    for op in ops:
      op_name = f"{dialect}.{op}"
      assert op_name in OFFICIAL_OPS
