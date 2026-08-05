"""Generated MLIR Official Dialects.

This module contains automatically generated OpSchemas for all official MLIR operations.
"""

import json
import importlib.resources as resources
from typing import Dict

from ml_switcheroo.core.mlir.dialect import OpSchema


def _load_official_ops() -> Dict[str, OpSchema]:
  """Loads the official MLIR operations from JSON."""
  data_path = resources.files("ml_switcheroo.core.mlir.data").joinpath("mlir_official_ops.json")

  with data_path.open("r", encoding="utf-8") as f:
    ops_data = json.load(f)

  return {f"{dialect}.{op}": OpSchema(name=f"{dialect}.{op}") for dialect, ops in ops_data.items() for op in ops}


OFFICIAL_OPS: Dict[str, OpSchema] = _load_official_ops()
