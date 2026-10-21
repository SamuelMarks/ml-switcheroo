"""Generated MLIR Official Dialects.

This module contains automatically generated OpSchemas for all official MLIR operations.
"""

import importlib.resources as resources
import json

from ml_switcheroo.core.mlir.dialect import OpSchema


def _load_official_ops() -> dict[str, OpSchema]:
  """Load the official MLIR operations from JSON.

  Returns:
      Dict[str, OpSchema]: Mapping of qualified operation names to OpSchema instances.
  """
  data_path = resources.files("ml_switcheroo.core.mlir.data").joinpath("mlir_official_ops.json")

  with data_path.open("r", encoding="utf-8") as f:
    ops_data = json.load(f)

  return {f"{dialect}.{op}": OpSchema(name=f"{dialect}.{op}") for dialect, ops in ops_data.items() for op in ops}


OFFICIAL_OPS: dict[str, OpSchema] = _load_official_ops()
