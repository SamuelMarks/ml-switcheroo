"""Import generation strategies for Python backend synthesis."""

from typing import Dict, List, cast

import libcst as cst

_FRAMEWORK_IMPORT_MAP: Dict[str, List[str]] = {
  "torch": [
    "import torch",
    "import torch.nn as nn",
  ],
  "jax": [
    "from flax import nnx",
    "import jax.numpy as jnp",
  ],
  "flax": [
    "from flax import nnx",
    "import jax.numpy as jnp",
  ],
  "flax_nnx": [
    "from flax import nnx",
    "import jax.numpy as jnp",
  ],
  "mlx": [
    "import mlx.core as mx",
    "import mlx.nn as nn",
  ],
  "keras": [
    "import keras",
    "import tensorflow as tf",
  ],
  "tensorflow": [
    "import keras",
    "import tensorflow as tf",
  ],
  "paxml": [
    "import praxis.layers as pl",
    "import praxis.layers.convolutions",
    "from praxis.base_layer import BaseLayer",
  ],
}


def generate_imports(framework: str) -> List[cst.SimpleStatementLine]:
  """Generate standard top-level import CST statements for the target framework.

  Args:
      framework: Target framework identifier (e.g. 'torch', 'jax', 'mlx').

  Returns:
      List of parsed LibCST SimpleStatementLine nodes for the target framework.
  """
  stmt_strs = _FRAMEWORK_IMPORT_MAP.get(framework, [])
  return [cast(cst.SimpleStatementLine, cst.parse_statement(s)) for s in stmt_strs]
