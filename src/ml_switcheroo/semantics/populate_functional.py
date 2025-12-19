"""
Script to populate 'semantics/k_functional.json'.

This script generates the Abstract Operation definitions for functional
transformations such as `vmap`, `grad`, and `jit`. It writes these
definitions to a new specification file in the semantics directory.
"""

import json
from pathlib import Path
from typing import Dict, Any


def create_functional_spec() -> None:
  """
  Defines and writes the abstract Functional API standard.

  Operations defined:
  - `vmap`: Vectorizing map. Standard arguments: `func`, `in_axes`, `out_axes`.
  - `grad`: Gradient calculation. Standard arguments: `func`, `argnums`.
  - `value_and_grad`: Value and Gradient.
  - `jit`: Just-In-Time compilation. Standard argument: `static_argnums`.
  """
  target_path = Path("src/ml_switcheroo/semantics/k_functional.json")

  # Abstract Definitions
  # We standardize on JAX naming conventions (in_axes) as they appear closer
  # to the mathematical/abstract definition compared to Torch's 'dims'.
  definitions: Dict[str, Any] = {
    "vmap": {
      "description": "Vectorizing map. Creates a function which maps 'func' over argument axes.",
      "std_args": ["func", "in_axes", "out_axes", "randomness"],
    },
    "grad": {
      "description": "Creates a function that evaluates the gradient of 'func'.",
      "std_args": ["func", "argnums", "has_aux"],
    },
    "value_and_grad": {
      "description": "Creates a function that evaluates both 'func' and the gradient of 'func'.",
      "std_args": ["func", "argnums", "has_aux"],
    },
    "jit": {
      "description": "Compiles a function for efficient execution (JIT/Graph mode).",
      "std_args": ["func", "static_argnums"],
    },
  }

  # Ensure directory exists
  target_path.parent.mkdir(parents=True, exist_ok=True)

  print(f"writing {len(definitions)} functional ops to {target_path}")
  target_path.write_text(json.dumps(definitions, indent=2))


if __name__ == "__main__":
  create_functional_spec()
