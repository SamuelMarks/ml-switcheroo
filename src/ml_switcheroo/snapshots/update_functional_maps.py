"""
Script to update Framework Snapshots with Functional Transform mappings.

Injects mappings for JAX/Torch functional transformations like `vmap`,
`grad`, and `jit` into the snapshot overlays.
"""

import json
from pathlib import Path


def update_framework_maps() -> None:
  """
  Iterates over Torch and JAX snapshots and injects functional API mappings.

  Torch Updates:
  - `vmap`: Maps `in_axes` -> `in_dims`.
  - `grad`: Maps to `torch.func.grad`.
  - `jit`: Maps to `torch.compile`.

  JAX Updates:
  - `vmap`: Maps `func` -> `fun`.
  - `grad`/`value_and_grad`: Maps `func` -> `fun`.
  - `jit`: Maps `func` -> `fun`.
  """
  snap_dir = Path("src/ml_switcheroo/snapshots")

  # --- 1. PyTorch Mappings (torch.func / torch.vmap) ---
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    torch_data = json.loads(torch_path.read_text())

    # Mappings
    torch_mappings = {
      "vmap": {
        "api": "torch.vmap",
        "args": {
          # Pivot: Standard 'in_axes' -> Torch 'in_dims'
          "in_axes": "in_dims",
          "out_axes": "out_dims",
        },
      },
      "grad": {
        "api": "torch.func.grad",
        # Torch doesn't strictly support has_aux in grad, usually requires functional wrapper
        # We map straight to func.grad
      },
      "jit": {"api": "torch.compile"},
    }

    torch_data["mappings"].update(torch_mappings)

    # Add import reference for torch.func if needed
    if "imports" not in torch_data:
      torch_data["imports"] = {}

    torch_data["imports"]["torch.func"] = {"alias": None, "root": "torch", "sub": "func"}

    torch_path.write_text(json.dumps(torch_data, indent=2))
    print(f"✅ Updated {torch_path}")

  # --- 2. JAX Mappings (Generic) ---
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    jax_data = json.loads(jax_path.read_text())

    jax_mappings = {
      "vmap": {
        "api": "jax.vmap",
        "args": {
          # Pivot: Standard 'func' -> JAX 'fun'
          "func": "fun"
        },
      },
      "grad": {"api": "jax.grad", "args": {"func": "fun"}},
      "value_and_grad": {"api": "jax.value_and_grad", "args": {"func": "fun"}},
      "jit": {"api": "jax.jit", "args": {"func": "fun"}},
    }

    jax_data["mappings"].update(jax_mappings)
    jax_path.write_text(json.dumps(jax_data, indent=2))
    print(f"✅ Updated {jax_path}")


if __name__ == "__main__":
  update_framework_maps()
