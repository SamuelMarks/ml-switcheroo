"""
Script to register 'Squeeze' and 'Unsqueeze' operations.

This update relies on the Generic Argument Refactoring capabilities of the
Core Rewriter. By defining the argument mapping in the JSON snapshot (`dim` -> `axis`),
we avoid writing a custom plugin. The rewriter will automatically rename
keyword arguments and map positional arguments if strict signatures are known.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_squeeze_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_Array_API)
  spec_path = sem_dir / "k_array_api.json"
  spec = {}
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

  # Define abstract operations
  spec["Squeeze"] = {
    "description": "Returns a tensor with all the dimensions of input of size 1 removed.",
    "std_args": ["input", "dim"],
  }
  spec["Unsqueeze"] = {
    "description": "Returns a new tensor with a dimension of size one inserted at the specified position.",
    "std_args": ["input", "dim"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Squeeze -> jax.numpy.squeeze
    # Map 'dim' to 'axis'
    snap["mappings"]["Squeeze"] = {"api": "jax.numpy.squeeze", "args": {"dim": "axis"}}

    # Unsqueeze -> jax.numpy.expand_dims
    # Map 'dim' to 'axis'
    snap["mappings"]["Unsqueeze"] = {"api": "jax.numpy.expand_dims", "args": {"dim": "axis"}}

    # Method / Function alias mappings
    snap["mappings"]["squeeze"] = snap["mappings"]["Squeeze"]
    snap["mappings"]["unsqueeze"] = snap["mappings"]["Unsqueeze"]

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["Squeeze"] = {"api": "torch.squeeze"}
    t_snap["mappings"]["Unsqueeze"] = {"api": "torch.unsqueeze"}

    t_snap["mappings"]["squeeze"] = {"api": "torch.squeeze"}
    t_snap["mappings"]["unsqueeze"] = {"api": "torch.unsqueeze"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_squeeze_semantics()
