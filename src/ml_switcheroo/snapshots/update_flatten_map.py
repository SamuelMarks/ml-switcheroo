"""
Script to register 'Flatten' operations into the Knowledge Base.

Registers:
1. `Flatten` (Generic)
2. `flatten_range` (Specific variant mapped to reshape)
3. `flatten_full` (Specific variant mapped to ravel)
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_flatten_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print(f"❌ Semantics directory not found at {sem_dir}")
    return

  # 1. Update Hub (K_Neural or K_Array - Flatten is borderline, usually Array)
  # We'll put it in K_Array_API as it is tensor manipulation
  spec_path = sem_dir / "k_array_api.json"

  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    # Define base abstractions used by the plugin for lookups
    spec["Flatten"] = {
      "description": "Flattens a tensor. Torch supports ranges, JAX supports ravel/reshape.",
      "std_args": ["input", "start_dim", "end_dim"],
    }
    spec["flatten_range"] = {
      "description": "Abstract handle for batch-preserving flatten logic.",
      "std_args": ["input", "shape"],
    }
    spec["flatten_full"] = {"description": "Abstract handle for full flattening.", "std_args": ["input"]}

    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"✅ Updated Spec: {spec_path.name}")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Main mapping triggers the plugin
    snap["mappings"]["Flatten"] = {"api": "jax.numpy.reshape", "requires_plugin": "flatten_range"}

    # Helper mappings used by the plugin context lookup
    snap["mappings"]["flatten_range"] = {"api": "jax.numpy.reshape"}
    snap["mappings"]["flatten_full"] = {"api": "jax.numpy.ravel"}

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    # Map torch.flatten to the Flatten abstract
    t_snap["mappings"]["Flatten"] = {"api": "torch.flatten"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_flatten_semantics()
