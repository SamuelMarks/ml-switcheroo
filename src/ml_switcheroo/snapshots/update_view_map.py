"""
Script to register 'View' operations with the semantics engine.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_view_semantics():
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

  spec["View"] = {
    "description": "Returns a new tensor with the same data as the self tensor but of a different shape.",
    "std_args": ["input", "shape"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Map to JAX Reshape + Plugin
    mapping = {"api": "jax.numpy.reshape", "requires_plugin": "view_semantics"}

    snap["mappings"]["View"] = mapping
    snap["mappings"]["view"] = mapping  # Method

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["View"] = {"api": "torch.Tensor.view"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_view_semantics()
