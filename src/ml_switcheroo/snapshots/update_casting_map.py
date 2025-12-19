"""
Script to register Type Casting operations.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_casting_semantics():
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

  # We define abstract "Cast" operations corresponding to the method calls
  cast_methods = ["float", "long", "int", "double", "half", "bool", "byte"]

  for m in cast_methods:
    # Uppercase key for Abstract ID (e.g., CastFloat)
    key = f"Cast{m.capitalize()}"
    spec[key] = {"description": f"Casts tensor to {m} type.", "std_args": []}

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # All map to the same plugin logic
    mapping = {
      "api": "astype",  # Nominal, the plugin does the work
      "requires_plugin": "type_methods",
    }

    for m in cast_methods:
      snap["mappings"][f"Cast{m.capitalize()}"] = mapping
      # Also map the lowercase method name used in discovery
      snap["mappings"][m] = mapping

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    for m in cast_methods:
      # e.g., torch.Tensor.float
      t_snap["mappings"][f"Cast{m.capitalize()}"] = {"api": f"torch.Tensor.{m}"}
      t_snap["mappings"][m] = {"api": f"torch.Tensor.{m}"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_casting_semantics()
