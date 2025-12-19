"""
Script to register 'Clamp' (or Clip) operations.

Addresses the mapping between:
1. PyTorch: `torch.clamp(input, min, max)` or `torch.clip(...)`
   - keywords: `min`, `max`
2. JAX/NumPy: `jax.numpy.clip(a, a_min, a_max)`
   - keywords: `a_min`, `a_max`

This semantic configuration enables the `PivotRewriter` to automatically
rename keyword arguments during transcoding.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_clamp_semantics():
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

  spec["Clamp"] = {
    "description": "Clamps all elements in input into the range [ min, max ].",
    "std_args": ["input", "min", "max"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # JAX uses clip(a, a_min, a_max)
    mapping = {"api": "jax.numpy.clip", "args": {"min": "a_min", "max": "a_max", "input": "a"}}

    snap["mappings"]["Clamp"] = mapping
    # Map common aliases usage -> abstract ID or direct map
    snap["mappings"]["clamp"] = mapping
    snap["mappings"]["clip"] = mapping

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    # Torch uses clamp or clip
    t_snap["mappings"]["Clamp"] = {"api": "torch.clamp"}
    t_snap["mappings"]["clamp"] = {"api": "torch.clamp"}
    t_snap["mappings"]["clip"] = {"api": "torch.clip"}  # Alias

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_clamp_semantics()
