"""
Script to register 'Reshape' and 'View' strategies into the Knowledge Base.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_shape_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (Abstract Spec)
  # Reshape is Array API (Math)
  spec_path = sem_dir / "k_array_api.json"

  spec = {}
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

  spec["Reshape"] = {
    "description": "Gives a new shape to an array without changing its data.",
    "std_args": ["x", "shape", "order"],
  }
  # Add alias concept
  spec["View"] = {
    "description": "Synonym for Reshape in Tensor frameworks (view logic handled by target behavior).",
    "std_args": ["x", "shape"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Spoke
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Map both to jax.numpy.reshape via Plugin
    mapping = {"api": "jax.numpy.reshape", "requires_plugin": "pack_shape_args"}

    snap["mappings"]["Reshape"] = mapping
    snap["mappings"]["View"] = mapping

    # Map specific tensor method
    # This allows PivotRewriter to find 'torch.Tensor.view' calls if mapped
    snap["mappings"]["view"] = mapping

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Spoke
  # Map 'view' to 'Reshape' concept implicitly by reverse indexing existing APIs
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    snap = json.loads(torch_path.read_text(encoding="utf-8"))

    # Explicit variants
    snap["mappings"]["Reshape"] = {"api": "torch.reshape"}
    snap["mappings"]["View"] = {"api": "torch.Tensor.view"}

    torch_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_shape_semantics()
