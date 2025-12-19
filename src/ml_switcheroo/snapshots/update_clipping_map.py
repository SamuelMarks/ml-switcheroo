"""
Script to register 'Gradient Clipping' operations.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_clipping_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_Optimization)
  spec_path = sem_dir / "k_optimization.json"

  # Create file if missing for this domain
  if not spec_path.exists():
    spec_path.write_text("{}", encoding="utf-8")

  spec = json.loads(spec_path.read_text(encoding="utf-8"))

  spec["ClipGradNorm"] = {
    "description": "Clips gradient norm of an iterable of parameters.",
    "std_args": ["parameters", "max_norm", "norm_type"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    snap["mappings"]["ClipGradNorm"] = {"api": "optax.clip_by_global_norm", "requires_plugin": "grad_clipper"}

    # Also map function names found in wild
    snap["mappings"]["clip_grad_norm_"] = snap["mappings"]["ClipGradNorm"]
    snap["mappings"]["clip_grad_norm"] = snap["mappings"]["ClipGradNorm"]

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["ClipGradNorm"] = {"api": "torch.nn.utils.clip_grad_norm_"}
    t_snap["mappings"]["clip_grad_norm_"] = {"api": "torch.nn.utils.clip_grad_norm_"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_clipping_semantics()
