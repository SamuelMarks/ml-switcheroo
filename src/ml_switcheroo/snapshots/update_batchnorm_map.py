"""
Script to register 'BatchNorm' operations with state unwrapping.
Handles 1d, 2d, and 3d variants mapping to generic Flax BatchNorm.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_batchnorm_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_Neural)
  spec_path = sem_dir / "k_neural_net.json"
  spec = {}
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

  bn_desc = "Batch Normalization. Plugin handles stateful return unwrapping for JAX."
  bn_args = ["num_features", "eps", "momentum", "affine", "track_running_stats"]

  # Register abstract ops for different dimensionalities
  # We unify them conceptually, but keep distinct keys for reverse lookup
  spec["BatchNorm"] = {"description": bn_desc, "std_args": bn_args}
  spec["BatchNorm1d"] = {"description": bn_desc, "std_args": bn_args}
  spec["BatchNorm2d"] = {"description": bn_desc, "std_args": bn_args}
  spec["BatchNorm3d"] = {"description": bn_desc, "std_args": bn_args}

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot (Spoke)
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # All convert to generic flax.nnx.BatchNorm
    # Arg mapping: eps -> epsilon, num_features -> num_features
    # momentum is handled differently (decay = 1 - momentum),
    # but for structural parity we map momentum -> momentum and assume runtime shim handles math if strictly needed.

    mapping = {"api": "flax.nnx.BatchNorm", "args": {"eps": "epsilon"}, "requires_plugin": "batch_norm_unwrap"}

    snap["mappings"]["BatchNorm"] = mapping
    snap["mappings"]["BatchNorm1d"] = mapping
    snap["mappings"]["BatchNorm2d"] = mapping
    snap["mappings"]["BatchNorm3d"] = mapping

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["BatchNorm"] = {"api": "torch.nn.BatchNorm2d"}  # Default assumption
    t_snap["mappings"]["BatchNorm1d"] = {"api": "torch.nn.BatchNorm1d"}
    t_snap["mappings"]["BatchNorm2d"] = {"api": "torch.nn.BatchNorm2d"}
    t_snap["mappings"]["BatchNorm3d"] = {"api": "torch.nn.BatchNorm3d"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_batchnorm_semantics()
