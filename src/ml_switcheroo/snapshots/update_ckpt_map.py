"""
Script to register 'Checkpoint Loading' operations.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_ckpt_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_IO or K_Neural)
  spec_path = sem_dir / "k_io.json"
  if not spec_path.exists():
    spec_path.write_text("{}", encoding="utf-8")

  spec = json.loads(spec_path.read_text(encoding="utf-8"))

  spec["LoadStateDict"] = {
    "description": "Copies parameters and buffers from state_dict into this module.",
    "std_args": ["state_dict", "strict"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Maps to custom KeyMapper usage
    snap["mappings"]["LoadStateDict"] = {"api": "KeyMapper.from_torch", "requires_plugin": "checkpoint_mapper"}

    # Method / Function alias mappings
    snap["mappings"]["load_state_dict"] = snap["mappings"]["LoadStateDict"]

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["LoadStateDict"] = {"api": "torch.nn.Module.load_state_dict"}
    t_snap["mappings"]["load_state_dict"] = {"api": "torch.nn.Module.load_state_dict"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_ckpt_semantics()
