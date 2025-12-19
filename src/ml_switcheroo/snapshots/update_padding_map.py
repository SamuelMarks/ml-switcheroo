"""
Script to register the 'Pad' operation and 'padding_converter' plugin
into the Distributed Knowledge Base.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_padding_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print(f"❌ Semantics directory not found at {sem_dir}")
    return

  # 1. Update Abstract Spec (Hub) -> k_neural_net.json
  spec_path = sem_dir / "k_neural_net.json"

  # Load or initialize
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
  else:
    spec = {}

  # Define Abstract Operation
  spec["Pad"] = {
    "description": "Pads a tensor. Plugin handles (left, right, top, bottom) -> nested tuple conversion.",
    "std_args": ["input", "pad", "mode", "value"],
  }

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
  print(f"✅ Updated Spec: {spec_path.name}")

  # 2. Update JAX Snapshot (Spoke)
  jax_snap_path = snap_dir / "jax_vlatest_map.json"
  if jax_snap_path.exists():
    snap = json.loads(jax_snap_path.read_text(encoding="utf-8"))

    # Inject Implementation
    snap["mappings"]["Pad"] = {"api": "jax.numpy.pad", "requires_plugin": "padding_converter"}

    jax_snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Snapshot: {jax_snap_path.name}")
  else:
    print(f"⚠️ JAX snapshot not found at {jax_snap_path}, verify sync is complete.")

  # 3. Update Torch Snapshot (Source Alias)
  # Ensure reverse lookup works found `torch.nn.functional.pad`
  torch_snap_path = snap_dir / "torch_vlatest_map.json"
  if torch_snap_path.exists():
    t_snap = json.loads(torch_snap_path.read_text(encoding="utf-8"))

    # We explicitly map both the functional path and the root alias
    t_snap["mappings"]["Pad"] = {"api": "torch.nn.functional.pad"}

    torch_snap_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Snapshot: {torch_snap_path.name}")


if __name__ == "__main__":
  update_padding_semantics()
