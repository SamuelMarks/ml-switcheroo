"""
Script to register Initializers.
While the logic is largely handled by the hardcoded mixin for specific tensor-stripping logic,
we register the APIs in the snapshots to ensure discovery scanners recognize them as 'known'
operations, even if the plugin overrides the detailed rewriting.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_init_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Hub
  spec_path = sem_dir / "k_neural_net.json"
  spec = {}
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

  inits = ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "constant", "zeros", "ones"]

  for i in inits:
    key = f"Init_{i}"
    spec[key] = {"description": f"Initializer: {i}", "std_args": ["tensor", "val"] if "constant" in i else ["tensor"]}

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    for i in inits:
      # Map underscore versions
      key = f"Init_{i}"
      api_name = f"torch.nn.init.{i}_"
      t_snap["mappings"][key] = {"api": api_name}
      # Map raw name for discovery
      t_snap["mappings"][f"{i}_"] = {"api": api_name}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot: {torch_path.name}")

  # JAX snapshot doesn't need explicit 'api' mappings here because
  # the InitializerMixin handles the specialized factory translation logic
  # instead of direct function-to-function mapping.


if __name__ == "__main__":
  update_init_semantics()
