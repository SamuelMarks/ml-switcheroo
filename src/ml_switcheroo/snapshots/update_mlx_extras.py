"""
Script to register MLX Compilation logic.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_mlx_extras():
  snap_dir = resolve_snapshots_dir()

  # 1. Update MLX Snapshot
  mlx_path = snap_dir / "mlx_vlatest_map.json"
  if mlx_path.exists():
    snap = json.loads(mlx_path.read_text(encoding="utf-8"))

    mappings = snap.setdefault("mappings", {})

    # Compile
    mappings["Compile"] = {"api": "mlx.core.compile", "requires_plugin": "mlx_compiler"}

    # Synchronize
    mappings["Synchronize"] = {
      "api": "mx.eval",  # Nominal API
      "requires_plugin": "mlx_synchronize",
    }

    mlx_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated MLX Snapshot: {mlx_path.name}")

  # 2. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["Compile"] = {"api": "torch.compile"}
    t_snap["mappings"]["Synchronize"] = {"api": "torch.cuda.synchronize"}

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot: {torch_path.name}")


if __name__ == "__main__":
  update_mlx_extras()
