"""
Script to register 'Scheduler' operations.
"""

import json
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


def update_scheduler_semantics():
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  if not sem_dir.exists():
    print("❌ Semantics directory missing.")
    return

  # 1. Update Hub (K_Optimization)
  spec_path = sem_dir / "k_optimization.json"
  if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
  else:
    spec = {}

  spec["StepLR"] = {"description": "Step learning rate scheduler.", "std_args": ["optimizer", "step_size", "gamma"]}
  spec["CosineAnnealingLR"] = {
    "description": "Cosine annealing scheduler.",
    "std_args": ["optimizer", "T_max", "eta_min"],
  }
  spec["SchedulerStep"] = {"description": "Step the scheduler.", "std_args": []}

  spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

  # 2. Update JAX Snapshot
  jax_path = snap_dir / "jax_vlatest_map.json"
  if jax_path.exists():
    snap = json.loads(jax_path.read_text(encoding="utf-8"))

    # Schedulers -> Optax with Rewire
    snap["mappings"]["StepLR"] = {"api": "optax.exponential_decay", "requires_plugin": "scheduler_rewire"}
    snap["mappings"]["CosineAnnealingLR"] = {"api": "optax.cosine_decay_schedule", "requires_plugin": "scheduler_rewire"}

    # Step method (.step())
    # We assume discovery maps "scheduler.step" to SchedulerStep
    # This requires dynamic discovery context or exact name match "step"
    snap["mappings"]["step"] = {"api": "noop", "requires_plugin": "scheduler_step_noop"}

    jax_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"✅ Updated JAX Snapshot at {jax_path.name}")

  # 3. Update Torch Snapshot
  torch_path = snap_dir / "torch_vlatest_map.json"
  if torch_path.exists():
    t_snap = json.loads(torch_path.read_text(encoding="utf-8"))

    t_snap["mappings"]["StepLR"] = {"api": "torch.optim.lr_scheduler.StepLR"}
    t_snap["mappings"]["CosineAnnealingLR"] = {"api": "torch.optim.lr_scheduler.CosineAnnealingLR"}
    t_snap["mappings"]["step"] = {"api": "torch.optim.lr_scheduler._LRScheduler.step"}  # Nominal

    torch_path.write_text(json.dumps(t_snap, indent=2), encoding="utf-8")
    print(f"✅ Updated Torch Snapshot at {torch_path.name}")


if __name__ == "__main__":
  update_scheduler_semantics()
