"""Generates a plan mapping ODL operations to framework snapshots."""

import json
import yaml
from pathlib import Path
from collections import defaultdict


def main():
  """Load snapshots and ODL operations, and generate a TODO plan."""
  odl_dir = Path("src/ml_switcheroo/semantics/odl")
  snapshots_dir = Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots")
  plan_path = Path("TODO_PLAN.md")

  # Maps lowercase op name to list of (framework, api_path)
  snapshot_index = defaultdict(list)

  print("Loading snapshots...")
  for snap_file in snapshots_dir.glob("*.json"):
    if snap_file.stat().st_size < 100:
      continue  # skip empty vunknown ones
    fw_name = snap_file.name.split("_v")[0]
    try:
      with open(snap_file, "r") as f:
        data = json.load(f)
        for cat, items in data.get("categories", {}).items():
          for item in items:
            name = item.get("name")
            api_path = item.get("api_path")
            if name:
              snapshot_index[name.lower()].append((fw_name, api_path))
    except Exception as e:
      print(f"Error loading {snap_file.name}: {e}")

  print("Loading ODL operations...")
  ops = []
  for yaml_file in odl_dir.glob("*.yaml"):
    with open(yaml_file, "r") as f:
      data = yaml.safe_load(f)
      if data and isinstance(data, dict):
        op_name = data.get("operation", yaml_file.stem)
        ops.append(op_name)

  ops.sort()

  print(f"Generating TODO_PLAN.md for {len(ops)} operations...")

  with open(plan_path, "w") as f:
    f.write("# Semantic Operations TODO Plan\n\n")
    f.write(
      "This plan audits each operation currently in `src/ml_switcheroo/semantics/odl` and grounds them against the extracted ML framework snapshots.\n\n"
    )

    found_ops = []
    missing_ops = []

    for op in ops:
      # try to find match
      op_lower = op.lower()
      matches = snapshot_index.get(op_lower, [])
      if not matches and op_lower.endswith("_"):
        matches = snapshot_index.get(op_lower[:-1], [])

      if matches:
        # Deduplicate by framework
        fws = {}
        for fw, path in matches:
          if fw not in fws:
            fws[fw] = path

        fws_str = ", ".join(f"`{fw}` ({path})" for fw, path in fws.items())
        found_ops.append(f"- [ ] **`{op}`**: Auto-fill definition using extracted docs/args from: {fws_str}.")
      else:
        missing_ops.append(
          f"- [ ] **`{op}`**: Not found in framework snapshots. Will require manual mapping or synthetic implementation."
        )

    f.write("## 1. Grounded Operations (Found in Snapshots)\n\n")
    for line in found_ops:
      f.write(line + "\n")

    f.write("\n## 2. Ungrounded Operations (Missing from Snapshots)\n\n")
    for line in missing_ops:
      f.write(line + "\n")

  print("Done!")


if __name__ == "__main__":
  main()
