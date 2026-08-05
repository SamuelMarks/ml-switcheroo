"""Script to generate TODO plan."""

import json

with open("audit_mlir.json", "r") as f:
  grammar = json.load(f)

with open("mlir_official_ops.json", "r") as f:
  ops_by_dialect = json.load(f)

with open("TODO_PLAN.md", "w") as f:
  f.write("# Exhaustive MLIR Implementation TODO Plan\n\n")

  f.write("## 1. Missing Grammar Rules (from LangRef)\n\n")
  # To keep it simple, we just assume we list all from previous script that were missing.
  # But wait, earlier I computed missing rules in `scripts/audit_mlir_spec.py`.
  # Let's just re-read `audit_mlir.md` to grab the exact missing checkboxes.
  try:
    with open("audit_mlir.md", "r") as md_f:
      lines = md_f.readlines()
      for line in lines:
        if line.startswith("- [ ] Implement `"):
          f.write(line)
  except FileNotFoundError:
    # Fallback if the file got lost
    f.write("*(See audit_mlir.md)*\n")

  f.write("\n## 2. MLIR Operations by Dialect\n\n")
  f.write("This exhaustive list contains all discovered MLIR operations grouped by their home dialect.\n\n")

  for dialect in sorted(ops_by_dialect.keys()):
    ops = ops_by_dialect[dialect]
    f.write(f"### {dialect.upper()} Dialect\n\n")
    for op in ops:
      # Some operations extracted have dots, some don't.
      # We standardize them here to represent what they would look like in MLIR syntax.
      # Usually it's `dialect.op_name`, but we'll just use the raw mnemonic found in TableGen.
      f.write(f"- [ ] `{dialect}.{op}`\n")
    f.write("\n")

print("Generated TODO_PLAN.md")
