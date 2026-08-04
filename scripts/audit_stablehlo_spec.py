"""Audit the StableHLO specification to find missing operations.

This script downloads the StableHLO spec and compares it with
implemented operations to identify and output missing ones.
"""

import urllib.request
import json
import re
from pathlib import Path

url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
  with urllib.request.urlopen(req) as response:
    content = response.read().decode("utf-8")
except Exception as e:
  print("Failed to download spec:", e)
  exit(1)

ops = []
for line in content.splitlines():
  m = re.match(r"^###\s+([a-z0-9_]+)$", line)
  if m:
    ops.append(m.group(1))

implemented = set()
definitions_dir = Path("src/ml_switcheroo/frameworks/definitions")
if (definitions_dir / "stablehlo.json").exists():
  with open(definitions_dir / "stablehlo.json") as f:
    data = json.load(f)
    implemented = set(data.keys())

missing = sorted(list(set(ops) - implemented))
bt = chr(96)

out = {"official_ops": sorted(ops), "implemented": sorted(list(implemented)), "missing": missing}

with open("audit_stablehlo.json", "w") as f:
  json.dump(out, f, indent=2)

with open("audit_stablehlo.md", "w") as f:
  f.write("# StableHLO Missing Operations\n\n")
  for op in missing:
    f.write("- [ ] Implement " + bt + op + bt + "\n")

print(f"Found {len(ops)} total ops.")
print(f"Implemented: {len(implemented)}")
print(f"Missing: {len(missing)}")
