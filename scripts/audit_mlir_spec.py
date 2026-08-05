"""Script to audit MLIR spec."""

import urllib.request
import json
import re
from pathlib import Path

url = "https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/docs/LangRef.md"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
  with urllib.request.urlopen(req) as response:
    content = response.read().decode("utf-8")
except Exception as e:
  print("Failed to download spec:", e)
  exit(1)

grammar_rules = {}

# Extract all blocks with ::=
in_block = False
block_lines: list[str] = []
for line in content.splitlines():
  if line.startswith("```"):
    if in_block:
      # process block
      block_content = "\n".join(block_lines)
      if "::=" in block_content:
        # Basic parsing of rules
        for rule_line in block_lines:
          if "::=" in rule_line and not rule_line.strip().startswith("//"):
            parts = rule_line.split("::=", 1)
            if len(parts) == 2:
              lhs = parts[0].strip()
              rhs = parts[1].strip()
              # Strip trailing comments from rhs
              rhs = re.sub(r"//.*$", "", rhs).strip()
              if (
                lhs
                and rhs
                and lhs
                not in [
                  "alternation",
                  "sequence",
                  "repetition0",
                  "repetition1",
                  "optionality",
                  "grouping",
                  "literal",
                  "example",
                ]
              ):
                grammar_rules[lhs] = rhs
      in_block = False
      block_lines = []
    else:
      in_block = True
      block_lines = []
  elif in_block:
    block_lines.append(line)

# Clean up multiline rules that might have continuation, although this naive parser just gets the first line.
# A more advanced parser would merge lines not containing ::=, but this is a good start.

with open("audit_mlir.json", "w") as f:
  json.dump(grammar_rules, f, indent=2)

# Now read the implemented grammar in src/ml_switcheroo/core/mlir/grammar.lark
lark_path = Path("src/ml_switcheroo/core/mlir/grammar.lark")
implemented_rules = set()
if lark_path.exists():
  with open(lark_path) as f:
    lark_content = f.read()
  for line in lark_content.splitlines():
    # Match lark rules like "rule: ..." or "RULE: ..."
    m = re.match(r"^([a-zA-Z_0-9]+)\s*:", line)
    if m:
      name = m.group(1).lower()
      implemented_rules.add(name)
      implemented_rules.add(name.replace("_", "-"))

# Some manual mapping of Lark tokens to MLIR LangRef terms
manual_mapping = {
  "string-literal": "string",
  "integer-literal": "number",
  "float-literal": "number",
  "bare-id": "identifier",
  "value-id": "val_id",
  "caret-id": "block_label",
  "symbol-ref-id": "sym_id",
  "type": "type",
  "region": "regions",
  "attribute-value": "attribute",
}

for mlir_term, lark_term in manual_mapping.items():
  if lark_term in implemented_rules:
    implemented_rules.add(mlir_term)
    implemented_rules.add(mlir_term.replace("-", "_"))

missing_rules = []
implemented_matches = []
for rule in grammar_rules.keys():
  rule_lower = rule.lower()
  # Normalize rules to ignore '-' vs '_'
  normalized = rule_lower.replace("-", "_")
  if rule_lower in implemented_rules or normalized in implemented_rules:
    implemented_matches.append(rule)
  else:
    missing_rules.append(rule)

with open("audit_mlir.md", "w") as f:
  f.write("# MLIR LangRef Missing Grammar Rules\n\n")
  for rule in sorted(missing_rules):
    f.write(f"- [ ] Implement `{rule}` (Spec: `{grammar_rules[rule]}`)\n")

print(f"Found {len(grammar_rules)} total MLIR grammar rules.")
print(f"Implemented (fuzzy match): {len(implemented_matches)}")
print(f"Missing: {len(missing_rules)}")
