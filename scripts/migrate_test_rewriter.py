"""
Migration script to update all tests to use TestRewriter (Pipeline) instead of legacy PivotRewriter.

Steps:
1. Replaces imports of PivotRewriter from core with TestRewriter from conftest.
2. Updates `tree.visit(rewriter)` pattern to `rewriter.convert(tree)`.
"""

import os
from pathlib import Path

TARGET_DIRS = ["tests/functionality", "tests/plugins", "tests/core/rewriter", "tests/codegen"]


def migrate_file(fpath: Path):
  content = fpath.read_text(encoding="utf-8")
  original_content = content

  # 1. Update Imports
  if "from ml_switcheroo.core.rewriter import PivotRewriter" in content:
    content = content.replace(
      "from ml_switcheroo.core.rewriter import PivotRewriter", "from tests.conftest import TestRewriter as PivotRewriter"
    )
    # Note: We alias as PivotRewriter to minimize churn in constructor calls i.e. PivotRewriter(...)

  # 2. Update Visit Pattern
  # Matches: tree.visit(rewriter) -> rewriter.convert(tree)
  # Matches: module.visit(rewriter) -> rewriter.convert(module)
  # Matches: cst.parse_module(code).visit(rewriter) -> rewriter.convert(cst.parse_module(code))

  # Simple heuristic replacements for common patterns found in codebase
  # Pattern A: return tree.visit(rewriter).code
  # Replacement: return rewriter.convert(tree).code
  content = content.replace("tree.visit(rewriter)", "rewriter.convert(tree)")

  # Pattern B: new_tree = tree.visit(rewriter)
  content = content.replace("new_tree = tree.visit(rewriter)", "new_tree = rewriter.convert(tree)")

  # Pattern C: module.visit(rewriter)
  content = content.replace("module.visit(rewriter)", "rewriter.convert(module)")

  # Pattern D: cst.parse_module(code).visit(rewriter)
  content = content.replace("cst.parse_module(code).visit(rewriter)", "rewriter.convert(cst.parse_module(code))")

  # Pattern E: (Structure Pass Test) return struct_pass.transform(module, context).code
  # This pattern is correct for explicit pass tests, no change needed.

  if content != original_content:
    print(f"Migrated {fpath}")
    fpath.write_text(content, encoding="utf-8")


def main():
  root = Path(__file__).parent.parent

  for d in TARGET_DIRS:
    dir_path = root / d
    if not dir_path.exists():
      continue

    for f in dir_path.glob("*.py"):
      migrate_file(f)


if __name__ == "__main__":
  main()
