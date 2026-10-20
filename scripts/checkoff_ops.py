"""Utility script to mark completed operator items in a markdown plan file.

Supports checking off either all backtick-quoted dotted operator identifiers
or an explicit subset passed via `--ops`.
"""

import typing
import argparse
import re
import sys
from pathlib import Path


def run(
  plan_file: Path = Path("TODO_PLAN.md"),
  ops: typing.Optional[typing.Sequence[str]] = None,
) -> int:
  """Mark matching unchecked operator items as completed (`[x]`) in `plan_file`.

  Args:
      plan_file: Path to the markdown checklist file.
      ops: Optional explicit list of dotted operator names to check off. If None,
          all unchecked dotted operator items (`- [ ] `dialect.op``) are checked off.

  Returns:
      Number of checklist replacements performed.
  """
  with open(plan_file, "r", encoding="utf-8") as f:
    content = f.read()

  if ops:
    total_count = 0
    for op in ops:
      escaped = re.escape(op)
      content, count = re.subn(rf"- \[ \] `({escaped})`", r"- [x] `\1`", content)
      total_count += count
  else:
    content, total_count = re.subn(
      r"- \[ \] `([a-zA-Z0-9_]+\.[a-zA-Z0-9_.]+)`",
      r"- [x] `\1`",
      content,
    )

  with open(plan_file, "w", encoding="utf-8") as f:
    f.write(content)

  return total_count


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
  """Command-line entrypoint for `checkoff_ops.py`.

  Args:
      argv: Optional command-line argument list.

  Returns:
      Exit status code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Check off completed operator items in a plan markdown file.")
  parser.add_argument(
    "--plan-file",
    type=Path,
    default=Path("TODO_PLAN.md"),
    help="Path to the markdown plan file.",
  )
  parser.add_argument(
    "--ops",
    nargs="*",
    default=None,
    help="Optional list of specific dotted operator names to check off.",
  )
  parsed = parser.parse_args(argv)
  run(plan_file=parsed.plan_file, ops=parsed.ops)
  return 0


if __name__ == "__main__":
  sys.exit(main())
