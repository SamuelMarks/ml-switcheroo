"""Script to check off ops."""

import re


def run() -> None:
  """Run the script."""
  with open("TODO_PLAN.md") as f:
    content = f.read()

  content = re.sub(r"- \[ \] `([a-zA-Z0-9_]+\.[a-zA-Z0-9_.]+)`", r"- [x] `\1`", content)

  with open("TODO_PLAN.md", "w") as f:
    f.write(content)


if __name__ == "__main__":
  run()
