"""Automated Docstring Coverage and Quality Checker.

Verifies 100% module, class, method, and function docstring presence,
prohibits placeholder docstrings, and enforces explicit argument (`Args:`)
documentation for all non-`self`/`cls` parameters in production modules
(`src/`, `scripts/`, and `docs/`).
"""

from __future__ import annotations
import typing

import argparse
import ast
import re
import sys
from pathlib import Path

PLACEHOLDER_DOCSTRINGS: frozenset[str] = frozenset(
  {
    "docstring.",
    "module docstring.",
    "function doc.",
    "class doc.",
    "init doc.",
    "dummy.",
    "dummy method.",
    "test module.",
    "auto-generated doc.",
    "run the script.",
  }
)


def is_test_path(path: Path) -> bool:
  """Determine whether a file path belongs to the test suite.

  Args:
      path: File path to inspect.

  Returns:
      True if the path is inside a `tests` directory or starts with `test_`.
  """
  return "tests" in path.parts or path.name.startswith("test_") or path.name == "conftest.py"


def extract_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> typing.List[str]:
  """Extract all non-`self`/`cls` parameter names from a function AST node.

  Args:
      node: Function or async function AST definition node.

  Returns:
      Ordered list of parameter identifier strings.
  """
  params: typing.List[str] = [
    arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if arg.arg not in ("self", "cls")
  ]
  if node.args.vararg is not None:
    params.append(node.args.vararg.arg)
  if node.args.kwarg is not None:
    params.append(node.args.kwarg.arg)
  return params


def _extract_args_section(doc: str) -> str:
  """Extract the `Args:` or `Parameters:` block from a docstring if present.

  Args:
      doc: Full docstring text.

  Returns:
      The extracted `Args:`/`Parameters:` section text, or the full docstring if no
      section header is present.
  """
  match = re.search(
    r"(?:^|\n)\s*(?:Args|Arguments|Parameters|Params):\s*\n(.*?)(?:\n\s*(?:Returns|Yields|Raises|Notes|Examples|Attributes):|\Z)",
    doc,
    re.DOTALL,
  )
  return match.group(1) if match else doc


def is_arg_documented(doc: str, arg_name: str) -> bool:
  """Check whether a parameter name is documented within a docstring.

  Args:
      doc: Normalized docstring content.
      arg_name: Parameter name to search for.

  Returns:
      True if the parameter (or its leading-underscore-stripped form) appears in `doc`.
  """
  target = _extract_args_section(doc)
  if re.search(rf"\b{re.escape(arg_name)}\b", target):
    return True
  stripped = arg_name.lstrip("_")
  if stripped and re.search(rf"\b{re.escape(stripped)}\b", target):
    return True
  return False


def has_non_none_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
  """Check whether a function AST node has a non-`None` return annotation.

  Args:
      node: Function or async function AST definition node.

  Returns:
      True if `node.returns` is present and not `None`.
  """
  if node.returns is None:
    return False
  if isinstance(node.returns, ast.Constant) and node.returns.value is None:
    return False
  if isinstance(node.returns, ast.Name) and node.returns.id == "None":
    return False
  return True


def is_return_documented(doc: str) -> bool:
  """Check whether a docstring includes a `Returns:` or `Yields:` section.

  Args:
      doc: Normalized docstring content.

  Returns:
      True if a `Returns:`, `Return:`, or `Yields:` section is present in `doc`.
  """
  return bool(re.search(r"(?:^|\n)\s*(?:Returns|Yields|Return):", doc))


def check_file(
  path: Path, check_args: typing.Optional[bool] = None, check_returns: typing.Optional[bool] = None
) -> typing.List[str]:
  """Validate docstrings, argument documentation, and return documentation for a Python file.

  Args:
      path: Path to the Python file to inspect.
      check_args: Whether to enforce `Args:` parameter documentation. If None,
          enabled for all files (including `tests/`).
      check_returns: Whether to enforce `Returns:` documentation for functions with
          non-`None` return annotations. If None, enabled when `check_args` is enabled.

  Returns:
      List of human-readable violation strings found in the file.
  """
  violations: typing.List[str] = []
  enforce_args = True if check_args is None else check_args
  enforce_returns = enforce_args if check_returns is None else check_returns

  source = path.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(path))

  mod_doc = ast.get_docstring(tree)
  if not mod_doc:
    violations.append(f"{path}:1: Missing module docstring")
  elif mod_doc.strip().lower() in PLACEHOLDER_DOCSTRINGS:
    violations.append(f"{path}:1: Placeholder module docstring '{mod_doc.strip()}'")

  for node in ast.walk(tree):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
      doc = ast.get_docstring(node)
      kind = "class" if isinstance(node, ast.ClassDef) else "function"
      if not doc:
        violations.append(f"{path}:{node.lineno}: Missing {kind} docstring for '{node.name}'")
        continue
      if doc.strip().lower() in PLACEHOLDER_DOCSTRINGS:
        violations.append(f"{path}:{node.lineno}: Placeholder {kind} docstring for '{node.name}': '{doc.strip()}'")

      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if enforce_args:
          params = extract_parameter_names(node)
          for param in params:
            if not is_arg_documented(doc, param):
              violations.append(f"{path}:{node.lineno}: Undocumented argument '{param}' in '{node.name}'")
        if enforce_returns and has_non_none_return(node) and not is_return_documented(doc):
          violations.append(f"{path}:{node.lineno}: Missing Returns section in '{node.name}'")

  return violations


def iter_python_files(targets: typing.Iterable[Path]) -> typing.List[Path]:
  """Collect all Python files from a sequence of file or directory paths.

  Args:
      targets: Iterable of file or directory paths to scan.

  Returns:
      Sorted list of discovered `.py` file paths.
  """
  files: typing.List[Path] = []
  for target in targets:
    if target.is_file() and target.suffix == ".py":
      files.append(target)
    elif target.is_dir():
      files.extend(sorted(target.rglob("*.py")))
  return files


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
  """Run the CLI docstring checker across target paths.

  Args:
      argv: Command-line arguments (defaults to `sys.argv[1:]` when None).

  Returns:
      Exit status code (0 if all checks pass, 1 if any violations are found).
  """
  parser = argparse.ArgumentParser(description="Verify 100% docstring and argument documentation quality.")
  parser.add_argument(
    "paths",
    nargs="*",
    default=["src", "scripts", "docs", "tests"],
    help="Files or directories to check.",
  )
  args = parser.parse_args(argv)

  targets = [Path(p) for p in args.paths]
  py_files = iter_python_files(targets)

  all_violations: typing.List[str] = []
  for py_file in py_files:
    all_violations.extend(check_file(py_file))

  if all_violations:
    for violation in all_violations:
      print(violation, file=sys.stderr)
    print(file=sys.stderr)
    print(f"Found {len(all_violations)} docstring violation(s).", file=sys.stderr)
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
