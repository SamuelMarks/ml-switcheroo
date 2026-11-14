"""Automated Docstring Coverage and Quality Checker.

Verifies 100% module, class, method, and function docstring presence,
prohibits placeholder/broken synthetic docstrings, enforces explicit `Args:`
parameter declarations (with stale-parameter detection), distinguishes `Yields:`
from `Returns:` on generators, and enforces `Raises:` and public `Attributes:`
documentation across production modules (`src/`, `scripts/`, and `docs/`).
"""

import argparse
import ast
import re
import sys
import typing
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
    "test docstring.",
    "auto-generated doc.",
    "run the script.",
    "execute implementation detail.",
    "computed test result or fixture value.",
    "test argument or fixture.",
    "converts .",
  }
)

PLACEHOLDER_SUBSTRINGS: tuple[str, ...] = (
  "execute implementation detail.",
  "computed test result or fixture value.",
  "test argument or fixture.",
  "converts .",
  "resulting `none` from",
  "mock implementation of  ",
)

_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
  re.compile(r"(?:^|\n)\s*[A-Za-z]+s\s+\.", re.IGNORECASE),
  re.compile(r"^execute\s+.+\s+helper\s+logic\.$", re.IGNORECASE),
  re.compile(r"^helper\s+to\b", re.IGNORECASE),
  re.compile(r"^test\s+helper\s+class\s+\w+\.$", re.IGNORECASE),
  re.compile(r"^test\s+stub\s+for\b", re.IGNORECASE),
  re.compile(r"^verif(?:y|ies)\s+(?:the\s+)?behavior\s+of\b", re.IGNORECASE),
  re.compile(r"^verify\s+[\w\s`']+\s+behavior\.$", re.IGNORECASE),
  re.compile(r":\s*the\s+[\w\s]+\s+\([^)\n]+\)\s+for\s+[\w\s]+\.", re.IGNORECASE),
)

_NL = chr(10)
_SECTION_BOUNDARY = rf"(?:{_NL}\s*(?:Args|Arguments|Parameters|Params|Returns|Return|Yields|Yield|Raises|Raise|Notes|Examples|Attributes):|\Z)"


def _match_placeholder_pattern(doc: str, first_line: str) -> str | None:
  """Match `first_line` or full `doc` against `_PLACEHOLDER_PATTERNS`.

  Args:
      doc: Full raw docstring content.
      first_line: First non-empty summary line of `doc`.

  Returns:
      Matched synthetic docstring fragment if found, otherwise `None`.
  """
  for pattern in _PLACEHOLDER_PATTERNS:
    if first_line and pattern.search(first_line):
      return first_line
  for pattern in _PLACEHOLDER_PATTERNS:
    match = pattern.search(doc)
    if match:
      return match.group(0).strip()
  return None


def _find_placeholder(doc: str) -> str | None:
  """Return the matched placeholder string if `doc` contains a placeholder docstring.

  Args:
      doc: Raw docstring text to inspect.

  Returns:
      The matched placeholder phrase if found, otherwise `None`.
  """
  normalized = doc.strip().lower()
  if normalized in PLACEHOLDER_DOCSTRINGS:
    return doc.strip()
  lines = [line.strip() for line in doc.strip().splitlines() if line.strip()]
  first_line = lines[0] if lines else ""
  if first_line.lower() in PLACEHOLDER_DOCSTRINGS:
    return first_line
  for sub in PLACEHOLDER_SUBSTRINGS:
    if sub in normalized:
      return sub
  return _match_placeholder_pattern(doc, first_line)


def is_test_path(path: Path) -> bool:
  """Determine whether a file path belongs to the test suite.

  Args:
      path: File path to inspect.

  Returns:
      True if the path is inside a `tests` directory or starts with `test_`.
  """
  return "tests" in path.parts or path.name.startswith("test_") or path.name == "conftest.py"


def extract_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
  """Extract all non-`self`/`cls` parameter names from a function AST node.

  Args:
      node: Function or async function AST definition node.

  Returns:
      Ordered list of parameter identifier strings.
  """
  params: list[str] = [
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
      The extracted `Args:`/`Parameters:` section text, or `""` when absent.
  """
  match = re.search(
    rf"(?:^|{_NL})[ \t]*(?:Args|Arguments|Parameters|Params):[ \t]*(?:{_NL}(.*?))?{_SECTION_BOUNDARY}",
    doc,
    re.DOTALL,
  )
  return (match.group(1) or "") if match else ""


def is_arg_documented(doc: str, arg_name: str) -> bool:
  """Check whether a parameter has an explicit `arg_name:` declaration in `Args:`.

  Args:
      doc: Normalized docstring content.
      arg_name: Parameter name to search for.

  Returns:
      True if `arg_name` (or its leading-underscore-stripped form) is declared in `Args:`.
  """
  target = _extract_args_section(doc)
  if not target:
    return False
  candidates = [re.escape(arg_name)]
  stripped = arg_name.lstrip("_")
  if stripped and stripped != arg_name:
    candidates.append(re.escape(stripped))
  pattern = rf"(?:^|{_NL})\s*\*{{0,2}}(?:{'|'.join(candidates)})\s*(?:\([^:{_NL}]*\))?\s*:"
  return bool(re.search(pattern, target))


def find_stale_args(doc: str, param_names: typing.Sequence[str]) -> list[str]:
  """Identify documented `Args:` parameter entries that do not exist in `param_names`.

  Args:
      doc: Full docstring content.
      param_names: Actual parameter names from the function signature.

  Returns:
      Ordered list of stale parameter names declared in `Args:`.
  """
  target = _extract_args_section(doc)
  if not target:
    return []
  non_empty = [line for line in target.splitlines() if line.strip()]
  if not non_empty:
    return []
  min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
  allowed = set(param_names) | {p.lstrip("_") for p in param_names if p.lstrip("_")} | {"self", "cls"}
  stale: list[str] = []
  entry_pat = rf"^\s*\*{{0,2}}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^:{_NL}]*\))?\s*:"
  for line in non_empty:
    if len(line) - len(line.lstrip()) == min_indent:
      match = re.match(entry_pat, line)
      if match and match.group(1) not in allowed:
        stale.append(match.group(1))
  return stale


def is_generator_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
  """Determine whether `node` directly contains `yield` or `yield from` expressions.

  Args:
      node: Function or async function AST definition node.

  Returns:
      True if `node` is a generator function (ignoring nested scopes).
  """
  stack: list[ast.AST] = list(node.body)
  while stack:
    current = stack.pop()
    if isinstance(current, (ast.Yield, ast.YieldFrom)):
      return True
    if not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
      stack.extend(ast.iter_child_nodes(current))
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
  return not (isinstance(node.returns, ast.Name) and node.returns.id == "None")


def is_return_documented(doc: str) -> bool:
  """Check whether a docstring includes a `Returns:` section.

  Args:
      doc: Normalized docstring content.

  Returns:
      True if a `Returns:` or `Return:` section is present in `doc`.
  """
  return bool(re.search(rf"(?:^|{_NL})\s*(?:Returns|Return):", doc))


def is_yield_documented(doc: str) -> bool:
  """Check whether a docstring includes a `Yields:` section.

  Args:
      doc: Normalized docstring content.

  Returns:
      True if a `Yields:` or `Yield:` section is present in `doc`.
  """
  return bool(re.search(rf"(?:^|{_NL})\s*(?:Yields|Yield):", doc))


def has_uncaught_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
  """Check whether `node` contains an uncaught `raise` statement in its direct scope.

  Ignores raises inside `try:` bodies that attach `except` handlers (unless re-raised
  inside an `except` or `finally` block).

  Args:
      node: Function or async function AST definition node.

  Returns:
      True if an uncaught `raise` statement exists in `node`.
  """
  stack: list[ast.AST] = list(node.body)
  while stack:
    current = stack.pop()
    if isinstance(current, ast.Raise):
      return True
    if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
      continue
    if isinstance(current, ast.Try):
      if not current.handlers:
        stack.extend(current.body)
      for handler in current.handlers:
        stack.extend(handler.body)
      stack.extend(current.orelse)
      stack.extend(current.finalbody)
    else:
      stack.extend(ast.iter_child_nodes(current))
  return False


def is_raises_documented(doc: str) -> bool:
  """Check whether a docstring includes a `Raises:` section.

  Args:
      doc: Normalized docstring content.

  Returns:
      True if a `Raises:` or `Raise:` section is present in `doc`.
  """
  return bool(re.search(rf"(?:^|{_NL})\s*(?:Raises|Raise):", doc))


def find_undocumented_class_attributes(node: ast.ClassDef, doc: str) -> list[str]:
  """Find public annotated class attributes lacking `Attributes:`, inline docs, or `Field(description=...)`.

  Args:
      node: Class AST definition node.
      doc: Class docstring text.

  Returns:
      Ordered list of undocumented public attribute names on `node`.
  """
  attr_match = re.search(rf"(?:^|{_NL})\s*Attributes:\s*{_NL}(.*?){_SECTION_BOUNDARY}", doc, re.DOTALL)
  attr_block = attr_match.group(1) if attr_match else ""

  undocumented: list[str] = []
  for idx, stmt in enumerate(node.body):
    if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
      continue
    attr_name = stmt.target.id
    if attr_name.startswith("_") or attr_name == "model_config":
      continue
    has_inline = False
    if idx + 1 < len(node.body):
      next_stmt = node.body[idx + 1]
      has_inline = (
        isinstance(next_stmt, ast.Expr)
        and isinstance(next_stmt.value, ast.Constant)
        and isinstance(next_stmt.value.value, str)
      )
    has_field_desc = isinstance(stmt.value, ast.Call) and any(kw.arg == "description" for kw in stmt.value.keywords)
    in_attr_section = bool(
      attr_block and re.search(rf"(?:^|{_NL})\s*{re.escape(attr_name)}\s*(?:\([^:{_NL}]*\))?\s*:", attr_block)
    )
    if not (has_inline or has_field_desc or in_attr_section):
      undocumented.append(attr_name)
  return undocumented


def _check_function_node(
  path: Path,
  node: ast.FunctionDef | ast.AsyncFunctionDef,
  doc: str,
  *,
  enforce_args: bool,
  enforce_returns: bool,
  enforce_raises: bool,
) -> list[str]:
  """Validate parameter, return/yield, and raises sections for a single function node.

  Args:
      path: Source file path.
      node: Function or async function AST node.
      doc: Extracted docstring for `node`.
      enforce_args: Whether to check `Args:` completeness and staleness.
      enforce_returns: Whether to check `Returns:` / `Yields:` presence.
      enforce_raises: Whether to check `Raises:` presence.

  Returns:
      List of violation messages for `node`.
  """
  violations: list[str] = []
  if enforce_args:
    params = extract_parameter_names(node)
    violations.extend(
      f"{path}:{node.lineno}: Undocumented argument '{param}' in '{node.name}'"
      for param in params
      if not is_arg_documented(doc, param)
    )
    violations.extend(
      f"{path}:{node.lineno}: Stale documented argument '{stale}' in '{node.name}'"
      for stale in find_stale_args(doc, params)
    )

  if enforce_returns and has_non_none_return(node):
    if is_generator_function(node):
      if not is_yield_documented(doc):
        violations.append(f"{path}:{node.lineno}: Missing Yields section in generator '{node.name}'")
    elif not is_return_documented(doc):
      violations.append(f"{path}:{node.lineno}: Missing Returns section in '{node.name}'")

  if enforce_raises and has_uncaught_raise(node) and not is_raises_documented(doc):
    violations.append(f"{path}:{node.lineno}: Missing Raises section in '{node.name}'")

  return violations


def check_file(
  path: Path,
  check_args: bool | None = None,
  check_returns: bool | None = None,
  *,
  check_raises: bool | None = None,
  check_attributes: bool | None = None,
) -> list[str]:
  """Validate docstrings, arguments, returns/yields, raises, and class attributes for a file.

  Args:
      path: Path to the Python file to inspect.
      check_args: Whether to enforce `Args:` parameter documentation (default: True).
      check_returns: Whether to enforce `Returns:` / `Yields:` documentation (default: matches `check_args`).
      check_raises: Whether to enforce `Raises:` documentation (default: True on non-test files).
      check_attributes: Whether to enforce public class `Attributes:` (default: True on non-test files).

  Returns:
      List of human-readable violation strings found in the file.
  """
  violations: list[str] = []
  is_prod = not is_test_path(path)
  enforce_args = True if check_args is None else check_args
  enforce_returns = enforce_args if check_returns is None else check_returns
  enforce_raises = is_prod if check_raises is None else check_raises
  enforce_attributes = is_prod if check_attributes is None else check_attributes

  source = path.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(path))

  mod_doc = ast.get_docstring(tree)
  if not mod_doc:
    violations.append(f"{path}:1: Missing module docstring")
  else:
    mod_placeholder = _find_placeholder(mod_doc)
    if mod_placeholder is not None:
      violations.append(f"{path}:1: Placeholder module docstring '{mod_placeholder}'")

  for node in ast.walk(tree):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
      doc = ast.get_docstring(node)
      kind = "class" if isinstance(node, ast.ClassDef) else "function"
      if not doc:
        violations.append(f"{path}:{node.lineno}: Missing {kind} docstring for '{node.name}'")
        continue
      node_placeholder = _find_placeholder(doc)
      if node_placeholder is not None:
        violations.append(f"{path}:{node.lineno}: Placeholder {kind} docstring for '{node.name}': '{node_placeholder}'")

      if isinstance(node, ast.ClassDef):
        if enforce_attributes:
          violations.extend(
            f"{path}:{node.lineno}: Undocumented public attribute '{attr_name}' in class '{node.name}'"
            for attr_name in find_undocumented_class_attributes(node, doc)
          )
      else:
        violations.extend(
          _check_function_node(
            path,
            node,
            doc,
            enforce_args=enforce_args,
            enforce_returns=enforce_returns,
            enforce_raises=enforce_raises,
          )
        )

  return violations


def iter_python_files(targets: typing.Iterable[Path]) -> list[Path]:
  """Collect all Python files from a sequence of file or directory paths.

  Args:
      targets: Iterable of file or directory paths to scan.

  Returns:
      Sorted list of discovered `.py` file paths.
  """
  files: list[Path] = []
  for target in targets:
    if target.is_file() and target.suffix == ".py":
      files.append(target)
    elif target.is_dir():
      files.extend(sorted(target.rglob("*.py")))
  return files


def main(argv: typing.Sequence[str] | None = None) -> int:
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

  all_violations: list[str] = []
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
