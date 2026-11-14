"""Per-Function, Statement, and Branch Coverage Verifier for ml-switcheroo.

Inspects a `coverage.py` JSON report (or generates one directly from a `.coverage`
data file) and verifies that every tracked file and every individual function or
method in `src/`, `scripts/`, and `docs/` achieves 100% statement and 100% branch
coverage.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import coverage
from pydantic import BaseModel, ConfigDict


class CoverageSummaryModel(BaseModel):
  """Summary metrics for a single file or function in a coverage JSON report.

  Attributes:
      num_statements: Total executable statements in scope.
      missing_lines: Count of unexecuted statements.
      num_branches: Total conditional branches in scope.
      missing_branches: Count of unexecuted conditional branches.
  """

  model_config = ConfigDict(extra="ignore")

  num_statements: int = 0
  missing_lines: int = 0
  num_branches: int = 0
  missing_branches: int = 0


class FunctionEntryModel(BaseModel):
  """Per-function coverage entry emitted by `coverage.py` (>=7.6).

  Attributes:
      start_line: 1-based line number where the function definition begins.
      missing_lines: List of 1-based line numbers missed inside the function.
      summary: Aggregated statement and branch counts for this function.
  """

  model_config = ConfigDict(extra="ignore")

  start_line: int
  missing_lines: list[int] = []
  summary: CoverageSummaryModel


class FileCoverageModel(BaseModel):
  """Per-file coverage entry containing file-level and function-level metrics.

  Attributes:
      summary: File-wide statement and branch coverage summary.
      missing_lines: Unexecuted 1-based line numbers across the file.
      functions: Mapping from function/method qualified name to its coverage entry.
  """

  model_config = ConfigDict(extra="ignore")

  summary: CoverageSummaryModel
  missing_lines: list[int] = []
  functions: dict[str, FunctionEntryModel] = {}


class CoverageReportModel(BaseModel):
  """Top-level `coverage.json` schema model.

  Attributes:
      files: Mapping from relative or absolute file path to its coverage entry.
  """

  model_config = ConfigDict(extra="ignore")

  files: dict[str, FileCoverageModel] = {}


def load_coverage_report(json_file: Path | None, data_file: Path) -> CoverageReportModel:
  """Load and validate a `CoverageReportModel` from JSON or a `.coverage` SQLite file.

  Args:
      json_file: Optional path to a pre-generated `coverage.json` report.
      data_file: Path to a `.coverage` data file used when `json_file` is `None`.

  Returns:
      Validated `CoverageReportModel` instance.

  Raises:
      FileNotFoundError: If neither `json_file` nor `data_file` exists on disk.
  """
  if json_file is not None:
    if not json_file.is_file():
      raise FileNotFoundError(f"Coverage JSON file not found: {json_file}")
    return CoverageReportModel.model_validate_json(json_file.read_text(encoding="utf-8"))

  if not data_file.is_file():
    raise FileNotFoundError(f"Coverage data file not found: {data_file}")

  cov = coverage.Coverage(data_file=str(data_file))
  cov.load()
  with tempfile.NamedTemporaryFile(mode="r+", suffix=".json", encoding="utf-8") as tmp:
    cov.json_report(outfile=tmp.name, show_contexts=False)
    raw_json = Path(tmp.name).read_text(encoding="utf-8")
  return CoverageReportModel.model_validate_json(raw_json)


def check_report_violations(report: CoverageReportModel) -> list[str]:
  """Collect all function-, statement-, and branch-level coverage violations.

  Args:
      report: Parsed `CoverageReportModel` to audit.

  Returns:
      Sorted list of human-readable violation messages (empty if 100% covered).
  """
  if not report.files:
    return ["Coverage report contains zero tracked files."]

  violations: list[str] = []
  for file_path in sorted(report.files):
    file_cov = report.files[file_path]
    for func_name in sorted(file_cov.functions):
      if not func_name:
        continue
      fn_entry = file_cov.functions[func_name]
      if fn_entry.summary.num_statements > 0 and fn_entry.summary.missing_lines > 0:
        violations.append(
          f"{file_path}:{fn_entry.start_line}: Function '{func_name}' missed "
          f"{fn_entry.summary.missing_lines} statement(s): {fn_entry.missing_lines}"
        )
      if fn_entry.summary.num_branches > 0 and fn_entry.summary.missing_branches > 0:
        violations.append(
          f"{file_path}:{fn_entry.start_line}: Function '{func_name}' missed "
          f"{fn_entry.summary.missing_branches} branch(es)"
        )

    if file_cov.summary.missing_lines > 0:
      violations.append(
        f"{file_path}:1: File missed {file_cov.summary.missing_lines} statement(s): {file_cov.missing_lines}"
      )
    if file_cov.summary.missing_branches > 0:
      violations.append(f"{file_path}:1: File missed {file_cov.summary.missing_branches} branch(es)")

  return violations


def main(argv: Sequence[str] | None = None) -> int:
  """Execute CLI verification of 100% function, line, and branch coverage.

  Args:
      argv: Command-line arguments (defaults to `sys.argv[1:]` when `None`).

  Returns:
      Exit code (`0` if 100% covered, `1` if missing files or violations exist).
  """
  parser = argparse.ArgumentParser(description="Verify 100% function, line, and branch test coverage.")
  parser.add_argument(
    "--json-file",
    type=Path,
    default=None,
    help="Optional path to an existing coverage.json report.",
  )
  parser.add_argument(
    "--data-file",
    type=Path,
    default=Path(".coverage"),
    help="Path to the .coverage data file (default: .coverage).",
  )
  args = parser.parse_args(argv)

  try:
    report = load_coverage_report(args.json_file, args.data_file)
  except FileNotFoundError as exc:
    print(str(exc), file=sys.stderr)
    return 1

  violations = check_report_violations(report)
  if violations:
    for violation in violations:
      print(violation, file=sys.stderr)
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
