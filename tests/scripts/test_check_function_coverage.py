"""Unit tests for `scripts/check_function_coverage.py`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import coverage
import pytest

from scripts.check_function_coverage import (
  CoverageReportModel,
  CoverageSummaryModel,
  FileCoverageModel,
  FunctionEntryModel,
  check_report_violations,
  load_coverage_report,
  main,
)


def _make_sample_report(
  *,
  fn_missing_lines: list[int] | None = None,
  fn_missing_branches: int = 0,
  file_missing_lines: list[int] | None = None,
  file_missing_branches: int = 0,
) -> CoverageReportModel:
  """Build a strongly-typed `CoverageReportModel` fixture for violation testing.

  Args:
      fn_missing_lines: Optional list of missed statement line numbers inside `compute`.
      fn_missing_branches: Number of missed branches inside `compute`.
      file_missing_lines: Optional list of missed statement line numbers at file scope.
      file_missing_branches: Number of missed branches at file scope.

  Returns:
      Constructed `CoverageReportModel` instance.
  """
  fn_lines = fn_missing_lines or []
  f_lines = file_missing_lines or []
  return CoverageReportModel(
    files={
      "src/pkg/mod.py": FileCoverageModel(
        summary=CoverageSummaryModel(
          num_statements=10,
          missing_lines=len(f_lines),
          num_branches=4,
          missing_branches=file_missing_branches,
        ),
        missing_lines=f_lines,
        functions={
          "": FunctionEntryModel(
            start_line=1,
            missing_lines=[],
            summary=CoverageSummaryModel(num_statements=2, missing_lines=0),
          ),
          "compute": FunctionEntryModel(
            start_line=5,
            missing_lines=fn_lines,
            summary=CoverageSummaryModel(
              num_statements=8,
              missing_lines=len(fn_lines),
              num_branches=4,
              missing_branches=fn_missing_branches,
            ),
          ),
          "empty_fn": FunctionEntryModel(
            start_line=20,
            missing_lines=[],
            summary=CoverageSummaryModel(num_statements=0, missing_lines=0, num_branches=0, missing_branches=0),
          ),
        },
      )
    }
  )


def test_check_report_violations_clean_and_empty() -> None:
  """Verify `check_report_violations` returns empty on 100% coverage and flags empty reports."""
  assert check_report_violations(CoverageReportModel(files={})) == ["Coverage report contains zero tracked files."]
  clean_report = _make_sample_report()
  assert check_report_violations(clean_report) == []


def test_check_report_violations_all_failure_modes() -> None:
  """Verify function-level and file-level statement and branch misses are reported."""
  bad_report = _make_sample_report(
    fn_missing_lines=[7, 8],
    fn_missing_branches=1,
    file_missing_lines=[7, 8],
    file_missing_branches=1,
  )
  violations = check_report_violations(bad_report)
  assert len(violations) == 4
  assert "Function 'compute' missed 2 statement(s): [7, 8]" in violations[0]
  assert "Function 'compute' missed 1 branch(es)" in violations[1]
  assert "File missed 2 statement(s): [7, 8]" in violations[2]
  assert "File missed 1 branch(es)" in violations[3]


def test_load_coverage_report_from_json_and_missing_errors(tmp_path: Path) -> None:
  """Verify loading from `--json-file` and raising `FileNotFoundError` when files are absent.

  Args:
      tmp_path: Temporary directory path fixture.
  """
  missing_json = tmp_path / "missing.json"
  missing_data = tmp_path / ".coverage_missing"
  with pytest.raises(FileNotFoundError, match="Coverage JSON file not found"):
    load_coverage_report(missing_json, missing_data)

  with pytest.raises(FileNotFoundError, match="Coverage data file not found"):
    load_coverage_report(None, missing_data)

  valid_json = tmp_path / "cov.json"
  valid_json.write_text(_make_sample_report().model_dump_json(), encoding="utf-8")
  loaded = load_coverage_report(valid_json, missing_data)
  assert "src/pkg/mod.py" in loaded.files


def test_load_coverage_report_from_sqlite_data_file(tmp_path: Path) -> None:
  """Verify generating and loading a report directly from a real `.coverage` data file.

  Args:
      tmp_path: Temporary directory path fixture.
  """
  mod_file = tmp_path / "sample_mod.py"
  mod_file.write_text(
    chr(10).join(
      [
        "def add_one(x: int) -> int:",
        "  if x > 0:",
        "    return x + 1",
        "  return x",
        "",
      ]
    ),
    encoding="utf-8",
  )
  data_file = tmp_path / ".coverage_sample"
  cov = coverage.Coverage(data_file=str(data_file), branch=True, source=[str(tmp_path)])
  cov.start()
  sys.path.insert(0, str(tmp_path))
  try:
    import importlib

    sample_mod = importlib.import_module("sample_mod")
    assert sample_mod.add_one(2) == 3
    assert sample_mod.add_one(0) == 0
  finally:
    sys.path.remove(str(tmp_path))
    sys.modules.pop("sample_mod", None)
    cov.stop()
    cov.save()

  report = load_coverage_report(None, data_file)
  assert check_report_violations(report) == []
  assert main(["--data-file", str(data_file)]) == 0


def test_main_exit_codes_and_dunder_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verify CLI exit codes for missing files, failing coverage, passing coverage, and `__main__`.

  Args:
      tmp_path: Temporary directory path fixture.
      capsys: Pytest stderr/stdout capture fixture.
  """
  assert main(["--json-file", str(tmp_path / "nope.json")]) == 1
  assert "Coverage JSON file not found" in capsys.readouterr().err

  failing_json = tmp_path / "fail.json"
  failing_json.write_text(_make_sample_report(fn_missing_lines=[6]).model_dump_json(), encoding="utf-8")
  assert main(["--json-file", str(failing_json)]) == 1
  assert "Function 'compute' missed 1 statement(s)" in capsys.readouterr().err

  passing_json = tmp_path / "pass.json"
  passing_json.write_text(_make_sample_report().model_dump_json(), encoding="utf-8")
  with patch.object(sys, "argv", ["scripts/check_function_coverage.py", "--json-file", str(passing_json)]):
    with pytest.raises(SystemExit) as exc_info:
      runpy.run_module("scripts.check_function_coverage", run_name="__main__")
    assert exc_info.value.code == 0
