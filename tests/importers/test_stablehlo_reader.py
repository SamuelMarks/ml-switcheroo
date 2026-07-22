"""Test suite for the Stablehlo Reader module."""

import pytest
from ml_switcheroo.importers.stablehlo_reader import StableHloSpecImporter


@pytest.fixture
def importer():
  """Provides a mock importer for testing."""
  return StableHloSpecImporter()


def test_parse_file_not_found(importer, tmp_path):
  """Parses file not found."""
  assert importer.parse_file(tmp_path / "missing.md") == {}


def test_parse_file_valid(importer, tmp_path):
  """Parses file valid."""
  md_file = tmp_path / "spec.md"
  md_file.write_text(
    "\n### `abs`\n#### Semantics\nComputes the absolute value.\n```mlir\n%result = stablehlo.abs %operand : tensor<...>\n```\n\n### `log_plus_one`\nLogs the plus one.\n```mlir\n%res = stablehlo.log_plus_one %a, %b : tensor<...>\n```\n    "
  )
  result = importer.parse_file(md_file)
  assert "Abs" in result
  assert result["Abs"]["description"] == "Computes the absolute value."
  assert result["Abs"]["std_args"] == ["operand"]
  assert result["Abs"]["variants"]["stablehlo"]["api"] == "stablehlo.abs"
  assert "LogPlusOne" in result
  assert result["LogPlusOne"]["std_args"] == ["a", "b"]
  assert result["LogPlusOne"]["variants"]["stablehlo"]["api"] == "stablehlo.logplusone"


def test_finalize_op_truncation(importer):
  """Verifies the behavior of finalize op truncation."""
  semantics = {}
  details = {"description": ["A" * 100, "B" * 100, "C" * 150], "raw_syntax": "%x, %y"}
  importer._finalize_op(semantics, "MyOp", details)
  desc = semantics["MyOp"]["description"]
  assert len(desc) == 300
  assert desc.endswith("...")
  assert semantics["MyOp"]["std_args"] == ["x", "y"]


def test_finalize_op_args_filtering(importer):
  """Verifies the behavior of finalize op arguments filtering."""
  semantics = {}
  details = {"description": ["Desc"], "raw_syntax": "%0 = stablehlo.add %lhs, %rhs, %result, %results"}
  importer._finalize_op(semantics, "Add", details)
  assert semantics["Add"]["std_args"] == ["lhs", "rhs"]
  assert semantics["Add"]["variants"]["stablehlo"]["api"] == "stablehlo.add"


def test_finalize_op_args_fallback(importer):
  """Verifies the behavior of finalize op arguments fallback."""
  semantics = {}
  details = {"description": ["Desc"], "raw_syntax": ""}
  importer._finalize_op(semantics, "Sub", details)
  assert semantics["Sub"]["std_args"] == ["input"]
  assert semantics["Sub"]["variants"]["stablehlo"]["api"] == "stablehlo.subtract"


def test_finalize_op_api_suffix(importer):
  """Verifies the behavior of finalize op API suffix."""
  semantics = {}
  importer._finalize_op(semantics, "Add", {})
  assert semantics["Add"]["variants"]["stablehlo"]["api"] == "stablehlo.add"
  importer._finalize_op(semantics, "Sub", {})
  assert semantics["Sub"]["variants"]["stablehlo"]["api"] == "stablehlo.subtract"
  importer._finalize_op(semantics, "Mul", {})
  assert semantics["Mul"]["variants"]["stablehlo"]["api"] == "stablehlo.multiply"
  importer._finalize_op(semantics, "Div", {})
  assert semantics["Div"]["variants"]["stablehlo"]["api"] == "stablehlo.divide"
  importer._finalize_op(semantics, "Pow", {})
  assert semantics["Pow"]["variants"]["stablehlo"]["api"] == "stablehlo.power"


def test_normalize_op_name(importer):
  """Verifies the behavior of normalize op name."""
  assert importer._normalize_op_name("abs") == "Abs"
  assert importer._normalize_op_name("add") == "Add"
  assert importer._normalize_op_name("subtract") == "Sub"
  assert importer._normalize_op_name("multiply") == "Mul"
  assert importer._normalize_op_name("divide") == "Div"
  assert importer._normalize_op_name("power") == "Pow"
  assert importer._normalize_op_name("log_plus_one") == "LogPlusOne"
  assert importer._normalize_op_name("custom_op_name") == "CustomOpName"
