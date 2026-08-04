"""Test suite for the Stablehlo Reader module."""

import pytest
from pathlib import Path
from typing import Dict, Any
from ml_switcheroo.importers.stablehlo_reader import StableHloSpecImporter


@pytest.fixture
def importer() -> StableHloSpecImporter:
  """Provides a mock importer for testing."""
  return StableHloSpecImporter()


def test_parse_file_not_found(importer: StableHloSpecImporter, tmp_path: Path) -> None:
  """Parses file not found."""
  assert importer.parse_file(tmp_path / "missing.md") == {}


def test_parse_file_valid(importer: StableHloSpecImporter, tmp_path: Path) -> None:
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


def test_parse_invalid_mlir(importer: StableHloSpecImporter, tmp_path: Path) -> None:
  """Parses invalid MLIR to trigger fallback exception block."""
  md_file = tmp_path / "spec_bad.md"
  # This syntax shouldn't parse cleanly, causing a failure that falls back to raw_syntax
  md_file.write_text("### `bad_op`\n```mlir\nstablehlo.bad_op ::: <<invalid>>\n```")
  result = importer.parse_file(md_file)
  assert "BadOp" in result
  # The parser failed, so it shouldn't extract std_args from the parsed_op
  assert result["BadOp"]["std_args"] == ["input"]


def test_finalize_op_truncation(importer: StableHloSpecImporter) -> None:
  """Verifies the behavior of finalize op truncation."""
  from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode

  semantics: Dict[str, Any] = {}
  parsed = OperationNode(
    name="stablehlo.my_op", results=[ValueNode(name="%res")], operands=[ValueNode(name="%x"), ValueNode(name="%y")]
  )
  details = {"description": ["A" * 100, "B" * 100, "C" * 150], "raw_syntax": "%x, %y", "parsed_op": parsed}
  importer._finalize_op(semantics, "MyOp", details)
  desc = semantics["MyOp"]["description"]
  assert len(desc) == 300
  assert desc.endswith("...")
  assert semantics["MyOp"]["std_args"] == ["x", "y"]


def test_finalize_op_args_filtering(importer: StableHloSpecImporter) -> None:
  """Verifies the behavior of finalize op arguments filtering."""
  from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode

  semantics: Dict[str, Any] = {}
  parsed = OperationNode(
    name="stablehlo.add",
    results=[ValueNode(name="%0"), ValueNode(name="%result"), ValueNode(name="%results")],
    operands=[ValueNode(name="%lhs"), ValueNode(name="%rhs")],
  )
  details = {"description": ["Desc"], "raw_syntax": "%0 = stablehlo.add %lhs, %rhs", "parsed_op": parsed}
  importer._finalize_op(semantics, "Add", details)
  assert semantics["Add"]["std_args"] == ["lhs", "rhs"]
  assert semantics["Add"]["variants"]["stablehlo"]["api"] == "stablehlo.add"


def test_finalize_op_args_fallback(importer: StableHloSpecImporter) -> None:
  """Verifies the behavior of finalize op arguments fallback."""
  semantics: Dict[str, Any] = {}
  details = {"description": ["Desc"], "raw_syntax": ""}
  importer._finalize_op(semantics, "Sub", details)
  assert semantics["Sub"]["std_args"] == ["input"]
  assert semantics["Sub"]["variants"]["stablehlo"]["api"] == "stablehlo.subtract"


def test_finalize_op_api_suffix(importer: StableHloSpecImporter) -> None:
  """Verifies the behavior of finalize op API suffix."""
  semantics: Dict[str, Any] = {}
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


def test_normalize_op_name(importer: StableHloSpecImporter) -> None:
  """Verifies the behavior of normalize op name."""
  assert importer._normalize_op_name("abs") == "Abs"
  assert importer._normalize_op_name("add") == "Add"
  assert importer._normalize_op_name("subtract") == "Sub"
  assert importer._normalize_op_name("multiply") == "Mul"
  assert importer._normalize_op_name("divide") == "Div"
  assert importer._normalize_op_name("power") == "Pow"
  assert importer._normalize_op_name("log_plus_one") == "LogPlusOne"
  assert importer._normalize_op_name("custom_op_name") == "CustomOpName"


def test_parse_markdown_edge_cases(importer: StableHloSpecImporter, tmp_path: Path) -> None:
  """Test edge cases in markdown parsing for branch coverage."""
  md_file = tmp_path / "spec_edge.md"
  md_file.write_text(
    "###\n\n"  # Empty h3
    "### **bold_name**\n\n"  # h3 with non text/code child
    "### `bad name`\n\n"  # h3 with invalid characters for op name
    "Some text before op.\n\n"  # Paragraph without current_op
    "```python\nprint(1)\n```\n\n"  # Code block without current_op
    "### `valid_op`\n"
    "Desc 1.\n\n"
    "Desc 2.\n"  # Append to existing description (will be ignored)
    "```javascript\nfoo\n```\n\n"  # Fence without mlir/stablehlo
    "```mlir\nnot stablehlo\n```\n\n"  # Fence with mlir but no stablehlo string
    "### `valid_op_2`\n"  # Finalize previous
  )
  result = importer.parse_file(md_file)
  assert "ValidOp" in result
  assert "ValidOp2" in result
  assert result["ValidOp"]["description"] == "Desc 1."
