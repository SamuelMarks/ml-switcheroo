"""
Tests for StableHLO Spec Importer.

Verifies:
1. Header regex parsing.
2. Description extraction and length truncation.
3. Syntax argument inference mapping.
4. Normalization of names (snake_case -> PascalCase).
5. Error handling for missing files.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from ml_switcheroo.importers.stablehlo_reader import StableHloSpecImporter

# We read the content from the sibling file mock_stablehlo.md usually,
# but for self-contained unit tests in CI where file placement might vary,
# we inject the content into a tmp_path file.
# NOTE: The long description is a single line to ensure the parser captures it fully
# before hitting the next blank line delimiter logic.
MOCK_CONTENT = """# StableHLO Specification

### `abs`

Computes the absolute value.
Returns a tensor of the same type.

#### Syntax

```mlir
%result = stablehlo.abs %operand : tensor<4xf32>
```

---

### `add`

Performs element-wise addition. This is a very long description that should continually go on and on to verify that the truncation logic works correctly because nobody wants a single line description in the generated json file that spans five hundred characters and breaks the terminal wrapping logic when printing summaries to the console log output. We need to ensure that this text is definitely longer than three hundred characters so that the logic inside the parser actually triggers the truncation suffix behavior during the unit test execution cycle.

#### Syntax

```mlir
%result = stablehlo.add %lhs, %rhs : tensor<2xi32>
```

### `log_plus_one`

Computes log(x + 1).

#### Inputs

No syntax block provided here.
"""


@pytest.fixture
def importer() -> StableHloSpecImporter:
  """Returns an instance of the importer."""
  return StableHloSpecImporter()


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
  """Creates a temporary spec file."""
  f = tmp_path / "spec.md"
  f.write_text(MOCK_CONTENT, encoding="utf-8")
  return f


def test_missing_file_returns_empty(importer: StableHloSpecImporter, tmp_path: Path) -> None:
  """Verify safe empty return on missing file."""
  missing = tmp_path / "ghost.md"
  # Patch the function reference inside the module, enabling verification of call.
  with patch("ml_switcheroo.importers.stablehlo_reader.log_error") as mock_log:
    res = importer.parse_file(missing)
    assert res == {}
    mock_log.assert_called_once()


def test_parse_simple_op(importer: StableHloSpecImporter, spec_file: Path) -> None:
  """Verify 'abs' parsing."""
  semantics = importer.parse_file(spec_file)

  assert "Abs" in semantics
  op = semantics["Abs"]

  # Check Description
  assert "Computes the absolute value." in op["description"]
  # Check Variant
  assert op["variants"]["stablehlo"]["api"] == "stablehlo.abs"
  # Check Argument Inference (%operand extracted from syntax string)
  assert op["std_args"] == ["operand"]


def test_parse_overrides_and_args(importer: StableHloSpecImporter, spec_file: Path) -> None:
  """
  Verify 'add' -> 'Add' override and multiple args extraction.
  Also verifies description truncation.
  """
  semantics = importer.parse_file(spec_file)

  assert "Add" in semantics
  op = semantics["Add"]

  # Check Override Normalization
  assert op["variants"]["stablehlo"]["api"] == "stablehlo.add"

  # Check Args (%lhs, %rhs)
  assert op["std_args"] == ["lhs", "rhs"]

  # Check Truncation
  assert len(op["description"]) <= 300
  assert op["description"].endswith("...")


def test_parse_complex_naming(importer: StableHloSpecImporter, spec_file: Path) -> None:
  """Verify 'log_plus_one' -> 'LogPlusOne' normalization and fallback args."""
  semantics = importer.parse_file(spec_file)

  assert "LogPlusOne" in semantics
  op = semantics["LogPlusOne"]

  # Check CamelCase conversion
  assert op["variants"]["stablehlo"]["api"] == "stablehlo.logplusone"

  # Verify fallback args (no syntax block present)
  assert op["std_args"] == ["input"]


def test_normalize_helper(importer: StableHloSpecImporter) -> None:
  """Unit test for name normalization logic."""
  assert importer._normalize_op_name("subtract") == "Sub"
  assert importer._normalize_op_name("dot_general") == "DotGeneral"
  assert importer._normalize_op_name("convert") == "Convert"
  assert importer._normalize_op_name("reduce_window") == "ReduceWindow"
