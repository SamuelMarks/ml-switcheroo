"""Test module."""

from pathlib import Path

from ml_switcheroo.importers.stablehlo_reader import StableHloSpecImporter


def test_stablehlo_reader_missing_file(tmp_path: Path) -> None:
  """Docstring."""
  importer = StableHloSpecImporter()
  res = importer.parse_file(tmp_path / "missing.md")
  assert res == {}


def test_stablehlo_reader_parse_markdown(tmp_path: Path) -> None:
  """Docstring."""
  importer = StableHloSpecImporter()

  md_content = """
### `abs`

Computes absolute value of `%operand`.

#### Syntax

```mlir
%result = "stablehlo.abs"(%operand) : (tensor<f32>) -> tensor<f32>
```

### `add`

Adds `%lhs` and `%rhs`.

#### Syntax

```mlir
%result = "stablehlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
```

### `subtract`

Subtracts `%rhs` from `%lhs`.

### `multiply`

Multiplies `%lhs` and `%rhs`.

### `divide`

Divides `%lhs` by `%rhs`.

### `power`

Power.

### `log_plus_one`

Log plus one.

```
%result = "stablehlo.log_plus_one"(%x) : (tensor<f32>) -> tensor<f32>
```

### `broken_syntax`

```mlir
stablehlo.broken_syntax ( ) [ ]
```

### `ignore_numbers`

```mlir
%result = "stablehlo.ignore_numbers"(%123) : (tensor<f32>) -> tensor<f32>
```
    """
  md_file = tmp_path / "spec.md"
  md_file.write_text(md_content)

  res = importer.parse_file(md_file)

  assert "Abs" in res
  assert res["Abs"]["description"] == "Computes absolute value of `%operand`."
  assert res["Abs"]["std_args"] == ["operand"]
  assert res["Abs"]["variants"]["stablehlo"]["api"] == "stablehlo.abs"

  assert "Add" in res
  assert res["Add"]["std_args"] == ["lhs", "rhs"]
  assert res["Add"]["variants"]["stablehlo"]["api"] == "stablehlo.add"

  assert "Sub" in res
  assert res["Sub"]["variants"]["stablehlo"]["api"] == "stablehlo.subtract"

  assert "Mul" in res
  assert res["Mul"]["variants"]["stablehlo"]["api"] == "stablehlo.multiply"

  assert "Div" in res
  assert res["Div"]["variants"]["stablehlo"]["api"] == "stablehlo.divide"

  assert "Pow" in res
  assert res["Pow"]["variants"]["stablehlo"]["api"] == "stablehlo.power"

  assert "LogPlusOne" in res
  assert res["LogPlusOne"]["variants"]["stablehlo"]["api"] == "stablehlo.logplusone"

  assert "BrokenSyntax" in res
  assert res["BrokenSyntax"]["std_args"] == ["input"]  # Fallback

  assert "IgnoreNumbers" in res
  assert res["IgnoreNumbers"]["std_args"] == ["input"]  # Fallback because %123 is ignored


def test_stablehlo_long_summary(tmp_path: Path) -> None:
  """Docstring."""
  importer = StableHloSpecImporter()
  long_desc = "A" * 310

  md_content = f"""
### `long_op`

{long_desc}

    """
  md_file = tmp_path / "spec.md"
  md_file.write_text(md_content)

  res = importer.parse_file(md_file)
  assert "LongOp" in res
  assert len(res["LongOp"]["description"]) == 300  # 297 + "..."
