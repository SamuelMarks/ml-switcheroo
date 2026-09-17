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


def test_stablehlo_extra_coverage(tmp_path: Path) -> None:
  """Docstring."""
  importer = StableHloSpecImporter()
  md_content = """
Intro text before any h3 to hit elif current_op false branch.

###

### *not code*

### `valid_op`

First paragraph.

Second paragraph to hit false branch of not current_def["description"].

```python
# just some other framework
```

```mlir
%result = "stablehlo.valid_op"(%x) : (tensor<f32>) -> tensor<f32>
// just a regular comment
```
"""
  md_file = tmp_path / "spec_extra.md"
  md_file.write_text(md_content)
  res = importer.parse_file(md_file)
  assert "ValidOp" in res


def test_stablehlo_all_missing_branches(tmp_path: Path) -> None:
  """Test remaining branches in stablehlo reader."""
  from unittest.mock import MagicMock, patch
  from markdown_it.token import Token

  importer = StableHloSpecImporter()

  # 1. No ops in file (covers 98->101)
  no_ops_file = tmp_path / "no_ops.md"
  no_ops_file.write_text("# Only top heading\nSome text\n")
  assert importer.parse_file(no_ops_file) == {}

  # 2. Module with empty operations containing stablehlo. (covers 90->85)
  empty_op_file = tmp_path / "empty_body.md"
  empty_op_file.write_text("""
### `empty_body_op`

```mlir
// stablehlo.
```
""")
  res = importer.parse_file(empty_op_file)
  assert "EmptyBodyOp" in res

  # 3. Synthetic tokens for 65->63 and 77->63
  t_h3 = Token("heading_open", "h3", 1)
  t_para = Token("paragraph_open", "p", 1)

  dummy_file = tmp_path / "dummy.md"
  dummy_file.write_text("dummy")

  with MagicMock() as mock_md:
    # t_h3 at end of tokens -> 65->63
    mock_md.parse.return_value = [t_h3]
    with patch("markdown_it.MarkdownIt", return_value=mock_md):
      assert importer._parse_markdown(dummy_file) == {}

    # t_para when op is active at end of tokens -> 77->63
    t_child = Token("text", "", 0)
    t_child.content = "my_op"
    t_inline = Token("inline", "", 0)
    t_inline.children = [t_child]
    mock_md.parse.return_value = [
      Token("heading_open", "h3", 1),
      t_inline,
      t_para,
    ]
    with patch("markdown_it.MarkdownIt", return_value=mock_md):
      parsed = importer._parse_markdown(dummy_file)
      assert "MyOp" in parsed
