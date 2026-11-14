"""End-to-end CLI entrypoint tests invoking `main()` against real `tmp_path` files (`ml_switcheroo.cli.__main__`)."""

import argparse
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import ml_switcheroo.cli.handlers.define as define_mod
from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.cli.commands import handle_gen_weight_script
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


def test_main_help_version_missing_and_unknown_args() -> None:
  """Verify `--version`, `--help`, missing subcommands, and unknown subcommand fallbacks."""
  with patch("sys.argv", ["ml-switcheroo"]), pytest.raises(SystemExit):
    main()

  with pytest.raises(SystemExit) as exc_ver:
    main(["--version"])
  assert exc_ver.value.code == 0

  with pytest.raises(SystemExit) as exc_help:
    main(["--help"])
  assert exc_help.value.code == 0

  with pytest.raises(SystemExit):
    main(["unknown_command"])

  with patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(command="unknown")):
    assert main(["unknown"]) == 0


def test_main_real_file_commands(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Invoke real CLI subcommands against `tmp_path` files and mini-configurations.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
      capsys: Pytest stdout/stderr capture fixture.
  """
  monkeypatch.chdir(tmp_path)

  # 1. `convert`: real Torch -> JAX transpilation to an output file on disk
  src_py = tmp_path / "input.py"
  src_py.write_text(f"import torch{_NL}y = torch.abs(x){_NL}", encoding="utf-8")
  out_py = tmp_path / "out.py"
  assert (
    main(["convert", str(src_py), "--source", "torch", "--target", "jax", "--out", str(out_py), "--config", "a=1"]) == 0
  )
  assert "jnp.abs(x)" in out_py.read_text(encoding="utf-8")

  # 2. `define`: real ODL YAML validation and copy into `tmp_path / "sem"`
  sem_dir = tmp_path / "sem"
  monkeypatch.setattr(define_mod, "resolve_semantics_dir", lambda: sem_dir)
  op_yaml = tmp_path / "op.yaml"
  op_yaml.write_text(yaml.dump({"operation": "CliTestOp", "description": "Op.", "variants": {}}), encoding="utf-8")
  assert main(["define", str(op_yaml)]) == 0
  assert (sem_dir / "odl" / "CliTestOp.yaml").exists()

  # 3. `gen-weight-script`: real Torch -> JAX weight migration script generation & failure case
  model_py = tmp_path / "model.py"
  model_py.write_text(
    _NL.join(
      [
        "import torch.nn as nn",
        "class Net(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.conv = nn.Conv2d(3, 8, 3)",
      ]
    ),
    encoding="utf-8",
  )
  weight_out = tmp_path / "migrate_weights.py"
  assert main(["gen-weight-script", str(model_py), "--out", str(weight_out), "--source", "torch", "--target", "jax"]) == 0
  assert "MAPPING_RULES" in weight_out.read_text(encoding="utf-8")
  assert handle_gen_weight_script(tmp_path / "missing_model.py", tmp_path / "unused.py") == 1

  # 4. `schema`, `completion`, `harvest`
  assert main(["schema"]) == 0
  assert '"OperationDef"' in capsys.readouterr().out or '"operation"' in capsys.readouterr().out
  assert main(["completion", "bash"]) == 0
  _ = capsys.readouterr()
  assert main(["harvest", str(tmp_path)]) == 0
  assert "Harvest complete." in capsys.readouterr().out

  # 5. `suggest` against an in-memory module in `sys.modules`
  mini_mod = types.ModuleType("cli_mini_mod")

  def sample_op(x: int) -> int:
    """Add one to `x`.

    Args:
        x: Input integer.

    Returns:
        `x + 1`.
    """
    return x + 1

  monkeypatch.setattr(mini_mod, "sample_op", sample_op, raising=False)
  monkeypatch.setitem(sys.modules, "cli_mini_mod", mini_mod)
  suggest_dir = tmp_path / "suggest_out"
  assert main(["suggest", "cli_mini_mod.*", "--out-dir", str(suggest_dir)]) == 0
  assert len(list(suggest_dir.glob("*.md"))) == 1

  # 6. `import-onnx` against a real Markdown file in `tmp_path`
  onnx_md = tmp_path / "Operators.md"
  onnx_md.write_text(
    _NL.join(
      [
        '### <a name="Relu"></a>**Relu**',
        "",
        "#### Summary",
        "",
        "Applies relu activation.",
        "",
        "#### Inputs",
        "",
        "<dl><dt>X : T</dt><dd>input</dd></dl>",
        "",
      ]
    ),
    encoding="utf-8",
  )
  onnx_out = tmp_path / "onnx_odl"
  assert (
    main(["import-onnx", str(onnx_md), "--out-dir", str(onnx_out), "--domain", "ai.onnx", "--opset-version", "21"]) == 0
  )

  # 7. `gen-docs`, `gen-tests`, `matrix`, and `ci` with a compact in-memory `SemanticsManager`
  mini_sem = SemanticsManager(load_defaults=False)
  mini_sem.data = {
    "Abs": {
      "operation": "Abs",
      "description": "Absolute value.",
      "std_args": ["x"],
      "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}},
    }
  }
  monkeypatch.setattr("ml_switcheroo.cli.handlers.dev.SemanticsManager", lambda: mini_sem)
  monkeypatch.setattr("ml_switcheroo.cli.handlers.verify.SemanticsManager", lambda: mini_sem)

  docs_md = tmp_path / "MIGRATION_GUIDE.md"
  assert main(["gen-docs", "--source", "torch", "--target", "jax", "--out", str(docs_md)]) == 0
  assert docs_md.exists()

  tests_py = tmp_path / "generated_tests.py"
  assert main(["gen-tests", "--out", str(tests_py)]) == 0
  assert tests_py.exists()

  with patch("sys.argv", ["ml-switcheroo", "matrix"]):
    assert main() == 0

  ci_report = tmp_path / "ci_report.json"
  (tmp_path / "tests").mkdir()
  (tmp_path / "tests" / "test_abs.py").write_text("def test_Abs(): pass", encoding="utf-8")
  assert main(["ci", "--json-report", str(ci_report)]) == 0
  assert ci_report.exists()

  # 8. `scaffold` against `tmp_path`
  with patch("ml_switcheroo.cli.handlers.scaffold.ConsensusEngine") as mock_ce:
    mock_ce.return_value.cluster.return_value = {"Abs": ["jax.numpy.abs"]}
    assert main(["scaffold", "jax"]) == 0
    assert (tmp_path / "jax_skeleton.json").exists()

  # 9. `verified-pipeline` reading real `tmp_path` files for both success and error outcomes
  vp_file = tmp_path / "vp.py"
  vp_file.write_text(f"import torch{_NL}", encoding="utf-8")
  with patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline", return_value={"status": "success"}):
    assert main(["verified-pipeline", str(vp_file)]) == 0
  with patch("ml_switcheroo.ingestion.verified_pipeline.run_verified_pipeline", return_value={"status": "failed"}):
    assert main(["verified-pipeline", str(vp_file)]) == 1

  # 10. `__main__` module execution
  with patch("sys.argv", ["ml-switcheroo", "schema"]), pytest.raises(SystemExit) as exc_main:
    runpy.run_module("ml_switcheroo.cli.__main__", run_name="__main__")
  assert exc_main.value.code == 0
