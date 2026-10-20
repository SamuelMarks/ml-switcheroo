"""Test suite for the Injector Fw module."""

import typing
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.tools.injector_fw import FrameworkInjector


@pytest.fixture
def target_json(tmp_path: Path) -> Path:
  """Execute target json helper logic.

  Args:
      tmp_path: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  defs_dir: Path = tmp_path / "definitions"
  defs_dir.mkdir()
  initial_data: typing.Dict[str, typing.Dict[str, str]] = {"OldOp": {"api": "torch.old"}}
  json_path: Path = defs_dir / "torch.json"
  json_path.write_text(json.dumps(initial_data), encoding="utf-8")
  return json_path


@pytest.fixture
def sample_variant() -> FrameworkVariant:
  """Execute sample variant helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return FrameworkVariant(api="torch.nn.functional.log_softmax", args={"dim": "dim"}, requires_plugin="custom_plugin")


def test_injector_updates_existing_json(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies the behavior of injector updates existing JSON.

  Args:
      target_json: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
  assert success is True
  content = json.loads(target_json.read_text())
  assert "OldOp" in content
  assert "LogSoftmax" in content
  entry = content["LogSoftmax"]
  assert entry["api"] == "torch.nn.functional.log_softmax"
  assert entry["args"] == {"dim": "dim"}
  assert entry["requires_plugin"] == "custom_plugin"


def test_injector_creates_new_file(tmp_path: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies the behavior of injector creates new file.

  Args:
      tmp_path: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  missing_path: Path = tmp_path / "definitions" / "new_fw.json"
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=missing_path):
    injector: FrameworkInjector = FrameworkInjector("new_fw", "NewOp", sample_variant)
    injector.inject(dry_run=False)
  assert missing_path.exists()
  content = json.loads(missing_path.read_text())
  assert "NewOp" in content
  assert content["NewOp"]["api"] == "torch.nn.functional.log_softmax"


def test_injector_dry_run(
  target_json: Path, sample_variant: FrameworkVariant, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verifies the behavior of injector dry run.

  Args:
      target_json: Test argument or fixture.
      sample_variant: Test argument or fixture.
      capsys: Test argument or fixture.
  """
  target_json.stat().st_mtime
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    injector.inject(dry_run=True)
  captured = capsys.readouterr()
  assert "[Dry Run] Writing to torch.json" in captured.out
  assert "LogSoftmax" in captured.out
  content = json.loads(target_json.read_text())
  assert "LogSoftmax" not in content


def test_injector_idempotency(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies behavior when entry already exists and is identical.

  Args:
      target_json: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  # Add the sample_variant to target_json
  content = json.loads(target_json.read_text())
  content["LogSoftmax"] = sample_variant.model_dump(exclude_none=True)
  target_json.write_text(json.dumps(content))

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
    assert success is True


def test_injector_update_existing_different(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Hits line 65 where existing entry is updated.

  Args:
      target_json: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  # Add the LogSoftmax but with different data
  content = json.loads(target_json.read_text())
  content["LogSoftmax"] = {"api": "torch.wrong"}
  target_json.write_text(json.dumps(content))

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
    assert success is True
    new_content = json.loads(target_json.read_text())
    assert new_content["LogSoftmax"]["api"] == "torch.nn.functional.log_softmax"


def test_injector_write_error(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies error handling during file write.

  Args:
      target_json: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  original_open = open

  def mock_open(
    file: Path | str,
    mode: str = "r",
    encoding: typing.Optional[str] = None,
  ) -> io.TextIOWrapper:
    """Mock open.

    Args:
        file: File path.
        mode: Open mode.
        encoding: File encoding.

    Returns:
        Computed test result or fixture value.
    """
    if mode == "w":
      raise OSError("Permission denied")
    return original_open(file, "r", encoding=encoding)

  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    with patch("builtins.open", side_effect=mock_open):
      injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
      success: bool = injector.inject(dry_run=False)
      assert success is False


def test_injector_load_corrupt_json(target_json: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies behavior when loading corrupt JSON.

  Args:
      target_json: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  target_json.write_text("{invalid json}")
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=target_json):
    injector: FrameworkInjector = FrameworkInjector("torch", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=False)
    assert success is True
    # The file should be overwritten
    content = json.loads(target_json.read_text())
    assert "LogSoftmax" in content
    assert "OldOp" not in content


# --- Merged from test_injector_fw_missing.py ---


def test_injector_fw_missing() -> None:
  """Verifies the behavior of injector framework missing."""
  from ml_switcheroo.core.dsl import FrameworkVariant
  from ml_switcheroo.tools.injector_fw.core import FrameworkInjector

  variant: FrameworkVariant = FrameworkVariant(api="foo")
  injector: FrameworkInjector = FrameworkInjector("jax", "bar", variant)
  with __import__("unittest.mock").mock.patch.object(
    injector, "_load_current", return_value={"bar": variant.model_dump(exclude_none=True)}
  ):
    assert injector.inject(dry_run=False) is True
  with __import__("unittest.mock").mock.patch.object(injector, "_load_current", return_value={"other": {}}):
    with __import__("unittest.mock").mock.patch("builtins.open", side_effect=OSError("fail")):
      assert injector.inject(dry_run=False) is False
  with __import__("unittest.mock").mock.patch(
    "builtins.open", __import__("unittest.mock").mock.mock_open(read_data="invalid json")
  ):
    assert injector._load_current() == {}


def test_injector_fw_updating() -> None:
  """Verifies the behavior of injector framework updating."""
  from ml_switcheroo.core.dsl import FrameworkVariant
  from ml_switcheroo.tools.injector_fw.core import FrameworkInjector

  variant: FrameworkVariant = FrameworkVariant(api="foo")
  injector: FrameworkInjector = FrameworkInjector("jax", "bar", variant)
  with __import__("unittest.mock").mock.patch.object(injector, "_load_current", return_value={"bar": {"api": "old_foo"}}):
    with __import__("unittest.mock").mock.patch("builtins.open", __import__("unittest.mock").mock.mock_open()):
      assert injector.inject(dry_run=False) is True


def test_injector_dry_run_nonexistent_parent_dir(tmp_path: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies dry-run behavior when target directory does not exist.

  Args:
      tmp_path: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  missing_dir_file: Path = tmp_path / "nonexistent_dir" / "target.json"
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=missing_dir_file):
    injector: FrameworkInjector = FrameworkInjector("target_fw", "LogSoftmax", sample_variant)
    success: bool = injector.inject(dry_run=True)
    assert success is True
    assert not missing_dir_file.parent.exists()


def test_injector_fw_load_non_dict_json(tmp_path: Path, sample_variant: FrameworkVariant) -> None:
  """Verifies that loading JSON containing a non-dict root returns an empty dict.

  Args:
      tmp_path: Test argument or fixture.
      sample_variant: Test argument or fixture.
  """
  json_file: Path = tmp_path / "list.json"
  json_file.write_text("[1, 2, 3]", encoding="utf-8")
  with patch("ml_switcheroo.tools.injector_fw.core.get_definitions_path", return_value=json_file):
    injector: FrameworkInjector = FrameworkInjector("target_fw", "LogSoftmax", sample_variant)
    assert injector._load_current() == {}
