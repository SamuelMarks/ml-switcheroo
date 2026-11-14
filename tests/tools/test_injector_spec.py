"""Unit tests for `StandardsInjector` using real `tmp_path` filesystem fixtures (`ml_switcheroo.tools.injector_spec`)."""

from pathlib import Path

import pytest
import yaml
from ml_switcheroo_ir.schema.ghost import SemanticTier

import ml_switcheroo.tools.injector_spec as injector_mod
from ml_switcheroo.core.dsl import OperationDef, OpType, ParameterDef
from ml_switcheroo.tools.injector_spec import StandardsInjector


def test_injector_writes_new_and_updates_existing_odl_files(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Verify `StandardsInjector.inject` creates/updates YAML files across tiers and supports `dry_run`.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
      capsys: Pytest stdout/stderr capture fixture.
  """
  monkeypatch.setattr(injector_mod, "resolve_semantics_dir", lambda: tmp_path)

  # 1. PascalCase op with default EXTRAS tier upgrades `self.tier` heuristic to NEURAL and writes `odl/LogSoftmax.yaml`
  op_pascal = OperationDef(
    operation="LogSoftmax",
    description="Log Softmax.",
    op_type=OpType.FUNCTION,
    std_args=["x", "dim"],
    variants={},
  )
  inj_new = StandardsInjector(op_pascal)
  assert inj_new.inject(dry_run=False) is True
  assert inj_new.tier == SemanticTier.NEURAL
  assert inj_new.found is True

  target_yaml = tmp_path / "odl" / "LogSoftmax.yaml"
  assert target_yaml.exists()
  loaded = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
  assert loaded == {
    "description": "Log Softmax.",
    "std_args": ["x", "dim"],
    "variants": {},
    "operation": "LogSoftmax",
  }

  # 2. Updating an existing file overwrites cleanly, and `dry_run=True` prints without mutating disk
  op_updated = OperationDef(
    operation="LogSoftmax",
    description="Updated description.",
    op_type=OpType.FUNCTION,
    std_args=["x"],
    variants={},
  )
  inj_dry = StandardsInjector(op_updated, tier=SemanticTier.NEURAL)
  assert inj_dry.inject(dry_run=True) is True
  assert "[Dry Run]" in capsys.readouterr().out
  assert yaml.safe_load(target_yaml.read_text(encoding="utf-8"))["description"] == "Log Softmax."

  assert inj_dry.inject(dry_run=False) is True
  assert yaml.safe_load(target_yaml.read_text(encoding="utf-8"))["description"] == "Updated description."

  # 3. Lowercase EXTRAS and slash-containing operation names preserve tier and sanitize `/` -> `_`
  op_slash = OperationDef(operation="io/save_checkpoint", description="Save.", std_args=[], variants={})
  inj_slash = StandardsInjector(op_slash, tier=SemanticTier.EXTRAS)
  assert inj_slash.inject(dry_run=False) is True
  assert inj_slash.tier == SemanticTier.EXTRAS
  assert (tmp_path / "odl" / "io_save_checkpoint.yaml").exists()


def test_serialize_op_and_args_edge_cases() -> None:
  """Verify `_serialize_op` optional metadata fields and `_serialize_args` parameter normalization."""
  op_cls = OperationDef(
    operation="CustomLayer",
    description="Class op.",
    variants={},
    op_type=OpType.CLASS,
    return_type="Tensor",
    is_inplace=True,
    output_shape_calc="lambda x: x.shape",
    std_args=[
      ParameterDef(name="a", type="int"),
      ParameterDef(name="b", type=None),
      "plain_c",
    ],
  )
  inj = StandardsInjector(op_cls)
  serialized = inj._serialize_op(op_cls)
  assert serialized["op_type"] == "class"
  assert serialized["return_type"] == "Tensor"
  assert serialized["is_inplace"] is True
  assert serialized["output_shape_calc"] == "lambda x: x.shape"

  std_args_out = serialized["std_args"]
  assert isinstance(std_args_out, list)
  assert std_args_out[0] == {"name": "a", "type": "int", "is_variadic": False, "kind": "positional_or_keyword"}
  assert std_args_out[1] == {"name": "b", "is_variadic": False, "kind": "positional_or_keyword"}
  assert std_args_out[2] == "plain_c"

  # Raw dict with `None` value stripped, and non-string/non-dict items skipped
  import collections.abc
  from unittest.mock import MagicMock

  mixed_raw: collections.abc.Sequence[str | dict[str, str | None]] = [{"name": "k", "type": None}, "s"]
  assert inj._serialize_args(mixed_raw) == [{"name": "k"}, "s"]

  non_str_arg = MagicMock()
  assert inj._serialize_args(["valid", non_str_arg]) == ["valid"]
