"""
Tests for ODL DSL Schema.

Updated to align with Schema Version 2, where `FrameworkVariant.api` is optional
to allow defining explicit failure variants with custom error messages.
"""

import pytest
from pydantic import ValidationError
from ml_switcheroo.core.dsl import (
  OperationDef,
  ParameterDef,
  FrameworkVariant,
  Rule,
  PluginScaffoldDef,
  PluginType,
)


def test_parameter_def_defaults():
  """Verify ParameterDef defaults correspond to specification."""
  p = ParameterDef(name="x")
  assert p.name == "x"
  assert p.type == "Any"
  assert p.doc is None
  assert p.default is None


def test_framework_variant_optional_api():
  """
  Verify API string is OPTIONAL.

  Previously this field was mandatory. It is now optional to allow
  explicitly blocking a framework variant (e.g. for operations that
  cannot be supported) by providing a `missing_message` instead.
  """
  # Should NOT raise ValidationError
  v = FrameworkVariant(missing_message="Not supported on this backend")
  assert v.api is None
  assert v.missing_message == "Not supported on this backend"


def test_framework_variant_casts():
  """Verify 'casts' dictionary is supported."""
  v = FrameworkVariant(api="foo", casts={"x": "int32"})
  assert v.casts["x"] == "int32"


def test_framework_variant_inject_args():
  """Verify 'inject_args' dictionary support."""
  v = FrameworkVariant(api="foo", inject_args={"eps": 1e-5, "flag": True})
  assert v.inject_args["eps"] == 1e-5
  assert v.inject_args["flag"] is True


def test_framework_variant_arg_values():
  """Verify 'arg_values' dictionary support."""
  v = FrameworkVariant(api="foo", arg_values={"reduction": {"mean": "'avg'", "sum": "'add'"}})
  assert v.arg_values["reduction"]["mean"] == "'avg'"
  assert v.arg_values["reduction"]["sum"] == "'add'"


def test_framework_variant_pack_to_tuple():
  """Verify 'pack_to_tuple' field support."""
  v = FrameworkVariant(api="foo", pack_to_tuple="axes")
  assert v.pack_to_tuple == "axes"


def test_framework_variant_select_index():
  """
  Verify 'output_select_index' validation.
  """
  v = FrameworkVariant(api="foo", output_select_index=0)
  assert v.output_select_index == 0

  v2 = FrameworkVariant(api="foo", output_select_index=1)
  assert v2.output_select_index == 1

  # Ensure type safety (should raise error if string provided for int field)
  with pytest.raises(ValidationError):
    FrameworkVariant(api="foo", output_select_index="zero")


def test_operation_def_structure():
  """Verify full hierarchical structure validation."""
  data = {
    "operation": "TestOp",
    "description": "A test op",
    "std_args": [{"name": "x", "type": "int"}],
    "variants": {"torch": {"api": "torch.test"}},
  }
  op = OperationDef(**data)
  assert op.operation == "TestOp"
  assert len(op.std_args) == 1
  assert op.std_args[0].type == "int"
  assert op.variants["torch"].api == "torch.test"


def test_rule_instantiation_by_name():
  """Verify Rule can be instantiated using is_val attribute directly."""
  r = Rule(if_arg="mode", is_val="nearest", use_api="jax.image.resize")
  assert r.if_arg == "mode"
  assert r.is_val == "nearest"
  assert r.use_api == "jax.image.resize"


def test_rule_instantiation_by_alias():
  """Verify Rule can be instantiated via dictionary validation using 'val' alias."""
  # Schema defines alias="val", not "is"
  data = {"if_arg": "flag", "val": True, "use_api": "target"}
  r = Rule.model_validate(data)
  assert r.is_val is True


def test_plugin_scaffold_with_rules():
  """Verify PluginScaffoldDef supports rules list."""
  rules = [
    Rule(if_arg="x", is_val=1, use_api="f1"),
    Rule(if_arg="x", is_val=2, use_api="f2"),
  ]
  scaffold = PluginScaffoldDef(name="test_plugin", type=PluginType.CALL, doc="Test", rules=rules)
  assert len(scaffold.rules) == 2
  assert scaffold.rules[1].use_api == "f2"
