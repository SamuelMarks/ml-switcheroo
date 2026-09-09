"""Tests for the Semantic Knowledge Base schema validator."""

import pytest

from ml_switcheroo.semantics.schema import (
  SemanticsFile,
  validate_yaml_schema,
  PluginTraits,
  StructuralTraits,
  FrameworkTraits,
)


def test_validate_yaml_schema_valid():
  """Test validating a well-formed YAML string."""
  yaml_str = """
    __frameworks__:
      pytorch:
        extends: "base"
        alias:
          module: "torch"
          name: "th"
    """
  result = validate_yaml_schema(yaml_str)
  assert isinstance(result, SemanticsFile)
  assert result.frameworks is not None
  assert "pytorch" in result.frameworks
  assert result.frameworks["pytorch"].extends == "base"
  assert result.frameworks["pytorch"].alias is not None
  assert result.frameworks["pytorch"].alias.module == "torch"
  assert result.frameworks["pytorch"].alias.name == "th"


def test_validate_yaml_schema_empty():
  """Test validating an empty YAML string."""
  result = validate_yaml_schema("")
  assert isinstance(result, SemanticsFile)
  assert result.frameworks is None
  assert result.imports is None


def test_validate_yaml_schema_invalid_yaml():
  """Test validating malformed YAML."""
  bad_yaml = """
    __frameworks__:
      - invalid
     indentation
    """
  with pytest.raises(ValueError, match="Invalid YAML content"):
    validate_yaml_schema(bad_yaml)


def test_validate_yaml_schema_invalid_schema():
  """Test validating YAML that fails schema rules."""
  bad_schema_yaml = """
    __frameworks__:
      pytorch:
        alias:
          module: "torch"
          # missing name
    """
  with pytest.raises(ValueError, match="Schema validation failed"):
    validate_yaml_schema(bad_schema_yaml)


def test_plugin_traits_default():
  """Test PluginTraits default instantiation."""
  traits = PluginTraits()
  assert traits.has_numpy_compatible_arrays is False
  assert traits.enforce_purity_analysis is False


def test_structural_traits_default():
  """Test StructuralTraits default instantiation."""
  traits = StructuralTraits()
  assert traits.requires_super_init is False
  assert traits.auto_strip_magic_args is False


def test_framework_traits_default():
  """Test FrameworkTraits default instantiation."""
  traits = FrameworkTraits()
  assert traits.extends is None
  assert isinstance(traits.traits, StructuralTraits)
  assert isinstance(traits.plugin_traits, PluginTraits)
