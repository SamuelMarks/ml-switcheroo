"""Tests for compiler registry mapping functions."""

from ml_switcheroo.core.compiler.registry import (
  get_backend_class,
  is_isa_target,
  is_isa_source,
  SassBackend,
  PythonBackend,
)


def test_get_backend_class():
  """Verifies the get_backend_class function."""
  # Known backend
  assert get_backend_class("sass") is SassBackend
  # Fallback
  assert get_backend_class("unknown_target") is PythonBackend


def test_is_isa_target():
  """Verifies the is_isa_target function."""
  assert is_isa_target("sass") is True
  assert is_isa_target("mlir") is True
  assert is_isa_target("python") is False


def test_is_isa_source():
  """Verifies the is_isa_source function."""
  assert is_isa_source("sass") is True
  assert is_isa_source("rdna") is True
  assert is_isa_source("python") is False
