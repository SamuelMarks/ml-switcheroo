"""Tests for compiler registry mapping functions."""

from ml_switcheroo.core.compiler.registry import (
  PythonBackend,
  NvidiaSassBackend,
  get_backend_class,
  is_isa_source,
  is_isa_target,
)


def test_get_backend_class():
  """Verifies the get_backend_class function."""
  # Known backend
  assert get_backend_class("nvidia_sass") is NvidiaSassBackend
  # Fallback
  assert get_backend_class("unknown_target") is PythonBackend


def test_is_isa_target():
  """Verifies the is_isa_target function."""
  assert is_isa_target("nvidia_sass") is True
  assert is_isa_target("mlir") is True
  assert is_isa_target("python") is False


def test_is_isa_source():
  """Verifies the is_isa_source function."""
  assert is_isa_source("nvidia_sass") is True
  assert is_isa_source("rdna") is True
  assert is_isa_source("python") is False
