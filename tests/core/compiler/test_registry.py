"""Unit tests for compiler backend/frontend registry helpers (`ml_switcheroo.core.compiler.registry`)."""

import pytest

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.registry import (
  GraphFrontend,
  NvidiaSassBackend,
  get_backend_class,
  instantiate_backend,
  is_isa_source,
  is_isa_target,
)


def test_compiler_registry_lookup_instantiation_predicates_and_abstract_frontend() -> None:
  """Verify `get_backend_class`, `instantiate_backend`, `is_isa_target`/`is_isa_source`, and `GraphFrontend`."""
  assert get_backend_class("nvidia_sass") is NvidiaSassBackend
  assert get_backend_class("unknown_target") is PythonBackend

  ir_cls = get_backend_class("ir")
  cuda_cls = get_backend_class("cuda")
  assert ir_cls is not None
  assert cuda_cls is not None
  assert isinstance(instantiate_backend(PythonBackend, "jax", None), PythonBackend)
  assert instantiate_backend(ir_cls, "ir", None) is not None
  assert instantiate_backend(ir_cls, "ml_switcheroo_ir", None) is not None
  assert instantiate_backend(cuda_cls, "cuda", None) is not None
  assert isinstance(instantiate_backend(NvidiaSassBackend, "nvidia_sass", None), NvidiaSassBackend)

  assert is_isa_target("nvidia_sass") is True
  assert is_isa_target("mlir") is True
  assert is_isa_target("python") is False

  assert is_isa_source("nvidia_sass") is True
  assert is_isa_source("rdna") is True
  assert is_isa_source("python") is False

  class _DummyFrontend(GraphFrontend):
    """Subclass inheriting unimplemented `parse_to_graph`."""

  with pytest.raises(NotImplementedError):
    _DummyFrontend().parse_to_graph("some code")
