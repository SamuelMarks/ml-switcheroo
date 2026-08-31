"""Test module."""

from typing import Dict, List, Optional

import libcst as cst

from ml_switcheroo.core.import_fixer.resolution import (
  ImportReq,
  ImportResolver,
  ResolutionPlan,
  _deduplicate,
  _QualNameScanner,
)


def test_importreq_signature() -> None:
  """Docstring."""
  req1: ImportReq = ImportReq(module="torch")
  assert req1.signature == "torch"

  req2: ImportReq = ImportReq(module="torch", subcomponent="nn")
  assert req2.signature == "torch.nn"

  req3: ImportReq = ImportReq(module="torch", subcomponent="nn", alias="nn")
  assert req3.signature == "torch.nn"

  req4: ImportReq = ImportReq(module="torch", subcomponent="nn", alias="my_nn")
  assert req4.signature == "torch.nn : my_nn"

  req5: ImportReq = ImportReq(module="jax.numpy", alias="jnp")
  assert req5.signature == "jax.numpy : jnp"

  req6: ImportReq = ImportReq(module="jax.numpy", alias="numpy")
  assert req6.signature == "jax.numpy"


def test_qual_name_scanner() -> None:
  """Docstring."""
  tree: cst.Module = cst.parse_module("import jax.numpy\nx = jax.numpy.sum()")
  scanner: _QualNameScanner = _QualNameScanner("jax.numpy")
  tree.visit(scanner)
  assert scanner.found is True

  tree2: cst.Module = cst.parse_module("x = jnp.sum()")
  scanner2: _QualNameScanner = _QualNameScanner("jax.numpy")
  tree2.visit(scanner2)
  assert scanner2.found is False

  # visit_Name
  tree3: cst.Module = cst.parse_module("jax_numpy_sum()")
  scanner3: _QualNameScanner = _QualNameScanner("jax_numpy_sum")
  tree3.visit(scanner3)
  assert scanner3.found is True

  # exception path - simulate get_full_name error (not easy with cst nodes but let's test early return)
  scanner4: _QualNameScanner = _QualNameScanner("jax.numpy")
  scanner4.found = True
  scanner4.visit_Attribute(cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")))
  scanner4.visit_Name(cst.Name("a"))
  assert scanner4.found is True

  # Bad Attribute visit
  scanner5: _QualNameScanner = _QualNameScanner("jax.numpy")

  # cst.Attribute needs correct types but we can pass something to trigger exception inside get_full_name if we wanted,
  # but `get_full_name` is robust. Let's just create an invalid node dynamically.
  class BadNode(cst.Attribute):
    """Docstring."""

    @property
    def value(self) -> cst.BaseExpression:
      """Docstring."""
      raise ValueError("bad")

  try:
    scanner5.visit_Attribute(BadNode(value=cst.Name("a"), attr=cst.Name("b")))
  except Exception:
    pass
  assert scanner5.found is False


class MockSemantics:
  """Docstring."""

  def get_framework_aliases(self) -> Dict[str, tuple[str, str]]:
    """Docstring."""
    return {"jax": ("jax.numpy", "jnp")}

  def get_import_map(self, target_fw: str) -> Dict[str, tuple[str, str, str]]:
    """Docstring."""
    if target_fw == "jax":
      return {
        "torch.nn": ("flax", "nnx", "nnx"),
      }
    return {}


def test_import_resolver() -> None:
  """Docstring."""
  sm: MockSemantics = MockSemantics()
  resolver: ImportResolver = ImportResolver(sm)

  # Test 1: Framework Base Check
  tree1: cst.Module = cst.parse_module("import jax\njax.sum()")
  plan: ResolutionPlan = resolver.resolve(tree1, "jax")
  assert any(r.signature == "jax" for r in plan.required_imports)

  # Test 2: Framework Alias Check - by alias name
  tree2: cst.Module = cst.parse_module("jnp.array([1, 2])")
  plan2: ResolutionPlan = resolver.resolve(tree2, "jax")
  assert any(r.signature == "jax.numpy : jnp" for r in plan2.required_imports)
  assert plan2.path_to_alias["jax.numpy"] == "jnp"

  # Test 2: Framework Alias Check - by mod path
  tree3: cst.Module = cst.parse_module("jax.numpy.array([1, 2])")
  plan3: ResolutionPlan = resolver.resolve(tree3, "jax")
  assert any(r.signature == "jax.numpy : jnp" for r in plan3.required_imports)

  # Test 3: Import Map Check - by alias name
  tree4: cst.Module = cst.parse_module("nnx.Linear()")
  plan4: ResolutionPlan = resolver.resolve(tree4, "jax")
  assert any(r.signature == "flax.nnx" for r in plan4.required_imports)
  assert plan4.mappings["torch.nn"].module == "flax"
  assert plan4.path_to_alias["flax.nnx"] == "nnx"

  # Test 3: Import Map Check - by full path
  tree5: cst.Module = cst.parse_module("flax.nnx.Linear()")
  plan5: ResolutionPlan = resolver.resolve(tree5, "jax")
  assert any(r.signature == "flax.nnx" for r in plan5.required_imports)

  # Test 3: Check_name fallback
  class MockSemantics2:
    """Docstring."""

    def get_framework_aliases(self) -> Dict[str, tuple[str, str]]:
      """Docstring."""
      return {}

    def get_import_map(self, target_fw: str) -> Dict[str, tuple[str, Optional[str], Optional[str]]]:
      """Docstring."""
      return {"torch.optim": ("optax", None, None)}

  sm2: MockSemantics2 = MockSemantics2()
  resolver2: ImportResolver = ImportResolver(sm2)
  tree6: cst.Module = cst.parse_module("optax.adam()")
  plan6: ResolutionPlan = resolver2.resolve(tree6, "jax")
  assert any(r.signature == "optax" for r in plan6.required_imports)

  # Check deduplicate
  reqs: List[ImportReq] = [ImportReq("a"), ImportReq("a"), ImportReq("b")]
  dedup: List[ImportReq] = _deduplicate(reqs)
  assert len(dedup) == 2
  assert dedup[0].signature == "a"
  assert dedup[1].signature == "b"
