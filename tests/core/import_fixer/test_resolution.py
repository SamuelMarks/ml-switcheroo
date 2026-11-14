"""Unit tests for ``ImportReq``, ``ResolutionPlan``, ``ImportResolver``, and scanners in ``resolution.py``."""

from unittest.mock import Mock

import libcst as cst
import pytest

from ml_switcheroo.core.import_fixer.resolution import (
  ImportReq,
  ImportResolver,
  ResolutionPlan,
  _deduplicate,
  _QualNameScanner,
)
from ml_switcheroo.semantics.manager import SemanticsManager


class StubSemanticsManager(SemanticsManager):
  """Lightweight ``SemanticsManager`` test double returning deterministic alias and import maps."""

  def __init__(
    self,
    aliases: dict[str, tuple[str, str]] | None = None,
    import_maps: dict[str, dict[str, tuple[str, str | None, str | None]]] | None = None,
  ) -> None:
    """Initialize the stub manager without loading filesystem defaults.

    Args:
        aliases: Mapping of framework names to ``(module_path, alias_name)`` pairs.
        import_maps: Per-target-framework mapping of source paths to ``(root, sub, alias)`` tuples.
    """
    super().__init__(load_defaults=False)
    self._aliases: dict[str, tuple[str, str]] = aliases if aliases is not None else {}
    self._import_maps: dict[str, dict[str, tuple[str, str | None, str | None]]] = (
      import_maps if import_maps is not None else {}
    )

  def get_framework_aliases(self) -> dict[str, tuple[str, str]]:
    """Return configured framework alias mappings.

    Returns:
        Dictionary mapping framework identifiers to ``(module_path, alias_name)`` pairs.
    """
    return self._aliases

  def get_import_map(self, target_framework: str) -> dict[str, tuple[str, str | None, str | None]]:
    """Return configured submodule import mappings for ``target_framework``.

    Args:
        target_framework: Target framework identifier.

    Returns:
        Dictionary mapping source module paths to ``(root, subcomponent, alias)`` tuples.
    """
    return self._import_maps.get(target_framework, {})


def test_import_req_signature() -> None:
  """Verify ``ImportReq.signature`` normalizes redundant aliases and retains explicit aliases."""
  assert ImportReq(module="torch").signature == "torch"
  assert ImportReq(module="torch", subcomponent="nn").signature == "torch.nn"
  assert ImportReq(module="torch", subcomponent="nn", alias="nn").signature == "torch.nn"
  assert ImportReq(module="torch", subcomponent="nn", alias="my_nn").signature == "torch.nn : my_nn"
  assert ImportReq(module="jax.numpy", alias="jnp").signature == "jax.numpy : jnp"
  assert ImportReq(module="jax.numpy", alias="numpy").signature == "jax.numpy"
  assert ImportReq(module="jax", alias="jax").signature == "jax"


def test_resolution_plan_initialization() -> None:
  """Verify ``ResolutionPlan`` supports both default empty collections and explicit values."""
  empty_plan = ResolutionPlan()
  assert empty_plan.required_imports == []
  assert empty_plan.mappings == {}
  assert empty_plan.path_to_alias == {}
  assert empty_plan.target_framework is None

  req = ImportReq(module="jax")
  populated_plan = ResolutionPlan(
    required_imports=[req],
    mappings={"torch": req},
    path_to_alias={"jax.numpy": "jnp"},
    target_framework="jax",
  )
  assert populated_plan.required_imports == [req]
  assert populated_plan.mappings == {"torch": req}
  assert populated_plan.path_to_alias == {"jax.numpy": "jnp"}
  assert populated_plan.target_framework == "jax"


def test_deduplicate() -> None:
  """Verify ``_deduplicate`` preserves insertion order while collapsing identical signatures."""
  reqs = [
    ImportReq(module="torch"),
    ImportReq(module="torch", alias="torch"),
    ImportReq(module="jax", subcomponent="numpy", alias="jnp"),
    ImportReq(module="jax", subcomponent="numpy", alias="jnp"),
  ]
  deduped = _deduplicate(reqs)
  assert len(deduped) == 2
  assert deduped[0].signature == "torch"
  assert deduped[1].signature == "jax.numpy : jnp"


def test_qual_name_scanner(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify ``_QualNameScanner`` matches exact names, attribute prefixes, early exits, and error paths.

  Args:
      monkeypatch: Pytest monkeypatch fixture used to simulate ``get_full_name`` failures.
  """
  tree = cst.parse_module(
    """import jax.numpy
x = jax.numpy.sum()
y = 5"""
  )

  scanner_prefix = _QualNameScanner("jax.numpy")
  tree.visit(scanner_prefix)
  assert scanner_prefix.found is True

  scanner_miss = _QualNameScanner("flax.nnx")
  tree.visit(scanner_miss)
  assert scanner_miss.found is False

  scanner_name = _QualNameScanner("y")
  tree.visit(scanner_name)
  assert scanner_name.found is True

  scanner_name_miss = _QualNameScanner("z")
  tree.visit(scanner_name_miss)
  assert scanner_name_miss.found is False

  scanner_already_found = _QualNameScanner("y")
  scanner_already_found.found = True
  scanner_already_found.visit_Name(cst.Name("y"))
  scanner_already_found.visit_Attribute(cst.Attribute(value=cst.Name("y"), attr=cst.Name("attr")))
  assert scanner_already_found.found is True

  monkeypatch.setattr(
    "ml_switcheroo.core.import_fixer.resolution.get_full_name",
    Mock(side_effect=ValueError("Simulated Error")),
  )
  scanner_err = _QualNameScanner("some.path")
  scanner_err.visit_Attribute(cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")))
  assert scanner_err.found is False


def test_import_resolver_aliases_and_submodules() -> None:
  """Verify ``ImportResolver.resolve`` detects base framework, alias, and submodule usages."""
  semantics = StubSemanticsManager(
    aliases={"jax": ("jax.numpy", "jnp")},
    import_maps={
      "jax": {
        "torch.nn": ("flax", "nnx", "nnx"),
        "torch.nn.functional": ("jax", "nn", "F"),
        "torch.lax": ("jax", "lax", None),
        "torch.optim": ("optax", None, None),
      }
    },
  )
  resolver = ImportResolver(semantics=semantics)

  plan_base = resolver.resolve(
    cst.parse_module(
      """import jax
jax.sum()"""
    ),
    "jax",
  )
  assert any(r.signature == "jax" for r in plan_base.required_imports)

  plan_alias = resolver.resolve(cst.parse_module("jnp.array([1, 2])"), "jax")
  assert any(r.signature == "jax.numpy : jnp" for r in plan_alias.required_imports)
  assert plan_alias.path_to_alias["jax.numpy"] == "jnp"

  plan_mod_path = resolver.resolve(cst.parse_module("jax.numpy.array([1, 2])"), "jax")
  assert any(r.signature == "jax.numpy : jnp" for r in plan_mod_path.required_imports)

  plan_nnx = resolver.resolve(cst.parse_module("nnx.Linear()"), "jax")
  assert any(r.signature == "flax.nnx" for r in plan_nnx.required_imports)
  assert plan_nnx.mappings["torch.nn"].module == "flax"
  assert plan_nnx.path_to_alias["flax.nnx"] == "nnx"

  plan_full_subpath = resolver.resolve(cst.parse_module("jax.nn.relu(x)"), "jax")
  assert any(r.signature == "jax.nn : F" for r in plan_full_subpath.required_imports)
  assert plan_full_subpath.path_to_alias["jax.nn"] == "F"

  plan_sub_no_alias = resolver.resolve(cst.parse_module("lax.sin(x)"), "jax")
  assert any(r.signature == "jax.lax" for r in plan_sub_no_alias.required_imports)

  plan_root_no_alias = resolver.resolve(cst.parse_module("optax.adam()"), "jax")
  assert any(r.signature == "optax" for r in plan_root_no_alias.required_imports)

  plan_unused = resolver.resolve(cst.parse_module("x = 1"), "jax")
  assert plan_unused.required_imports == []

  plan_no_alias_fw = resolver.resolve(cst.parse_module("x = 1"), "torch")
  assert plan_no_alias_fw.required_imports == []
