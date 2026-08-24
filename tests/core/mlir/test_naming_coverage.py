"""Tests for MLIR naming coverage."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_naming_collision():
  """Test naming collision."""
  ctx = NamingContext()
  # To hit 100 and 123: py_name="coll", collision so attempt="_coll"
  ctx._used_names["coll"] = True
  res1 = ctx.register("%coll", hint="coll")
  assert res1 == "_coll"

  # To hit 119: attempt="_coll2", but _coll2 is also used
  ctx._used_names["coll2"] = True
  ctx._used_names["_coll2"] = True
  res2 = ctx.register("%coll2", hint="coll2")
  # attempt="_coll2" -> in used_names -> count loop
  # prefix = coll2, count = 0 -> coll2_0
  assert res2 == "_coll2_0"


def test_name_registry_fallback_loop():
  """Docstring."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  reg = NamingContext()
  # ssa_name="v". base="v". py_name="v".
  # v is an identifier, not in reserved, but IN used_names if we put it there
  # attempt="_v".
  # if attempt in used_names -> count loop
  # prefix="v". else -> prefix="_v".
  # attempt = "_v_0".
  reg._used_names["v"] = True
  reg._used_names["_v"] = True
  reg._used_names["_v_0"] = True
  reg._used_names["_v_1"] = True

  name = reg.register("v")
  assert name == "_v_2"


def test_name_registry_hint_starts_without_underscore():
  """Docstring."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  reg = NamingContext()
  reg._used_names["_myhint"] = True
  reg._used_names["_myhint_0"] = True
  name = reg.register("ssa1", hint="myhint")
  assert name == "_myhint_1"


def test_name_registry_hint_starts_with_underscore():
  """Docstring."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  reg = NamingContext()
  reg._used_names["_myhint"] = True
  reg._used_names["_myhint_0"] = True
  name = reg.register("ssa1", hint="_myhint")
  assert name == "_myhint_1"


def test_name_registry_hint_starts_with_underscore_loop():
  """Docstring."""
  from ml_switcheroo.core.mlir.naming import NamingContext

  reg = NamingContext()
  reg._used_names["_ssa1"] = True
  reg._used_names["_a"] = True
  name = reg.register("%a")
  assert name == "_a_0"
