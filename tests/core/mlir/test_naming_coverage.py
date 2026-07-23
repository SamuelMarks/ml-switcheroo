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
