"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_missing_more():
  """Docstring."""
  ctx = NamingContext()

  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["v"] = "v"
  ctx._used_names["_v"] = "v"
  ctx._used_names["_v_0"] = "v"
  res10 = ctx.register("nonssa")
  assert res10 == "_v_1"
