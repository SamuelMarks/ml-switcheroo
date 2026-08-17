"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_missing_last():
  """Docstring."""
  ctx = NamingContext()

  ctx._used_names["a"] = "%a"
  res1 = ctx.register("%a", hint="a")
  assert res1 == "_a"

  res2 = ctx.register("%other", hint=".b")
  assert res2 == "_b"
