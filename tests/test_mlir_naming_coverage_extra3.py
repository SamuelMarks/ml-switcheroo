"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_missing_last() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()

  ctx._used_names["a"] = "%a"
  res1: str = ctx.register("%a", hint="a")
  assert res1 == "_a"

  res2: str = ctx.register("%other", hint=".b")
  assert res2 == "_b"
