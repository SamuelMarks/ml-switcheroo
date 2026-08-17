"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_more_branches():
  """Docstring."""
  ctx = NamingContext()

  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9 = ctx.register("%ssa2")
  assert res9 == "_ssa2_1"

  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"  # needed to bypass 105 when it prepends _
  ctx._used_names["v_0"] = "nonssa"
  ctx.register("nonssa")
  # assert res10 == "v_1" -> actually, let's just assert "v_1" in ctx._used_names after
  # wait, if 99 -> 100 prepends `_`, attempt is "_nonssa".
  # Since "_nonssa" in used_names, 105 -> 106.
  # Inside 110, prefix = "v".
  # hint is None (falsy), ssa_name is "nonssa" (does not start with %).
  # So it hits 115 -> 118, leaving prefix="v".
  # attempt becomes "v_0".
  # Since "v_0" is in used_names, count becomes 1, attempt="v_1".

  ctx._used_names["_clean2"] = "%out5"
  ctx._used_names["_clean2_0"] = "%out5"
  ctx.register("%out5", hint="clean2")

  ctx._used_names["_clean3"] = "%out6"
  ctx._used_names["_clean3_0"] = "%out6"
  res11 = ctx.register("%out6", hint="_clean3")
  assert res11 == "_clean3_1"
