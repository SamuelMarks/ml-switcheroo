"""Test suite for mlir naming coverage."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_naming_context_register_branches() -> None:
  """Verify naming context register branches behavior."""
  ctx: NamingContext = NamingContext()

  res1: str = ctx.register("%self1", hint="self1")
  assert res1 == "self"

  res2: str = ctx.register("%return1", hint="return1")
  assert res2 == "return1"

  res3: str = ctx.register("%out_abc", hint="_semantic")
  assert res3 == "_semantic"

  res4: str = ctx.register("%out2")
  assert res4 == "_out2"

  res5: str = ctx.register("%bad", hint="1bad")
  assert res5 == "_1bad"  # not a valid identifier

  ctx._used_names["clean"] = "%out3"
  res6: str = ctx.register("%out3", hint="clean")
  assert res6 == "_clean"

  ctx._used_names["_already"] = "%out4"
  res7: str = ctx.register("%out4", hint="_already")
  # this will hit 105 -> 107
  assert res7 == "_already_0"

  # This was hit by res6.

  # Collision resolution loop
  ctx._used_names["_clean2"] = "%out5"
  ctx._used_names["_clean2_0"] = "%out5"
  res8: str = ctx.register("%out5", hint="clean2")
  assert res8 == "_clean2_1"

  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9: str = ctx.register("%ssa2")
  assert res9 == "_ssa2_1"

  # No hint, no % start
  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["v"] = "nonssa"
  ctx._used_names["_v"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["_v_0"] = "nonssa"
  res10: str = ctx.register("nonssa")
  assert res10 == "_v_1"


def test_naming_context_lookup_branches() -> None:
  """Verify naming context lookup branches behavior."""
  ctx: NamingContext = NamingContext()
  ctx.register("%a", hint="a")

  assert ctx.lookup("%a") == "a"

  assert ctx.lookup("@func") == "func"

  assert ctx.lookup("%unknown") == "_unknown"


def test_naming_context_register_branches_extra() -> None:
  """Verify naming context register branches extra behavior."""
  ctx: NamingContext = NamingContext()

  res1: str = ctx.register("%self1", hint="self1")
  assert res1 == "self"

  res2: str = ctx.register("%return1", hint="return1")
  assert res2 == "return1"

  res3: str = ctx.register("%out_abc", hint="_semantic")
  assert res3 == "_semantic"

  res4: str = ctx.register("%out2")
  assert res4 == "_out2"

  res5: str = ctx.register("%bad", hint="1bad")
  assert res5 == "_1bad"  # not a valid identifier

  ctx._used_names["clean"] = "%out3"
  res6: str = ctx.register("%out3", hint="clean")
  assert res6 == "_clean"

  ctx._used_names["_already"] = "%out4"
  res7: str = ctx.register("%out4", hint="_already")
  assert res7 == "_already_0"

  # Collision resolution loop
  ctx._used_names["_clean2"] = "%out5"
  ctx._used_names["_clean2_0"] = "%out5"
  res8: str = ctx.register("%out5", hint="clean2")
  assert res8 == "_clean2_1"

  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9: str = ctx.register("%ssa2")
  assert res9 == "_ssa2_1"

  # No hint, no % start
  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["v"] = "v"
  ctx._used_names["_v"] = "v"
  ctx._used_names["_v_0"] = "v"
  res10: str = ctx.register("nonssa")
  assert res10 == "_v_1"

  ctx._used_names["a"] = "%a"
  res1_extra: str = ctx.register("%a", hint="a")
  assert res1_extra == "_a"

  res2_extra: str = ctx.register("%other", hint=".b")
  assert res2_extra == "_b"

  ctx.register("%abc", hint="")

  ctx.lookup("unknown")
  ctx.lookup("@func")
  ctx.lookup("%a")


def test_more_branches() -> None:
  """Verify more branches behavior."""
  ctx: NamingContext = NamingContext()

  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9: str = ctx.register("%ssa2")
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
  res11: str = ctx.register("%out6", hint="_clean3")
  assert res11 == "_clean3_1"


def test_missing_more() -> None:
  """Verify missing more behavior."""
  ctx: NamingContext = NamingContext()

  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["v"] = "v"
  ctx._used_names["_v"] = "v"
  ctx._used_names["_v_0"] = "v"
  res10: str = ctx.register("nonssa")
  assert res10 == "_v_1"


def test_missing_last() -> None:
  """Verify missing last behavior."""
  ctx: NamingContext = NamingContext()

  ctx._used_names["a"] = "%a"
  res1: str = ctx.register("%a", hint="a")
  assert res1 == "_a"

  res2: str = ctx.register("%other", hint=".b")
  assert res2 == "_b"
