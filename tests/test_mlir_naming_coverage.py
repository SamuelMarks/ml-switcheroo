"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_naming_context_register_branches() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()

  # 70 -> 71 (match ends with digits)
  # 73 -> 74 (safe to strip digits)
  res1: str = ctx.register("%self1", hint="self1")
  assert res1 == "self"

  # 73 -> 76 (stripped name is reserved)
  res2: str = ctx.register("%return1", hint="return1")
  assert res2 == "return1"

  # 88 -> 94 (semantic hint, already starts with _)
  res3: str = ctx.register("%out_abc", hint="_semantic")
  assert res3 == "_semantic"

  # 91 -> 94 (no hint, starts with %)
  res4: str = ctx.register("%out2")
  assert res4 == "_out2"

  # 97 -> 99 (invalid identifier)
  res5: str = ctx.register("%bad", hint="1bad")
  assert res5 == "_1bad"  # not a valid identifier

  # 99 -> 100 (needs _)
  ctx._used_names["clean"] = "%out3"
  res6: str = ctx.register("%out3", hint="clean")
  assert res6 == "_clean"

  # 99 -> 102 (already has _)
  ctx._used_names["_already"] = "%out4"
  res7: str = ctx.register("%out4", hint="_already")
  # this will hit 105 -> 107
  assert res7 == "_already_0"

  # 105 -> 125 (attempt valid and not used)
  # This was hit by res6.

  # 111 -> 112 -> 113 -> 114
  # Collision resolution loop
  ctx._used_names["_clean2"] = "%out5"
  ctx._used_names["_clean2_0"] = "%out5"
  res8: str = ctx.register("%out5", hint="clean2")
  assert res8 == "_clean2_1"

  # 111 -> 115 -> 116
  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9: str = ctx.register("%ssa2")
  assert res9 == "_ssa2_1"

  # 115 -> 118
  # No hint, no % start
  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["v"] = "nonssa"
  ctx._used_names["_v"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["_v_0"] = "nonssa"
  res10: str = ctx.register("nonssa")
  assert res10 == "_v_1"

  # 141 -> 142 (lookup branch inside register indirectly, let's just make sure)


def test_naming_context_lookup_branches() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  ctx.register("%a", hint="a")

  # 141 -> 142
  assert ctx.lookup("%a") == "a"

  # 141 -> 145 (not in map)
  # 145 -> 146
  assert ctx.lookup("@func") == "func"

  # 145 -> 149
  assert ctx.lookup("%unknown") == "_unknown"


# --- Merged from test_mlir_naming_coverage_final.py ---


def test_naming_context_register_branches_extra() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()

  # 70 -> 71 (match ends with digits)
  # 73 -> 74 (safe to strip digits)
  res1: str = ctx.register("%self1", hint="self1")
  assert res1 == "self"

  # 73 -> 76 (stripped name is reserved)
  res2: str = ctx.register("%return1", hint="return1")
  assert res2 == "return1"

  # 88 -> 94 (semantic hint, already starts with _)
  res3: str = ctx.register("%out_abc", hint="_semantic")
  assert res3 == "_semantic"

  # 91 -> 94 (no hint, starts with %)
  res4: str = ctx.register("%out2")
  assert res4 == "_out2"

  # 97 -> 99 (invalid identifier)
  res5: str = ctx.register("%bad", hint="1bad")
  assert res5 == "_1bad"  # not a valid identifier

  # 99 -> 100 (needs _)
  ctx._used_names["clean"] = "%out3"
  res6: str = ctx.register("%out3", hint="clean")
  assert res6 == "_clean"

  # 99 -> 102 (already has _)
  ctx._used_names["_already"] = "%out4"
  res7: str = ctx.register("%out4", hint="_already")
  assert res7 == "_already_0"

  # 111 -> 112 -> 113 -> 114
  # Collision resolution loop
  ctx._used_names["_clean2"] = "%out5"
  ctx._used_names["_clean2_0"] = "%out5"
  res8: str = ctx.register("%out5", hint="clean2")
  assert res8 == "_clean2_1"

  # 111 -> 115 -> 116
  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9: str = ctx.register("%ssa2")
  assert res9 == "_ssa2_1"

  # 115 -> 118
  # No hint, no % start
  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["v"] = "v"
  ctx._used_names["_v"] = "v"
  ctx._used_names["_v_0"] = "v"
  res10: str = ctx.register("nonssa")
  assert res10 == "_v_1"

  # 88 -> 94
  ctx._used_names["a"] = "%a"
  res1_extra: str = ctx.register("%a", hint="a")
  assert res1_extra == "_a"

  res2_extra: str = ctx.register("%other", hint=".b")
  assert res2_extra == "_b"

  # 63 -> 65: hint is falsy
  ctx.register("%abc", hint="")

  # 141 -> 145 -> 149
  ctx.lookup("unknown")
  ctx.lookup("@func")
  ctx.lookup("%a")


# --- Merged from test_mlir_naming_coverage_extra.py ---


def test_more_branches() -> None:
  """Docstring."""
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


# --- Merged from test_mlir_naming_coverage_extra2.py ---


def test_missing_more() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()

  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["v"] = "v"
  ctx._used_names["_v"] = "v"
  ctx._used_names["_v_0"] = "v"
  res10: str = ctx.register("nonssa")
  assert res10 == "_v_1"


# --- Merged from test_mlir_naming_coverage_extra3.py ---


def test_missing_last() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()

  ctx._used_names["a"] = "%a"
  res1: str = ctx.register("%a", hint="a")
  assert res1 == "_a"

  res2: str = ctx.register("%other", hint=".b")
  assert res2 == "_b"
