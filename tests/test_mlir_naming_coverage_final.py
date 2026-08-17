"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_naming_context_register_branches():
  """Docstring."""
  ctx = NamingContext()

  # 70 -> 71 (match ends with digits)
  # 73 -> 74 (safe to strip digits)
  res1 = ctx.register("%self1", hint="self1")
  assert res1 == "self"

  # 73 -> 76 (stripped name is reserved)
  res2 = ctx.register("%return1", hint="return1")
  assert res2 == "return1"

  # 88 -> 94 (semantic hint, already starts with _)
  res3 = ctx.register("%out_abc", hint="_semantic")
  assert res3 == "_semantic"

  # 91 -> 94 (no hint, starts with %)
  res4 = ctx.register("%out2")
  assert res4 == "_out2"

  # 97 -> 99 (invalid identifier)
  res5 = ctx.register("%bad", hint="1bad")
  assert res5 == "_1bad"  # not a valid identifier

  # 99 -> 100 (needs _)
  ctx._used_names["clean"] = "%out3"
  res6 = ctx.register("%out3", hint="clean")
  assert res6 == "_clean"

  # 99 -> 102 (already has _)
  ctx._used_names["_already"] = "%out4"
  res7 = ctx.register("%out4", hint="_already")
  assert res7 == "_already_0"

  # 111 -> 112 -> 113 -> 114
  # Collision resolution loop
  ctx._used_names["_clean2"] = "%out5"
  ctx._used_names["_clean2_0"] = "%out5"
  res8 = ctx.register("%out5", hint="clean2")
  assert res8 == "_clean2_1"

  # 111 -> 115 -> 116
  ctx._used_names["_ssa2"] = "%ssa2"
  ctx._used_names["_ssa2_0"] = "%ssa2"
  res9 = ctx.register("%ssa2")
  assert res9 == "_ssa2_1"

  # 115 -> 118
  # No hint, no % start
  ctx._used_names["nonssa"] = "nonssa"
  ctx._used_names["_nonssa"] = "nonssa"
  ctx._used_names["v"] = "v"
  ctx._used_names["_v"] = "v"
  ctx._used_names["_v_0"] = "v"
  res10 = ctx.register("nonssa")
  assert res10 == "_v_1"

  # 88 -> 94
  ctx._used_names["a"] = "%a"
  res1 = ctx.register("%a", hint="a")
  assert res1 == "_a"

  res2 = ctx.register("%other", hint=".b")
  assert res2 == "_b"

  # 63 -> 65: hint is falsy
  ctx.register("%abc", hint="")

  # 141 -> 145 -> 149
  ctx.lookup("unknown")
  ctx.lookup("@func")
  ctx.lookup("%a")
