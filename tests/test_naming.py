"""Module docstring."""

from ml_switcheroo.core.mlir.naming import NamingContext


def test_naming_context_init() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  assert ctx._map == {}
  assert ctx._used_names == {}
  assert "return" in ctx._reserved
  assert "self" not in ctx._reserved


def test_naming_context_register_no_hint() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  # If ssa_name starts with %, replaces with _
  assert ctx.register("%0") == "_0"
  assert "%0" in ctx._map
  assert ctx._used_names["_0"] is True


def test_naming_context_register_with_hint_strip_digits() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  # Strip digits if candidate is available
  assert ctx.register("%self0", hint="self0") == "self"
  assert ctx._used_names["self"] is True

  # Strip digits but candidate used (gets prefixed because hint does not match ssa_name)
  assert ctx.register("%x1", hint="self1") == "_self1"  # 'self' is used


def test_naming_context_register_with_semantic_hint() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  # If hint does not match ssa_name (e.g. op type hint), adds _
  assert ctx.register("%1", hint="flatten") == "_flatten"


def test_naming_context_register_collision_and_fallback() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  # Reserve a name manually to force collision
  ctx._used_names["_myvar"] = True

  # Prepend _ if not already, but it's used, so indexed fallback
  assert ctx.register("%myvar") == "_myvar_0"
  assert ctx.register("%myvar_another", hint="myvar") == "myvar"


def test_naming_context_register_reserved() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  assert ctx.register("%return", hint="return") == "_return"


def test_naming_context_register_invalid_identifier() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  # hint has bad chars
  assert ctx.register("%2", hint="invalid-name") == "_invalid-name_0"


def test_naming_context_lookup() -> None:
  """Docstring."""
  ctx: NamingContext = NamingContext()
  ctx.register("%val", hint="val")

  assert ctx.lookup("%val") == "val"
  assert ctx.lookup("@my_func") == "my_func"
  assert ctx.lookup("%unregistered") == "_unregistered"
