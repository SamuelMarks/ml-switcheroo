"""Test module for SymbolType data structures.

This module validates the correctness of the symbol type classes (`SymbolType`, `TensorType`,
`ModuleType`, `UnionType`, and `Scope`) used by the SymbolTableAnalyzer to track and compare
type states during AST traversal.
"""

from ml_switcheroo.analysis.symbol_types import ModuleType, Scope, SymbolType, TensorType, UnionType


def test_symbol_type() -> None:
  """Test basic generic SymbolType behavior.

  Verifies that base `SymbolType` instances with identical names compare as equal,
  and differ from instances with other names or entirely different object types.
  """
  sym1: SymbolType = SymbolType()
  sym1.name = "Unknown"

  sym2: SymbolType = SymbolType()
  sym2.name = "Unknown"

  sym3: SymbolType = SymbolType()
  sym3.name = "Other"

  assert str(sym1) == "Unknown"
  assert sym1 == sym2
  assert sym1 != sym3
  assert sym1 != "Not a SymbolType"


def test_tensor_type() -> None:
  """Test framework-specific TensorType behavior.

  Verifies that `TensorType` correctly stores the framework name, automatically sets
  its base name to "Tensor", and implements equality such that tensors from the
  same framework are equal, but tensors from different frameworks are not.
  """
  t1: TensorType = TensorType(framework="torch")
  t2: TensorType = TensorType(framework="torch")
  t3: TensorType = TensorType(framework="jax")

  assert t1.name == "Tensor"
  assert t1 == t2
  assert t1 != t3
  assert t1 != "Not a TensorType"


def test_module_type() -> None:
  """Test ModuleType behavior for imported libraries.

  Verifies that `ModuleType` correctly tracks the import path (e.g. `torch.nn`),
  automatically sets its base name to "Module", and checks equality based on the path.
  """
  m1: ModuleType = ModuleType(path="torch.nn")
  m2: ModuleType = ModuleType(path="torch.nn")
  m3: ModuleType = ModuleType(path="jax.numpy")

  assert m1.name == "Module"
  assert m1 == m2
  assert m1 != m3
  assert m1 != "Not a ModuleType"


def test_union_type() -> None:
  """Test UnionType behavior for merged branch states.

  Verifies the string representation and equality logic of `UnionType`. Because
  equality is currently based on the set of the string representations of its
  member types, a Union of a PyTorch Tensor and a JAX Tensor (both stringifying
  to "Tensor") evaluates as equal to a Union containing only one of them.
  """
  t_torch: TensorType = TensorType(framework="torch")
  t_jax: TensorType = TensorType(framework="jax")

  u1: UnionType = UnionType(types=[t_torch, t_jax])
  u2: UnionType = UnionType(types=[t_jax, t_torch])
  u3: UnionType = UnionType(types=[t_torch])

  assert str(u1) == "Union[Tensor]"  # they both evaluate to "Tensor" string
  assert u1 == u2  # they contain same string representations
  assert u1 == u3  # Wait, is u1 == u3 because the string repr is just "Tensor"?
  # Let's check the eq implementation: return set(str(t) for t in self.types) == set(str(t) for t in other.types)
  # Yes, since both are "Tensor", the set is {"Tensor"}. So they are equal.

  assert u1 != "Not a UnionType"


def test_scope() -> None:
  """Test hierarchical Scope resolution.

  Verifies that a child `Scope` can correctly set and get its own local symbols,
  and transparently fall back to its parent `Scope` when looking up symbols
  that exist only in higher scopes. Also verifies that `snapshot()` only returns
  the local symbols.
  """
  root: Scope = Scope(name="root")
  root.set("x", TensorType(framework="torch"))

  child: Scope = Scope(parent=root, name="child")
  child.set("y", TensorType(framework="jax"))

  assert root.get("x") == TensorType(framework="torch")
  assert root.get("y") is None

  assert child.get("x") == TensorType(framework="torch")
  assert child.get("y") == TensorType(framework="jax")
  assert child.get("z") is None

  snap: dict[str, SymbolType] = child.snapshot()
  assert snap["y"] == TensorType(framework="jax")
  assert "x" not in snap
