"""
Tests for Infix and Prefix Operation Rewriting.

Verifies that:
1.  Binary operations (div, matmul) are rewritten to infix symbols (/, @).
2.  Unary operations (neg, bitwise_not) are rewritten to prefix symbols (-, ~).
3.  All supported Python operators map correctly.
4.  Invalid argument counts (arity mismatch) trigger proper fallbacks.
5.  Complex expressions are parenthesized correctly.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


class MockInfixSemantics(SemanticsManager):
  """
  Mock semantics defining a wide range of binary and unary operators
  to test the `_rewrite_as_infix` method comprehensively.
  """

  def __init__(self):
    super().__init__()
    self.data = {}
    self._reverse_index = {}

    # Define helper to inject ops
    def inject(name, s_api, op, arity=2):
      args = ["x", "y"] if arity == 2 else ["x"]
      variants = {
        "torch": {"api": s_api},
        "jax": {"transformation_type": "infix", "operator": op},
      }
      self.data[name] = {"variants": variants, "std_args": args}
      self._reverse_index[s_api] = (name, self.data[name])

    # --- Binary Operators ---
    inject("div", "torch.div", "/")
    inject("add", "torch.add", "+")
    inject("sub", "torch.sub", "-")
    inject("mul", "torch.mul", "*")
    inject("pow", "torch.pow", "**")
    inject("matmul", "torch.matmul", "@")
    inject("bit_and", "torch.bitwise_and", "&")
    inject("bit_or", "torch.bitwise_or", "|")
    inject("bit_xor", "torch.bitwise_xor", "^")
    inject("lshift", "torch.left_shift", "<<")
    inject("rshift", "torch.right_shift", ">>")
    # Missing Modulo/FloorDivide in Torch API standardly, but defining mock
    inject("mod", "torch.fmod", "%")

    # --- Unary Operators ---
    inject("neg", "torch.neg", "-", arity=1)
    inject("invert", "torch.bitwise_not", "~", arity=1)
    inject("logical_not", "torch.logical_not", "not", arity=1)

    # --- Invalid ---
    inject("bad_op", "torch.bad", "???")


@pytest.fixture
def engine():
  return ASTEngine(semantics=MockInfixSemantics(), source="torch", target="jax", strict_mode=True)


def test_infix_arithmetic_ops(engine):
  """Verify standard arithmetic generation."""
  assert "a / b" in engine.run("torch.div(a, b)").code
  assert "a + b" in engine.run("torch.add(a, b)").code
  assert "a - b" in engine.run("torch.sub(a, b)").code
  assert "a * b" in engine.run("torch.mul(a, b)").code
  assert "a ** b" in engine.run("torch.pow(a, b)").code
  assert "a % b" in engine.run("torch.fmod(a, b)").code


def test_infix_bitwise_ops(engine):
  """Verify bitwise operator generation."""
  assert "a & b" in engine.run("torch.bitwise_and(a, b)").code
  assert "a | b" in engine.run("torch.bitwise_or(a, b)").code
  assert "a ^ b" in engine.run("torch.bitwise_xor(a, b)").code
  assert "a << b" in engine.run("torch.left_shift(a, b)").code
  assert "a >> b" in engine.run("torch.right_shift(a, b)").code


def test_infix_matmul_reordering(engine):
  """Verify binary infix handles argument normalization first."""
  # Swapped args in call
  code = "res = torch.matmul(y=mat_b, x=mat_a)"
  result = engine.run(code)
  assert "mat_a @ mat_b" in result.code


def test_unary_operators(engine):
  """Verify unary prefix rewrite."""
  assert "-x" in engine.run("torch.neg(x)").code
  assert "~x" in engine.run("torch.bitwise_not(x)").code
  assert "not x" in engine.run("torch.logical_not(x)").code


def test_unary_complex_expression_parens(engine):
  """Verify unary operator applied to a complex expression adds parens."""
  code = "y = torch.neg(a + b)"
  result = engine.run(code)
  # -(a + b)
  assert "-(a + b)" in result.code


def test_infix_invalid_arg_count_binary(engine):
  """
  Input:  torch.div(a)  <- Missing arg for binary op.
  Expect: Fallback to original code w/ Error Marker.
  """
  code = "res = torch.div(start_val)"
  result = engine.run(code)

  # Check preservation
  assert "torch.div(start_val)" in result.code
  # Check error reporting
  assert "args=1" in str(result.errors) or len(result.errors) > 0
  assert "# Reason: Infix/Prefix transformation failed" in result.code


def test_infix_invalid_operator_symbol(engine):
  """
  Input:  torch.bad(a, b) maps to symbol '???'.
  Expect: Fallback and reporting.
  """
  code = "res = torch.bad(a, b)"
  result = engine.run(code)

  assert "torch.bad(a, b)" in result.code
  assert "# Reason: Infix/Prefix transformation failed" in result.code
  assert "Unsupported binary operator symbol: ???" in result.code
