"""Tests for the API helpers mixin used by transformers."""

import libcst as _cst

from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin

cst = _cst


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize the dummy context."""
    self.target_framework = "flax_nnx"
    self.alias_map = {"my_root": "canonical_root"}


class DummyTransformer(ApiHelpersMixin):
  """Dummy transformer using the API helpers mixin."""

  def __init__(self) -> None:
    """Initialize the dummy transformer."""
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "context",
      getattr({"v": getattr({"v": DummyContext}, "__getitem__")("v")()}, "__getitem__")("v"),
    )

  def _cst_to_string(self, node: cst.CSTNode) -> str:
    """Convert a CST node to a string.

    Args:
        node (cst.CSTNode): The node.

    Returns:
        str: String value.
    """
    return getattr(node, "value", str(node))


def test_api_helpers_get_qualified_name_short() -> None:
  """Verify api helpers get qualified name short behavior."""
  p = DummyTransformer()
  node = cst.Name("my_root")
  res = p._get_qualified_name(node)
  assert res == "canonical_root"


def test_api_helpers_inject_stmts_to_body_simple() -> None:
  """Verify api helpers inject stmts to body simple behavior."""
  p = DummyTransformer()
  node = cst.parse_statement("def foo(): pass")
  assert isinstance(node, cst.FunctionDef)
  new_stmts = getattr(
    {
      "v": [
        getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_statement}, "__getitem__")("v")(
                      getattr({"v": "x = 1"}, "__getitem__")("v")
                    )
                  },
                  "__getitem__",
                )("v").body
              },
              "__getitem__",
            )("v")[0]
          },
          "__getitem__",
        )("v")
      ]
    },
    "__getitem__",
  )("v")
  res = getattr(
    {
      "v": getattr({"v": getattr({"v": p}, "__getitem__")("v")._inject_stmts_to_body}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": new_stmts}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(res.body, cst.IndentedBlock)
  assert len(res.body.body) == 2


def test_api_helpers_inject_argument_to_signature_default_comma() -> None:
  """Verify api helpers inject argument to signature default comma behavior."""
  p = DummyTransformer()
  node = cst.parse_statement("def foo(self): pass")
  assert isinstance(node, cst.FunctionDef)
  # Provide a dummy method to simulate create_dotted_name since it creates AST nodes
  setattr(p, "_create_dotted_name", lambda name: cst.Name(name))
  res = p._inject_argument_to_signature(node, "new_arg", None)

  params = list(res.params.params)
  assert len(params) == 2
  assert params[0].name.value == "self"
  assert isinstance(params[0].comma, cst.Comma)
  assert params[1].name.value == "new_arg"
