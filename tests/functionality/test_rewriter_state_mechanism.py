"""Test suite for the Rewriter State Mechanism module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockStateSemantics(SemanticsManager):
  """Test helper class MockStateSemantics."""

  def __init__(self) -> None:
    """Initializes the MockStateSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self._providers = {}
    self._source_registry = {}
    self._known_rng_methods: set[str] = set()
    self._validation_status = {}
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "framework_configs",
      getattr(
        {
          "v": {
            "tensorflow": {"stateful_call": {"method": "apply", "prepend_arg": "variables"}},
            "mlx": getattr(
              {
                "v": {
                  "stateful_call": {"method": "call_fn", "prepend_arg": "ctx"},
                  "traits": getattr(
                    {
                      "v": {
                        "inject_magic_args": getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": (
                                    getattr({"v": "ctx"}, "__getitem__")("v"),
                                    getattr({"v": "custom.Context"}, "__getitem__")("v"),
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v"),
                        "module_base": getattr({"v": "mlx.nn.Module"}, "__getitem__")("v"),
                      }
                    },
                    "__getitem__",
                  )("v"),
                }
              },
              "__getitem__",
            )("v"),
            "torch": {"traits": {"module_base": "torch.nn.Module"}},
          }
        },
        "__getitem__",
      )("v"),
    )
    self._inject("Linear", SemanticTier.NEURAL, "torch", "torch.Linear", "tensorflow", "func.Dense")
    getattr(
      getattr(
        {
          "v": getattr(
            {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")["Linear"]},
            "__getitem__",
          )("v")["variants"]
        },
        "__getitem__",
      )("v"),
      "__setitem__",
    )("mlx", getattr({"v": {"api": getattr({"v": "custom.Layer"}, "__getitem__")("v")}}, "__getitem__")("v"))

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, tier: SemanticTier, s_fw: str, s_api: str, t_fw: str, t_api: str) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        tier: Test argument or fixture.
        s_fw: Test argument or fixture.
        s_api: Test argument or fixture.
        t_fw: Test argument or fixture.
        t_api: Test argument or fixture.
    """
    variants = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "variants": getattr({"v": variants}, "__getitem__")("v"),
            "std_args": getattr({"v": [getattr({"v": "x"}, "__getitem__")("v")]}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v").get},
                          "__getitem__",
                        )("v")(getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def is_verified(self, _id: str) -> bool:
    """Mock is_verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockStateSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="tensorflow", strict_mode=False)
  return TestRewriter(semantics, config)


def rewrite_code(rewriter: TestRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_signature_injection_missing_arg(rewriter: TestRewriter) -> None:
  """Verifies the behavior of signature injection missing argument.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "\nclass Net:\n    def __init__(self):\n        self.layer = torch.Linear(10, 10)\n\n    def forward(self, x):\n        return self.layer(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result


def test_signature_no_injection_if_present(rewriter: TestRewriter) -> None:
  """Verifies the behavior of signature no injection if present.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "\nclass Net:\n    def __init__(self):\n        self.layer = torch.Linear(10, 10)\n\n    def forward(self, variables, x):\n        return self.layer(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "self.layer.apply(variables, x)" in result
  assert "Injected missing state argument" not in result


def test_custom_trait_injection() -> None:
  """Verifies the behavior of custom trait injection."""
  semantics = MockStateSemantics()
  config = RuntimeConfig(target_framework="mlx", strict_mode=False)
  custom_rewriter = TestRewriter(semantics, config)
  code: str = "\nclass Net(torch.nn.Module):\n    def __init__(self):\n        self.layer = torch.Linear(10)\n\n    def forward(self, input):\n        return self.layer(input)\n"
  result: str = rewrite_code(custom_rewriter, code)
  assert "def forward(self, ctx, input):" in result or "def forward(self, input):" not in result
  assert "self.layer.call_fn(ctx, input)" in result
