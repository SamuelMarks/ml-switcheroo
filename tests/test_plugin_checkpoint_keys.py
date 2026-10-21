"""Test suite for plugin checkpoint keys."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.checkpoint_keys import transform_checkpoint_keys


def test_transform_checkpoint_keys_positional_arg() -> None:
  """Verify transform checkpoint keys positional arg behavior."""
  # model.load_state_dict(state)
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(state)"), cst.Call)
  ctx = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node = transform_checkpoint_keys(node, ctx)

  getattr(
    {"v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v")._preamble_injector}, "__getitem__")("v").assert_called},
    "__getitem__",
  )("v")()
  assert ctx.metadata.get("key_mapper_injected") is True

  code = getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Module}, "__getitem__")("v")(
            body=getattr(
              {
                "v": [
                  getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleStatementLine}, "__getitem__")(
                        "v"
                      )(
                        body=getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Expr}, "__getitem__")("v")(
                                    value=getattr({"v": new_node}, "__getitem__")("v")
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v")
                      )
                    },
                    "__getitem__",
                  )("v")
                ]
              },
              "__getitem__",
            )("v")
          )
        },
        "__getitem__",
      )("v").code
    },
    "__getitem__",
  )("v")
  assert code.strip() == "KeyMapper.from_torch(state)"


def test_transform_checkpoint_keys_keyword_arg() -> None:
  """Verify transform checkpoint keys keyword arg behavior."""
  # model.load_state_dict(state_dict=state_var, strict=True)
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(state_dict=state_var, strict=True)"), cst.Call)
  ctx = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node = transform_checkpoint_keys(node, ctx)

  getattr(
    {"v": getattr({"v": getattr({"v": ctx}, "__getitem__")("v")._preamble_injector}, "__getitem__")("v").assert_called},
    "__getitem__",
  )("v")()
  assert ctx.metadata.get("key_mapper_injected") is True

  code = getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Module}, "__getitem__")("v")(
            body=getattr(
              {
                "v": [
                  getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleStatementLine}, "__getitem__")(
                        "v"
                      )(
                        body=getattr(
                          {
                            "v": [
                              getattr(
                                {
                                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Expr}, "__getitem__")("v")(
                                    value=getattr({"v": new_node}, "__getitem__")("v")
                                  )
                                },
                                "__getitem__",
                              )("v")
                            ]
                          },
                          "__getitem__",
                        )("v")
                      )
                    },
                    "__getitem__",
                  )("v")
                ]
              },
              "__getitem__",
            )("v")
          )
        },
        "__getitem__",
      )("v").code
    },
    "__getitem__",
  )("v")
  assert code.strip() == "KeyMapper.from_torch(state_var)"


def test_transform_checkpoint_keys_no_args() -> None:
  """Verify transform checkpoint keys no args behavior."""
  # model.load_state_dict()
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict()"), cst.Call)
  ctx = HookContext(
    semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock()
  )  # {}, inject_preamble=MagicMock())

  new_node = transform_checkpoint_keys(node, ctx)

  # Should return original node
  assert new_node is node
  assert not getattr(ctx._preamble_injector, "called")


def test_transform_checkpoint_keys_already_injected() -> None:
  """Verify transform checkpoint keys already injected behavior."""
  node = cst.ensure_type(cst.parse_expression("model.load_state_dict(state)"), cst.Call)
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(), preamble_injector=MagicMock())
  ctx.metadata = {"key_mapper_injected": True}

  transform_checkpoint_keys(node, ctx)

  assert not getattr(ctx._preamble_injector, "called")
