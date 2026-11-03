"""Test extra corner cases for the MLIR parser."""

from tests.conftest import _test_getattr

from lark import Token, Tree

from ml_switcheroo.core.mlir.parser import MlirTransformer, OperationNode


def test_mlir_transformer_operation_extras() -> None:
  """Test corner cases inside MlirTransformer.operation method."""
  transformer = MlirTransformer()

  children = _test_getattr(
    {
      "v": [
        Tree("operands", [Tree("operand_list", [Tree("operand", [Token("VAL_ID", "%0")])])]),
        Tree("op_tail", []),
        Tree("function_type", [Token("TYPE", "i32")]),
        _test_getattr(
          {
            "v": _test_getattr({"v": Tree}, "__getitem__")("v")(
              _test_getattr({"v": "successor_list"}, "__getitem__")("v"),
              _test_getattr(
                {
                  "v": [
                    _test_getattr(
                      {
                        "v": _test_getattr({"v": Tree}, "__getitem__")("v")(
                          _test_getattr({"v": "successor"}, "__getitem__")("v"),
                          _test_getattr(
                            {
                              "v": [
                                _test_getattr(
                                  {
                                    "v": _test_getattr({"v": Token}, "__getitem__")("v")(
                                      _test_getattr({"v": "OTHER_TOKEN"}, "__getitem__")("v"),
                                      _test_getattr({"v": "dummy"}, "__getitem__")("v"),
                                    )
                                  },
                                  "__getitem__",
                                )("v")
                              ]
                            },
                            "__getitem__",
                          )("v"),
                        )
                      },
                      "__getitem__",
                    )("v")
                  ]
                },
                "__getitem__",
              )("v"),
            )
          },
          "__getitem__",
        )("v"),
      ]
    },
    "__getitem__",
  )("v")
  op: OperationNode = transformer.operation(children)
  assert len(op.operands) == 1
  assert op.operands[0].name == "%0"

  children2 = [
    Tree("operands", []),
    Tree(
      "function_type",
      [
        Tree("unknown_data", []),
        Tree("type_list_parens", [Token("OTHER", "dummy"), Token("TYPE", "f32")]),
        Token("ARROW", "->"),
        Tree("unknown_data", []),
        Tree("type_list_parens", [Token("OTHER", "dummy"), Token("TYPE", "f32")]),
      ],
    ),
  ]
  op2: OperationNode = transformer.operation(children2)
  assert len(op2.result_types) == 1
  assert op2.result_types[0].body == "f32"

  op3: OperationNode = transformer.operation([])
  assert op3.name == ""


def test_mlir_transformer_op_result_list_extra() -> None:
  """Test corner case with unexpected nodes in op_result_list."""
  transformer = MlirTransformer()
  children = _test_getattr(
    {
      "v": [
        _test_getattr(
          {
            "v": _test_getattr({"v": Tree}, "__getitem__")("v")(
              _test_getattr({"v": "op_result_list"}, "__getitem__")("v"),
              _test_getattr(
                {
                  "v": [
                    _test_getattr(
                      {
                        "v": _test_getattr({"v": Tree}, "__getitem__")("v")(
                          _test_getattr({"v": "op_result"}, "__getitem__")("v"),
                          _test_getattr(
                            {
                              "v": [
                                _test_getattr(
                                  {
                                    "v": _test_getattr({"v": Token}, "__getitem__")("v")(
                                      _test_getattr({"v": "VAL_ID"}, "__getitem__")("v"),
                                      _test_getattr({"v": "%1"}, "__getitem__")("v"),
                                    )
                                  },
                                  "__getitem__",
                                )("v")
                              ]
                            },
                            "__getitem__",
                          )("v"),
                        )
                      },
                      "__getitem__",
                    )("v"),
                    _test_getattr(
                      {
                        "v": _test_getattr({"v": Tree}, "__getitem__")("v")(
                          _test_getattr({"v": "unknown_node"}, "__getitem__")("v"),
                          _test_getattr({"v": []}, "__getitem__")("v"),
                        )
                      },
                      "__getitem__",
                    )("v"),
                  ]
                },
                "__getitem__",
              )("v"),
            )
          },
          "__getitem__",
        )("v")
      ]
    },
    "__getitem__",
  )("v")
  op: OperationNode = transformer.operation(children)
  assert len(op.results) == 1
  assert op.results[0].name == "%1"
