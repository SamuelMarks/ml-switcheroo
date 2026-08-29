"""Test module."""

from unittest.mock import MagicMock, patch

import libcst as cst


def test_attention_packing_pos_args() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.attention_packing import repack_attn_keras, repack_attn_torch

  node: cst.BaseExpression = cst.parse_expression("MultiheadAttention(a, b, c)")
  ctx: MagicMock = MagicMock()
  ctx.get_mapping.return_value = {"api": "tf.keras.layers.MultiHeadAttention"}
  repack_attn_keras(node, ctx)

  ctx.get_mapping.return_value = {"api": "torch.nn.MultiheadAttention"}
  repack_attn_torch(node, ctx)


def test_casting_no_traits() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock()
  ctx.get_framework_config.return_value = {}
  pass


def test_device_allocator_value_error() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.device_allocator import _parse_device_args

  _parse_device_args(cst.parse_expression("torch.device('cuda:foo')"))
  _parse_device_args(cst.parse_expression("torch.device('cuda', 1)"))


def test_device_checks_exception() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.device_checks import transform_cuda_check

  ctx: MagicMock = MagicMock()
  node: cst.BaseExpression = cst.parse_expression("tensor.is_cuda")
  with patch("ml_switcheroo.plugins.device_checks.get_adapter", side_effect=Exception("mock")):
    assert transform_cuda_check(node, ctx) == node


def test_flatten_branches() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.flatten import transform_flatten

  ctx: MagicMock = MagicMock()
  ctx.framework = "jax"
  node: cst.BaseExpression = cst.parse_expression("flatten()")
  # 58: node has no args
  transform_flatten(node, ctx)

  ctx.framework = "numpy"
  node = cst.parse_expression("flatten(x)")
  ctx.lookup_api.side_effect = lambda x: None
  # 122: lookup_api("flatten") or lookup_api("Flatten")
  # 129: return node when target_api is None
  transform_flatten(node, ctx)

  ctx.framework = "tf"
  ctx.lookup_api.side_effect = lambda x: "tf.reshape"
  node = cst.parse_expression("flatten(x, start_dim=1, end_dim=-1)")
  # 153: end_dim < 0
  transform_flatten(node, ctx)

  node = cst.parse_expression("flatten(x, start_dim=0, end_dim=2)")
  # 207: tuple kwargs
  ctx.framework = "torch"
  ctx.lookup_api.side_effect = lambda x: "torch.flatten"
  transform_flatten(node, ctx)


def test_loss_wrapper_branches() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx: MagicMock = MagicMock()
  ctx.framework = "keras"
  # 90, 97: lookup_api None -> return node
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("loss(y_true, y_pred)")
  transform_loss_reduction(node, ctx)

  node = cst.parse_expression("obj.loss(y_true, y_pred)")
  transform_loss_reduction(node, ctx)

  # 120-122: args keyword mapping fallback
  ctx.lookup_api.return_value = "keras.losses.MSE"
  node = cst.parse_expression("obj.loss(y_true=a, y_pred=b)")
  transform_loss_reduction(node, ctx)


def test_optimizer_step_branches() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("opt(lr=0.1)")
  transform_optimizer_init(node, ctx)


def test_padding_branches() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.padding import transform_padding

  ctx: MagicMock = MagicMock()
  ctx.framework = "jax"
  node: cst.BaseExpression = cst.parse_expression("pad()")
  transform_padding(node, ctx)


def test_state_flag_injection_branches() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.state_flag_injection import capture_eval_state

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  # 149
  node: cst.BaseExpression = cst.parse_expression("model.eval()")
  capture_eval_state(node, ctx)


def test_static_unroll_branches() -> None:
  """Docstring."""
  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  ctx: MagicMock = MagicMock()
  node: cst.BaseStatement = cst.parse_statement("for i in range(1): pass")
  # 102-103
  unroll_static_loops(node, ctx)


def test_plugin_init() -> None:
  """Docstring."""
  import importlib
  from unittest.mock import patch

  with patch("pkgutil.iter_modules") as mock_iter:
    mock_iter.return_value = [(None, "_hidden", False), (None, "my_utils", False), (None, "bad_plugin", False)]

    with patch("importlib.import_module", side_effect=Exception("mock err")):
      import ml_switcheroo.plugins

      importlib.reload(ml_switcheroo.plugins)


def test_loss_wrapper_missing_lines() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx: MagicMock = MagicMock()
  ctx.framework = "keras"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("loss()")
  transform_loss_reduction(node, ctx)

  ctx.lookup_api.return_value = "keras.losses.MSE"
  ctx.get_mapping.return_value = {"args": {"custom": "my_val"}}
  node2: cst.BaseExpression = cst.parse_expression("loss(custom=1)")
  transform_loss_reduction(node2, ctx)


def test_nnx_to_torch_params_unsupported() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name

  node: cst.BaseExpression = cst.parse_expression("foo()")
  _extract_leaf_name(node)


def test_optimizer_step_missing() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("opt()")
  transform_optimizer_init(node, ctx)


def test_padding_missing() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding

  ctx: MagicMock = MagicMock()
  ctx.get_framework_config.return_value = None
  _supports_numpy_padding(ctx)

  ctx.framework = "jax"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("pad()")
  transform_padding(node, ctx)

  ctx.framework = "jax"
  ctx.lookup_api.return_value = "jax.numpy.pad"
  node2: cst.BaseExpression = cst.parse_expression("pad(x, ((1, 2),))")
  transform_padding(node2, ctx)


def test_state_flag_injection_missing() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.plugins.state_flag_injection import _get_func_name

  node: cst.BaseExpression = cst.parse_expression("foo()")
  _get_func_name(node)


def test_static_unroll_missing() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  ctx: MagicMock = MagicMock()
  node: cst.BaseStatement = cst.parse_statement("for i in range(1): pass")
  # Empty loop body (pass is empty after filter?)
  # actually pass is a SimpleStatementLine with Pass
  unroll_static_loops(node, ctx)


def test_loss_wrapper_branches_real() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx: MagicMock = MagicMock()
  ctx.framework = "keras"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("loss(y_true, y_pred)")
  transform_loss_reduction(node, ctx)

  ctx.lookup_api.return_value = "keras.losses.MSE"
  ctx.get_mapping.return_value = {"args": {"custom": "my_val"}}
  node3: cst.BaseExpression = cst.parse_expression("loss(custom=a)")
  transform_loss_reduction(node3, ctx)


def test_nnx_to_torch_params_real() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name

  node: cst.BaseExpression = cst.parse_expression("1")
  _extract_leaf_name(node)


def test_optimizer_step_init_real() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("opt(lr=1)")
  transform_optimizer_init(node, ctx)


def test_padding_real() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding

  ctx: MagicMock = MagicMock()
  ctx.get_framework_config.return_value = None
  _supports_numpy_padding(ctx)

  ctx.framework = "jax"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("pad()")
  transform_padding(node, ctx)

  ctx.lookup_api.return_value = "jax.numpy.pad"
  node_pad: cst.BaseExpression = cst.parse_expression("pad(a, pad_width=[(1,1)])")
  transform_padding(node_pad, ctx)

  node3: cst.BaseExpression = cst.parse_expression("pad(a, pads=[(1,1)])")
  transform_padding(node3, ctx)


def test_state_flag_eval_real() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.state_flag_injection import capture_eval_state

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  node: cst.BaseExpression = cst.parse_expression("obj.foo()")
  capture_eval_state(node, ctx)


def test_static_unroll_real() -> None:
  """Docstring."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  ctx: MagicMock = MagicMock()
  node: cst.BaseStatement = cst.parse_statement("for i in range(1):\n  pass")
  node_changed: cst.For = node.with_changes(body=cst.IndentedBlock(body=[]))
  unroll_static_loops(node_changed, ctx)
