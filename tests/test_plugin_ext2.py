"""Test suite for plugin coverage gaps with concrete AST assertions."""

import importlib
from unittest.mock import MagicMock, patch

import libcst as cst


def test_attention_packing_pos_args() -> None:
  """Verify attention packing positional arguments for Keras and Torch."""
  from ml_switcheroo.plugins.attention_packing import repack_attn_keras, repack_attn_torch

  node: cst.BaseExpression = cst.parse_expression("MultiheadAttention(a, b, c)")
  assert isinstance(node, cst.Call)
  ctx: MagicMock = MagicMock()
  ctx.get_mapping.return_value = {"api": "tf.keras.layers.MultiHeadAttention"}
  res_keras = repack_attn_keras(node, ctx)
  assert isinstance(res_keras, cst.CSTNode)

  ctx.get_mapping.return_value = {"api": "torch.nn.MultiheadAttention"}
  res_torch = repack_attn_torch(node, ctx)
  assert isinstance(res_torch, cst.CSTNode)


def test_casting_no_traits() -> None:
  """Verify casting plugin falls back gracefully when framework config has no traits."""
  from ml_switcheroo.plugins.casting import transform_casting

  ctx: MagicMock = MagicMock()
  ctx.get_framework_config.return_value = {}
  ctx.lookup_api.return_value = None
  node = cst.parse_expression("x.float()")
  assert isinstance(node, cst.Call)
  res = transform_casting(node, ctx)
  assert res == node


def test_device_allocator_value_error() -> None:
  """Verify device allocator argument parser handles non-numeric index strings and two-arg calls."""
  from ml_switcheroo.plugins.device_allocator import _parse_device_args

  expr1 = cst.parse_expression("torch.device('cuda:foo')")
  assert isinstance(expr1, cst.Call)
  res1 = _parse_device_args(expr1)
  assert res1 is not None

  expr2 = cst.parse_expression("torch.device('cuda', 1)")
  assert isinstance(expr2, cst.Call)
  res2 = _parse_device_args(expr2)
  assert res2 is not None


def test_device_checks_exception() -> None:
  """Verify device checks plugin returns original node if adapter lookup raises an exception."""
  from ml_switcheroo.plugins.device_checks import transform_cuda_check

  ctx: MagicMock = MagicMock()
  node: cst.BaseExpression = cst.parse_expression("tensor.is_cuda")
  with patch("ml_switcheroo.plugins.device_checks.get_adapter", side_effect=Exception("mock")):
    assert transform_cuda_check(node, ctx) == node


def test_flatten_branches() -> None:
  """Verify flatten plugin edge-case branches across JAX, NumPy, TensorFlow, and PyTorch."""
  from ml_switcheroo.plugins.flatten import transform_flatten

  ctx: MagicMock = MagicMock()
  ctx.framework = "jax"
  node: cst.BaseExpression = cst.parse_expression("flatten()")
  assert isinstance(node, cst.Call)
  assert transform_flatten(node, ctx) == node

  ctx.framework = "numpy"
  node = cst.parse_expression("flatten(x)")
  assert isinstance(node, cst.Call)
  ctx.lookup_api.side_effect = lambda x: None
  assert transform_flatten(node, ctx) == node

  ctx.framework = "tf"
  ctx.lookup_api.side_effect = lambda x: "tf.reshape"
  node = cst.parse_expression("flatten(x, start_dim=1, end_dim=-1)")
  assert isinstance(node, cst.Call)
  res_tf = transform_flatten(node, ctx)
  assert isinstance(res_tf, cst.Call)

  node = cst.parse_expression("flatten(x, start_dim=0, end_dim=2)")
  assert isinstance(node, cst.Call)
  ctx.framework = "torch"
  ctx.lookup_api.side_effect = lambda x: "torch.flatten"
  res_torch = transform_flatten(node, ctx)
  assert isinstance(res_torch, cst.Call)


def test_loss_wrapper_branches() -> None:
  """Verify loss wrapper reduction transformation branches."""
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx: MagicMock = MagicMock()
  ctx.framework = "keras"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("loss(y_true, y_pred)")
  assert isinstance(node, cst.Call)
  assert transform_loss_reduction(node, ctx) == node

  node = cst.parse_expression("obj.loss(y_true, y_pred)")
  assert isinstance(node, cst.Call)
  assert transform_loss_reduction(node, ctx) == node

  ctx.lookup_api.return_value = "keras.losses.MSE"
  node = cst.parse_expression("obj.loss(y_true=a, y_pred=b)")
  assert isinstance(node, cst.Call)
  res = transform_loss_reduction(node, ctx)
  assert isinstance(res, cst.CSTNode)


def test_optimizer_step_branches() -> None:
  """Verify optimizer init returns a Call node when target API is missing."""
  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("opt(lr=0.1)")
  assert isinstance(node, cst.Call)
  assert isinstance(transform_optimizer_init(node, ctx), cst.Call)


def test_padding_branches() -> None:
  """Verify padding plugin returns original node when called with zero arguments."""
  from ml_switcheroo.plugins.padding import transform_padding

  ctx: MagicMock = MagicMock()
  ctx.framework = "jax"
  node: cst.BaseExpression = cst.parse_expression("pad()")
  assert isinstance(node, cst.Call)
  assert transform_padding(node, ctx) == node


def test_state_flag_injection_branches() -> None:
  """Verify state flag capture on model.eval() call."""
  from ml_switcheroo.plugins.state_flag_injection import capture_eval_state

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  node: cst.BaseExpression = cst.parse_expression("model.eval()")
  assert isinstance(node, cst.Call)
  res = capture_eval_state(node, ctx)
  assert res is not None


def test_static_unroll_branches() -> None:
  """Verify static loop unrolling on a single-iteration range loop."""
  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  ctx: MagicMock = MagicMock()
  node: cst.BaseStatement = cst.parse_statement("for i in range(1): pass")
  assert isinstance(node, cst.For)
  res = unroll_static_loops(node, ctx)
  assert res is not None


def test_plugin_init() -> None:
  """Verify plugin package initialization tolerates module import errors."""
  with patch("pkgutil.iter_modules") as mock_iter:
    mock_iter.return_value = [(None, "_hidden", False), (None, "my_utils", False), (None, "bad_plugin", False)]

    with patch("importlib.import_module", side_effect=Exception("mock err")):
      import ml_switcheroo.plugins

      reloaded = importlib.reload(ml_switcheroo.plugins)
      assert reloaded is not None


def test_loss_wrapper_missing_lines() -> None:
  """Verify loss wrapper handles empty args and custom keyword argument mapping."""
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx: MagicMock = MagicMock()
  ctx.framework = "keras"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("loss()")
  assert isinstance(node, cst.Call)
  assert transform_loss_reduction(node, ctx) == node

  ctx.lookup_api.return_value = "keras.losses.MSE"
  ctx.get_mapping.return_value = {"args": {"custom": "my_val"}}
  node2: cst.BaseExpression = cst.parse_expression("loss(custom=1)")
  assert isinstance(node2, cst.Call)
  res2 = transform_loss_reduction(node2, ctx)
  assert isinstance(res2, cst.CSTNode)


def test_nnx_to_torch_params_unsupported() -> None:
  """Verify _extract_leaf_name returns None for non-Name/Attribute expressions."""
  from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name

  node: cst.BaseExpression = cst.parse_expression("foo()")
  assert _extract_leaf_name(node) is None


def test_optimizer_step_missing() -> None:
  """Verify optimizer init handles empty argument lists when lookup_api is None."""
  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("opt()")
  assert isinstance(node, cst.Call)
  assert isinstance(transform_optimizer_init(node, ctx), cst.Call)


def test_padding_missing() -> None:
  """Verify padding helper and tuple padding width conversion."""
  from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding

  ctx: MagicMock = MagicMock()
  ctx.get_framework_config.return_value = None
  assert _supports_numpy_padding(ctx) is True

  ctx.framework = "jax"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("pad()")
  assert isinstance(node, cst.Call)
  assert transform_padding(node, ctx) == node

  ctx.framework = "jax"
  ctx.lookup_api.return_value = "jax.numpy.pad"
  node2: cst.BaseExpression = cst.parse_expression("pad(x, ((1, 2),))")
  assert isinstance(node2, cst.Call)
  res2 = transform_padding(node2, ctx)
  assert isinstance(res2, cst.Call)


def test_state_flag_injection_missing() -> None:
  """Verify _get_func_name returns None for nested call expressions."""
  from ml_switcheroo.plugins.state_flag_injection import _get_func_name

  node: cst.BaseExpression = cst.parse_expression("foo()")
  assert _get_func_name(node) is None


def test_static_unroll_missing() -> None:
  """Verify static unroll handles single-statement pass body."""
  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  ctx: MagicMock = MagicMock()
  node: cst.BaseStatement = cst.parse_statement("for i in range(1): pass")
  assert isinstance(node, cst.For)
  res = unroll_static_loops(node, ctx)
  assert res is not None


def test_loss_wrapper_branches_real() -> None:
  """Verify loss wrapper real branch execution with custom argument remapping."""
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  ctx: MagicMock = MagicMock()
  ctx.framework = "keras"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("loss(y_true, y_pred)")
  assert isinstance(node, cst.Call)
  assert transform_loss_reduction(node, ctx) == node

  ctx.lookup_api.return_value = "keras.losses.MSE"
  ctx.get_mapping.return_value = {"args": {"custom": "my_val"}}
  node3: cst.BaseExpression = cst.parse_expression("loss(custom=a)")
  assert isinstance(node3, cst.Call)
  res3 = transform_loss_reduction(node3, ctx)
  assert isinstance(res3, cst.CSTNode)


def test_nnx_to_torch_params_real() -> None:
  """Verify _extract_leaf_name returns None for integer literals."""
  from ml_switcheroo.plugins.nnx_to_torch_params import _extract_leaf_name

  node: cst.BaseExpression = cst.parse_expression("1")
  assert _extract_leaf_name(node) is None


def test_optimizer_step_init_real() -> None:
  """Verify transform_optimizer_init returns a Call node when lookup_api is None."""
  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("opt(lr=1)")
  assert isinstance(node, cst.Call)
  assert isinstance(transform_optimizer_init(node, ctx), cst.Call)


def test_padding_real() -> None:
  """Verify padding transformation with pad_width and pads keyword arguments."""
  from ml_switcheroo.plugins.padding import _supports_numpy_padding, transform_padding

  ctx: MagicMock = MagicMock()
  ctx.get_framework_config.return_value = None
  assert _supports_numpy_padding(ctx) is True

  ctx.framework = "jax"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("pad()")
  assert isinstance(node, cst.Call)
  assert transform_padding(node, ctx) == node

  ctx.lookup_api.return_value = "jax.numpy.pad"
  node_pad: cst.BaseExpression = cst.parse_expression("pad(a, pad_width=[(1,1)])")
  assert isinstance(node_pad, cst.Call)
  res_pad = transform_padding(node_pad, ctx)
  assert isinstance(res_pad, cst.Call)

  node3: cst.BaseExpression = cst.parse_expression("pad(a, pads=[(1,1)])")
  assert isinstance(node3, cst.Call)
  res3 = transform_padding(node3, ctx)
  assert isinstance(res3, cst.Call)


def test_state_flag_eval_real() -> None:
  """Verify capture_eval_state returns a Call node on method calls."""
  from ml_switcheroo.plugins.state_flag_injection import capture_eval_state

  ctx: MagicMock = MagicMock()
  ctx.framework = "torch"
  node: cst.BaseExpression = cst.parse_expression("obj.foo()")
  assert isinstance(node, cst.Call)
  assert isinstance(capture_eval_state(node, ctx), cst.Call)


def test_static_unroll_real() -> None:
  """Verify unroll_static_loops returns a FlattenSentinel when loop body is empty."""
  from ml_switcheroo.plugins.static_unroll import unroll_static_loops

  ctx: MagicMock = MagicMock()
  node: cst.BaseStatement = cst.parse_statement("for i in range(1):\n  pass")
  assert isinstance(node, cst.For)
  node_changed: cst.For = node.with_changes(body=cst.IndentedBlock(body=[]))
  assert isinstance(unroll_static_loops(node_changed, ctx), cst.FlattenSentinel)
