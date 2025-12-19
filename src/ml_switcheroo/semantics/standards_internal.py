"""
Internal Standards Definition (Tier C).

Contains curated definitions for operations not covered by
upstream bodies (ONNX/Array API).
"""

INTERNAL_OPS = {
  # --- Import Configurations (Feature 024) ---
  "__imports__": {
    "torchvision": {"variants": {"torch": {"root": "torchvision", "alias": None}}},
    "torchvision.transforms": {"variants": {"torch": {"root": "torchvision", "sub": "transforms", "alias": "T"}}},
  },
  # --- Namespaces (Fix for Strict Mode) ---
  "TorchFunctional": {
    "description": "Torch Functional Namespace",
    "std_args": [],
    "variants": {
      "torch": {"api": "torch.nn.functional"},
      "jax": {"api": "jax.nn"},
    },
  },
  # --- Optimizers ---
  "Adam": {
    "description": "Adaptive Moment Estimation.",
    "std_args": ["params", "lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"],
  },
  "SGD": {
    "description": "Stochastic Gradient Descent.",
    "std_args": ["params", "lr", "momentum", "dampening", "weight_decay", "nesterov"],
  },
  "RMSprop": {
    "description": "Root Mean Square Propagation.",
    "std_args": ["params", "lr", "rho", "eps", "weight_decay", "momentum", "centered"],
  },
  "Adagrad": {
    "description": "Adaptive Gradient Algorithm.",
    "std_args": ["params", "lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"],
  },
  "AdamW": {
    "description": "Adam with decoupled weight decay.",
    "std_args": ["params", "lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"],
  },
  # --- Vision Transforms ---
  "Resize": {
    "description": "Resize the input image to the given size.",
    "std_args": ["size", "interpolation", "antialias"],
  },
  "Normalize": {
    "description": "Normalize a tensor image with mean and standard deviation.",
    "std_args": ["mean", "std", "inplace"],
  },
  "ToTensor": {
    "description": "Convert a PIL Image or numpy.ndarray to tensor.",
    "std_args": [],
  },
  "CenterCrop": {
    "description": "Crops the given image at the center.",
    "std_args": ["size"],
  },
  "RandomCrop": {
    "description": "Crop the given image at a random location.",
    "std_args": ["size", "padding", "pad_if_needed", "fill", "padding_mode"],
  },
  "RandomHorizontalFlip": {
    "description": "Horizontally flip the image randomly with a given probability.",
    "std_args": ["p"],
  },
  "RandomVerticalFlip": {
    "description": "Vertically flip the image randomly with a given probability.",
    "std_args": ["p"],
  },
  "Pad": {
    "description": "Pad the given image on all sides.",
    "std_args": ["padding", "fill", "padding_mode"],
  },
  "Grayscale": {
    "description": "Convert image to grayscale.",
    "std_args": ["num_output_channels"],
  },
  # --- State Containers ---
  "register_buffer": {
    "description": "Registers a persistent buffer (non-parameter state).",
    "std_args": ["name", "tensor", "persistent"],
  },
  "register_parameter": {
    "description": "Registers a learnable parameter.",
    "std_args": ["name", "param"],
  },
  "state_dict": {
    "description": "Returns a dictionary containing a whole state of the module.",
    "std_args": ["destination", "prefix", "keep_vars"],
  },
  "load_state_dict": {
    "description": "Copies parameters and buffers from state_dict into this module.",
    "std_args": ["state_dict", "strict"],
  },
  "parameters": {
    "description": "Returns an iterator over module parameters.",
    "std_args": ["recurse"],
  },
  # --- Neural Layers (Complex) ---
  "MultiheadAttention": {
    "description": "Multi-head attention mechanism. Supports repacking query/key/value via plugins.",
    "std_args": [
      "embed_dim",
      "num_heads",
      "dropout",
      "bias",
      "add_bias_kv",
      "add_zero_attn",
      "kdim",
      "vdim",
      "batch_first",
    ],
    # Static wiring for core frameworks to ensure 'sync' picks them up without heuristics
    "variants": {
      "torch": {"api": "torch.nn.MultiheadAttention"},
      "keras": {
        "api": "keras.layers.MultiHeadAttention",
        "args": {"embed_dim": "key_dim"},
        "requires_plugin": "repack_attention_call",
      },
      "flax_nnx": {"api": "flax.nnx.MultiHeadAttention", "requires_plugin": "repack_attention_call"},
    },
  },
  # --- Functional Transforms (Merged from populate_functional.py) ---
  "vmap": {
    "description": "Vectorizing map. Creates a function which maps 'func' over argument axes.",
    "std_args": ["func", "in_axes", "out_axes", "randomness"],
  },
  "grad": {
    "description": "Creates a function that evaluates the gradient of 'func'.",
    "std_args": ["func", "argnums", "has_aux"],
  },
  "value_and_grad": {
    "description": "Creates a function that evaluates both 'func' and the gradient of 'func'.",
    "std_args": ["func", "argnums", "has_aux"],
  },
  "jit": {
    "description": "Compiles a function for efficient execution (JIT/Graph mode).",
    "std_args": ["func", "static_argnums"],
  },
}
