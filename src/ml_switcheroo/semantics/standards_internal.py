"""
Internal Standards Definition (Hub of the Knowledge Base).

This module defines the "Golden Set" of abstract operations that are not
covered by (or require overrides on top of) upstream standards like ONNX
or the Python Array API.

It defines:
1.  **Abstract Schema**: The standard argument names (`std_args`) and docstrings.
2.  **Implementation Variants**: Currently empty, as all implementations have been
    distributed to their respective Framework Adapters in `src/ml_switcheroo/frameworks/`.

Refactor Status:
- All Frameworks: Mappings moved to `definitions` property in Adapter classes.
"""

INTERNAL_OPS = {
  # ============================================================================
  # 1. Config & Namespaces (Infrastructure)
  # ============================================================================
  "__imports__": {},
  "TorchFunctional": {
    "description": "Torch Functional Namespace",
    "std_args": [],
    "variants": {},
  },
  # ============================================================================
  # 2. Optimization & Learning Rate Schedulers (Tier C)
  # ============================================================================
  "Adam": {
    "description": "Adaptive Moment Estimation.",
    "std_args": ["params", "lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"],
    "variants": {},
  },
  "SGD": {
    "description": "Stochastic Gradient Descent.",
    "std_args": ["params", "lr", "momentum", "dampening", "weight_decay", "nesterov"],
    "variants": {},
  },
  "RMSprop": {
    "description": "Root Mean Square Propagation.",
    "std_args": ["params", "lr", "rho", "eps", "weight_decay", "momentum", "centered"],
    "variants": {},
  },
  # Schedulers
  "StepLR": {
    "description": "Decays the learning rate of each parameter group by gamma every step_size epochs.",
    "std_args": ["optimizer", "step_size", "gamma"],
    "variants": {},
  },
  "CosineAnnealingLR": {
    "description": "Set the learning rate of each parameter group using a cosine annealing schedule.",
    "std_args": ["optimizer", "T_max"],
    "variants": {},
  },
  # Optimization Verbs
  "ClipGradNorm": {
    "description": "Clips gradient norm of an iterable of parameters.",
    "std_args": ["parameters", "max_norm"],
    "variants": {},
  },
  "step": {
    "description": "Performs a single optimization step.",
    "std_args": [],
    "variants": {},
  },
  "zero_grad": {
    "description": "Sets the gradients of all optimized torch.Tensor s to zero.",
    "std_args": [],
    "variants": {},
  },
  # ============================================================================
  # 3. Array API Overrides & Utilities (Array Manipulation)
  # ============================================================================
  "randn": {
    "description": "Returns a tensor filled with random numbers from a normal distribution.",
    "std_args": ["shape"],
    "variants": {},
  },
  "Clamp": {
    "description": "Clamp all elements in input into the range [min, max].",
    "std_args": ["input", "min", "max"],
    "variants": {},
  },
  "Gather": {
    "description": "Gathers values along an axis specified by dim.",
    "std_args": ["input", "dim", "index"],
    "variants": {},
  },
  "Scatter": {
    "description": "Writes all values from the tensor src into self at the indices specified in index.",
    "std_args": ["input", "dim", "index", "src"],
    "variants": {},
  },
  "Flatten": {
    "description": "Flattens input by reshaping it into a one-dimensional tensor.",
    "std_args": ["input", "start_dim", "end_dim"],
    "variants": {},
  },
  "Reshape": {
    "description": "Returns a tensor with the same data and number of elements as input, but with the specified shape.",
    "std_args": ["x", "shape"],
    "variants": {},
  },
  "View": {
    "description": "Returns a new tensor with the same data as the self tensor but of a different shape.",
    "std_args": ["input", "shape"],
    "variants": {},
  },
  "Squeeze": {
    "description": "Returns a tensor with all the dimensions of input of size 1 removed.",
    "std_args": ["input", "dim"],
    "variants": {},
  },
  "Unsqueeze": {
    "description": "Returns a new tensor with a dimension of size one inserted at the specified position.",
    "std_args": ["input", "dim"],
    "variants": {},
  },
  "TopK": {
    "description": "Returns the k largest elements of the given input tensor along a given dimension.",
    "std_args": ["input", "k", "dim"],
    "variants": {},
  },
  "ArgMax": {
    "description": "Returns the indices of the maximum value of all elements in the input tensor.",
    "std_args": ["input", "dim", "keepdim"],
    "variants": {},
  },
  "ArgMin": {
    "description": "Returns the indices of the minimum value of all elements in the input tensor.",
    "std_args": ["input", "dim", "keepdim"],
    "variants": {},
  },
  "Pad": {
    "description": "Pads tensor.",
    "std_args": ["input", "pad", "mode", "value"],
    "variants": {},
  },
  "Einsum": {
    "description": "Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention.",
    "std_args": ["equation", "operands"],
    "variants": {},
  },
  "permute_dims": {
    "description": "Permutes the dimensions of the input.",
    "std_args": ["x", "axes"],
    "variants": {},
  },
  # Casting (Type Mapping)
  "CastFloat": {
    "description": "Cast tensor to float32",
    "std_args": [],
    "variants": {},
  },
  "CastDouble": {
    "description": "Cast tensor to float64",
    "std_args": [],
    "variants": {},
  },
  "CastHalf": {
    "description": "Cast tensor to float16",
    "std_args": [],
    "variants": {},
  },
  "CastLong": {
    "description": "Cast tensor to int64",
    "std_args": [],
    "variants": {},
  },
  "CastInt": {
    "description": "Cast tensor to int32",
    "std_args": [],
    "variants": {},
  },
  "CastBool": {
    "description": "Cast tensor to bool",
    "std_args": [],
    "variants": {},
  },
  "size": {
    "description": "Get tensor shape",
    "std_args": [],
    "variants": {},
  },
  # ============================================================================
  # 4. Neural Layers (Specific Mappings and Overrides)
  # ============================================================================
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
    "variants": {},
  },
  "Embedding": {
    "description": "Lookup table for storing embeddings of a fixed dictionary and size.",
    "std_args": [
      "num_embeddings",
      "embedding_dim",
      "padding_idx",
      "max_norm",
      "norm_type",
      "scale_grad_by_freq",
      "sparse",
    ],
    "variants": {},
  },
  "Linear": {
    "description": "Applies a linear transformation to the incoming data",
    "std_args": ["in_features", "out_features", "bias"],
    "variants": {},
  },
  "Sequential": {
    "description": "A sequential container.",
    "std_args": ["layers"],
    "variants": {},
  },
  "BatchNorm": {
    "description": "Batch Normalization.",
    "std_args": ["input", "eps"],
    "variants": {},
  },
  "LayerNorm": {
    "description": "Applies Layer Normalization over a mini-batch of inputs.",
    "std_args": ["normalized_shape", "eps", "elementwise_affine", "bias"],
    "variants": {},
  },
  "GELU": {
    "description": "Gaussian Error Linear Unit.",
    "std_args": ["input"],
    "variants": {},
  },
  "OneHot": {
    "description": "One-hot encoding.",
    "std_args": ["input", "num_classes"],
    "variants": {},
  },
  "CrossEntropyLoss": {
    "description": "Cross Entropy Loss.",
    "std_args": ["input", "target", "weight"],
    "variants": {},
  },
  "MSELoss": {
    "description": "Mean Squared Error.",
    "std_args": ["input", "target"],
    "variants": {},
  },
  # ============================================================================
  # 5. Vision Transforms
  # ============================================================================
  "Resize": {
    "description": "Resize the input image to the given size.",
    "std_args": ["size", "interpolation", "antialias"],
    "variants": {},
  },
  "Normalize": {
    "description": "Normalize a tensor image with mean and standard deviation.",
    "std_args": ["mean", "std", "inplace"],
    "variants": {},
  },
  "ToTensor": {
    "description": "Convert a PIL Image or numpy.ndarray to tensor.",
    "std_args": [],
    "variants": {},
  },
  "CenterCrop": {
    "description": "Crops the given image at the center.",
    "std_args": ["size"],
    "variants": {},
  },
  "RandomCrop": {
    "description": "Crop the given image at a random location.",
    "std_args": ["size", "padding", "pad_if_needed", "fill", "padding_mode"],
    "variants": {},
  },
  "RandomHorizontalFlip": {
    "description": "Horizontally flip the image randomly with a given probability.",
    "std_args": ["p"],
    "variants": {},
  },
  "RandomVerticalFlip": {
    "description": "Vertically flip the image randomly with a given probability.",
    "std_args": ["p"],
    "variants": {},
  },
  "Grayscale": {
    "description": "Convert image to grayscale.",
    "std_args": ["num_output_channels"],
    "variants": {},
  },
  # ============================================================================
  # 6. State Containers & Extras
  # ============================================================================
  "no_grad": {
    "description": "Context-manager that disabled gradient calculation.",
    "std_args": [],
    "variants": {},
  },
  "enable_grad": {
    "description": "Context-manager that enables gradient calculation.",
    "std_args": [],
    "variants": {},
  },
  "register_buffer": {
    "description": "Registers a persistent buffer (non-parameter state).",
    "std_args": ["name", "tensor", "persistent"],
    "variants": {},
  },
  "register_parameter": {
    "description": "Registers a learnable parameter.",
    "std_args": ["name", "param"],
    "variants": {},
  },
  "state_dict": {
    "description": "Returns a dictionary containing a whole state of the module.",
    "std_args": ["destination", "prefix", "keep_vars"],
    "variants": {},
  },
  "load_state_dict": {
    "description": "Copies parameters and buffers from state_dict into this module.",
    "std_args": ["state_dict", "strict"],
    "variants": {},
  },
  "parameters": {
    "description": "Returns an iterator over module parameters.",
    "std_args": ["recurse"],
    "variants": {},
  },
  "DataLoader": {
    "description": "Data loading utility.",
    "std_args": ["dataset"],
    "variants": {},
  },
  "LoadStateDict": {
    "description": "Loading state utility for mappings.",
    "std_args": [],
    "variants": {},
  },
  "Param": {
    "description": "Container for trainable parameter.",
    "std_args": ["value"],
    "variants": {},
  },
  "Variable": {
    "description": "Generic state container (Trainable or Non-Trainable).",
    "std_args": ["value"],
    "variants": {},
  },
  "Cache": {
    "description": "Container for mutable state (non-grad).",
    "std_args": ["value"],
    "variants": {},
  },
  # ============================================================================
  # 7. Functional Transforms
  # ============================================================================
  "vmap": {
    "description": "Vectorizing map. Creates a function which maps 'func' over argument axes.",
    "std_args": ["func", "in_axes", "out_axes", "randomness"],
    "variants": {},
  },
  "grad": {
    "description": "Creates a function that evaluates the gradient of 'func'.",
    "std_args": ["func", "argnums", "has_aux"],
    "variants": {},
  },
  "value_and_grad": {
    "description": "Creates a function that evaluates both 'func' and the gradient of 'func'.",
    "std_args": ["func", "argnums", "has_aux"],
    "variants": {},
  },
  "jit": {
    "description": "Compiles a function for efficient execution (JIT/Graph mode).",
    "std_args": ["func", "static_argnums"],
    "variants": {},
  },
  "Compile": {
    "description": "Alias for JIT compilation.",
    "std_args": ["func"],
    "variants": {},
  },
  "Synchronize": {
    "description": "Blocks until computation is finished (Barrier).",
    "std_args": [],
    "variants": {},
  },
}
