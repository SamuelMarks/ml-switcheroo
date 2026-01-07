"""
Internal Standards Definition (Hub of the Knowledge Base).

This module defines the "Golden Set" of abstract operations (The Hub).
It defines the **Abstract Schema**: Standard argument names (`std_args`) and docstrings.

It does **NOT** contain implementation details (variants).
Implementations must be defined in Framework Adapters or Snapshot Overlay files.
"""

from ml_switcheroo.core.dsl import OpType

# ============================================================================
# 1. Array API (Math)
# ============================================================================
MATH_OPS = {
  "randn": {
    "description": "Returns a tensor filled with random numbers from a normal distribution.",
    "std_args": ["shape"],
  },
  "Clamp": {
    "description": "Clamp all elements in input into the range [min, max].",
    "std_args": ["input", "min", "max"],
  },
  "View": {
    "description": "Returns a new tensor with the same data and elements but different shape.",
    "std_args": ["input", "shape"],
  },
  "permute_dims": {
    "description": "Permutes tensor dimensions.",
    "std_args": ["input", {"name": "dims", "type": "int", "is_variadic": True}],
  },
  "Abs": {
    "description": "Calculates the absolute value.",
    "std_args": ["x"],
  },
  "Mean": {
    "description": "Calculates the mean value.",
    "std_args": ["x"],
  },
  "Sum": {
    "description": "Calculates the sum value.",
    "std_args": ["x"],
  },
  "Add": {
    "description": "Element-wise addition.",
    "std_args": ["x", "y"],
  },
  "Sub": {
    "description": "Element-wise subtraction.",
    "std_args": ["x", "y"],
  },
  "Mul": {
    "description": "Element-wise multiplication.",
    "std_args": ["x", "y"],
  },
  "Div": {
    "description": "Element-wise division.",
    "std_args": ["x", "y"],
  },
  "Pow": {
    "description": "Element-wise power.",
    "std_args": ["x", "y"],
  },
  "exp": {
    "description": "Exponential.",
    "std_args": ["x"],
  },
  "log": {
    "description": "Logarithm.",
    "std_args": ["x"],
  },
  "sqrt": {
    "description": "Square root.",
    "std_args": ["x"],
  },
  "square": {
    "description": "Square.",
    "std_args": ["x"],
  },
  "Gather": {
    "description": "Gathers values along an axis specified by dim.",
    "std_args": ["input", "dim", "index"],
  },
  "Scatter": {
    "description": "Writes all values from the tensor src into indices specified.",
    "std_args": ["input", "dim", "index", "src"],
  },
  "Flatten": {
    "description": "Flattens input by reshaping it into a one-dimensional tensor.",
    "std_args": ["input", "start_dim", "end_dim"],
  },
  "Reshape": {
    "description": "Returns a tensor with the same data and number of elements as input, but with the specified shape.",
    "std_args": ["input", "shape"],
  },
  "Squeeze": {
    "description": "Returns a tensor with all the dimensions of input of size 1 removed.",
    "std_args": ["input", "dim"],
  },
  "Unsqueeze": {
    "description": "Returns a new tensor with a dimension of size one inserted at the specified position.",
    "std_args": ["input", "dim"],
  },
  "TopK": {
    "description": "Returns the k largest elements of the given input tensor along a given dimension.",
    "std_args": ["input", "k", "dim", "largest", "sorted"],
  },
  "ArgMax": {
    "description": "Returns the indices of the maximum value of all elements in the input tensor.",
    "std_args": ["input", "dim", "keepdim"],
  },
  "ArgMin": {
    "description": "Returns the indices of the minimum value of all elements in the input tensor.",
    "std_args": ["input", "dim", "keepdim"],
  },
  "Pad": {
    "description": "Pads input tensor.",
    "std_args": ["input", "pad", "mode", "value"],
  },
  "Einsum": {
    "description": "Sums the product of the elements of the input operands along dimensions specified using a notation.",
    "std_args": ["equation", "operands"],
  },
  "OneHot": {
    "description": "Takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes).",
    "std_args": ["tensor", "num_classes"],
  },
  "max": {
    "description": "Element-wise maximum or reduction.",
    "std_args": ["x"],
  },
  "min": {
    "description": "Element-wise minimum or reduction.",
    "std_args": ["x"],
  },
  "relu": {
    "description": "Rectified Linear Unit.",
    "std_args": ["x"],
  },
  "TorchFunctional": {
    "description": "Abstract Functional Namespace (e.g. F in torch.nn.functional)",
    "std_args": [],
  },
  "Float32": {"description": "32-bit floating point type.", "std_args": []},
  "Float64": {"description": "64-bit floating point type (Double).", "std_args": []},
  "Float16": {"description": "16-bit floating point type (Half).", "std_args": []},
  "Int64": {"description": "64-bit signed integer type (Long).", "std_args": []},
  "Int32": {"description": "32-bit signed integer type (Int).", "std_args": []},
  "Int16": {"description": "16-bit signed integer type (Short).", "std_args": []},
  "UInt8": {"description": "8-bit unsigned integer type (Byte).", "std_args": []},
  "Bool": {"description": "Boolean type.", "std_args": []},
  "size": {"description": "Get tensor shape", "std_args": []},
  "data_ptr": {
    "description": "Get valid data pointer or buffer access.",
    "std_args": [],
  },
  "CastFloat": {
    "description": "Cast tensor to float32",
    "std_args": ["x"],
    "metadata": {"target_type": "Float32"},
  },
  "CastDouble": {
    "description": "Cast tensor to float64",
    "std_args": ["x"],
    "metadata": {"target_type": "Float64"},
  },
  "CastHalf": {
    "description": "Cast tensor to float16",
    "std_args": ["x"],
    "metadata": {"target_type": "Float16"},
  },
  "CastLong": {
    "description": "Cast tensor to int64",
    "std_args": ["x"],
    "metadata": {"target_type": "Int64"},
  },
  "CastInt": {
    "description": "Cast tensor to int32",
    "std_args": ["x"],
    "metadata": {"target_type": "Int32"},
  },
  "CastShort": {
    "description": "Cast tensor to int16",
    "std_args": ["x"],
    "metadata": {"target_type": "Int16"},
  },
  "CastByte": {
    "description": "Cast tensor to uint8",
    "std_args": ["x"],
    "metadata": {"target_type": "UInt8"},
  },
  "CastBool": {
    "description": "Cast tensor to bool",
    "std_args": ["x"],
    "metadata": {"target_type": "Bool"},
  },
  "CastChar": {
    "description": "Cast tensor to int8/char",
    "std_args": ["x"],
    "metadata": {"target_type": "Int8"},
  },
}

# ============================================================================
# 2. Neural (Layers/Model/State)
# ============================================================================
NEURAL_OPS = {
  "MultiheadAttention": {
    "description": "Multi-head attention mechanism.",
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
  },
  "Embedding": {
    "description": "Lookup table for storing embeddings.",
    "std_args": [
      "num_embeddings",
      "embedding_dim",
      "padding_idx",
      "max_norm",
      "norm_type",
      "scale_grad_by_freq",
      "sparse",
    ],
  },
  "Linear": {
    "description": "Applies a linear transformation to the incoming data",
    "std_args": ["in_features", "out_features", "bias"],
  },
  "Conv2d": {
    "description": "Applies a 2D convolution over an input signal composed of several input planes.",
    "std_args": [
      "in_channels",
      "out_channels",
      "kernel_size",
      "stride",
      "padding",
      "dilation",
      "groups",
      "bias",
      "padding_mode",
    ],
  },
  "MaxPool2d": {
    "description": "Applies a 2D max pooling over an input signal composed of several input planes.",
    "std_args": [
      "kernel_size",
      "stride",
      "padding",
      "dilation",
      "return_indices",
      "ceil_mode",
    ],
  },
  "Dropout": {
    "description": "During training, randomly zeroes some of the elements of the input tensor with probability p.",
    "std_args": ["p", "inplace"],
  },
  "Sequential": {
    "description": "A sequential container.",
    "std_args": ["layers"],
  },
  "BatchNorm": {
    "description": "Batch Normalization.",
    "std_args": ["input", "eps"],
  },
  "LayerNorm": {
    "description": "Applies Layer Normalization over a mini-batch of inputs.",
    "std_args": ["normalized_shape", "eps", "elementwise_affine", "bias"],
  },
  "GELU": {
    "description": "Gaussian Error Linear Unit.",
    "std_args": ["input"],
  },
  "ReLU": {
    "description": "Rectified Linear Unit.",
    "std_args": [],
  },
  "softmax": {
    "description": "Applies the Softmax function to an n-dimensional input Tensor.",
    "std_args": ["input", "dim"],
  },
  "log_softmax": {
    "description": "Applies the LogSoftmax function to an n-dimensional input Tensor.",
    "std_args": ["input", "dim"],
  },
  "CrossEntropyLoss": {
    "description": "Cross Entropy Loss.",
    "std_args": ["input", "target", "weight"],
  },
  "MSELoss": {
    "description": "Mean Squared Error.",
    "std_args": ["input", "target"],
  },
  "register_buffer": {
    "description": "Registers a persistent buffer.",
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
  "LoadStateDict": {
    "description": "Loading state utility for mappings.",
    "std_args": [],
  },
  "Param": {
    "description": "Container for trainable parameter.",
    "std_args": ["value"],
  },
  "Variable": {
    "description": "Generic state container.",
    "std_args": ["value"],
  },
  "Cache": {
    "description": "Container for mutable state.",
    "std_args": ["value"],
  },
}

# ============================================================================
# 3. Extras (Optimizers, Contexts, IO, Device)
# ============================================================================
EXTRAS_OPS = {
  "__imports__": {},
  "Adam": {
    "description": "Adaptive Moment Estimation optimizer.",
    "std_args": [
      "params",
      "lr",
      "beta1",
      "beta2",
      "eps",
      "weight_decay",
      "amsgrad",
    ],
  },
  "SGD": {
    "description": "Stochastic Gradient Descent optimizer.",
    "std_args": ["params", "lr", "momentum", "dampening", "weight_decay", "nesterov"],
  },
  "RMSprop": {
    "description": "Root Mean Square Propagation optimizer.",
    "std_args": ["params", "lr", "rho", "eps", "weight_decay", "momentum", "centered"],
  },
  "StepLR": {
    "description": "Decays the learning rate of each parameter group by gamma every step_size epochs.",
    "std_args": ["optimizer", "step_size", "gamma"],
  },
  "CosineAnnealingLR": {
    "description": "Set the learning rate of each parameter group using a cosine annealing schedule.",
    "std_args": ["optimizer", "T_max"],
  },
  "ClipGradNorm": {
    "description": "Clips gradient norm of an iterable of parameters.",
    "std_args": ["parameters", "max_norm"],
  },
  "step": {
    "description": "Performs a single optimization step.",
    "std_args": [],
  },
  "zero_grad": {
    "description": "Sets the gradients of all optimized parameters to zero.",
    "std_args": [],
  },
  "no_grad": {
    "description": "Context-manager that disabled gradient calculation.",
    "op_type": OpType.CONTEXT,
    "std_args": [],
  },
  "enable_grad": {
    "description": "Context-manager that enables gradient calculation.",
    "op_type": OpType.CONTEXT,
    "std_args": [],
  },
  "Resize": {
    "description": "Resize the input image to the given size.",
    "std_args": ["size"],
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
  "Grayscale": {
    "description": "Convert image to grayscale.",
    "std_args": ["num_output_channels"],
  },
  # --- IO Operations ---
  "Save": {
    "description": "Serialize object to disk.",
    "std_args": ["obj", "f"],
  },
  "Load": {
    "description": "Deserialize object from disk.",
    "std_args": ["f"],
  },
  # --- Device Management ---
  "Device": {
    "description": "Abstract Device placement context.",
    "std_args": ["type", "index"],
  },
  "CudaAvailable": {
    "description": "Checks if a CUDA device is available.",
    "std_args": [],
    "return_type": "bool",
  },
  # ---------------------
  "DataLoader": {
    "description": "Foundational PyTorch Data Loader. Mapped to GenericDataLoader shim via Plugin.",
    "std_args": [
      "dataset",
      "batch_size",
      "shuffle",
      "sampler",
      "batch_sampler",
      "num_workers",
      "collate_fn",
      "pin_memory",
      "drop_last",
      "timeout",
      "worker_init_fn",
      "multiprocessing_context",
      "generator",
      "prefetch_factor",
      "persistent_workers",
      "pin_memory_device",
    ],
  },
  "torch.utils": {
    "description": "Torch Utilities Namespace",
    "std_args": [],
  },
  "torch.utils.data": {
    "description": "Torch Data Utilities Namespace",
    "std_args": [],
  },
  "vmap": {
    "description": "Vectorizing map.",
    "std_args": ["func", "in_axes", "out_axes", "randomness"],
  },
  "grad": {
    "description": "Evaluates gradient.",
    "std_args": ["func", "argnums", "has_aux"],
  },
  "value_and_grad": {
    "description": "Evaluates value and gradient.",
    "std_args": ["func", "argnums", "has_aux"],
  },
  "jit": {
    "description": "JIT Compilation.",
    "std_args": ["func", "static_argnums"],
  },
  "Compile": {
    "description": "JIT Alias.",
    "std_args": ["func"],
  },
  "Synchronize": {
    "description": "Execution Barrier.",
    "std_args": [],
  },
}

# Merged Defaults for Backwards Compatibility
INTERNAL_OPS = {
  **MATH_OPS,
  **NEURAL_OPS,
  **EXTRAS_OPS,
  "SiLU": {
    "description": "Sigmoid Linear Unit activation function.",
    "std_args": [
      {
        "name": "x",
        "type": "Tensor",
      },
    ],
  },
  "ModuleList": {
    "description": "Holds submodules in a list.",
    "std_args": [
      {
        "name": "modules",
        "type": "List[Module]",
      },
    ],
  },
  "TensorType": {
    "description": "Abstract Type Annotation for Tensors/Arrays.",
    "std_args": [],
  },
  "Arange": {
    "description": "Returns evenly spaced values within a given interval.",
    "std_args": [
      {
        "name": "start",
        "type": "int",
      },
      {
        "name": "stop",
        "type": "int",
      },
      {
        "name": "step",
        "type": "int",
        "default": "1",
      },
      {
        "name": "dtype",
        "type": "dtype",
      },
    ],
  },
  "Ones": {
    "description": "Returns a new tensor of given shape filled with ones.",
    "std_args": [
      {
        "name": "shape",
        "type": "Tuple[int, ...]",
      },
      {
        "name": "dtype",
        "type": "dtype",
      },
    ],
  },
  "Concatenate": {
    "description": "Joins a sequence of arrays along an existing axis.",
    "std_args": [
      {
        "name": "tensors",
        "type": "List[Tensor]",
      },
      {
        "name": "axis",
        "type": "int",
        "default": "0",
      },
    ],
  },
  "Zeros": {
    "description": "Returns a tensor filled with the scalar value 0, with the shape defined by the argument.",
    "std_args": [
      {
        "name": "shape",
        "type": "Tuple[int, ...]",
      },
      {
        "name": "dtype",
        "type": "dtype",
        "default": "None",
      },
    ],
  },
  "RandInt": {
    "description": "Generates integers uniformly distributed in the range [low, high).",
    "std_args": [
      {
        "name": "low",
        "type": "int",
      },
      {
        "name": "high",
        "type": "int",
      },
      {
        "name": "shape",
        "type": "Tuple[int, ...]",
      },
      {
        "name": "dtype",
        "type": "dtype",
        "default": "None",
      },
    ],
  },
  "Array": {
    "description": "Creates a tensor/array from a list or numeric data.",
    "std_args": [
      {
        "name": "data",
        "type": "List[Any]",
      },
      {
        "name": "dtype",
        "type": "dtype",
        "default": "None",
      },
    ],
  },
  "Pad": {
    "description": "Pads a tensor. Plugin handles conversion between flat padding (Torch) and tuple-padding (NumPy).",
    "std_args": [
      {
        "name": "input",
        "type": "Tensor",
      },
      {
        "name": "pad",
        "type": "Union[Tuple[int, ...], List[int]]",
      },
      {
        "name": "mode",
        "type": "str",
        "default": '"constant"',
      },
      {
        "name": "value",
        "type": "float",
        "default": "0.0",
      },
    ],
  },
  "AssertClose": {
    "description": "Asserts that two tensors are numerically close.",
    "std_args": [
      {
        "name": "actual",
        "type": "Tensor",
      },
      {
        "name": "expected",
        "type": "Tensor",
      },
      {
        "name": "rtol",
        "type": "float",
        "default": "1e-5",
      },
      {
        "name": "atol",
        "type": "float",
        "default": "1e-8",
      },
    ],
  },
  "A": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Abs": {
    "description": "Absolute takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where absolute value, y = abs(x), is applied to the tensor elementwise.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Acos": {
    "description": "Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Acosh": {
    "description": "Calculates the hyperbolic arccosine of the given input tensor element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "ActivityRegularization": {
    "description": "Layer that applies an update to the cost function based input activity.\n\nArgs:\n    l1: L1 regularization factor (positive float).\n    l2: L2 regularization factor (positive float).\n\nInput shape:\n    Arbitrary. Use the keyword argument `input_shape`\n    (tuple of integers, does not include the samples axis)\n    when using this layer as the first layer in a model.\n\nOutput shape:\n    Same shape as input.",
    "std_args": [
      {"name": "l1", "type": "Any"},
      {"name": "l2", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Adadelta": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "eps", "type": "Any"},
      {"name": "lr", "type": "Any"},
      {"name": "rho", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
    ],
    "variants": {},
  },
  "Adafactor": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "lr", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
    ],
    "variants": {},
  },
  "Adagrad": {
    "description": "Auto-discovered via Consensus (Score: 4.0)",
    "std_args": [
      {"name": "eps", "type": "Any"},
      {"name": "initial_accumulator_value", "type": "Any"},
      {"name": "lr", "type": "Any"},
    ],
    "variants": {},
  },
  "Adam": {
    "description": "Auto-discovered via Consensus (Score: 4.0)",
    "std_args": [
      {"name": "eps", "type": "Any"},
      {"name": "lr", "type": "Any"},
    ],
    "variants": {},
  },
  "Adamax": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "eps", "type": "Any"},
      {"name": "lr", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
    ],
    "variants": {},
  },
  "Adamw": {
    "description": "Auto-discovered via Consensus (Score: 4.0)",
    "std_args": [
      {"name": "eps", "type": "Any"},
      {"name": "lr", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
    ],
    "variants": {},
  },
  "AdaptiveAveragePooling1D": {
    "description": 'Adaptive average pooling operation for 1D temporal or spatial data.\n\nThis layer applies an adaptive average pooling operation, which pools the\ninput such that the output has a target length specified by `output_size`,\nregardless of the input length. The kernel size and stride are automatically\ncomputed to achieve the target output size.\n\nArgs:\n    output_size: Integer specifying the target output length.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, length, channels)`.\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, length)`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, `"channels_last"` is used.\n\nInput shape:\n    - If `data_format="channels_last"`: 3D tensor\n        `(batch_size, length, channels)`\n    - If `data_format="channels_first"`: 3D tensor\n        `(batch_size, channels, length)`\n\nOutput shape:\n    - If `data_format="channels_last"`:\n        `(batch_size, output_length, channels)`\n    - If `data_format="channels_first"`:\n        `(batch_size, channels, output_length)`\n\nExamples:\n    >>> import numpy as np\n    >>> input_seq = np.random.rand(1, 64, 3)\n    >>> layer = AdaptiveAveragePooling1D(output_size=32)\n    >>> output_seq = layer(input_seq)\n    >>> output_seq.shape\n    (1, 32, 3)',
    "std_args": [
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AdaptiveAveragePooling2D": {
    "description": 'Adaptive average pooling operation for 2D spatial data.\n\nThis layer applies an adaptive average pooling operation, which pools the\ninput such that the output has a target spatial size specified by\n`output_size`, regardless of the input spatial size. The kernel size\nand stride are automatically computed to achieve the target output size.\n\nArgs:\n    output_size: Integer or tuple of 2 integers specifying the\n        target output size.\n        If an integer, the same value is used for both height and width.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, height, width, channels)`.\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, `"channels_last"` is used.\n\nInput shape:\n    - If `data_format="channels_last"`: 4D tensor\n        `(batch_size, height, width, channels)`\n    - If `data_format="channels_first"`: 4D tensor\n        `(batch_size, channels, height, width)`\n\nOutput shape:\n    - If `data_format="channels_last"`:\n        `(batch_size, output_height, output_width, channels)`\n    - If `data_format="channels_first"`:\n        `(batch_size, channels, output_height, output_width)`\n\nExamples:\n    >>> import numpy as np\n    >>> input_img = np.random.rand(1, 64, 64, 3)\n    >>> layer = AdaptiveAveragePooling2D(output_size=32)\n    >>> output_img = layer(input_img)\n    >>> output_img.shape\n    (1, 32, 32, 3)',
    "std_args": [
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AdaptiveAveragePooling3D": {
    "description": 'Adaptive average pooling operation for 3D volumetric data.\n\nThis layer applies an adaptive average pooling operation, which pools the\ninput such that the output has a target spatial size specified by\n`output_size`, regardless of the input spatial size. The kernel size\nand stride are automatically computed to achieve the target output size.\n\nArgs:\n    output_size: Integer or tuple of 3 integers specifying the\n        target output size.\n        If an integer, the same value is used for depth, height, and width.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, depth, height, width, channels)`.\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, depth, height, width)`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, `"channels_last"` is used.\n\nInput shape:\n    - If `data_format="channels_last"`: 5D tensor\n        `(batch_size, depth, height, width, channels)`\n    - If `data_format="channels_first"`: 5D tensor\n        `(batch_size, channels, depth, height, width)`\n\nOutput shape:\n    - If `data_format="channels_last"`:\n        `(batch_size, output_depth, output_height, output_width, channels)`\n    - If `data_format="channels_first"`:\n        `(batch_size, channels, output_depth, output_height, output_width)`\n\nExamples:\n    >>> import numpy as np\n    >>> input_vol = np.random.rand(1, 32, 32, 32, 3)\n    >>> layer = AdaptiveAveragePooling3D(output_size=16)\n    >>> output_vol = layer(input_vol)\n    >>> output_vol.shape\n    (1, 16, 16, 16, 3)',
    "std_args": [
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AdaptiveGradClipState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "AdaptiveMaxPooling1D": {
    "description": 'Adaptive max pooling operation for 1D temporal or spatial data.\n\nThis layer applies an adaptive max pooling operation, which pools the\ninput such that the output has a target length specified by `output_size`,\nregardless of the input length. The kernel size and stride are automatically\ncomputed to achieve the target output size.\n\nArgs:\n    output_size: Integer specifying the target output length.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, length, channels)`.\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, length)`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, `"channels_last"` is used.\n\nInput shape:\n    - If `data_format="channels_last"`: 3D tensor\n        `(batch_size, length, channels)`\n    - If `data_format="channels_first"`: 3D tensor\n        `(batch_size, channels, length)`\n\nOutput shape:\n    - If `data_format="channels_last"`:\n        `(batch_size, output_length, channels)`\n    - If `data_format="channels_first"`:\n        `(batch_size, channels, output_length)`\n\nExamples:\n    >>> import numpy as np\n    >>> input_seq = np.random.rand(1, 64, 3)\n    >>> layer = AdaptiveMaxPooling1D(output_size=32)\n    >>> output_seq = layer(input_seq)\n    >>> output_seq.shape\n    (1, 32, 3)',
    "std_args": [
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AdaptiveMaxPooling2D": {
    "description": 'Adaptive max pooling operation for 2D spatial data.\n\nThis layer applies an adaptive max pooling operation, which pools the\ninput such that the output has a target spatial size specified by\n`output_size`, regardless of the input spatial size. The kernel size\nand stride are automatically computed to achieve the target output size.\n\nArgs:\n    output_size: Integer or tuple of 2 integers specifying the\n        target output size.\n        If an integer, the same value is used for both height and width.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, height, width, channels)`.\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, `"channels_last"` is used.\n\nInput shape:\n    - If `data_format="channels_last"`: 4D tensor\n        `(batch_size, height, width, channels)`\n    - If `data_format="channels_first"`: 4D tensor\n        `(batch_size, channels, height, width)`\n\nOutput shape:\n    - If `data_format="channels_last"`:\n        `(batch_size, output_height, output_width, channels)`\n    - If `data_format="channels_first"`:\n        `(batch_size, channels, output_height, output_width)`\n\nExamples:\n    >>> import numpy as np\n    >>> input_img = np.random.rand(1, 64, 64, 3)\n    >>> layer = AdaptiveMaxPooling2D(output_size=32)\n    >>> output_img = layer(input_img)\n    >>> output_img.shape\n    (1, 32, 32, 3)',
    "std_args": [
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AdaptiveMaxPooling3D": {
    "description": 'Adaptive max pooling operation for 3D volumetric data.\n\nThis layer applies an adaptive max pooling operation, which pools the\ninput such that the output has a target spatial size specified by\n`output_size`, regardless of the input spatial size. The kernel size\nand stride are automatically computed to achieve the target output size.\n\nArgs:\n    output_size: Integer or tuple of 3 integers specifying the\n        target output size.\n        If an integer, the same value is used for depth, height, and width.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, depth, height, width, channels)`.\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, depth, height, width)`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, `"channels_last"` is used.\n\nInput shape:\n    - If `data_format="channels_last"`: 5D tensor\n        `(batch_size, depth, height, width, channels)`\n    - If `data_format="channels_first"`: 5D tensor\n        `(batch_size, channels, depth, height, width)`\n\nOutput shape:\n    - If `data_format="channels_last"`:\n        `(batch_size, output_depth, output_height, output_width, channels)`\n    - If `data_format="channels_first"`:\n        `(batch_size, channels, output_depth, output_height, output_width)`\n\nExamples:\n    >>> import numpy as np\n    >>> input_vol = np.random.rand(1, 32, 32, 32, 3)\n    >>> layer = AdaptiveMaxPooling3D(output_size=16)\n    >>> output_vol = layer(input_vol)\n    >>> output_vol.shape\n    (1, 16, 16, 16, 3)',
    "std_args": [
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Add": {
    "description": "Performs element-wise binary addition (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md). (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "AddDecayedWeightsState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "AddNoiseState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "AdditiveAttention": {
    "description": "Additive attention layer, a.k.a. Bahdanau-style attention.\n\nInputs are a list with 2 or 3 elements:\n1. A `query` tensor of shape `(batch_size, Tq, dim)`.\n2. A `value` tensor of shape `(batch_size, Tv, dim)`.\n3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none\n    supplied, `value` will be used as `key`.\n\nThe calculation follows the steps:\n1. Calculate attention scores using `query` and `key` with shape\n    `(batch_size, Tq, Tv)` as a non-linear sum\n    `scores = reduce_sum(tanh(query + key), axis=-1)`.\n2. Use scores to calculate a softmax distribution with shape\n    `(batch_size, Tq, Tv)`.\n3. Use the softmax distribution to create a linear combination of `value`\n    with shape `(batch_size, Tq, dim)`.\n\nArgs:\n    use_scale: If `True`, will create a scalar variable to scale the\n        attention scores.\n    dropout: Float between 0 and 1. Fraction of the units to drop for the\n        attention scores. Defaults to `0.0`.\n\nCall arguments:\n    inputs: List of the following tensors:\n        - `query`: Query tensor of shape `(batch_size, Tq, dim)`.\n        - `value`: Value tensor of shape `(batch_size, Tv, dim)`.\n        - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If\n            not given, will use `value` for both `key` and `value`, which is\n            the most common case.\n    mask: List of the following tensors:\n        - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.\n            If given, the output will be zero at the positions where\n            `mask==False`.\n        - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.\n            If given, will apply the mask such that values at positions\n             where `mask==False` do not contribute to the result.\n    return_attention_scores: bool, it `True`, returns the attention scores\n        (after masking and softmax) as an additional output argument.\n    training: Python boolean indicating whether the layer should behave in\n        training mode (adding dropout) or in inference mode (no dropout).\n    use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds\n        a mask such that position `i` cannot attend to positions `j > i`.\n        This prevents the flow of information from the future towards the\n        past. Defaults to `False`.\n\nOutput:\n    Attention outputs of shape `(batch_size, Tq, dim)`.\n    (Optional) Attention scores after masking and softmax with shape\n        `(batch_size, Tq, Tv)`.",
    "std_args": [
      {"name": "use_scale", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Alphadropout": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "ApplyIfFiniteState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ArgMax": {
    "description": "Computes the indices of the max elements of the input tensor's element along the provided axis. The resulting tensor has the same rank as the input if keepdims equals 1. If keepdims equals 0, then the resulting tensor has the reduced dimension pruned. If select_last_index is True (default False), th...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "axis", "type": "int"},
      {"name": "keepdims", "type": "int"},
      {"name": "select_last_index", "type": "int"},
    ],
    "variants": {},
  },
  "ArgMin": {
    "description": "Computes the indices of the min elements of the input tensor's element along the provided axis. The resulting tensor has the same rank as the input if keepdims equals 1. If keepdims equals 0, then the resulting tensor has the reduced dimension pruned. If select_last_index is True (default False), th...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "axis", "type": "int"},
      {"name": "keepdims", "type": "int"},
      {"name": "select_last_index", "type": "int"},
    ],
    "variants": {},
  },
  "ArrayLike": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "value", "type": "Any"},
    ],
    "variants": {},
  },
  "Asin": {
    "description": "Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Asinh": {
    "description": "Calculates the hyperbolic arcsine of the given input tensor element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Atan": {
    "description": "Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Atanh": {
    "description": "Calculates the hyperbolic arctangent of the given input tensor element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Atom": {
    "description": "Auto-generated from sass_code_defs",
    "std_args": [],
    "variants": {},
  },
  "Attention": {
    "description": "Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed. This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V. For self attention, `kv_sequence_length` equals to `q_sequence_length`. ...",
    "std_args": [
      {"name": "Q", "type": "Any"},
      {"name": "K", "type": "Any"},
      {"name": "V", "type": "Any"},
      {"name": "attn_mask", "type": "Any"},
      {"name": "past_key", "type": "Any"},
      {"name": "past_value", "type": "Any"},
      {"name": "nonpad_kv_seqlen", "type": "int"},
      {"name": "is_causal", "type": "int"},
      {"name": "kv_num_heads", "type": "int"},
      {"name": "q_num_heads", "type": "int"},
      {"name": "qk_matmul_output_mode", "type": "int"},
      {"name": "scale", "type": "float"},
      {"name": "softcap", "type": "float"},
      {"name": "softmax_precision", "type": "int"},
    ],
    "variants": {},
  },
  "AugMix": {
    "description": 'Performs the AugMix data augmentation technique.\n\nAugMix aims to produce images with variety while preserving the image\nsemantics and local statistics. During the augmentation process,\nthe same augmentation is applied across all images in the batch\nin num_chains different ways, with each chain consisting of\nchain_depth augmentations.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nReferences:\n    - [AugMix paper](https://arxiv.org/pdf/1912.02781)\n    - [Official Code](https://github.com/google-research/augmix)\n\nArgs:\n    value_range: the range of values the incoming images will have.\n        Represented as a two number tuple written (low, high).\n        This is typically either `(0, 1)` or `(0, 255)` depending\n        on how your preprocessing pipeline is set up.\n    num_chains: an integer representing the number of different chains to\n        be mixed, defaults to 3.\n    chain_depth: an integer representing the maximum number of\n        transformations to be applied in each chain. The actual number\n        of transformations in each chain will be sampled randomly\n        from the range `[0, `chain_depth`]`. Defaults to 3.\n    factor: The strength of the augmentation as a normalized value\n        between 0 and 1. Default is 0.3.\n    alpha: a float value used as the probability coefficients for the\n        Beta and Dirichlet distributions, defaults to 1.0.\n    all_ops: Use all operations (including random_brightness,\n        random_color_degeneration, random_contrast and random_sharpness).\n        Default is True.\n    interpolation: The interpolation method to use for resizing operations.\n        Options include `"nearest"`, `"bilinear"`. Default is `"bilinear"`.\n    seed: Integer. Used to create a random seed.',
    "std_args": [
      {"name": "value_range", "type": "Any"},
      {"name": "num_chains", "type": "Any"},
      {"name": "chain_depth", "type": "Any"},
      {"name": "factor", "type": "Any"},
      {"name": "alpha", "type": "Any"},
      {"name": "all_ops", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AutoContrast": {
    "description": "Performs the auto-contrast operation on an image.\n\nAuto contrast stretches the values of an image across the entire available\n`value_range`. This makes differences between pixels more obvious. An\nexample of this is if an image only has values `[0, 1]` out of the range\n`[0, 255]`, auto contrast will change the `1` values to be `255`.\n\nThis layer is active at both training and inference time.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    value_range: Range of values the incoming images will have.\n        Represented as a two number tuple written `(low, high)`.\n        This is typically either `(0, 1)` or `(0, 255)` depending\n        on how your preprocessing pipeline is set up.\n        Defaults to `(0, 255)`.",
    "std_args": [
      {"name": "value_range", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AuxOutput": {
    "description": "Auxiliary outputs from flex_attention operation.",
    "std_args": [],
    "variants": {},
  },
  "AveragePooling1D": {
    "description": 'Average pooling for temporal data.\n\nDownsamples the input representation by taking the average value over the\nwindow defined by `pool_size`. The window is shifted by `strides`.  The\nresulting output when using "valid" padding option has a shape of:\n`output_shape = (input_shape - pool_size + 1) / strides)`\n\nThe resulting output shape when using the "same" padding option is:\n`output_shape = input_shape / strides`\n\nArgs:\n    pool_size: int, size of the max pooling window.\n    strides: int or None. Specifies how much the pooling window moves\n        for each pooling step. If None, it will default to `pool_size`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    3D tensor with shape `(batch_size, steps, features)`.\n- If `data_format="channels_first"`:\n    3D tensor with shape `(batch_size, features, steps)`.\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    3D tensor with shape `(batch_size, downsampled_steps, features)`.\n- If `data_format="channels_first"`:\n    3D tensor with shape `(batch_size, features, downsampled_steps)`.\n\nExamples:\n\n`strides=1` and `padding="valid"`:\n\n>>> x = np.array([1., 2., 3., 4., 5.])\n>>> x = np.reshape(x, [1, 5, 1])\n>>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,\n...    strides=1, padding="valid")\n>>> avg_pool_1d(x)\n\n`strides=2` and `padding="valid"`:\n\n>>> x = np.array([1., 2., 3., 4., 5.])\n>>> x = np.reshape(x, [1, 5, 1])\n>>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,\n...    strides=2, padding="valid")\n>>> avg_pool_1d(x)\n\n`strides=1` and `padding="same"`:\n\n>>> x = np.array([1., 2., 3., 4., 5.])\n>>> x = np.reshape(x, [1, 5, 1])\n>>> avg_pool_1d = keras.layers.AveragePooling1D(pool_size=2,\n...    strides=1, padding="same")\n>>> avg_pool_1d(x)',
    "std_args": [
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AveragePooling2D": {
    "description": 'Average pooling operation for 2D spatial data.\n\nDownsamples the input along its spatial dimensions (height and width)\nby taking the average value over an input window\n(of size defined by `pool_size`) for each channel of the input.\nThe window is shifted by `strides` along each dimension.\n\nThe resulting output when using the `"valid"` padding option has a spatial\nshape (number of rows or columns) of:\n`output_shape = math.floor((input_shape - pool_size) / strides) + 1`\n(when `input_shape >= pool_size`)\n\nThe resulting output shape when using the `"same"` padding option is:\n`output_shape = input_shape`\n\nArgs:\n    pool_size: int or tuple of 2 integers, factors by which to downscale\n        (dim1, dim2). If only one integer is specified, the same\n        window length will be used for all dimensions.\n    strides: int or tuple of 2 integers, or None. Strides values. If None,\n        it will default to `pool_size`. If only one int is specified, the\n        same stride size will be used for all dimensions.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    4D tensor with shape `(batch_size, height, width, channels)`.\n- If `data_format="channels_first"`:\n    4D tensor with shape `(batch_size, channels, height, width)`.\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    4D tensor with shape\n    `(batch_size, pooled_height, pooled_width, channels)`.\n- If `data_format="channels_first"`:\n    4D tensor with shape\n    `(batch_size, channels, pooled_height, pooled_width)`.\n\nExamples:\n\n`strides=(1, 1)` and `padding="valid"`:\n\n>>> x = np.array([[1., 2., 3.],\n...               [4., 5., 6.],\n...               [7., 8., 9.]])\n>>> x = np.reshape(x, [1, 3, 3, 1])\n>>> avg_pool_2d = keras.layers.AveragePooling2D(pool_size=(2, 2),\n...    strides=(1, 1), padding="valid")\n>>> avg_pool_2d(x)\n\n`strides=(2, 2)` and `padding="valid"`:\n\n>>> x = np.array([[1., 2., 3., 4.],\n...              [5., 6., 7., 8.],\n...              [9., 10., 11., 12.]])\n>>> x = np.reshape(x, [1, 3, 4, 1])\n>>> avg_pool_2d = keras.layers.AveragePooling2D(pool_size=(2, 2),\n...    strides=(2, 2), padding="valid")\n>>> avg_pool_2d(x)\n\n`stride=(1, 1)` and `padding="same"`:\n\n>>> x = np.array([[1., 2., 3.],\n...                  [4., 5., 6.],\n...                  [7., 8., 9.]])\n>>> x = np.reshape(x, [1, 3, 3, 1])\n>>> avg_pool_2d = keras.layers.AveragePooling2D(pool_size=(2, 2),\n...    strides=(1, 1), padding="same")\n>>> avg_pool_2d(x)',
    "std_args": [
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AveragePooling3D": {
    "description": 'Average pooling operation for 3D data (spatial or spatio-temporal).\n\nDownsamples the input along its spatial dimensions (depth, height, and\nwidth) by taking the average value over an input window (of size defined by\n`pool_size`) for each channel of the input. The window is shifted by\n`strides` along each dimension.\n\nArgs:\n    pool_size: int or tuple of 3 integers, factors by which to downscale\n        (dim1, dim2, dim3). If only one integer is specified, the same\n        window length will be used for all dimensions.\n    strides: int or tuple of 3 integers, or None. Strides values. If None,\n        it will default to `pool_size`. If only one int is specified, the\n        same stride size will be used for all dimensions.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`\n\nExample:\n\n```python\ndepth = 30\nheight = 30\nwidth = 30\nchannels = 3\n\ninputs = keras.layers.Input(shape=(depth, height, width, channels))\nlayer = keras.layers.AveragePooling3D(pool_size=3)\noutputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)\n```',
    "std_args": [
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "AvgPool1d": {
    "description": "Applies a 1D average pooling over an input signal composed of several input planes.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "ceil_mode", "type": "Any"},
      {"name": "count_include_pad", "type": "Any"},
    ],
    "variants": {},
  },
  "AvgPool2d": {
    "description": "Applies a 2D average pooling over an input signal composed of several input planes.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "ceil_mode", "type": "Any"},
      {"name": "count_include_pad", "type": "Any"},
      {"name": "divisor_override", "type": "Any"},
    ],
    "variants": {},
  },
  "AvgPool3d": {
    "description": "Applies a 3D average pooling over an input signal composed of several input planes.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "ceil_mode", "type": "Any"},
      {"name": "count_include_pad", "type": "Any"},
      {"name": "divisor_override", "type": "Any"},
    ],
    "variants": {},
  },
  "Avgpool": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "kernel_size", "type": "Union"},
      {"name": "padding", "type": "Union"},
      {"name": "stride", "type": "Union"},
    ],
    "variants": {},
  },
  "B": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "BLOCK_M": {
    "description": "Thread block size for the sequence length dimension of Q in forward pass.",
    "std_args": [],
    "variants": {},
  },
  "BatchNorm": {
    "description": "Batch Normalization.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "BatchNormalization": {
    "description": "Carries out batch normalization as described in the paper https://arxiv.org/abs/1502.03167. Depending on the mode it is being run, There are five required inputs 'X', 'scale', 'B', 'input_mean' and 'input_var'. Note that 'input_mean' and 'input_var' are expected to be the estimated statistics in inf...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "scale", "type": "Any"},
      {"name": "B", "type": "Any"},
      {"name": "input_mean", "type": "Any"},
      {"name": "input_var", "type": "Any"},
      {"name": "epsilon", "type": "float"},
      {"name": "momentum", "type": "float"},
      {"name": "training_mode", "type": "int"},
    ],
    "variants": {},
  },
  "Bernoulli": {
    "description": "Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number, where an output of 1 is produced with probability p and an output of 0 is produced with pro...",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "dtype", "type": "int"},
      {"name": "seed", "type": "float"},
    ],
    "variants": {},
  },
  "Bidirectional": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Bilinear": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "bias", "type": "bool"},
    ],
    "variants": {},
  },
  "Bool": {
    "description": "Boolean type.",
    "std_args": [],
    "variants": {},
  },
  "C": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "CTCLoss": {
    "description": "The Connectionist Temporal Classification loss.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "blank", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "zero_infinity", "type": "Any"},
    ],
    "variants": {},
  },
  "Cache": {
    "description": "Container for mutable state.",
    "std_args": [
      {"name": "value", "type": "Any"},
    ],
    "variants": {},
  },
  "Call": {
    "description": "Auto-generated from sass_code_defs",
    "std_args": [],
    "variants": {},
  },
  "Cast": {
    "description": "The operator casts the elements of a given input tensor to a data type specified by the 'to' argument and returns an output tensor of the same size in the converted type. The 'to' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message. Casting from s...",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "round_mode", "type": "str"},
      {"name": "saturate", "type": "int"},
      {"name": "to", "type": "int"},
    ],
    "variants": {},
  },
  "CastBool": {
    "description": "Cast tensor to bool",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastByte": {
    "description": "Cast tensor to uint8",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastChar": {
    "description": "Cast tensor to int8/char",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastDouble": {
    "description": "Cast tensor to float64",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastFloat": {
    "description": "Cast tensor to float32",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastHalf": {
    "description": "Cast tensor to float16",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastInt": {
    "description": "Cast tensor to int32",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastLong": {
    "description": "Cast tensor to int64",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CastShort": {
    "description": "Cast tensor to int16",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "CategoryEncoding": {
    "description": 'A preprocessing layer which encodes integer features.\n\nThis layer provides options for condensing data into a categorical encoding\nwhen the total number of tokens are known in advance. It accepts integer\nvalues as inputs, and it outputs a dense or sparse representation of those\ninputs. For integer inputs where the total number of tokens is not known,\nuse `keras.layers.IntegerLookup` instead.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nExamples:\n\n**One-hot encoding data**\n\n>>> layer = keras.layers.CategoryEncoding(\n...           num_tokens=4, output_mode="one_hot")\n>>> layer([3, 2, 0, 1])\narray([[0., 0., 0., 1.],\n        [0., 0., 1., 0.],\n        [1., 0., 0., 0.],\n        [0., 1., 0., 0.]]>\n\n**Multi-hot encoding data**\n\n>>> layer = keras.layers.CategoryEncoding(\n...           num_tokens=4, output_mode="multi_hot")\n>>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])\narray([[1., 1., 0., 0.],\n        [1., 0., 0., 0.],\n        [0., 1., 1., 0.],\n        [0., 1., 0., 1.]]>\n\n**Using weighted inputs in `"count"` mode**\n\n>>> layer = keras.layers.CategoryEncoding(\n...           num_tokens=4, output_mode="count")\n>>> count_weights = np.array([[.1, .2], [.1, .1], [.2, .3], [.4, .2]])\n>>> layer([[0, 1], [0, 0], [1, 2], [3, 1]], count_weights=count_weights)\n  array([[0.1, 0.2, 0. , 0. ],\n         [0.2, 0. , 0. , 0. ],\n         [0. , 0.2, 0.3, 0. ],\n         [0. , 0.2, 0. , 0.4]]>\n\nArgs:\n    num_tokens: The total number of tokens the layer should support. All\n        inputs to the layer must integers in the range `0 <= value <\n        num_tokens`, or an error will be thrown.\n    output_mode: Specification for the output of the layer.\n        Values can be `"one_hot"`, `"multi_hot"` or `"count"`,\n        configuring the layer as follows:\n            - `"one_hot"`: Encodes each individual element in the input\n                into an array of `num_tokens` size, containing a 1 at the\n                element index. If the last dimension is size 1, will encode\n                on that dimension. If the last dimension is not size 1,\n                will append a new dimension for the encoded output.\n            - `"multi_hot"`: Encodes each sample in the input into a single\n                array of `num_tokens` size, containing a 1 for each\n                vocabulary term present in the sample. Treats the last\n                dimension as the sample dimension, if input shape is\n                `(..., sample_length)`, output shape will be\n                `(..., num_tokens)`.\n            - `"count"`: Like `"multi_hot"`, but the int array contains a\n                count of the number of times the token at that index\n                appeared in the sample.\n        For all output modes, currently only output up to rank 2 is\n        supported.\n        Defaults to `"multi_hot"`.\n    sparse: Whether to return a sparse tensor; for backends that support\n        sparse tensors.\n\nCall arguments:\n    inputs: A 1D or 2D tensor of integer inputs.\n    count_weights: A tensor in the same shape as `inputs` indicating the\n        weight for each sample value when summing up in `count` mode.\n        Not used in `"multi_hot"` or `"one_hot"` modes.',
    "std_args": [
      {"name": "num_tokens", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Ceil": {
    "description": "Ceil takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the ceil is, y = ceil(x), is applied to the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Celu": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "alpha", "type": "Any"},
    ],
    "variants": {},
  },
  "CenterCrop": {
    "description": "Crops the given image at the center.",
    "std_args": [
      {"name": "size", "type": "Any"},
    ],
    "variants": {},
  },
  "Clamp": {
    "description": "Clamp all elements in input into the range [min, max].",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "min", "type": "Any"},
      {"name": "max", "type": "Any"},
    ],
    "variants": {},
  },
  "Clip": {
    "description": "Clip operator limits the given input within an interval. The interval is specified by the inputs 'min' and 'max'. They default to numeric_limits::lowest() and numeric_limits::max(), respectively. When 'min' is greater than 'max', the clip operator sets all the 'input' values to the value of 'max'. T...",
    "std_args": [
      {"name": "input", "type": "Tensor"},
      {"name": "min", "type": "Tensor"},
      {"name": "max", "type": "Tensor"},
    ],
    "variants": {},
  },
  "ClipByGlobalNormState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ClipGradNorm": {
    "description": "Clips gradient norm of an iterable of parameters.",
    "std_args": [
      {"name": "parameters", "type": "Any"},
      {"name": "max_norm", "type": "Any"},
    ],
    "variants": {},
  },
  "ClipState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Compile": {
    "description": "JIT Alias.",
    "std_args": [
      {"name": "func", "type": "Any"},
    ],
    "variants": {},
  },
  "ComplexWarning": {
    "description": "The warning raised when casting a complex dtype to a real dtype.",
    "std_args": [],
    "variants": {},
  },
  "Compress": {
    "description": "Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index. In case axis is not provided, input is flattened before elements are selected. Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html",
    "std_args": [
      {"name": "input", "type": "Tensor"},
      {"name": "condition", "type": "Any"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "ComputeCv": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Concat": {
    "description": "Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.",
    "std_args": [
      {"name": "inputs", "type": "Tensor"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "ConditionallyMaskState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ConditionallyTransformState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Constant": {
    "description": "This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value, or value_* must be specified.",
    "std_args": [
      {"name": "sparse_value", "type": "Tensor"},
      {"name": "value", "type": "Tensor"},
      {"name": "value_float", "type": "float"},
      {"name": "value_floats", "type": "List[float]"},
      {"name": "value_int", "type": "int"},
      {"name": "value_ints", "type": "List[int]"},
      {"name": "value_string", "type": "str"},
      {"name": "value_strings", "type": "List[str]"},
    ],
    "variants": {},
  },
  "ControlVariate": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Conv": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
    ],
    "variants": {},
  },
  "Conv1DTranspose": {
    "description": '1D transposed convolution layer.\n\nThe need for transposed convolutions generally arise from the desire to use\na transformation going in the opposite direction of a normal convolution,\ni.e., from something that has the shape of the output of some convolution\nto something that has the shape of its input while maintaining a\nconnectivity pattern that is compatible with said convolution.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the transpose convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        transposed convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the transposed convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    output_padding: An integer tuple/list of 1 integer specifying the\n        amount of padding along the time dimension of the output tensor.\n        The amount of output padding must be lower than the stride.\n        If set to `None` (default), the output shape is inferred.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: An integer tuple/list of 1 integer, specifying\n        the dilation rate to use for dilated convolution.\n        Currently, specifying a `dilation_rate` value != 1 is\n        incompatible with specifying a stride value != 1.\n        Also dilation rate larger than 1 is not currently supported.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, steps, channels)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, channels, steps)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, new_steps, filters)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, filters, new_steps)`\n\nReturns:\n    A 3D tensor representing\n    `activation(conv1d_transpose(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nReferences:\n- [A guide to convolution arithmetic for deep learning](\n    https://arxiv.org/abs/1603.07285v1)\n- [Deconvolutional Networks](\n    https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)\n\nExample:\n\n>>> x = np.random.rand(4, 10, 128)\n>>> y = keras.layers.Conv1DTranspose(32, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 21, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Conv1d": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "in_channels", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "Conv2DTranspose": {
    "description": '2D transposed convolution layer.\n\nThe need for transposed convolutions generally arise from the desire to use\na transformation going in the opposite direction of a normal convolution,\ni.e., from something that has the shape of the output of some convolution\nto something that has the shape of its input while maintaining a\nconnectivity pattern that is compatible with said convolution.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the transposed convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        transposed convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the transposed convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    output_padding: An integer or tuple/list of 2 integers,\n        specifying the amount of padding along the height and width\n        of the output tensor.\n        Can be a single integer to specify the same value for all\n        spatial dimensions.\n        The amount of output padding along a given dimension must be\n        lower than the stride along that same dimension.\n        If set to `None` (default), the output shape is inferred.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch_size, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n     dilation_rate: An integer or tuple/list of 2 integers,\n        specifying the dilation rate for\n        all spatial dimensions for dilated convolution.\n        Specifying different dilation rates\n        for different dimensions is not supported.\n        Currently, specifying any `dilation_rate` value != 1 is\n        incompatible with specifying any stride value != 1.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, height, width, channels)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`\n\nReturns:\n    A 4D tensor representing\n    `activation(conv2d_transpose(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nReferences:\n- [A guide to convolution arithmetic for deep learning](\n    https://arxiv.org/abs/1603.07285v1)\n- [Deconvolutional Networks](\n    https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)\n\nExample:\n\n>>> x = np.random.rand(4, 10, 8, 128)\n>>> y = keras.layers.Conv2DTranspose(32, 2, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 20, 16, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Conv2d": {
    "description": "Applies a 2D convolution over an input signal composed of several input planes.",
    "std_args": [
      {"name": "in_channels", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
    ],
    "variants": {},
  },
  "Conv3DTranspose": {
    "description": '3D transposed convolution layer.\n\nThe need for transposed convolutions generally arise from the desire to use\na transformation going in the opposite direction of a normal convolution,\ni.e., from something that has the shape of the output of some convolution\nto something that has the shape of its input while maintaining a\nconnectivity pattern that is compatible with said convolution.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the transposed convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        transposed convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the transposed convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n     output_padding: An integer or tuple/list of 3 integers,\n        specifying the amount of padding along the depth, height, and\n        width.\n        Can be a single integer to specify the same value for all\n        spatial dimensions.\n        The amount of output padding along a given dimension must be\n        lower than the stride along that same dimension.\n        If set to `None` (default), the output shape is inferred.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    dilation_rate: an integer or tuple/list of 3 integers, specifying\n        the dilation rate to use for dilated convolution.\n        Can be a single integer to specify the same value for\n        all spatial dimensions.\n        Currently, specifying any `dilation_rate` value != 1 is\n        incompatible with specifying any stride value != 1.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, new_spatial_dim1, new_spatial_dim2, new_spatial_dim3,\n    filters)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, filters, new_spatial_dim1, new_spatial_dim2,\n    new_spatial_dim3)`\n\nReturns:\n    A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nReferences:\n- [A guide to convolution arithmetic for deep learning](\n    https://arxiv.org/abs/1603.07285v1)\n- [Deconvolutional Networks](\n    https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)\n\nExample:\n\n>>> x = np.random.rand(4, 10, 8, 12, 128)\n>>> y = keras.layers.Conv3DTranspose(32, 2, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 20, 16, 24, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Conv3d": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "in_channels", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "ConvLSTM1D": {
    "description": '1D Convolutional LSTM.\n\nSimilar to an LSTM layer, but the input transformations\nand recurrent transformations are both convolutional.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of\n        the convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the\n        same height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 1 integers, specifying the dilation\n        rate to use for dilated convolution.\n    activation: Activation function to use. By default hyperbolic tangent\n        activation function is applied (`tanh(x)`).\n    recurrent_activation: Activation function to use for the recurrent step.\n    use_bias: Boolean, whether the layer uses a bias vector.\n    kernel_initializer: Initializer for the `kernel` weights matrix,\n        used for the linear transformation of the inputs.\n    recurrent_initializer: Initializer for the `recurrent_kernel` weights\n        matrix, used for the linear transformation of the recurrent state.\n    bias_initializer: Initializer for the bias vector.\n    unit_forget_bias: Boolean. If `True`, add 1 to the bias of\n        the forget gate at initialization.\n        Use in combination with `bias_initializer="zeros"`.\n        This is recommended in [Jozefowicz et al., 2015](\n        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)\n    kernel_regularizer: Regularizer function applied to the `kernel` weights\n        matrix.\n    recurrent_regularizer: Regularizer function applied to the\n        `recurrent_kernel` weights matrix.\n    bias_regularizer: Regularizer function applied to the bias vector.\n    activity_regularizer: Regularizer function applied to.\n    kernel_constraint: Constraint function applied to the `kernel` weights\n        matrix.\n    recurrent_constraint: Constraint function applied to the\n        `recurrent_kernel` weights matrix.\n    bias_constraint: Constraint function applied to the bias vector.\n    dropout: Float between 0 and 1. Fraction of the units to drop for the\n        linear transformation of the inputs.\n    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop\n        for the linear transformation of the recurrent state.\n    seed: Random seed for dropout.\n    return_sequences: Boolean. Whether to return the last output\n        in the output sequence, or the full sequence. Default: `False`.\n    return_state: Boolean. Whether to return the last state in addition\n        to the output. Default: `False`.\n    go_backwards: Boolean (default: `False`).\n        If `True`, process the input sequence backwards and return the\n        reversed sequence.\n    stateful: Boolean (default False). If `True`, the last state\n        for each sample at index i in a batch will be used as initial\n        state for the sample of index i in the following batch.\n    unroll: Boolean (default: `False`).\n        If `True`, the network will be unrolled,\n        else a symbolic loop will be used.\n        Unrolling can speed-up a RNN,\n        although it tends to be more memory-intensive.\n        Unrolling is only suitable for short sequences.\n\n\nCall arguments:\n    inputs: A 4D tensor.\n    initial_state: List of initial state tensors to be passed to the first\n        call of the cell.\n    mask: Binary tensor of shape `(samples, timesteps)` indicating whether a\n        given timestep should be masked.\n    training: Python boolean indicating whether the layer should behave in\n        training mode or in inference mode.\n        This is only relevant if `dropout` or `recurrent_dropout` are set.\n\nInput shape:\n\n- If `data_format="channels_first"`:\n    4D tensor with shape: `(samples, time, channels, rows)`\n- If `data_format="channels_last"`:\n    4D tensor with shape: `(samples, time, rows, channels)`\n\nOutput shape:\n\n- If `return_state`: a list of tensors. The first tensor is the output.\n    The remaining tensors are the last states,\n    each 3D tensor with shape: `(samples, filters, new_rows)` if\n    `data_format=\'channels_first\'`\n    or shape: `(samples, new_rows, filters)` if\n    `data_format=\'channels_last\'`.\n    `rows` values might have changed due to padding.\n- If `return_sequences`: 4D tensor with shape: `(samples, timesteps,\n    filters, new_rows)` if data_format=\'channels_first\'\n    or shape: `(samples, timesteps, new_rows, filters)` if\n    `data_format=\'channels_last\'`.\n- Else, 3D tensor with shape: `(samples, filters, new_rows)` if\n    `data_format=\'channels_first\'`\n    or shape: `(samples, new_rows, filters)` if\n    `data_format=\'channels_last\'`.\n\nReferences:\n\n- [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)\n    (the current implementation does not include the feedback loop on the\n    cells output).',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "recurrent_activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "recurrent_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "unit_forget_bias", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "recurrent_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "recurrent_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "recurrent_dropout", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "return_sequences", "type": "Any"},
      {"name": "return_state", "type": "Any"},
      {"name": "go_backwards", "type": "Any"},
      {"name": "stateful", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ConvLSTM2D": {
    "description": '2D Convolutional LSTM.\n\nSimilar to an LSTM layer, but the input transformations\nand recurrent transformations are both convolutional.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the convolution).\n    kernel_size: int or tuple/list of 2 integers, specifying the size of the\n        convolution window.\n    strides: int or tuple/list of 2 integers, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 2 integers, specifying the dilation\n        rate to use for dilated convolution.\n    activation: Activation function to use. By default hyperbolic tangent\n        activation function is applied (`tanh(x)`).\n    recurrent_activation: Activation function to use for the recurrent step.\n    use_bias: Boolean, whether the layer uses a bias vector.\n    kernel_initializer: Initializer for the `kernel` weights matrix,\n        used for the linear transformation of the inputs.\n    recurrent_initializer: Initializer for the `recurrent_kernel` weights\n        matrix, used for the linear transformation of the recurrent state.\n    bias_initializer: Initializer for the bias vector.\n    unit_forget_bias: Boolean. If `True`, add 1 to the bias of the forget\n        gate at initialization.\n        Use in combination with `bias_initializer="zeros"`.\n        This is recommended in [Jozefowicz et al., 2015](\n        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)\n    kernel_regularizer: Regularizer function applied to the `kernel` weights\n        matrix.\n    recurrent_regularizer: Regularizer function applied to the\n        `recurrent_kernel` weights matrix.\n    bias_regularizer: Regularizer function applied to the bias vector.\n    activity_regularizer: Regularizer function applied to.\n    kernel_constraint: Constraint function applied to the `kernel` weights\n        matrix.\n    recurrent_constraint: Constraint function applied to the\n        `recurrent_kernel` weights matrix.\n    bias_constraint: Constraint function applied to the bias vector.\n    dropout: Float between 0 and 1. Fraction of the units to drop for the\n        linear transformation of the inputs.\n    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop\n        for the linear transformation of the recurrent state.\n    seed: Random seed for dropout.\n    return_sequences: Boolean. Whether to return the last output\n        in the output sequence, or the full sequence. Default: `False`.\n    return_state: Boolean. Whether to return the last state in addition\n        to the output. Default: `False`.\n    go_backwards: Boolean (default: `False`).\n        If `True`, process the input sequence backwards and return the\n        reversed sequence.\n    stateful: Boolean (default False). If `True`, the last state\n        for each sample at index i in a batch will be used as initial\n        state for the sample of index i in the following batch.\n    unroll: Boolean (default: `False`).\n        If `True`, the network will be unrolled,\n        else a symbolic loop will be used.\n        Unrolling can speed-up a RNN,\n        although it tends to be more memory-intensive.\n        Unrolling is only suitable for short sequences.\n\n\nCall arguments:\n    inputs: A 5D tensor.\n    mask: Binary tensor of shape `(samples, timesteps)` indicating whether a\n        given timestep should be masked.\n    training: Python boolean indicating whether the layer should behave in\n        training mode or in inference mode.\n        This is only relevant if `dropout` or `recurrent_dropout` are set.\n    initial_state: List of initial state tensors to be passed to the first\n        call of the cell.\n\nInput shape:\n\n- If `data_format=\'channels_first\'`:\n    5D tensor with shape: `(samples, time, channels, rows, cols)`\n- If `data_format=\'channels_last\'`:\n    5D tensor with shape: `(samples, time, rows, cols, channels)`\n\nOutput shape:\n\n- If `return_state`: a list of tensors. The first tensor is the output.\n    The remaining tensors are the last states,\n    each 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if\n    `data_format=\'channels_first\'`\n    or shape: `(samples, new_rows, new_cols, filters)` if\n    `data_format=\'channels_last\'`. `rows` and `cols` values might have\n    changed due to padding.\n- If `return_sequences`: 5D tensor with shape: `(samples, timesteps,\n    filters, new_rows, new_cols)` if data_format=\'channels_first\'\n    or shape: `(samples, timesteps, new_rows, new_cols, filters)` if\n    `data_format=\'channels_last\'`.\n- Else, 4D tensor with shape: `(samples, filters, new_rows, new_cols)` if\n    `data_format=\'channels_first\'`\n    or shape: `(samples, new_rows, new_cols, filters)` if\n    `data_format=\'channels_last\'`.\n\nReferences:\n\n- [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)\n    (the current implementation does not include the feedback loop on the\n    cells output).',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "recurrent_activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "recurrent_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "unit_forget_bias", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "recurrent_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "recurrent_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "recurrent_dropout", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "return_sequences", "type": "Any"},
      {"name": "return_state", "type": "Any"},
      {"name": "go_backwards", "type": "Any"},
      {"name": "stateful", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ConvLSTM3D": {
    "description": '3D Convolutional LSTM.\n\nSimilar to an LSTM layer, but the input transformations\nand recurrent transformations are both convolutional.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the convolution).\n    kernel_size: int or tuple/list of 3 integers, specifying the size of the\n        convolution window.\n    strides: int or tuple/list of 3 integers, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 3 integers, specifying the dilation\n        rate to use for dilated convolution.\n    activation: Activation function to use. By default hyperbolic tangent\n        activation function is applied (`tanh(x)`).\n    recurrent_activation: Activation function to use for the recurrent step.\n    use_bias: Boolean, whether the layer uses a bias vector.\n    kernel_initializer: Initializer for the `kernel` weights matrix,\n        used for the linear transformation of the inputs.\n    recurrent_initializer: Initializer for the `recurrent_kernel` weights\n        matrix, used for the linear transformation of the recurrent state.\n    bias_initializer: Initializer for the bias vector.\n    unit_forget_bias: Boolean. If `True`, add 1 to the bias of the forget\n        gate at initialization.\n        Use in combination with `bias_initializer="zeros"`.\n        This is recommended in [Jozefowicz et al., 2015](\n        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)\n    kernel_regularizer: Regularizer function applied to the `kernel` weights\n        matrix.\n    recurrent_regularizer: Regularizer function applied to the\n        `recurrent_kernel` weights matrix.\n    bias_regularizer: Regularizer function applied to the bias vector.\n    activity_regularizer: Regularizer function applied to.\n    kernel_constraint: Constraint function applied to the `kernel` weights\n        matrix.\n    recurrent_constraint: Constraint function applied to the\n        `recurrent_kernel` weights matrix.\n    bias_constraint: Constraint function applied to the bias vector.\n    dropout: Float between 0 and 1. Fraction of the units to drop for the\n        linear transformation of the inputs.\n    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop\n        for the linear transformation of the recurrent state.\n    seed: Random seed for dropout.\n    return_sequences: Boolean. Whether to return the last output\n        in the output sequence, or the full sequence. Default: `False`.\n    return_state: Boolean. Whether to return the last state in addition\n        to the output. Default: `False`.\n    go_backwards: Boolean (default: `False`).\n        If `True`, process the input sequence backwards and return the\n        reversed sequence.\n    stateful: Boolean (default False). If `True`, the last state\n        for each sample at index i in a batch will be used as initial\n        state for the sample of index i in the following batch.\n    unroll: Boolean (default: `False`).\n        If `True`, the network will be unrolled,\n        else a symbolic loop will be used.\n        Unrolling can speed-up a RNN,\n        although it tends to be more memory-intensive.\n        Unrolling is only suitable for short sequences.\n\n\nCall arguments:\n    inputs: A 6D tensor.\n    mask: Binary tensor of shape `(samples, timesteps)` indicating whether a\n        given timestep should be masked.\n    training: Python boolean indicating whether the layer should behave in\n        training mode or in inference mode.\n        This is only relevant if `dropout` or `recurrent_dropout` are set.\n    initial_state: List of initial state tensors to be passed to the first\n        call of the cell.\n\nInput shape:\n\n- If `data_format=\'channels_first\'`:\n    5D tensor with shape: `(samples, time, channels, *spatial_dims)`\n- If `data_format=\'channels_last\'`:\n    5D tensor with shape: `(samples, time, *spatial_dims, channels)`\n\nOutput shape:\n\n- If `return_state`: a list of tensors. The first tensor is the output.\n    The remaining tensors are the last states,\n    each 4D tensor with shape: `(samples, filters, *spatial_dims)` if\n    `data_format=\'channels_first\'`\n    or shape: `(samples, *spatial_dims, filters)` if\n    `data_format=\'channels_last\'`.\n- If `return_sequences`: 5D tensor with shape: `(samples, timesteps,\n    filters, *spatial_dims)` if data_format=\'channels_first\'\n    or shape: `(samples, timesteps, *spatial_dims, filters)` if\n    `data_format=\'channels_last\'`.\n- Else, 4D tensor with shape: `(samples, filters, *spatial_dims)` if\n    `data_format=\'channels_first\'`\n    or shape: `(samples, *spatial_dims, filters)` if\n    `data_format=\'channels_last\'`.\n\nReferences:\n\n- [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)\n    (the current implementation does not include the feedback loop on the\n    cells output).',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "recurrent_activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "recurrent_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "unit_forget_bias", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "recurrent_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "recurrent_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "recurrent_dropout", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "return_sequences", "type": "Any"},
      {"name": "return_state", "type": "Any"},
      {"name": "go_backwards", "type": "Any"},
      {"name": "stateful", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ConvT": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ConvTranspose1d": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "in_channels", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "ConvTranspose2d": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "in_channels", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "ConvTranspose3d": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "in_channels", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "Convolution1D": {
    "description": '1D convolution layer (e.g. temporal convolution).\n\nThis layer creates a convolution kernel that is convolved with the layer\ninput over a single spatial (or temporal) dimension to produce a tensor of\noutputs. If `use_bias` is True, a bias vector is created and added to the\noutputs. Finally, if `activation` is not `None`, it is applied to the\noutputs as well.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, `"valid"`, `"same"` or `"causal"`(case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n        `"causal"` results in causal(dilated) convolutions, e.g. `output[t]`\n        does not depend on`input[t+1:]`. Useful when modeling temporal data\n        where the model should not violate the temporal order.\n        See [WaveNet: A Generative Model for Raw Audio, section2.1](\n        https://arxiv.org/abs/1609.03499).\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 1 integers, specifying the dilation\n        rate to use for dilated convolution.\n    groups: A positive int specifying the number of groups in which the\n        input is split along the channel axis. Each group is convolved\n        separately with `filters // groups` filters. The output is the\n        concatenation of all the `groups` results along the channel axis.\n        Input channels and `filters` must both be divisible by `groups`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, steps, channels)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, channels, steps)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, new_steps, filters)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, filters, new_steps)`\n\nReturns:\n    A 3D tensor representing `activation(conv1d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nExample:\n\n>>> # The inputs are 128-length vectors with 10 timesteps, and the\n>>> # batch size is 4.\n>>> x = np.random.rand(4, 10, 128)\n>>> y = keras.layers.Conv1D(32, 3, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 8, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Convolution1DTranspose": {
    "description": '1D transposed convolution layer.\n\nThe need for transposed convolutions generally arise from the desire to use\na transformation going in the opposite direction of a normal convolution,\ni.e., from something that has the shape of the output of some convolution\nto something that has the shape of its input while maintaining a\nconnectivity pattern that is compatible with said convolution.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the transpose convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        transposed convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the transposed convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    output_padding: An integer tuple/list of 1 integer specifying the\n        amount of padding along the time dimension of the output tensor.\n        The amount of output padding must be lower than the stride.\n        If set to `None` (default), the output shape is inferred.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: An integer tuple/list of 1 integer, specifying\n        the dilation rate to use for dilated convolution.\n        Currently, specifying a `dilation_rate` value != 1 is\n        incompatible with specifying a stride value != 1.\n        Also dilation rate larger than 1 is not currently supported.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, steps, channels)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, channels, steps)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, new_steps, filters)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, filters, new_steps)`\n\nReturns:\n    A 3D tensor representing\n    `activation(conv1d_transpose(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nReferences:\n- [A guide to convolution arithmetic for deep learning](\n    https://arxiv.org/abs/1603.07285v1)\n- [Deconvolutional Networks](\n    https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)\n\nExample:\n\n>>> x = np.random.rand(4, 10, 128)\n>>> y = keras.layers.Conv1DTranspose(32, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 21, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Convolution2D": {
    "description": '2D convolution layer.\n\nThis layer creates a convolution kernel that is convolved with the layer\ninput over a 2D spatial (or temporal) dimension (height and width) to\nproduce a tensor of outputs. If `use_bias` is True, a bias vector is created\nand added to the outputs. Finally, if `activation` is not `None`, it is\napplied to the outputs as well.\n\nNote on numerical precision: While in general Keras operation execution\nresults are identical across backends up to 1e-7 precision in float32,\n`Conv2D` operations may show larger variations. Due to the large\nnumber of element-wise multiplications and additions in convolution\noperations, especially with large inputs or kernel sizes, accumulated\nfloating-point differences can exceed this 1e-7 threshold. These variations\nare particularly noticeable when using different backends (e.g., TensorFlow\nvs JAX) or different hardware.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the convolution).\n    kernel_size: int or tuple/list of 2 integer, specifying the size of the\n        convolution window.\n    strides: int or tuple/list of 2 integer, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch_size, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    dilation_rate: int or tuple/list of 2 integers, specifying the dilation\n        rate to use for dilated convolution.\n    groups: A positive int specifying the number of groups in which the\n        input is split along the channel axis. Each group is convolved\n        separately with `filters // groups` filters. The output is the\n        concatenation of all the `groups` results along the channel axis.\n        Input channels and `filters` must both be divisible by `groups`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, height, width, channels)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`\n\nReturns:\n    A 4D tensor representing `activation(conv2d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 10, 128)\n>>> y = keras.layers.Conv2D(32, 3, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 8, 8, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Convolution2DTranspose": {
    "description": '2D transposed convolution layer.\n\nThe need for transposed convolutions generally arise from the desire to use\na transformation going in the opposite direction of a normal convolution,\ni.e., from something that has the shape of the output of some convolution\nto something that has the shape of its input while maintaining a\nconnectivity pattern that is compatible with said convolution.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the transposed convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        transposed convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the transposed convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    output_padding: An integer or tuple/list of 2 integers,\n        specifying the amount of padding along the height and width\n        of the output tensor.\n        Can be a single integer to specify the same value for all\n        spatial dimensions.\n        The amount of output padding along a given dimension must be\n        lower than the stride along that same dimension.\n        If set to `None` (default), the output shape is inferred.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch_size, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n     dilation_rate: An integer or tuple/list of 2 integers,\n        specifying the dilation rate for\n        all spatial dimensions for dilated convolution.\n        Specifying different dilation rates\n        for different dimensions is not supported.\n        Currently, specifying any `dilation_rate` value != 1 is\n        incompatible with specifying any stride value != 1.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, height, width, channels)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`\n\nReturns:\n    A 4D tensor representing\n    `activation(conv2d_transpose(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nReferences:\n- [A guide to convolution arithmetic for deep learning](\n    https://arxiv.org/abs/1603.07285v1)\n- [Deconvolutional Networks](\n    https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)\n\nExample:\n\n>>> x = np.random.rand(4, 10, 8, 128)\n>>> y = keras.layers.Conv2DTranspose(32, 2, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 20, 16, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Convolution3D": {
    "description": '3D convolution layer.\n\nThis layer creates a convolution kernel that is convolved with the layer\ninput over a 3D spatial (or temporal) dimension (width,height and depth) to\nproduce a tensor of outputs. If `use_bias` is True, a bias vector is created\nand added to the outputs. Finally, if `activation` is not `None`, it is\napplied to the outputs as well.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the convolution).\n    kernel_size: int or tuple/list of 3 integer, specifying the size of the\n        convolution window.\n    strides: int or tuple/list of 3 integer, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 3 integers, specifying the dilation\n        rate to use for dilated convolution.\n    groups: A positive int specifying the number of groups in which the\n        input is split along the channel axis. Each group is convolved\n        separately with `filters // groups` filters. The output is the\n        concatenation of all the `groups` results along the channel axis.\n        Input channels and `filters` must both be divisible by `groups`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, new_spatial_dim1, new_spatial_dim2, new_spatial_dim3,\n    filters)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, filters, new_spatial_dim1, new_spatial_dim2,\n    new_spatial_dim3)`\n\nReturns:\n    A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 10, 10, 128)\n>>> y = keras.layers.Conv3D(32, 3, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 8, 8, 8, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Convolution3DTranspose": {
    "description": '3D transposed convolution layer.\n\nThe need for transposed convolutions generally arise from the desire to use\na transformation going in the opposite direction of a normal convolution,\ni.e., from something that has the shape of the output of some convolution\nto something that has the shape of its input while maintaining a\nconnectivity pattern that is compatible with said convolution.\n\nArgs:\n    filters: int, the dimension of the output space (the number of filters\n        in the transposed convolution).\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        transposed convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the transposed convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n     output_padding: An integer or tuple/list of 3 integers,\n        specifying the amount of padding along the depth, height, and\n        width.\n        Can be a single integer to specify the same value for all\n        spatial dimensions.\n        The amount of output padding along a given dimension must be\n        lower than the stride along that same dimension.\n        If set to `None` (default), the output shape is inferred.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    dilation_rate: an integer or tuple/list of 3 integers, specifying\n        the dilation rate to use for dilated convolution.\n        Can be a single integer to specify the same value for\n        all spatial dimensions.\n        Currently, specifying any `dilation_rate` value != 1 is\n        incompatible with specifying any stride value != 1.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    kernel_initializer: Initializer for the convolution kernel. If `None`,\n        the default initializer (`"glorot_uniform"`) will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    kernel_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, new_spatial_dim1, new_spatial_dim2, new_spatial_dim3,\n    filters)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, filters, new_spatial_dim1, new_spatial_dim2,\n    new_spatial_dim3)`\n\nReturns:\n    A 5D tensor representing `activation(conv3d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nReferences:\n- [A guide to convolution arithmetic for deep learning](\n    https://arxiv.org/abs/1603.07285v1)\n- [Deconvolutional Networks](\n    https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)\n\nExample:\n\n>>> x = np.random.rand(4, 10, 8, 12, 128)\n>>> y = keras.layers.Conv3DTranspose(32, 2, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 20, 16, 24, 32)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Cos": {
    "description": "Calculates the cosine of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Cosh": {
    "description": "Calculates the hyperbolic cosine of the given input tensor element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "CosineAnnealingLR": {
    "description": "Set the learning rate of each parameter group using a cosine annealing schedule.",
    "std_args": [
      {"name": "optimizer", "type": "Any"},
      {"name": "T_max", "type": "Any"},
    ],
    "variants": {},
  },
  "CosineSimilarity": {
    "description": "Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "dim", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "Cosinesimilarity": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "dim", "type": "int"},
      {"name": "reduction", "type": "Literal"},
    ],
    "variants": {},
  },
  "Cropping1D": {
    "description": "Cropping layer for 1D input (e.g. temporal sequence).\n\nIt crops along the time dimension (axis 1).\n\nExample:\n\n>>> input_shape = (2, 3, 2)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> x\n[[[ 0  1]\n  [ 2  3]\n  [ 4  5]]\n [[ 6  7]\n  [ 8  9]\n  [10 11]]]\n>>> y = keras.layers.Cropping1D(cropping=1)(x)\n>>> y\n[[[2 3]]\n [[8 9]]]\n\nArgs:\n    cropping: Int, or tuple of int (length 2), or dictionary.\n        - If int: how many units should be trimmed off at the beginning and\n          end of the cropping dimension (axis 1).\n        - If tuple of 2 ints: how many units should be trimmed off at the\n          beginning and end of the cropping dimension\n          (`(left_crop, right_crop)`).\n\nInput shape:\n    3D tensor with shape `(batch_size, axis_to_crop, features)`\n\nOutput shape:\n    3D tensor with shape `(batch_size, cropped_axis, features)`",
    "std_args": [
      {"name": "cropping", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Cropping2D": {
    "description": 'Cropping layer for 2D input (e.g. picture).\n\nIt crops along spatial dimensions, i.e. height and width.\n\nExample:\n\n>>> input_shape = (2, 28, 28, 3)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> y = keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)\n>>> y.shape\n(2, 24, 20, 3)\n\nArgs:\n    cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.\n        - If int: the same symmetric cropping is applied to height and\n          width.\n        - If tuple of 2 ints: interpreted as two different symmetric\n          cropping values for height and width:\n          `(symmetric_height_crop, symmetric_width_crop)`.\n        - If tuple of 2 tuples of 2 ints: interpreted as\n          `((top_crop, bottom_crop), (left_crop, right_crop))`.\n    data_format: A string, one of `"channels_last"` (default) or\n        `"channels_first"`. The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, height, width, channels)` while `"channels_first"`\n        corresponds to inputs with shape\n        `(batch_size, channels, height, width)`.\n        When unspecified, uses `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json` (if exists). Defaults to\n        `"channels_last"`.\n\nInput shape:\n    4D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, height, width, channels)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, channels, height, width)`\n\nOutput shape:\n    4D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, cropped_height, cropped_width, channels)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, channels, cropped_height, cropped_width)`',
    "std_args": [
      {"name": "cropping", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Cropping3D": {
    "description": 'Cropping layer for 3D data (e.g. spatial or spatio-temporal).\n\nExample:\n\n>>> input_shape = (2, 28, 28, 10, 3)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> y = keras.layers.Cropping3D(cropping=(2, 4, 2))(x)\n>>> y.shape\n(2, 24, 20, 6, 3)\n\nArgs:\n    cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.\n        - If int: the same symmetric cropping is applied to depth, height,\n          and width.\n        - If tuple of 3 ints: interpreted as three different symmetric\n          cropping values for depth, height, and width:\n          `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.\n        - If tuple of 3 tuples of 2 ints: interpreted as\n          `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,\n          right_dim2_crop), (left_dim3_crop, right_dim3_crop))`.\n    data_format: A string, one of `"channels_last"` (default) or\n        `"channels_first"`. The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        When unspecified, uses `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json` (if exists). Defaults to\n        `"channels_last"`.\n\nInput shape:\n    5D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, first_axis_to_crop, second_axis_to_crop,\n      third_axis_to_crop, channels)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, channels, first_axis_to_crop, second_axis_to_crop,\n      third_axis_to_crop)`\n\nOutput shape:\n    5D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, first_cropped_axis, second_cropped_axis,\n      third_cropped_axis, channels)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, channels, first_cropped_axis, second_cropped_axis,\n      third_cropped_axis)`',
    "std_args": [
      {"name": "cropping", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "CrossEntropyLoss": {
    "description": "Cross Entropy Loss.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "weight", "type": "Any"},
    ],
    "variants": {},
  },
  "CudaAvailable": {
    "description": "Checks if a CUDA device is available.",
    "std_args": [],
    "variants": {},
  },
  "CumSum": {
    "description": "Performs cumulative sum of the input elements along the given axis. By default, it will do the sum inclusively meaning the first element is copied as is. Through an `exclusive` attribute, this behavior can change to exclude the first element. It can also perform summation in the opposite direction o...",
    "std_args": [
      {"name": "x", "type": "Tensor"},
      {"name": "axis", "type": "Any"},
      {"name": "exclusive", "type": "int"},
      {"name": "reverse", "type": "int"},
    ],
    "variants": {},
  },
  "CutMix": {
    "description": "CutMix data augmentation technique.\n\nCutMix is a data augmentation method where patches are cut and pasted\nbetween two images in the dataset, while the labels are also mixed\nproportionally to the area of the patches.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nReferences:\n   - [CutMix paper]( https://arxiv.org/abs/1905.04899).\n\nArgs:\n    factor: A single float or a tuple of two floats between 0 and 1.\n        If a tuple of numbers is passed, a `factor` is sampled\n        between the two values.\n        If a single float is passed, a value between 0 and the passed\n        float is sampled. These values define the range from which the\n        mixing weight is sampled. A higher factor increases the variability\n        in patch sizes, leading to more diverse and larger mixed patches.\n        Defaults to 1.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "CvExpectedValue": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "CvState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "DataLoader": {
    "description": "Foundational PyTorch Data Loader. Mapped to GenericDataLoader shim via Plugin.",
    "std_args": [
      {"name": "dataset", "type": "Any"},
      {"name": "batch_size", "type": "Any"},
      {"name": "shuffle", "type": "Any"},
      {"name": "sampler", "type": "Any"},
      {"name": "batch_sampler", "type": "Any"},
      {"name": "num_workers", "type": "Any"},
      {"name": "collate_fn", "type": "Any"},
      {"name": "pin_memory", "type": "Any"},
      {"name": "drop_last", "type": "Any"},
      {"name": "timeout", "type": "Any"},
      {"name": "worker_init_fn", "type": "Any"},
      {"name": "multiprocessing_context", "type": "Any"},
      {"name": "generator", "type": "Any"},
      {"name": "prefetch_factor", "type": "Any"},
      {"name": "persistent_workers", "type": "Any"},
      {"name": "pin_memory_device", "type": "Any"},
    ],
    "variants": {},
  },
  "DataParallel": {
    "description": "Implements data parallelism at the module level.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "module", "type": "Any"},
      {"name": "device_ids", "type": "Any"},
      {"name": "output_device", "type": "Any"},
      {"name": "dim", "type": "Any"},
    ],
    "variants": {},
  },
  "Dense": {
    "description": 'Just your regular densely-connected NN layer.\n\n`Dense` implements the operation:\n`output = activation(dot(input, kernel) + bias)`\nwhere `activation` is the element-wise activation function\npassed as the `activation` argument, `kernel` is a weights matrix\ncreated by the layer, and `bias` is a bias vector created by the layer\n(only applicable if `use_bias` is `True`). When this layer is\nfollowed by a `BatchNormalization` layer, it is recommended to set\n`use_bias=False` as `BatchNormalization` has its own bias term.\n\nNote: If the input to the layer has a rank greater than 2, `Dense`\ncomputes the dot product between the `inputs` and the `kernel` along the\nlast axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).\nFor example, if input has dimensions `(batch_size, d0, d1)`, then we create\na `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2\nof the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are\n`batch_size * d0` such sub-tensors). The output in this case will have\nshape `(batch_size, d0, units)`.\n\nArgs:\n    units: Positive integer, dimensionality of the output space.\n    activation: Activation function to use.\n        If you don\'t specify anything, no activation is applied\n        (ie. "linear" activation: `a(x) = x`).\n    use_bias: Boolean, whether the layer uses a bias vector.\n    kernel_initializer: Initializer for the `kernel` weights matrix.\n    bias_initializer: Initializer for the bias vector.\n    kernel_regularizer: Regularizer function applied to\n        the `kernel` weights matrix.\n    bias_regularizer: Regularizer function applied to the bias vector.\n    activity_regularizer: Regularizer function applied to\n        the output of the layer (its "activation").\n    kernel_constraint: Constraint function applied to\n        the `kernel` weights matrix.\n    bias_constraint: Constraint function applied to the bias vector.\n    lora_rank: Optional integer. If set, the layer\'s forward pass\n        will implement LoRA (Low-Rank Adaptation)\n        with the provided rank. LoRA sets the layer\'s kernel\n        to non-trainable and replaces it with a delta over the\n        original kernel, obtained via multiplying two lower-rank\n        trainable matrices. This can be useful to reduce the\n        computation cost of fine-tuning large dense layers.\n        You can also enable LoRA on an existing\n        `Dense` layer by calling `layer.enable_lora(rank)`.\n    lora_alpha: Optional integer. If set, this parameter scales the\n        low-rank adaptation delta (computed as the product of two lower-rank\n        trainable matrices) during the forward pass. The delta is scaled by\n        `lora_alpha / lora_rank`, allowing you to fine-tune the strength of\n        the LoRA adjustment independently of `lora_rank`.\n\nInput shape:\n    N-D tensor with shape: `(batch_size, ..., input_dim)`.\n    The most common situation would be\n    a 2D input with shape `(batch_size, input_dim)`.\n\nOutput shape:\n    N-D tensor with shape: `(batch_size, ..., units)`.\n    For instance, for a 2D input with shape `(batch_size, input_dim)`,\n    the output would have shape `(batch_size, units)`.',
    "std_args": [
      {"name": "units", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "lora_rank", "type": "Any"},
      {"name": "lora_alpha", "type": "Any"},
      {"name": "quantization_config", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "DepthwiseConv1D": {
    "description": '1D depthwise convolution layer.\n\nDepthwise convolution is a type of convolution in which each input channel\nis convolved with a different kernel (called a depthwise kernel). You can\nunderstand depthwise convolution as the first step in a depthwise separable\nconvolution.\n\nIt is implemented via the following steps:\n\n- Split the input into individual channels.\n- Convolve each channel with an individual depthwise kernel with\n  `depth_multiplier` output channels.\n- Concatenate the convolved outputs along the channels axis.\n\nUnlike a regular 1D convolution, depthwise convolution does not mix\ninformation across different input channels.\n\nThe `depth_multiplier` argument determines how many filters are applied to\none input channel. As such, it controls the amount of output channels that\nare generated per input channel in the depthwise step.\n\nArgs:\n    kernel_size: int or tuple/list of 1 integer, specifying the size of the\n        depthwise convolution window.\n    strides: int or tuple/list of 1 integer, specifying the stride length\n        of the convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    depth_multiplier: The number of depthwise convolution output channels\n        for each input channel. The total number of depthwise convolution\n        output channels will be equal to `input_channel * depth_multiplier`.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 1 integers, specifying the dilation\n        rate to use for dilated convolution.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    depthwise_initializer: Initializer for the convolution kernel.\n        If `None`, the default initializer (`"glorot_uniform"`)\n        will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    depthwise_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    depthwise_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, steps, channels)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, channels, steps)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape:\n    `(batch_shape, new_steps, channels * depth_multiplier)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape:\n    `(batch_shape, channels * depth_multiplier, new_steps)`\n\nReturns:\n    A 3D tensor representing\n    `activation(depthwise_conv1d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 12)\n>>> y = keras.layers.DepthwiseConv1D(3, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 4, 36)',
    "std_args": [
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "depth_multiplier", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "depthwise_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "depthwise_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "depthwise_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "DepthwiseConv2D": {
    "description": '2D depthwise convolution layer.\n\nDepthwise convolution is a type of convolution in which each input channel\nis convolved with a different kernel (called a depthwise kernel). You can\nunderstand depthwise convolution as the first step in a depthwise separable\nconvolution.\n\nIt is implemented via the following steps:\n\n- Split the input into individual channels.\n- Convolve each channel with an individual depthwise kernel with\n  `depth_multiplier` output channels.\n- Concatenate the convolved outputs along the channels axis.\n\nUnlike a regular 2D convolution, depthwise convolution does not mix\ninformation across different input channels.\n\nThe `depth_multiplier` argument determines how many filters are applied to\none input channel. As such, it controls the amount of output channels that\nare generated per input channel in the depthwise step.\n\nArgs:\n    kernel_size: int or tuple/list of 2 integer, specifying the size of the\n        depthwise convolution window.\n    strides: int or tuple/list of 2 integer, specifying the stride length\n        of the depthwise convolution. `strides > 1` is incompatible with\n        `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    depth_multiplier: The number of depthwise convolution output channels\n        for each input channel. The total number of depthwise convolution\n        output channels will be equal to `input_channel * depth_multiplier`.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file\n        at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 2 integers, specifying the dilation\n        rate to use for dilated convolution.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    depthwise_initializer: Initializer for the convolution kernel.\n        If `None`, the default initializer (`"glorot_uniform"`)\n        will be used.\n    bias_initializer: Initializer for the bias vector. If `None`, the\n        default initializer (`"zeros"`) will be used.\n    depthwise_regularizer: Optional regularizer for the convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    depthwise_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape). Constraints\n        are not safe to use when doing asynchronous distributed training.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, height, width, channels)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape:\n    `(batch_size, new_height, new_width, channels * depth_multiplier)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape:\n    `(batch_size, channels * depth_multiplier, new_height, new_width)`\n\nReturns:\n    A 4D tensor representing\n    `activation(depthwise_conv2d(inputs, kernel) + bias)`.\n\nRaises:\n    ValueError: when both `strides > 1` and `dilation_rate > 1`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 10, 12)\n>>> y = keras.layers.DepthwiseConv2D(kernel_size=3, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 8, 8, 12)',
    "std_args": [
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "depth_multiplier", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "depthwise_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "depthwise_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "depthwise_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Det": {
    "description": "Det calculates determinant of a square matrix or batches of square matrices. Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions, and the inner-most 2 dimensions form square matrices. The output is a tensor of shape `[*]`, containing the determinants of all in...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Device": {
    "description": "Abstract Device placement context.",
    "std_args": [
      {"name": "type", "type": "Any"},
      {"name": "index", "type": "Any"},
    ],
    "variants": {},
  },
  "Discretization": {
    "description": 'A preprocessing layer which buckets continuous features by ranges.\n\nThis layer will place each element of its input data into one of several\ncontiguous ranges and output an integer index indicating which range each\nelement was placed in.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    Any array of dimension 2 or higher.\n\nOutput shape:\n    Same as input shape.\n\nArguments:\n    bin_boundaries: A list of bin boundaries.\n        The leftmost and rightmost bins\n        will always extend to `-inf` and `inf`,\n        so `bin_boundaries=[0., 1., 2.]`\n        generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`,\n        and `[2., +inf)`.\n        If this option is set, `adapt()` should not be called.\n    num_bins: The integer number of bins to compute.\n        If this option is set, `bin_boundaries` should not be set and\n        `adapt()` should be called to learn the bin boundaries.\n    epsilon: Error tolerance, typically a small fraction\n        close to zero (e.g. 0.01). Higher values of epsilon increase\n        the quantile approximation, and hence result in more\n        unequal buckets, but could improve performance\n        and resource consumption.\n    output_mode: Specification for the output of the layer.\n        Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or\n        `"count"` configuring the layer as follows:\n        - `"int"`: Return the discretized bin indices directly.\n        - `"one_hot"`: Encodes each individual element in the\n            input into an array the same size as `num_bins`,\n            containing a 1 at the input\'s bin\n            index. If the last dimension is size 1, will encode on that\n            dimension.  If the last dimension is not size 1,\n            will append a new dimension for the encoded output.\n        - `"multi_hot"`: Encodes each sample in the input into a\n            single array the same size as `num_bins`,\n            containing a 1 for each bin index\n            index present in the sample.\n            Treats the last dimension as the sample\n            dimension, if input shape is `(..., sample_length)`,\n            output shape will be `(..., num_tokens)`.\n        - `"count"`: As `"multi_hot"`, but the int array contains\n            a count of the number of times the bin index appeared\n            in the sample.\n        Defaults to `"int"`.\n    sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,\n        and `"count"` output modes. Only supported with TensorFlow\n        backend. If `True`, returns a `SparseTensor` instead of\n        a dense `Tensor`. Defaults to `False`.\n\nExamples:\n\nDiscretize float values based on provided buckets.\n>>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])\n>>> layer = Discretization(bin_boundaries=[0., 1., 2.])\n>>> layer(input)\narray([[0, 2, 3, 1],\n       [1, 3, 2, 1]])\n\nDiscretize float values based on a number of buckets to compute.\n>>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])\n>>> layer = Discretization(num_bins=4, epsilon=0.01)\n>>> layer.adapt(input)\n>>> layer(input)\narray([[0, 2, 3, 2],\n       [1, 3, 3, 1]])',
    "std_args": [
      {"name": "bin_boundaries", "type": "Any"},
      {"name": "num_bins", "type": "Any"},
      {"name": "epsilon", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "name", "type": "Any"},
    ],
    "variants": {},
  },
  "Div": {
    "description": "Performs element-wise binary division (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md). (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Dropout": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "lr", "type": "Any"},
    ],
    "variants": {},
  },
  "Dropout2d": {
    "description": "Randomly zero out entire channels.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "p", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "Dropout3d": {
    "description": "Randomly zero out entire channels.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "p", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "Einsum": {
    "description": "An einsum of the form `term1, term2 -> output-term` produces an output tensor using the following equation ``` output[output-term] = reduce-sum( input1[term1] * input2[term2] ) ``` where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2) that do not ...",
    "std_args": [
      {"name": "Inputs", "type": "Tensor"},
      {"name": "equation", "type": "str"},
    ],
    "variants": {},
  },
  "EinsumDense": {
    "description": 'A layer that uses `einsum` as the backing computation.\n\nThis layer can perform einsum calculations of arbitrary dimensionality.\n\nArgs:\n    equation: An equation describing the einsum to perform.\n        This equation must be a valid einsum string of the form\n        `ab,bc->ac`, `...ab,bc->...ac`, or\n        `ab...,bc->ac...` where \'ab\', \'bc\', and \'ac\' can be any valid einsum\n        axis expression sequence.\n    output_shape: The expected shape of the output tensor\n        (excluding the batch dimension and any dimensions\n        represented by ellipses). You can specify `None` for any dimension\n        that is unknown or can be inferred from the input shape.\n    activation: Activation function to use. If you don\'t specify anything,\n        no activation is applied\n        (that is, a "linear" activation: `a(x) = x`).\n    bias_axes: A string containing the output dimension(s)\n        to apply a bias to. Each character in the `bias_axes` string\n        should correspond to a character in the output portion\n        of the `equation` string.\n    kernel_initializer: Initializer for the `kernel` weights matrix.\n    bias_initializer: Initializer for the bias vector.\n    kernel_regularizer: Regularizer function applied to the `kernel` weights\n        matrix.\n    bias_regularizer: Regularizer function applied to the bias vector.\n    kernel_constraint: Constraint function applied to the `kernel` weights\n        matrix.\n    bias_constraint: Constraint function applied to the bias vector.\n    lora_rank: Optional integer. If set, the layer\'s forward pass\n        will implement LoRA (Low-Rank Adaptation)\n        with the provided rank. LoRA sets the layer\'s kernel\n        to non-trainable and replaces it with a delta over the\n        original kernel, obtained via multiplying two lower-rank\n        trainable matrices\n        (the factorization happens on the last dimension).\n        This can be useful to reduce the\n        computation cost of fine-tuning large dense layers.\n        You can also enable LoRA on an existing\n        `EinsumDense` layer by calling `layer.enable_lora(rank)`.\n     lora_alpha: Optional integer. If set, this parameter scales the\n        low-rank adaptation delta (computed as the product of two lower-rank\n        trainable matrices) during the forward pass. The delta is scaled by\n        `lora_alpha / lora_rank`, allowing you to fine-tune the strength of\n        the LoRA adjustment independently of `lora_rank`.\n    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.\n\nExamples:\n\n**Biased dense layer with einsums**\n\nThis example shows how to instantiate a standard Keras dense layer using\neinsum operations. This example is equivalent to\n`keras.layers.Dense(64, use_bias=True)`.\n\n>>> layer = keras.layers.EinsumDense("ab,bc->ac",\n...                                       output_shape=64,\n...                                       bias_axes="c")\n>>> input_tensor = keras.Input(shape=[32])\n>>> output_tensor = layer(input_tensor)\n>>> output_tensor.shape\n(None, 64)\n\n**Applying a dense layer to a sequence**\n\nThis example shows how to instantiate a layer that applies the same dense\noperation to every element in a sequence. Here, the `output_shape` has two\nvalues (since there are two non-batch dimensions in the output); the first\ndimension in the `output_shape` is `None`, because the sequence dimension\n`b` has an unknown shape.\n\n>>> layer = keras.layers.EinsumDense("abc,cd->abd",\n...                                       output_shape=(None, 64),\n...                                       bias_axes="d")\n>>> input_tensor = keras.Input(shape=[32, 128])\n>>> output_tensor = layer(input_tensor)\n>>> output_tensor.shape\n(None, 32, 64)\n\n**Applying a dense layer to a sequence using ellipses**\n\nThis example shows how to instantiate a layer that applies the same dense\noperation to every element in a sequence, but uses the ellipsis notation\ninstead of specifying the batch and sequence dimensions.\n\nBecause we are using ellipsis notation and have specified only one axis, the\n`output_shape` arg is a single value. When instantiated in this way, the\nlayer can handle any number of sequence dimensions - including the case\nwhere no sequence dimension exists.\n\n>>> layer = keras.layers.EinsumDense("...x,xy->...y",\n...                                       output_shape=64,\n...                                       bias_axes="y")\n>>> input_tensor = keras.Input(shape=[32, 128])\n>>> output_tensor = layer(input_tensor)\n>>> output_tensor.shape\n(None, 32, 64)',
    "std_args": [
      {"name": "equation", "type": "Any"},
      {"name": "output_shape", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "bias_axes", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "lora_rank", "type": "Any"},
      {"name": "lora_alpha", "type": "Any"},
      {"name": "gptq_unpacked_column_size", "type": "Any"},
      {"name": "quantization_config", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Elu": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "alpha", "type": "Any"},
    ],
    "variants": {},
  },
  "EmaState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Embedding": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Equal": {
    "description": "Returns the tensor resulted from performing the `equal` logical operation elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Equalization": {
    "description": 'Preprocessing layer for histogram equalization on image channels.\n\nHistogram equalization is a technique to adjust image intensities to\nenhance contrast by effectively spreading out the most frequent\nintensity values. This layer applies equalization on a channel-wise\nbasis, which can improve the visibility of details in images.\n\nThis layer works with both grayscale and color images, performing\nequalization independently on each color channel. At inference time,\nthe equalization is consistently applied.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    value_range: Optional list/tuple of 2 floats specifying the lower\n        and upper limits of the input data values. Defaults to `[0, 255]`.\n        If the input image has been scaled, use the appropriate range\n        (e.g., `[0.0, 1.0]`). The equalization will be scaled to this\n        range, and output values will be clipped accordingly.\n    bins: Integer specifying the number of histogram bins to use for\n        equalization. Defaults to 256, which is suitable for 8-bit images.\n        Larger values can provide more granular intensity redistribution.\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format,\n    or `(..., channels, height, width)`, in `"channels_first"` format.\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., target_height, target_width, channels)`,\n    or `(..., channels, target_height, target_width)`,\n    in `"channels_first"` format.\n\nExample:\n\n```python\n# Create an equalization layer for standard 8-bit images\nequalizer = keras.layers.Equalization()\n\n# An image with uneven intensity distribution\nimage = [...] # your input image\n\n# Apply histogram equalization\nequalized_image = equalizer(image)\n\n# For images with custom value range\ncustom_equalizer = keras.layers.Equalization(\n    value_range=[0.0, 1.0],  # for normalized images\n    bins=128  # fewer bins for more subtle equalization\n)\ncustom_equalized = custom_equalizer(normalized_image)\n```',
    "std_args": [
      {"name": "value_range", "type": "Any"},
      {"name": "bins", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Erf": {
    "description": "Computes the error function of the given input tensor element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Exp": {
    "description": "Calculates the exponential of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Flatten": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "FlaxLayer": {
    "description": 'Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.\n\nThis layer enables the use of Flax components in the form of\n[`flax.linen.Module`](\n    https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)\ninstances within Keras when using JAX as the backend for Keras.\n\nThe module method to use for the forward pass can be specified via the\n`method` argument and is `__call__` by default. This method must take the\nfollowing arguments with these exact names:\n\n- `self` if the method is bound to the module, which is the case for the\n    default of `__call__`, and `module` otherwise to pass the module.\n- `inputs`: the inputs to the model, a JAX array or a `PyTree` of arrays.\n- `training` *(optional)*: an argument specifying if we\'re in training mode\n    or inference mode, `True` is passed in training mode.\n\n`FlaxLayer` handles the non-trainable state of your model and required RNGs\nautomatically. Note that the `mutable` parameter of\n[`flax.linen.Module.apply()`](\n    https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply)\nis set to `DenyList(["params"])`, therefore making the assumption that all\nthe variables outside of the "params" collection are non-trainable weights.\n\nThis example shows how to create a `FlaxLayer` from a Flax `Module` with\nthe default `__call__` method and no training argument:\n\n```python\nclass MyFlaxModule(flax.linen.Module):\n    @flax.linen.compact\n    def __call__(self, inputs):\n        x = inputs\n        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)\n        x = flax.linen.relu(x)\n        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))\n        x = x.reshape((x.shape[0], -1))  # flatten\n        x = flax.linen.Dense(features=200)(x)\n        x = flax.linen.relu(x)\n        x = flax.linen.Dense(features=10)(x)\n        x = flax.linen.softmax(x)\n        return x\n\nflax_module = MyFlaxModule()\nkeras_layer = FlaxLayer(flax_module)\n```\n\nThis example shows how to wrap the module method to conform to the required\nsignature. This allows having multiple input arguments and a training\nargument that has a different name and values. This additionally shows how\nto use a function that is not bound to the module.\n\n```python\nclass MyFlaxModule(flax.linen.Module):\n    @flax.linen.compact\n    def forward(self, input1, input2, deterministic):\n        ...\n        return outputs\n\ndef my_flax_module_wrapper(module, inputs, training):\n    input1, input2 = inputs\n    return module.forward(input1, input2, not training)\n\nflax_module = MyFlaxModule()\nkeras_layer = FlaxLayer(\n    module=flax_module,\n    method=my_flax_module_wrapper,\n)\n```\n\nArgs:\n    module: An instance of `flax.linen.Module` or subclass.\n    method: The method to call the model. This is generally a method in the\n        `Module`. If not provided, the `__call__` method is used. `method`\n        can also be a function not defined in the `Module`, in which case it\n        must take the `Module` as the first argument. It is used for both\n        `Module.init` and `Module.apply`. Details are documented in the\n        `method` argument of [`flax.linen.Module.apply()`](\n          https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.apply).\n    variables: A `dict` containing all the variables of the module in the\n        same format as what is returned by [`flax.linen.Module.init()`](\n          https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.init).\n        It should contain a "params" key and, if applicable, other keys for\n        collections of variables for non-trainable state. This allows\n        passing trained parameters and learned non-trainable state or\n        controlling the initialization. If `None` is passed, the module\'s\n        `init` function is called at build time to initialize the variables\n        of the model.',
    "std_args": [
      {"name": "module", "type": "Any"},
      {"name": "method", "type": "Any"},
      {"name": "variables", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Float16": {
    "description": "16-bit floating point type (Half).",
    "std_args": [],
    "variants": {},
  },
  "Float32": {
    "description": "32-bit floating point type.",
    "std_args": [],
    "variants": {},
  },
  "Float64": {
    "description": "64-bit floating point type (Double).",
    "std_args": [],
    "variants": {},
  },
  "Floor": {
    "description": "Floor takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the floor is, y = floor(x), is applied to the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "FusedMultiplyAdd": {
    "description": "Auto-generated from sass_vunknown_map.json",
    "std_args": [],
    "variants": {},
  },
  "GELU": {
    "description": "Gaussian Error Linear Unit.",
    "std_args": [
      {"name": "input", "type": "Any"},
    ],
    "variants": {},
  },
  "GRU": {
    "description": "Computes an one-layer GRU. This operator is usually supported via some custom implementation such as CuDNN. Notations: * `X` - input tensor * `z` - update gate * `r` - reset gate * `h` - hidden gate * `t` - time step (t-1 means previous time step) * `W[zrh]` - W parameter weight matrix for update, r...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "W", "type": "Tensor"},
      {"name": "R", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
      {"name": "sequence_lens", "type": "Any"},
      {"name": "initial_h", "type": "Tensor"},
      {"name": "activation_alpha", "type": "List[float]"},
      {"name": "activation_beta", "type": "List[float]"},
      {"name": "activations", "type": "List[str]"},
      {"name": "clip", "type": "float"},
      {"name": "direction", "type": "str"},
      {"name": "hidden_size", "type": "int"},
      {"name": "layout", "type": "int"},
      {"name": "linear_before_reset", "type": "int"},
    ],
    "variants": {},
  },
  "Gather": {
    "description": "Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates them in an output tensor of rank q + (r - 1). It is an indexing operation that indexes into the input `data`...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "indices", "type": "Any"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "GaussianDropout": {
    "description": "Apply multiplicative 1-centered Gaussian noise.\n\nAs it is a regularization layer, it is only active at training time.\n\nArgs:\n    rate: Float, drop probability (as with `Dropout`).\n        The multiplicative noise will have\n        standard deviation `sqrt(rate / (1 - rate))`.\n    seed: Integer, optional random seed to enable deterministic behavior.\n\nCall arguments:\n    inputs: Input tensor (of any rank).\n    training: Python boolean indicating whether the layer should behave in\n        training mode (adding dropout) or in inference mode (doing nothing).",
    "std_args": [
      {"name": "rate", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GaussianNoise": {
    "description": "Apply additive zero-centered Gaussian noise.\n\nThis is useful to mitigate overfitting\n(you could see it as a form of random data augmentation).\nGaussian Noise (GS) is a natural choice as corruption process\nfor real valued inputs.\n\nAs it is a regularization layer, it is only active at training time.\n\nArgs:\n    stddev: Float, standard deviation of the noise distribution.\n    seed: Integer, optional random seed to enable deterministic behavior.\n\nCall arguments:\n    inputs: Input tensor (of any rank).\n    training: Python boolean indicating whether the layer should behave in\n        training mode (adding noise) or in inference mode (doing nothing).",
    "std_args": [
      {"name": "stddev", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Gaussiannll": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "eps", "type": "float"},
      {"name": "full", "type": "bool"},
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "Gelu": {
    "description": "Auto-discovered via Consensus (Score: 4.0)",
    "std_args": [
      {"name": "approximate", "type": "Any"},
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalAveragePooling1D": {
    "description": 'Global average pooling operation for temporal data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        temporal dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nCall arguments:\n    inputs: A 3D tensor.\n    mask: Binary tensor of shape `(batch_size, steps)` indicating whether\n        a given step should be masked (excluded from the average).\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    3D tensor with shape:\n    `(batch_size, steps, features)`\n- If `data_format=\'channels_first\'`:\n    3D tensor with shape:\n    `(batch_size, features, steps)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, features)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        3D tensor with shape `(batch_size, 1, features)`\n    - If `data_format="channels_first"`:\n        3D tensor with shape `(batch_size, features, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 3, 4)\n>>> y = keras.layers.GlobalAveragePooling1D()(x)\n>>> y.shape\n(2, 4)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalAveragePooling2D": {
    "description": 'Global average pooling operation for 2D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, height, weight)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    4D tensor with shape:\n    `(batch_size, height, width, channels)`\n- If `data_format=\'channels_first\'`:\n    4D tensor with shape:\n    `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        4D tensor with shape `(batch_size, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        4D tensor with shape `(batch_size, channels, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 3)\n>>> y = keras.layers.GlobalAveragePooling2D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalAveragePooling3D": {
    "description": 'Global average pooling operation for 3D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format=\'channels_first\'`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        5D tensor with shape `(batch_size, 1, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        5D tensor with shape `(batch_size, channels, 1, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 4, 3)\n>>> y = keras.layers.GlobalAveragePooling3D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalAvgPool1D": {
    "description": 'Global average pooling operation for temporal data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        temporal dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nCall arguments:\n    inputs: A 3D tensor.\n    mask: Binary tensor of shape `(batch_size, steps)` indicating whether\n        a given step should be masked (excluded from the average).\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    3D tensor with shape:\n    `(batch_size, steps, features)`\n- If `data_format=\'channels_first\'`:\n    3D tensor with shape:\n    `(batch_size, features, steps)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, features)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        3D tensor with shape `(batch_size, 1, features)`\n    - If `data_format="channels_first"`:\n        3D tensor with shape `(batch_size, features, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 3, 4)\n>>> y = keras.layers.GlobalAveragePooling1D()(x)\n>>> y.shape\n(2, 4)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalAvgPool2D": {
    "description": 'Global average pooling operation for 2D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, height, weight)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    4D tensor with shape:\n    `(batch_size, height, width, channels)`\n- If `data_format=\'channels_first\'`:\n    4D tensor with shape:\n    `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        4D tensor with shape `(batch_size, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        4D tensor with shape `(batch_size, channels, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 3)\n>>> y = keras.layers.GlobalAveragePooling2D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalAvgPool3D": {
    "description": 'Global average pooling operation for 3D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format=\'channels_first\'`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        5D tensor with shape `(batch_size, 1, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        5D tensor with shape `(batch_size, channels, 1, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 4, 3)\n>>> y = keras.layers.GlobalAveragePooling3D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalMaxPool1D": {
    "description": 'Global max pooling operation for temporal data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        temporal dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    3D tensor with shape:\n    `(batch_size, steps, features)`\n- If `data_format=\'channels_first\'`:\n    3D tensor with shape:\n    `(batch_size, features, steps)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, features)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        3D tensor with shape `(batch_size, 1, features)`\n    - If `data_format="channels_first"`:\n        3D tensor with shape `(batch_size, features, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 3, 4)\n>>> y = keras.layers.GlobalMaxPooling1D()(x)\n>>> y.shape\n(2, 4)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalMaxPool2D": {
    "description": 'Global max pooling operation for 2D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, height, weight)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    4D tensor with shape:\n    `(batch_size, height, width, channels)`\n- If `data_format=\'channels_first\'`:\n    4D tensor with shape:\n    `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        4D tensor with shape `(batch_size, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        4D tensor with shape `(batch_size, channels, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 3)\n>>> y = keras.layers.GlobalMaxPooling2D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalMaxPool3D": {
    "description": 'Global max pooling operation for 3D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format=\'channels_first\'`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        5D tensor with shape `(batch_size, 1, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        5D tensor with shape `(batch_size, channels, 1, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 4, 3)\n>>> y = keras.layers.GlobalMaxPooling3D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalMaxPooling1D": {
    "description": 'Global max pooling operation for temporal data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        temporal dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    3D tensor with shape:\n    `(batch_size, steps, features)`\n- If `data_format=\'channels_first\'`:\n    3D tensor with shape:\n    `(batch_size, features, steps)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, features)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        3D tensor with shape `(batch_size, 1, features)`\n    - If `data_format="channels_first"`:\n        3D tensor with shape `(batch_size, features, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 3, 4)\n>>> y = keras.layers.GlobalMaxPooling1D()(x)\n>>> y.shape\n(2, 4)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalMaxPooling2D": {
    "description": 'Global max pooling operation for 2D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, height, weight)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    4D tensor with shape:\n    `(batch_size, height, width, channels)`\n- If `data_format=\'channels_first\'`:\n    4D tensor with shape:\n    `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        4D tensor with shape `(batch_size, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        4D tensor with shape `(batch_size, channels, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 3)\n>>> y = keras.layers.GlobalMaxPooling2D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "GlobalMaxPooling3D": {
    "description": 'Global max pooling operation for 3D data.\n\nArgs:\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n    keepdims: A boolean, whether to keep the temporal dimension or not.\n        If `keepdims` is `False` (default), the rank of the tensor is\n        reduced for spatial dimensions. If `keepdims` is `True`, the\n        spatial dimension are retained with length 1.\n        The behavior is the same as for `tf.reduce_mean` or `np.mean`.\n\nInput shape:\n\n- If `data_format=\'channels_last\'`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format=\'channels_first\'`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `keepdims=False`:\n    2D tensor with shape `(batch_size, channels)`.\n- If `keepdims=True`:\n    - If `data_format="channels_last"`:\n        5D tensor with shape `(batch_size, 1, 1, 1, channels)`\n    - If `data_format="channels_first"`:\n        5D tensor with shape `(batch_size, channels, 1, 1, 1)`\n\nExample:\n\n>>> x = np.random.rand(2, 4, 5, 4, 3)\n>>> y = keras.layers.GlobalMaxPooling3D()(x)\n>>> y.shape\n(2, 3)',
    "std_args": [
      {"name": "data_format", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Glu": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "dim", "type": "Any"},
    ],
    "variants": {},
  },
  "Grayscale": {
    "description": "Convert image to grayscale.",
    "std_args": [
      {"name": "num_output_channels", "type": "Any"},
    ],
    "variants": {},
  },
  "Greater": {
    "description": "Returns the tensor resulted from performing the `greater` logical operation elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Group": {
    "description": "An :class:`mlx.core.distributed.Group` represents a group of independent mlx",
    "std_args": [],
    "variants": {},
  },
  "GroupNormalization": {
    "description": "A GroupNormalization function. Carries out group normalization as described in the paper https://arxiv.org/abs/1803.08494 This operator transforms input according to ``` y = scale * (x - mean) / sqrt(variance + epsilon) + bias, ``` where the mean and variance are computed per instance per group of c...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "scale", "type": "Tensor"},
      {"name": "bias", "type": "Tensor"},
      {"name": "epsilon", "type": "float"},
      {"name": "num_groups", "type": "int"},
      {"name": "stash_type", "type": "int"},
    ],
    "variants": {},
  },
  "GroupQueryAttention": {
    "description": "Grouped Query Attention layer.\n\nThis is an implementation of grouped-query attention introduced by\n[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here\n`num_key_value_heads` denotes number of groups, setting\n`num_key_value_heads` to 1 is equivalent to multi-query attention, and\nwhen `num_key_value_heads` is equal to `num_query_heads` it is equivalent\nto multi-head attention.\n\nThis layer first projects `query`, `key`, and `value` tensors. Then, `key`\nand `value` are repeated to match the number of heads of `query`.\n\nThen, the `query` is scaled and dot-producted with `key` tensors. These are\nsoftmaxed to obtain attention probabilities. The value tensors are then\ninterpolated by these probabilities and concatenated back to a single\ntensor.\n\nArgs:\n    head_dim: Size of each attention head.\n    num_query_heads: Number of query attention heads.\n    num_key_value_heads: Number of key and value attention heads.\n    dropout: Dropout probability.\n    use_bias: Boolean, whether the dense layers use bias vectors/matrices.\n    flash_attention: If `None`, the layer attempts to use flash\n        attention for faster and more memory-efficient attention\n        computations when possible. This behavior can be configured using\n        `keras.config.enable_flash_attention()` or\n        `keras.config.disable_flash_attention()`.\n    kernel_initializer: Initializer for dense layer kernels.\n    bias_initializer: Initializer for dense layer biases.\n    kernel_regularizer: Regularizer for dense layer kernels.\n    bias_regularizer: Regularizer for dense layer biases.\n    activity_regularizer: Regularizer for dense layer activity.\n    kernel_constraint: Constraint for dense layer kernels.\n    bias_constraint: Constraint for dense layer kernels.\n    seed: Optional integer to seed the dropout layer.\n\nCall arguments:\n    query: Query tensor of shape `(batch_dim, target_seq_len, feature_dim)`,\n        where `batch_dim` is batch size, `target_seq_len` is the length of\n        target sequence, and `feature_dim` is dimension of feature.\n    value: Value tensor of shape `(batch_dim, source_seq_len, feature_dim)`,\n        where `batch_dim` is batch size, `source_seq_len` is the length of\n        source sequence, and `feature_dim` is dimension of feature.\n    key: Optional key tensor of shape\n        `(batch_dim, source_seq_len, feature_dim)`. If not given, will use\n        `value` for both `key` and `value`, which is most common case.\n    attention_mask: A boolean mask of shape\n        `(batch_dim, target_seq_len, source_seq_len)`, that prevents\n        attention to certain positions. The boolean mask specifies which\n        query elements can attend to which key elements, where 1 indicates\n        attention and 0 indicates no attention. Broadcasting can happen for\n        the missing batch dimensions and the head dimension.\n    return_attention_scores: A boolean to indicate whether the output\n        should be `(attention_output, attention_scores)` if `True`, or\n        `attention_output` if `False`. Defaults to `False`.\n    training: Python boolean indicating whether the layer should behave in\n        training mode (adding dropout) or in inference mode (no dropout).\n        Will go with either using the training mode of the parent\n        layer/model or `False` (inference) if there is no parent layer.\n    use_causal_mask: A boolean to indicate whether to apply a causal mask to\n        prevent tokens from attending to future tokens (e.g., used in a\n        decoder Transformer).\n\nReturns:\n    attention_output: Result of the computation, of shape\n        `(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len`\n        is for target sequence length and `feature_dim` is the query input\n        last dim.\n    attention_scores: (Optional) attention coefficients of shape\n        `(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.",
    "std_args": [
      {"name": "head_dim", "type": "Any"},
      {"name": "num_query_heads", "type": "Any"},
      {"name": "num_key_value_heads", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "flash_attention", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Groupnorm": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
      {"name": "eps", "type": "Any"},
      {"name": "num_groups", "type": "Any"},
    ],
    "variants": {},
  },
  "Grucell": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "HardSigmoid": {
    "description": "HardSigmoid takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)), is applied to the tensor elementwise.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "alpha", "type": "float"},
      {"name": "beta", "type": "float"},
    ],
    "variants": {},
  },
  "HardSwish": {
    "description": "HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x), where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Hardshrink": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Hardswish": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Hardtanh": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "HashedCrossing": {
    "description": "A preprocessing layer which crosses features using the \"hashing trick\".\n\nThis layer performs crosses of categorical features using the \"hashing\ntrick\". Conceptually, the transformation can be thought of as:\n`hash(concatenate(features)) % num_bins`.\n\nThis layer currently only performs crosses of scalar inputs and batches of\nscalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and\n`()`.\n\n**Note:** This layer wraps `tf.keras.layers.HashedCrossing`. It cannot\nbe used as part of the compiled computation graph of a model with\nany backend other than TensorFlow.\nIt can however be used with any backend when running eagerly.\nIt can also always be used as part of an input preprocessing pipeline\nwith any backend (outside the model itself), which is how we recommend\nto use this layer.\n\n**Note:** This layer is safe to use inside a `tf.data` pipeline\n(independently of which backend you're using).\n\nArgs:\n    num_bins: Number of hash bins.\n    output_mode: Specification for the output of the layer. Values can be\n        `\"int\"`, or `\"one_hot\"` configuring the layer as follows:\n        - `\"int\"`: Return the integer bin indices directly.\n        - `\"one_hot\"`: Encodes each individual element in the input into an\n            array the same size as `num_bins`, containing a 1 at the input's\n            bin index. Defaults to `\"int\"`.\n    sparse: Boolean. Only applicable to `\"one_hot\"` mode and only valid\n        when using the TensorFlow backend. If `True`, returns\n        a `SparseTensor` instead of a dense `Tensor`. Defaults to `False`.\n    **kwargs: Keyword arguments to construct a layer.\n\nExamples:\n\n**Crossing two scalar features.**\n\n>>> layer = keras.layers.HashedCrossing(\n...     num_bins=5)\n>>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])\n>>> feat2 = np.array([101, 101, 101, 102, 102])\n>>> layer((feat1, feat2))\narray([1, 4, 1, 1, 3])\n\n**Crossing and one-hotting two scalar features.**\n\n>>> layer = keras.layers.HashedCrossing(\n...     num_bins=5, output_mode='one_hot')\n>>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])\n>>> feat2 = np.array([101, 101, 101, 102, 102])\n>>> layer((feat1, feat2))\narray([[0., 1., 0., 0., 0.],\n        [0., 0., 0., 0., 1.],\n        [0., 1., 0., 0., 0.],\n        [0., 1., 0., 0., 0.],\n        [0., 0., 0., 1., 0.]], dtype=float32)",
    "std_args": [
      {"name": "num_bins", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Hashing": {
    "description": "A preprocessing layer which hashes and bins categorical features.\n\nThis layer transforms categorical inputs to hashed output. It element-wise\nconverts a ints or strings to ints in a fixed range. The stable hash\nfunction uses `tensorflow::ops::Fingerprint` to produce the same output\nconsistently across all platforms.\n\nThis layer uses [FarmHash64](https://github.com/google/farmhash) by default,\nwhich provides a consistent hashed output across different platforms and is\nstable across invocations, regardless of device and context, by mixing the\ninput bits thoroughly.\n\nIf you want to obfuscate the hashed output, you can also pass a random\n`salt` argument in the constructor. In that case, the layer will use the\n[SipHash64](https://github.com/google/highwayhash) hash function, with\nthe `salt` value serving as additional input to the hash function.\n\n**Note:** This layer internally uses TensorFlow. It cannot\nbe used as part of the compiled computation graph of a model with\nany backend other than TensorFlow.\nIt can however be used with any backend when running eagerly.\nIt can also always be used as part of an input preprocessing pipeline\nwith any backend (outside the model itself), which is how we recommend\nto use this layer.\n\n**Note:** This layer is safe to use inside a `tf.data` pipeline\n(independently of which backend you're using).\n\n**Example (FarmHash64)**\n\n>>> layer = keras.layers.Hashing(num_bins=3)\n>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]\n>>> layer(inp)\narray([[1],\n        [0],\n        [1],\n        [1],\n        [2]])>\n\n**Example (FarmHash64) with a mask value**\n\n>>> layer = keras.layers.Hashing(num_bins=3, mask_value='')\n>>> inp = [['A'], ['B'], [''], ['C'], ['D']]\n>>> layer(inp)\narray([[1],\n        [1],\n        [0],\n        [2],\n        [2]])\n\n**Example (SipHash64)**\n\n>>> layer = keras.layers.Hashing(num_bins=3, salt=[133, 137])\n>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]\n>>> layer(inp)\narray([[1],\n        [2],\n        [1],\n        [0],\n        [2]])\n\n**Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**\n\n>>> layer = keras.layers.Hashing(num_bins=3, salt=133)\n>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]\n>>> layer(inp)\narray([[0],\n        [0],\n        [2],\n        [1],\n        [0]])\n\nArgs:\n    num_bins: Number of hash bins. Note that this includes the `mask_value`\n        bin, so the effective number of bins is `(num_bins - 1)`\n        if `mask_value` is set.\n    mask_value: A value that represents masked inputs, which are mapped to\n        index 0. `None` means no mask term will be added and the\n        hashing will start at index 0. Defaults to `None`.\n    salt: A single unsigned integer or None.\n        If passed, the hash function used will be SipHash64,\n        with these values used as an additional input\n        (known as a \"salt\" in cryptography).\n        These should be non-zero. If `None`, uses the FarmHash64 hash\n        function. It also supports tuple/list of 2 unsigned\n        integer numbers, see reference paper for details.\n        Defaults to `None`.\n    output_mode: Specification for the output of the layer. Values can be\n        `\"int\"`, `\"one_hot\"`, `\"multi_hot\"`, or\n        `\"count\"` configuring the layer as follows:\n        - `\"int\"`: Return the integer bin indices directly.\n        - `\"one_hot\"`: Encodes each individual element in the input into an\n            array the same size as `num_bins`, containing a 1\n            at the input's bin index. If the last dimension is size 1,\n            will encode on that dimension.\n            If the last dimension is not size 1, will append a new\n            dimension for the encoded output.\n        - `\"multi_hot\"`: Encodes each sample in the input into a\n            single array the same size as `num_bins`,\n            containing a 1 for each bin index\n            index present in the sample. Treats the last dimension\n            as the sample dimension, if input shape is\n            `(..., sample_length)`, output shape will be\n            `(..., num_tokens)`.\n        - `\"count\"`: As `\"multi_hot\"`, but the int array contains a count of\n            the number of times the bin index appeared in the sample.\n        Defaults to `\"int\"`.\n    sparse: Boolean. Only applicable to `\"one_hot\"`, `\"multi_hot\"`,\n        and `\"count\"` output modes. Only supported with TensorFlow\n        backend. If `True`, returns a `SparseTensor` instead of\n        a dense `Tensor`. Defaults to `False`.\n    **kwargs: Keyword arguments to construct a layer.\n\nInput shape:\n    A single string, a list of strings, or an `int32` or `int64` tensor\n    of shape `(batch_size, ...,)`.\n\nOutput shape:\n    An `int32` tensor of shape `(batch_size, ...)`.\n\nReference:\n\n- [SipHash with salt](https://www.131002.net/siphash/siphash.pdf)",
    "std_args": [
      {"name": "num_bins", "type": "Any"},
      {"name": "mask_value", "type": "Any"},
      {"name": "salt", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "INTEGER": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Identity": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "InjectHyperparamsState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "InjectStatefulHyperparamsState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "InputLayer": {
    "description": 'This is the class from which all layers inherit.\n\nA layer is a callable object that takes as input one or more tensors and\nthat outputs one or more tensors. It involves *computation*, defined\nin the `call()` method, and a *state* (weight variables). State can be\ncreated:\n\n* in `__init__()`, for instance via `self.add_weight()`;\n* in the optional `build()` method, which is invoked by the first\n  `__call__()` to the layer, and supplies the shape(s) of the input(s),\n  which may not have been known at initialization time.\n\nLayers are recursively composable: If you assign a Layer instance as an\nattribute of another Layer, the outer layer will start tracking the weights\ncreated by the inner layer. Nested layers should be instantiated in the\n`__init__()` method or `build()` method.\n\nUsers will just instantiate a layer and then treat it as a callable.\n\nArgs:\n    trainable: Boolean, whether the layer\'s variables should be trainable.\n    name: String name of the layer.\n    dtype: The dtype of the layer\'s computations and weights. Can also be a\n        `keras.DTypePolicy`, which allows the computation and weight dtype\n        to differ. Defaults to `None`. `None` means to use\n        `keras.config.dtype_policy()`, which is a `float32` policy unless\n        set to different value (via `keras.config.set_dtype_policy()`).\n\nAttributes:\n    name: The name of the layer (string).\n    dtype: Dtype of the layer\'s weights. Alias of `layer.variable_dtype`.\n    variable_dtype: Dtype of the layer\'s weights.\n    compute_dtype: The dtype of the layer\'s computations.\n        Layers automatically cast inputs to this dtype, which causes\n        the computations and output to also be in this dtype.\n        When mixed precision is used with a\n        `keras.DTypePolicy`, this will be different\n        than `variable_dtype`.\n    trainable_weights: List of variables to be included in backprop.\n    non_trainable_weights: List of variables that should not be\n        included in backprop.\n    weights: The concatenation of the lists trainable_weights and\n        non_trainable_weights (in this order).\n    trainable: Whether the layer should be trained (boolean), i.e.\n        whether its potentially-trainable weights should be returned\n        as part of `layer.trainable_weights`.\n    input_spec: Optional (list of) `InputSpec` object(s) specifying the\n        constraints on inputs that can be accepted by the layer.\n\nWe recommend that descendants of `Layer` implement the following methods:\n\n* `__init__()`: Defines custom layer attributes, and creates layer weights\n    that do not depend on input shapes, using `add_weight()`,\n    or other state.\n* `build(self, input_shape)`: This method can be used to create weights that\n    depend on the shape(s) of the input(s), using `add_weight()`, or other\n    state. `__call__()` will automatically build the layer\n    (if it has not been built yet) by calling `build()`.\n* `call(self, *args, **kwargs)`: Called in `__call__` after making\n    sure `build()` has been called. `call()` performs the logic of applying\n    the layer to the input arguments.\n    Two reserved keyword arguments you can optionally use in `call()` are:\n        1. `training` (boolean, whether the call is in inference mode or\n            training mode).\n        2. `mask` (boolean tensor encoding masked timesteps in the input,\n            used e.g. in RNN layers).\n    A typical signature for this method is `call(self, inputs)`, and user\n    could optionally add `training` and `mask` if the layer need them.\n* `get_config(self)`: Returns a dictionary containing the configuration\n    used to initialize this layer. If the keys differ from the arguments\n    in `__init__()`, then override `from_config(self)` as well.\n    This method is used when saving\n    the layer or a model that contains this layer.\n\nExamples:\n\nHere\'s a basic example: a layer with two variables, `w` and `b`,\nthat returns `y = w . x + b`.\nIt shows how to implement `build()` and `call()`.\nVariables set as attributes of a layer are tracked as weights\nof the layers (in `layer.weights`).\n\n```python\nclass SimpleDense(Layer):\n    def __init__(self, units=32):\n        super().__init__()\n        self.units = units\n\n    # Create the state of the layer (weights)\n    def build(self, input_shape):\n        self.kernel = self.add_weight(\n            shape=(input_shape[-1], self.units),\n            initializer="glorot_uniform",\n            trainable=True,\n            name="kernel",\n        )\n        self.bias = self.add_weight(\n            shape=(self.units,),\n            initializer="zeros",\n            trainable=True,\n            name="bias",\n        )\n\n    # Defines the computation\n    def call(self, inputs):\n        return ops.matmul(inputs, self.kernel) + self.bias\n\n# Instantiates the layer.\nlinear_layer = SimpleDense(4)\n\n# This will also call `build(input_shape)` and create the weights.\ny = linear_layer(ops.ones((2, 2)))\nassert len(linear_layer.weights) == 2\n\n# These weights are trainable, so they\'re listed in `trainable_weights`:\nassert len(linear_layer.trainable_weights) == 2\n```\n\nBesides trainable weights, updated via backpropagation during training,\nlayers can also have non-trainable weights. These weights are meant to\nbe updated manually during `call()`. Here\'s a example layer that computes\nthe running sum of its inputs:\n\n```python\nclass ComputeSum(Layer):\n\n  def __init__(self, input_dim):\n      super(ComputeSum, self).__init__()\n      # Create a non-trainable weight.\n      self.total = self.add_weight(\n        shape=(),\n        initializer="zeros",\n        trainable=False,\n        name="total",\n      )\n\n  def call(self, inputs):\n      self.total.assign(self.total + ops.sum(inputs))\n      return self.total\n\nmy_sum = ComputeSum(2)\nx = ops.ones((2, 2))\ny = my_sum(x)\n\nassert my_sum.weights == [my_sum.total]\nassert my_sum.non_trainable_weights == [my_sum.total]\nassert my_sum.trainable_weights == []\n```',
    "std_args": [
      {"name": "shape", "type": "Any"},
      {"name": "batch_size", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "ragged", "type": "Any"},
      {"name": "batch_shape", "type": "Any"},
      {"name": "input_tensor", "type": "Any"},
      {"name": "optional", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "InputSpec": {
    "description": "Specifies the rank, dtype and shape of every input to a layer.\n\nLayers can expose (if appropriate) an `input_spec` attribute:\nan instance of `InputSpec`, or a nested structure of `InputSpec` instances\n(one per input tensor). These objects enable the layer to run input\ncompatibility checks for input structure, input rank, input shape, and\ninput dtype for the first argument of `Layer.__call__`.\n\nA `None` entry in a shape is compatible with any dimension.\n\nArgs:\n    dtype: Expected dtype of the input.\n    shape: Shape tuple, expected shape of the input\n        (may include `None` for dynamic axes).\n        Includes the batch size.\n    ndim: Integer, expected rank of the input.\n    max_ndim: Integer, maximum rank of the input.\n    min_ndim: Integer, minimum rank of the input.\n    axes: Dictionary mapping integer axes to\n        a specific dimension value.\n    allow_last_axis_squeeze: If `True`, allow inputs of rank N+1 as long\n        as the last axis of the input is 1, as well as inputs of rank N-1\n        as long as the last axis of the spec is 1.\n    name: Expected key corresponding to this input when passing data as\n        a dictionary.\n    optional: Boolean, whether the input is optional or not.\n        An optional input can accept `None` values.\n\nExample:\n\n```python\nclass MyLayer(Layer):\n    def __init__(self):\n        super().__init__()\n        # The layer will accept inputs with\n        # shape (*, 28, 28) & (*, 28, 28, 1)\n        # and raise an appropriate error message otherwise.\n        self.input_spec = InputSpec(\n            shape=(None, 28, 28, 1),\n            allow_last_axis_squeeze=True)\n```",
    "std_args": [
      {"name": "dtype", "type": "Any"},
      {"name": "shape", "type": "Any"},
      {"name": "ndim", "type": "Any"},
      {"name": "max_ndim", "type": "Any"},
      {"name": "min_ndim", "type": "Any"},
      {"name": "axes", "type": "Any"},
      {"name": "allow_last_axis_squeeze", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "optional", "type": "Any"},
    ],
    "variants": {},
  },
  "Instancenorm": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
      {"name": "eps", "type": "Any"},
      {"name": "num_features", "type": "Any"},
    ],
    "variants": {},
  },
  "Int16": {
    "description": "16-bit signed integer type (Short).",
    "std_args": [],
    "variants": {},
  },
  "Int32": {
    "description": "32-bit signed integer type (Int).",
    "std_args": [],
    "variants": {},
  },
  "Int64": {
    "description": "64-bit signed integer type (Long).",
    "std_args": [],
    "variants": {},
  },
  "IntegerLookup": {
    "description": 'A preprocessing layer that maps integers to (possibly encoded) indices.\n\nThis layer maps a set of arbitrary integer input tokens into indexed integer\noutput via a table-based vocabulary lookup. The layer\'s output indices will\nbe contiguously arranged up to the maximum vocab size, even if the input\ntokens are non-continguous or unbounded. The layer supports multiple options\nfor encoding the output via `output_mode`, and has optional support for\nout-of-vocabulary (OOV) tokens and masking.\n\nThe vocabulary for the layer must be either supplied on construction or\nlearned via `adapt()`. During `adapt()`, the layer will analyze a data set,\ndetermine the frequency of individual integer tokens, and create a\nvocabulary from them. If the vocabulary is capped in size, the most frequent\ntokens will be used to create the vocabulary and all others will be treated\nas OOV.\n\nThere are two possible output modes for the layer.  When `output_mode` is\n`"int"`, input integers are converted to their index in the vocabulary (an\ninteger).  When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`,\ninput integers are encoded into an array where each dimension corresponds to\nan element in the vocabulary.\n\nThe vocabulary can optionally contain a mask token as well as an OOV token\n(which can optionally occupy multiple indices in the vocabulary, as set\nby `num_oov_indices`).\nThe position of these tokens in the vocabulary is fixed. When `output_mode`\nis `"int"`, the vocabulary will begin with the mask token at index 0,\nfollowed by OOV indices, followed by the rest of the vocabulary. When\n`output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will\nbegin with OOV indices and instances of the mask token will be dropped.\n\n**Note:** This layer uses TensorFlow internally. It cannot\nbe used as part of the compiled computation graph of a model with\nany backend other than TensorFlow.\nIt can however be used with any backend when running eagerly.\nIt can also always be used as part of an input preprocessing pipeline\nwith any backend (outside the model itself), which is how we recommend\nto use this layer.\n\n**Note:** This layer is safe to use inside a `tf.data` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    max_tokens: Maximum size of the vocabulary for this layer. This should\n        only be specified when adapting the vocabulary or when setting\n        `pad_to_max_tokens=True`. If None, there is no cap on the size of\n        the vocabulary. Note that this size includes the OOV\n        and mask tokens. Defaults to `None`.\n    num_oov_indices: The number of out-of-vocabulary tokens to use.\n        If this value is more than 1, OOV inputs are modulated to\n        determine their OOV value.\n        If this value is 0, OOV inputs will cause an error when calling\n        the layer. Defaults to `1`.\n    mask_token: An integer token that represents masked inputs. When\n        `output_mode` is `"int"`, the token is included in vocabulary\n        and mapped to index 0. In other output modes,\n        the token will not appear in the vocabulary and instances\n        of the mask token in the input will be dropped.\n        If set to None, no mask term will be added. Defaults to `None`.\n    oov_token: Only used when `invert` is `True`. The token to return\n        for OOV indices. Defaults to `-1`.\n    vocabulary: Optional. Either an array of integers or a string path to a\n        text file. If passing an array, can pass a tuple, list,\n        1D NumPy array, or 1D tensor containing the integer vocbulary terms.\n        If passing a file path, the file should contain one line per term\n        in the vocabulary. If this argument is set,\n        there is no need to `adapt()` the layer.\n    vocabulary_dtype: The dtype of the vocabulary terms.\n        Only `vocabulary_dtype=\'int64\'` is supported at this time.\n        Defaults to `"int64"`.\n    idf_weights: Only valid when `output_mode` is `"tf_idf"`.\n        A tuple, list, 1D NumPy array, or 1D tensor or the same length\n        as the vocabulary, containing the floating point inverse document\n        frequency weights, which will be multiplied by per sample term\n        counts for the final TF-IDF weight.\n        If the `vocabulary` argument is set, and `output_mode` is\n        `"tf_idf"`, this argument must be supplied.\n    invert: Only valid when `output_mode` is `"int"`.\n        If `True`, this layer will map indices to vocabulary items\n        instead of mapping vocabulary items to indices.\n        Defaults to `False`.\n    output_mode: Specification for the output of the layer. Values can be\n        `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"`\n        configuring the layer as follows:\n        - `"int"`: Return the vocabulary indices of the input tokens.\n        - `"one_hot"`: Encodes each individual element in the input into an\n            array the same size as the vocabulary,\n            containing a 1 at the element index. If the last dimension\n            is size 1, will encode on that dimension.\n            If the last dimension is not size 1, will append a new\n            dimension for the encoded output.\n        - `"multi_hot"`: Encodes each sample in the input into a single\n            array the same size as the vocabulary,\n            containing a 1 for each vocabulary term present in the sample.\n            Treats the last dimension as the sample dimension,\n            if input shape is `(..., sample_length)`,\n            output shape will be `(..., num_tokens)`.\n        - `"count"`: As `"multi_hot"`, but the int array contains\n            a count of the number of times the token at that index\n            appeared in the sample.\n        - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is\n            applied to find the value in each token slot.\n        For `"int"` output, the output shape matches the input shape.\n        For `"one_hot"` output, the output shape is\n        `input_shape + (vocabulary_size,)`, where `input_shape` may\n        have arbitrary rank. For other output modes (`"multi_hot"`,\n        `"count"`, `"tf_idf"`), the output shape is `(batch_size,\n        vocabulary_size)`. Defaults to `"int"`.\n    pad_to_max_tokens: Only applicable when `output_mode` is `"multi_hot"`,\n        `"count"`, or `"tf_idf"`. If `True`, the output will have\n        its feature axis padded to `max_tokens` even if the number\n        of unique tokens in the vocabulary is less than `max_tokens`,\n        resulting in a tensor of shape `(batch_size, max_tokens)`\n        regardless of vocabulary size. Defaults to `False`.\n    sparse: Boolean. Only applicable to `"multi_hot"`, `"count"`, and\n        `"tf_idf"` output modes. Only supported with TensorFlow\n        backend. If `True`, returns a `SparseTensor`\n        instead of a dense `Tensor`. Defaults to `False`.\n\nExamples:\n\n**Creating a lookup layer with a known vocabulary**\n\nThis example creates a lookup layer with a pre-existing vocabulary.\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([[12, 1138, 42], [42, 1000, 36]])  # Note OOV tokens\n>>> layer = IntegerLookup(vocabulary=vocab)\n>>> layer(data)\narray([[1, 3, 4],\n       [4, 0, 2]])\n\n**Creating a lookup layer with an adapted vocabulary**\n\nThis example creates a lookup layer and generates the vocabulary by\nanalyzing the dataset.\n\n>>> data = np.array([[12, 1138, 42], [42, 1000, 36]])\n>>> layer = IntegerLookup()\n>>> layer.adapt(data)\n>>> layer.get_vocabulary()\n[-1, 42, 1138, 1000, 36, 12]\n\nNote that the OOV token -1 have been added to the vocabulary. The remaining\ntokens are sorted by frequency (42, which has 2 occurrences, is first) then\nby inverse sort order.\n\n>>> data = np.array([[12, 1138, 42], [42, 1000, 36]])\n>>> layer = IntegerLookup()\n>>> layer.adapt(data)\n>>> layer(data)\narray([[5, 2, 1],\n       [1, 3, 4]])\n\n**Lookups with multiple OOV indices**\n\nThis example demonstrates how to use a lookup layer with multiple OOV\nindices.  When a layer is created with more than one OOV index, any OOV\ntokens are hashed into the number of OOV buckets, distributing OOV tokens in\na deterministic fashion across the set.\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([[12, 1138, 42], [37, 1000, 36]])\n>>> layer = IntegerLookup(vocabulary=vocab, num_oov_indices=2)\n>>> layer(data)\narray([[2, 4, 5],\n       [1, 0, 3]])\n\nNote that the output for OOV token 37 is 1, while the output for OOV token\n1000 is 0. The in-vocab terms have their output index increased by 1 from\nearlier examples (12 maps to 2, etc) in order to make space for the extra\nOOV token.\n\n**One-hot output**\n\nConfigure the layer with `output_mode=\'one_hot\'`. Note that the first\n`num_oov_indices` dimensions in the ont_hot encoding represent OOV values.\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([12, 36, 1138, 42, 7])  # Note OOV tokens\n>>> layer = IntegerLookup(vocabulary=vocab, output_mode=\'one_hot\')\n>>> layer(data)\narray([[0., 1., 0., 0., 0.],\n        [0., 0., 1., 0., 0.],\n        [0., 0., 0., 1., 0.],\n        [0., 0., 0., 0., 1.],\n        [1., 0., 0., 0., 0.]], dtype=float32)\n\n**Multi-hot output**\n\nConfigure the layer with `output_mode=\'multi_hot\'`. Note that the first\n`num_oov_indices` dimensions in the multi_hot encoding represent OOV tokens\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([[12, 1138, 42, 42],\n...                  [42,    7, 36,  7]])  # Note OOV tokens\n>>> layer = IntegerLookup(vocabulary=vocab, output_mode=\'multi_hot\')\n>>> layer(data)\narray([[0., 1., 0., 1., 1.],\n       [1., 0., 1., 0., 1.]], dtype=float32)\n\n**Token count output**\n\nConfigure the layer with `output_mode=\'count\'`. As with multi_hot output,\nthe first `num_oov_indices` dimensions in the output represent OOV tokens.\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([[12, 1138, 42, 42],\n...                  [42,    7, 36,  7]])  # Note OOV tokens\n>>> layer = IntegerLookup(vocabulary=vocab, output_mode=\'count\')\n>>> layer(data)\narray([[0., 1., 0., 1., 2.],\n       [2., 0., 1., 0., 1.]], dtype=float32)\n\n**TF-IDF output**\n\nConfigure the layer with `output_mode=\'tf_idf\'`. As with multi_hot output,\nthe first `num_oov_indices` dimensions in the output represent OOV tokens.\n\nEach token bin will output `token_count * idf_weight`, where the idf weights\nare the inverse document frequency weights per token. These should be\nprovided along with the vocabulary. Note that the `idf_weight` for OOV\ntokens will default to the average of all idf weights passed in.\n\n>>> vocab = [12, 36, 1138, 42]\n>>> idf_weights = [0.25, 0.75, 0.6, 0.4]\n>>> data = np.array([[12, 1138, 42, 42],\n...                  [42,    7, 36,  7]])  # Note OOV tokens\n>>> layer = IntegerLookup(\n...     output_mode=\'tf_idf\', vocabulary=vocab, idf_weights=idf_weights)\n>>> layer(data)\narray([[0.  , 0.25, 0.  , 0.6 , 0.8 ],\n        [1.0 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)\n\nTo specify the idf weights for oov tokens, you will need to pass the entire\nvocabulary including the leading oov token.\n\n>>> vocab = [-1, 12, 36, 1138, 42]\n>>> idf_weights = [0.9, 0.25, 0.75, 0.6, 0.4]\n>>> data = np.array([[12, 1138, 42, 42],\n...                  [42,    7, 36,  7]])  # Note OOV tokens\n>>> layer = IntegerLookup(\n...     output_mode=\'tf_idf\', vocabulary=vocab, idf_weights=idf_weights)\n>>> layer(data)\narray([[0.  , 0.25, 0.  , 0.6 , 0.8 ],\n        [1.8 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)\n\nWhen adapting the layer in `"tf_idf"` mode, each input sample will\nbe considered a document, and IDF weight per token will be\ncalculated as:\n`log(1 + num_documents / (1 + token_document_count))`.\n\n**Inverse lookup**\n\nThis example demonstrates how to map indices to tokens using this layer.\n(You can also use `adapt()` with `inverse=True`, but for simplicity we\'ll\npass the vocab in this example.)\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([[1, 3, 4], [4, 0, 2]])\n>>> layer = IntegerLookup(vocabulary=vocab, invert=True)\n>>> layer(data)\narray([[  12, 1138,   42],\n       [  42,   -1,   36]])\n\nNote that the first index correspond to the oov token by default.\n\n**Forward and inverse lookup pairs**\n\nThis example demonstrates how to use the vocabulary of a standard lookup\nlayer to create an inverse lookup layer.\n\n>>> vocab = [12, 36, 1138, 42]\n>>> data = np.array([[12, 1138, 42], [42, 1000, 36]])\n>>> layer = IntegerLookup(vocabulary=vocab)\n>>> i_layer = IntegerLookup(\n...     vocabulary=layer.get_vocabulary(), invert=True)\n>>> int_data = layer(data)\n>>> i_layer(int_data)\narray([[  12, 1138,   42],\n       [  42,   -1,   36]])\n\nIn this example, the input token 1000 resulted in an output of -1, since\n1000 was not in the vocabulary - it got represented as an OOV, and all OOV\ntokens are returned as -1 in the inverse layer. Also, note that for the\ninverse to work, you must have already set the forward layer vocabulary\neither directly or via `adapt()` before calling `get_vocabulary()`.',
    "std_args": [
      {"name": "max_tokens", "type": "Any"},
      {"name": "num_oov_indices", "type": "Any"},
      {"name": "mask_token", "type": "Any"},
      {"name": "oov_token", "type": "Any"},
      {"name": "vocabulary", "type": "Any"},
      {"name": "vocabulary_dtype", "type": "Any"},
      {"name": "idf_weights", "type": "Any"},
      {"name": "invert", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "pad_to_max_tokens", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "IsInf": {
    "description": "Map infinity to true and other values to false.",
    "std_args": [
      {"name": "X", "type": "Any"},
      {"name": "detect_negative", "type": "int"},
      {"name": "detect_positive", "type": "int"},
    ],
    "variants": {},
  },
  "IsNaN": {
    "description": "Returns which elements of the input are NaN.",
    "std_args": [
      {"name": "X", "type": "Any"},
    ],
    "variants": {},
  },
  "JaxLayer": {
    "description": 'Keras Layer that wraps a JAX model.\n\nThis layer enables the use of JAX components within Keras when using JAX as\nthe backend for Keras.\n\n## Model function\n\nThis layer accepts JAX models in the form of a function, `call_fn`, which\nmust take the following arguments with these exact names:\n\n- `params`: trainable parameters of the model.\n- `state` (*optional*): non-trainable state of the model. Can be omitted if\n    the model has no non-trainable state.\n- `rng` (*optional*): a `jax.random.PRNGKey` instance. Can be omitted if the\n    model does not need RNGs, neither during training nor during inference.\n- `inputs`: inputs to the model, a JAX array or a `PyTree` of arrays.\n- `training` (*optional*): an argument specifying if we\'re in training mode\n    or inference mode, `True` is passed in training mode. Can be omitted if\n    the model behaves the same in training mode and inference mode.\n\nThe `inputs` argument is mandatory. Inputs to the model must be provided via\na single argument. If the JAX model takes multiple inputs as separate\narguments, they must be combined into a single structure, for instance in a\n`tuple` or a `dict`.\n\n## Model weights initialization\n\nThe initialization of the `params` and `state` of the model can be handled\nby this layer, in which case the `init_fn` argument must be provided. This\nallows the model to be initialized dynamically with the right shape.\nAlternatively, and if the shape is known, the `params` argument and\noptionally the `state` argument can be used to create an already initialized\nmodel.\n\nThe `init_fn` function, if provided, must take the following arguments with\nthese exact names:\n\n- `rng`: a `jax.random.PRNGKey` instance.\n- `inputs`: a JAX array or a `PyTree` of arrays with placeholder values to\n    provide the shape of the inputs.\n- `training` (*optional*): an argument specifying if we\'re in training mode\n    or inference mode. `True` is always passed to `init_fn`. Can be omitted\n    regardless of whether `call_fn` has a `training` argument.\n\n## Models with non-trainable state\n\nFor JAX models that have non-trainable state:\n\n- `call_fn` must have a `state` argument\n- `call_fn` must return a `tuple` containing the outputs of the model and\n    the new non-trainable state of the model\n- `init_fn` must return a `tuple` containing the initial trainable params of\n    the model and the initial non-trainable state of the model.\n\nThis code shows a possible combination of `call_fn` and `init_fn` signatures\nfor a model with non-trainable state. In this example, the model has a\n`training` argument and an `rng` argument in `call_fn`.\n\n```python\ndef stateful_call(params, state, rng, inputs, training):\n    outputs = ...\n    new_state = ...\n    return outputs, new_state\n\ndef stateful_init(rng, inputs):\n    initial_params = ...\n    initial_state = ...\n    return initial_params, initial_state\n```\n\n## Models without non-trainable state\n\nFor JAX models with no non-trainable state:\n\n- `call_fn` must not have a `state` argument\n- `call_fn` must return only the outputs of the model\n- `init_fn` must return only the initial trainable params of the model.\n\nThis code shows a possible combination of `call_fn` and `init_fn` signatures\nfor a model without non-trainable state. In this example, the model does not\nhave a `training` argument and does not have an `rng` argument in `call_fn`.\n\n```python\ndef stateless_call(params, inputs):\n    outputs = ...\n    return outputs\n\ndef stateless_init(rng, inputs):\n    initial_params = ...\n    return initial_params\n```\n\n## Conforming to the required signature\n\nIf a model has a different signature than the one required by `JaxLayer`,\none can easily write a wrapper method to adapt the arguments. This example\nshows a model that has multiple inputs as separate arguments, expects\nmultiple RNGs in a `dict`, and has a `deterministic` argument with the\nopposite meaning of `training`. To conform, the inputs are combined in a\nsingle structure using a `tuple`, the RNG is split and used the populate the\nexpected `dict`, and the Boolean flag is negated:\n\n```python\ndef my_model_fn(params, rngs, input1, input2, deterministic):\n    ...\n    if not deterministic:\n        dropout_rng = rngs["dropout"]\n        keep = jax.random.bernoulli(dropout_rng, dropout_rate, x.shape)\n        x = jax.numpy.where(keep, x / dropout_rate, 0)\n        ...\n    ...\n    return outputs\n\ndef my_model_wrapper_fn(params, rng, inputs, training):\n    input1, input2 = inputs\n    rng1, rng2 = jax.random.split(rng)\n    rngs = {"dropout": rng1, "preprocessing": rng2}\n    deterministic = not training\n    return my_model_fn(params, rngs, input1, input2, deterministic)\n\nkeras_layer = JaxLayer(my_model_wrapper_fn, params=initial_params)\n```\n\n## Usage with Haiku modules\n\n`JaxLayer` enables the use of [Haiku](https://dm-haiku.readthedocs.io)\ncomponents in the form of\n[`haiku.Module`](https://dm-haiku.readthedocs.io/en/latest/api.html#module).\nThis is achieved by transforming the module per the Haiku pattern and then\npassing `module.apply` in the `call_fn` parameter and `module.init` in the\n`init_fn` parameter if needed.\n\nIf the model has non-trainable state, it should be transformed with\n[`haiku.transform_with_state`](\n  https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform_with_state).\nIf the model has no non-trainable state, it should be transformed with\n[`haiku.transform`](\n  https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.transform).\nAdditionally, and optionally, if the module does not use RNGs in "apply", it\ncan be transformed with\n[`haiku.without_apply_rng`](\n  https://dm-haiku.readthedocs.io/en/latest/api.html#without-apply-rng).\n\nThe following example shows how to create a `JaxLayer` from a Haiku module\nthat uses random number generators via `hk.next_rng_key()` and takes a\ntraining positional argument:\n\n```python\nclass MyHaikuModule(hk.Module):\n    def __call__(self, x, training):\n        x = hk.Conv2D(32, (3, 3))(x)\n        x = jax.nn.relu(x)\n        x = hk.AvgPool((1, 2, 2, 1), (1, 2, 2, 1), "VALID")(x)\n        x = hk.Flatten()(x)\n        x = hk.Linear(200)(x)\n        if training:\n            x = hk.dropout(rng=hk.next_rng_key(), rate=0.3, x=x)\n        x = jax.nn.relu(x)\n        x = hk.Linear(10)(x)\n        x = jax.nn.softmax(x)\n        return x\n\ndef my_haiku_module_fn(inputs, training):\n    module = MyHaikuModule()\n    return module(inputs, training)\n\ntransformed_module = hk.transform(my_haiku_module_fn)\n\nkeras_layer = JaxLayer(\n    call_fn=transformed_module.apply,\n    init_fn=transformed_module.init,\n)\n```\n\nArgs:\n    call_fn: The function to call the model. See description above for the\n        list of arguments it takes and the outputs it returns.\n    init_fn: the function to call to initialize the model. See description\n        above for the list of arguments it takes and the outputs it returns.\n        If `None`, then `params` and/or `state` must be provided.\n  params: A `PyTree` containing all the model trainable parameters. This\n        allows passing trained parameters or controlling the initialization.\n        If both `params` and `state` are `None`, `init_fn` is called at\n        build time to initialize the trainable parameters of the model.\n  state: A `PyTree` containing all the model non-trainable state. This\n        allows passing learned state or controlling the initialization. If\n        both `params` and `state` are `None`, and `call_fn` takes a `state`\n        argument, then `init_fn` is called at build time to initialize the\n        non-trainable state of the model.\n  seed: Seed for random number generator. Optional.\n  dtype: The dtype of the layer\'s computations and weights. Can also be a\n        `keras.DTypePolicy`. Optional. Defaults to the default policy.',
    "std_args": [
      {"name": "call_fn", "type": "Any"},
      {"name": "init_fn", "type": "Any"},
      {"name": "params", "type": "Any"},
      {"name": "state", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Kldiv": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "L1": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "LOWER_RIGHT": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "LSTM": {
    "description": "Computes an one-layer LSTM. This operator is usually supported via some custom implementation such as CuDNN. Notations: * `X` - input tensor * `i` - input gate * `o` - output gate * `f` - forget gate * `c` - cell gate * `t` - time step (t-1 means previous time step) * `W[iofc]` - W parameter weight ...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "W", "type": "Tensor"},
      {"name": "R", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
      {"name": "sequence_lens", "type": "Any"},
      {"name": "initial_h", "type": "Tensor"},
      {"name": "initial_c", "type": "Tensor"},
      {"name": "P", "type": "Tensor"},
      {"name": "activation_alpha", "type": "List[float]"},
      {"name": "activation_beta", "type": "List[float]"},
      {"name": "activations", "type": "List[str]"},
      {"name": "clip", "type": "float"},
      {"name": "direction", "type": "str"},
      {"name": "hidden_size", "type": "int"},
      {"name": "input_forget", "type": "int"},
      {"name": "layout", "type": "int"},
    ],
    "variants": {},
  },
  "Lambda": {
    "description": "Wraps arbitrary expressions as a `Layer` object.\n\nThe `Lambda` layer exists so that arbitrary expressions can be used\nas a `Layer` when constructing Sequential\nand Functional API models. `Lambda` layers are best suited for simple\noperations or quick experimentation. For more advanced use cases,\nprefer writing new subclasses of `Layer`.\n\nWARNING: `Lambda` layers have (de)serialization limitations!\n\nThe main reason to subclass `Layer` instead of using a\n`Lambda` layer is saving and inspecting a model. `Lambda` layers\nare saved by serializing the Python bytecode, which is fundamentally\nnon-portable and potentially unsafe.\nThey should only be loaded in the same environment where\nthey were saved. Subclassed layers can be saved in a more portable way\nby overriding their `get_config()` method. Models that rely on\nsubclassed Layers are also often easier to visualize and reason about.\n\nExample:\n\n```python\n# add a x -> x^2 layer\nmodel.add(Lambda(lambda x: x ** 2))\n```\n\nArgs:\n    function: The function to be evaluated. Takes input tensor as first\n        argument.\n    output_shape: Expected output shape from function. This argument\n        can usually be inferred if not explicitly provided.\n        Can be a tuple or function. If a tuple, it only specifies\n        the first dimension onward; sample dimension is assumed\n        either the same as the input:\n        `output_shape = (input_shape[0], ) + output_shape` or,\n        the input is `None` and the sample dimension is also `None`:\n        `output_shape = (None, ) + output_shape`.\n        If a function, it specifies the\n        entire shape as a function of the input shape:\n        `output_shape = f(input_shape)`.\n    mask: Either None (indicating no masking) or a callable with the same\n        signature as the `compute_mask` layer method, or a tensor\n        that will be returned as output mask regardless\n        of what the input is.\n    arguments: Optional dictionary of keyword arguments to be passed to the\n        function.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "output_shape", "type": "Any"},
      {"name": "mask", "type": "Any"},
      {"name": "arguments", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Layer": {
    "description": 'This is the class from which all layers inherit.\n\nA layer is a callable object that takes as input one or more tensors and\nthat outputs one or more tensors. It involves *computation*, defined\nin the `call()` method, and a *state* (weight variables). State can be\ncreated:\n\n* in `__init__()`, for instance via `self.add_weight()`;\n* in the optional `build()` method, which is invoked by the first\n  `__call__()` to the layer, and supplies the shape(s) of the input(s),\n  which may not have been known at initialization time.\n\nLayers are recursively composable: If you assign a Layer instance as an\nattribute of another Layer, the outer layer will start tracking the weights\ncreated by the inner layer. Nested layers should be instantiated in the\n`__init__()` method or `build()` method.\n\nUsers will just instantiate a layer and then treat it as a callable.\n\nArgs:\n    trainable: Boolean, whether the layer\'s variables should be trainable.\n    name: String name of the layer.\n    dtype: The dtype of the layer\'s computations and weights. Can also be a\n        `keras.DTypePolicy`, which allows the computation and weight dtype\n        to differ. Defaults to `None`. `None` means to use\n        `keras.config.dtype_policy()`, which is a `float32` policy unless\n        set to different value (via `keras.config.set_dtype_policy()`).\n\nAttributes:\n    name: The name of the layer (string).\n    dtype: Dtype of the layer\'s weights. Alias of `layer.variable_dtype`.\n    variable_dtype: Dtype of the layer\'s weights.\n    compute_dtype: The dtype of the layer\'s computations.\n        Layers automatically cast inputs to this dtype, which causes\n        the computations and output to also be in this dtype.\n        When mixed precision is used with a\n        `keras.DTypePolicy`, this will be different\n        than `variable_dtype`.\n    trainable_weights: List of variables to be included in backprop.\n    non_trainable_weights: List of variables that should not be\n        included in backprop.\n    weights: The concatenation of the lists trainable_weights and\n        non_trainable_weights (in this order).\n    trainable: Whether the layer should be trained (boolean), i.e.\n        whether its potentially-trainable weights should be returned\n        as part of `layer.trainable_weights`.\n    input_spec: Optional (list of) `InputSpec` object(s) specifying the\n        constraints on inputs that can be accepted by the layer.\n\nWe recommend that descendants of `Layer` implement the following methods:\n\n* `__init__()`: Defines custom layer attributes, and creates layer weights\n    that do not depend on input shapes, using `add_weight()`,\n    or other state.\n* `build(self, input_shape)`: This method can be used to create weights that\n    depend on the shape(s) of the input(s), using `add_weight()`, or other\n    state. `__call__()` will automatically build the layer\n    (if it has not been built yet) by calling `build()`.\n* `call(self, *args, **kwargs)`: Called in `__call__` after making\n    sure `build()` has been called. `call()` performs the logic of applying\n    the layer to the input arguments.\n    Two reserved keyword arguments you can optionally use in `call()` are:\n        1. `training` (boolean, whether the call is in inference mode or\n            training mode).\n        2. `mask` (boolean tensor encoding masked timesteps in the input,\n            used e.g. in RNN layers).\n    A typical signature for this method is `call(self, inputs)`, and user\n    could optionally add `training` and `mask` if the layer need them.\n* `get_config(self)`: Returns a dictionary containing the configuration\n    used to initialize this layer. If the keys differ from the arguments\n    in `__init__()`, then override `from_config(self)` as well.\n    This method is used when saving\n    the layer or a model that contains this layer.\n\nExamples:\n\nHere\'s a basic example: a layer with two variables, `w` and `b`,\nthat returns `y = w . x + b`.\nIt shows how to implement `build()` and `call()`.\nVariables set as attributes of a layer are tracked as weights\nof the layers (in `layer.weights`).\n\n```python\nclass SimpleDense(Layer):\n    def __init__(self, units=32):\n        super().__init__()\n        self.units = units\n\n    # Create the state of the layer (weights)\n    def build(self, input_shape):\n        self.kernel = self.add_weight(\n            shape=(input_shape[-1], self.units),\n            initializer="glorot_uniform",\n            trainable=True,\n            name="kernel",\n        )\n        self.bias = self.add_weight(\n            shape=(self.units,),\n            initializer="zeros",\n            trainable=True,\n            name="bias",\n        )\n\n    # Defines the computation\n    def call(self, inputs):\n        return ops.matmul(inputs, self.kernel) + self.bias\n\n# Instantiates the layer.\nlinear_layer = SimpleDense(4)\n\n# This will also call `build(input_shape)` and create the weights.\ny = linear_layer(ops.ones((2, 2)))\nassert len(linear_layer.weights) == 2\n\n# These weights are trainable, so they\'re listed in `trainable_weights`:\nassert len(linear_layer.trainable_weights) == 2\n```\n\nBesides trainable weights, updated via backpropagation during training,\nlayers can also have non-trainable weights. These weights are meant to\nbe updated manually during `call()`. Here\'s a example layer that computes\nthe running sum of its inputs:\n\n```python\nclass ComputeSum(Layer):\n\n  def __init__(self, input_dim):\n      super(ComputeSum, self).__init__()\n      # Create a non-trainable weight.\n      self.total = self.add_weight(\n        shape=(),\n        initializer="zeros",\n        trainable=False,\n        name="total",\n      )\n\n  def call(self, inputs):\n      self.total.assign(self.total + ops.sum(inputs))\n      return self.total\n\nmy_sum = ComputeSum(2)\nx = ops.ones((2, 2))\ny = my_sum(x)\n\nassert my_sum.weights == [my_sum.total]\nassert my_sum.non_trainable_weights == [my_sum.total]\nassert my_sum.trainable_weights == []\n```',
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "LayerNorm": {
    "description": "Applies Layer Normalization over a mini-batch of inputs.",
    "std_args": [
      {"name": "normalized_shape", "type": "Any"},
      {"name": "eps", "type": "Any"},
      {"name": "elementwise_affine", "type": "Any"},
      {"name": "bias", "type": "Any"},
    ],
    "variants": {},
  },
  "LayerNormalization": {
    "description": "This is layer normalization defined in ONNX as function. The overall computation can be split into two stages. The first stage is standardization, which makes the normalized elements have zero mean and unit variances. The computation required by standardization can be described by the following equa...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "Scale", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
      {"name": "axis", "type": "int"},
      {"name": "epsilon", "type": "float"},
      {"name": "stash_type", "type": "int"},
    ],
    "variants": {},
  },
  "LazyConvTranspose1d": {
    "description": "A :class:`torch.nn.ConvTranspose1d` module with lazy initialization of the ``in_channels`` argument.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "LazyConvTranspose2d": {
    "description": "A :class:`torch.nn.ConvTranspose2d` module with lazy initialization of the ``in_channels`` argument.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "LazyConvTranspose3d": {
    "description": "A :class:`torch.nn.ConvTranspose3d` module with lazy initialization of the ``in_channels`` argument.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "out_channels", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "groups", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "LeakyRelu": {
    "description": "LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`, `f(x) = x for x >= 0`, is applied to the data tensor elementwise.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "alpha", "type": "float"},
    ],
    "variants": {},
  },
  "Less": {
    "description": "Returns the tensor resulted from performing the `less` logical operation elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Linear": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
      {"name": "in_features", "type": "Any"},
      {"name": "out_features", "type": "Any"},
    ],
    "variants": {},
  },
  "LinearT": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Lion": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "b1", "type": "Any"},
      {"name": "b2", "type": "Any"},
      {"name": "lr", "type": "Any"},
      {"name": "mask", "type": "Any"},
      {"name": "mu_dtype", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
    ],
    "variants": {},
  },
  "List": {
    "description": "A Module that implements a mutable sequence.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "it", "type": "Any"},
    ],
    "variants": {},
  },
  "Load": {
    "description": "Deserialize object from disk.",
    "std_args": [
      {"name": "f", "type": "Any"},
    ],
    "variants": {},
  },
  "Log": {
    "description": "Calculates the natural log of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "LogSoftmax": {
    "description": 'The operator computes the log of softmax values for the given input: LogSoftmax(input, axis) = Log(Softmax(input, axis=axis)) The "axis" attribute indicates the dimension along which LogSoftmax will be performed. The output tensor has the same shape and contains the LogSoftmax values of the correspo...',
    "std_args": [
      {"name": "input", "type": "Tensor"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "Logcosh": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "reduction", "type": "Literal"},
    ],
    "variants": {},
  },
  "Logsigmoid": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Logsumexp": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "b", "type": "Any"},
      {"name": "dim", "type": "Any"},
      {"name": "keepdim", "type": "Any"},
      {"name": "return_sign", "type": "Any"},
      {"name": "where", "type": "Any"},
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "Lstmcell": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "M": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "MA": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "MSELoss": {
    "description": "Mean Squared Error.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
    ],
    "variants": {},
  },
  "MarginRankingLoss": {
    "description": "Creates a criterion that measures the loss given",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "margin", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
    ],
    "variants": {},
  },
  "Marginranking": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "margin", "type": "float"},
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "MaskedNode": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "MaskedState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Masking": {
    "description": "Masks a sequence by using a mask value to skip timesteps.\n\nFor each timestep in the input tensor (dimension #1 in the tensor),\nif all values in the input tensor at that timestep\nare equal to `mask_value`, then the timestep will be masked (skipped)\nin all downstream layers (as long as they support masking).\n\nIf any downstream layer does not support masking yet receives such\nan input mask, an exception will be raised.\n\nExample:\n\nConsider a NumPy data array `x` of shape `(samples, timesteps, features)`,\nto be fed to an LSTM layer. You want to mask timestep #3 and #5 because you\nlack data for these timesteps. You can:\n\n- Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`\n- Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:\n\n```python\nsamples, timesteps, features = 32, 10, 8\ninputs = np.random.random([samples, timesteps, features]).astype(np.float32)\ninputs[:, 3, :] = 0.\ninputs[:, 5, :] = 0.\n\nmodel = keras.models.Sequential()\nmodel.add(keras.layers.Masking(mask_value=0.0))\nmodel.add(keras.layers.LSTM(32))\noutput = model(inputs)\n# The time step 3 and 5 will be skipped from LSTM calculation.\n```\n\nNote: in the Keras masking convention, a masked timestep is denoted by\na mask value of `False`, while a non-masked (i.e. usable) timestep\nis denoted by a mask value of `True`.",
    "std_args": [
      {"name": "mask_value", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "MatMul": {
    "description": "Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Max": {
    "description": "Element-wise max of each of the input tensors (with Numpy-style broadcasting support). All inputs and outputs must have the same data type. This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "data_0", "type": "Tensor"},
    ],
    "variants": {},
  },
  "MaxNumBoundingBoxes": {
    "description": "Ensure the maximum number of bounding boxes.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    max_number: Desired output number of bounding boxes.\n    padding_value: The padding value of the `boxes` and `labels` in\n        `bounding_boxes`. Defaults to `-1`.",
    "std_args": [
      {"name": "max_number", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxPool1d": {
    "description": "Applies a 1D max pooling over an input signal composed of several input planes.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "return_indices", "type": "Any"},
      {"name": "ceil_mode", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxPool2d": {
    "description": "Applies a 2D max pooling over an input signal composed of several input planes.",
    "std_args": [
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "return_indices", "type": "Any"},
      {"name": "ceil_mode", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxPool3d": {
    "description": "Applies a 3D max pooling over an input signal composed of several input planes.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "return_indices", "type": "Any"},
      {"name": "ceil_mode", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxPooling1D": {
    "description": 'Max pooling operation for 1D temporal data.\n\nDownsamples the input representation by taking the maximum value over a\nspatial window of size `pool_size`. The window is shifted by `strides`.\n\nThe resulting output when using the `"valid"` padding option has a shape of:\n`output_shape = (input_shape - pool_size + 1) / strides)`.\n\nThe resulting output shape when using the `"same"` padding option is:\n`output_shape = input_shape / strides`\n\nArgs:\n    pool_size: int, size of the max pooling window.\n    strides: int or None. Specifies how much the pooling window moves\n        for each pooling step. If None, it will default to `pool_size`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    3D tensor with shape `(batch_size, steps, features)`.\n- If `data_format="channels_first"`:\n    3D tensor with shape `(batch_size, features, steps)`.\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    3D tensor with shape `(batch_size, downsampled_steps, features)`.\n- If `data_format="channels_first"`:\n    3D tensor with shape `(batch_size, features, downsampled_steps)`.\n\nExamples:\n\n`strides=1` and `padding="valid"`:\n\n>>> x = np.array([1., 2., 3., 4., 5.])\n>>> x = np.reshape(x, [1, 5, 1])\n>>> max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,\n...    strides=1, padding="valid")\n>>> max_pool_1d(x)\n\n`strides=2` and `padding="valid"`:\n\n>>> x = np.array([1., 2., 3., 4., 5.])\n>>> x = np.reshape(x, [1, 5, 1])\n>>> max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,\n...    strides=2, padding="valid")\n>>> max_pool_1d(x)\n\n`strides=1` and `padding="same"`:\n\n>>> x = np.array([1., 2., 3., 4., 5.])\n>>> x = np.reshape(x, [1, 5, 1])\n>>> max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,\n...    strides=1, padding="same")\n>>> max_pool_1d(x)',
    "std_args": [
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxPooling2D": {
    "description": 'Max pooling operation for 2D spatial data.\n\nDownsamples the input along its spatial dimensions (height and width)\nby taking the maximum value over an input window\n(of size defined by `pool_size`) for each channel of the input.\nThe window is shifted by `strides` along each dimension.\n\nThe resulting output when using the `"valid"` padding option has a spatial\nshape (number of rows or columns) of:\n`output_shape = math.floor((input_shape - pool_size) / strides) + 1`\n(when `input_shape >= pool_size`)\n\nThe resulting output shape when using the `"same"` padding option is:\n`output_shape = math.floor((input_shape - 1) / strides) + 1`\n\nArgs:\n    pool_size: int or tuple of 2 integers, factors by which to downscale\n        (dim1, dim2). If only one integer is specified, the same\n        window length will be used for all dimensions.\n    strides: int or tuple of 2 integers, or None. Strides values. If None,\n        it will default to `pool_size`. If only one int is specified, the\n        same stride size will be used for all dimensions.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    4D tensor with shape `(batch_size, height, width, channels)`.\n- If `data_format="channels_first"`:\n    4D tensor with shape `(batch_size, channels, height, width)`.\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    4D tensor with shape\n    `(batch_size, pooled_height, pooled_width, channels)`.\n- If `data_format="channels_first"`:\n    4D tensor with shape\n    `(batch_size, channels, pooled_height, pooled_width)`.\n\nExamples:\n\n`strides=(1, 1)` and `padding="valid"`:\n\n>>> x = np.array([[1., 2., 3.],\n...               [4., 5., 6.],\n...               [7., 8., 9.]])\n>>> x = np.reshape(x, [1, 3, 3, 1])\n>>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),\n...    strides=(1, 1), padding="valid")\n>>> max_pool_2d(x)\n\n`strides=(2, 2)` and `padding="valid"`:\n\n>>> x = np.array([[1., 2., 3., 4.],\n...               [5., 6., 7., 8.],\n...               [9., 10., 11., 12.]])\n>>> x = np.reshape(x, [1, 3, 4, 1])\n>>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),\n...    strides=(2, 2), padding="valid")\n>>> max_pool_2d(x)\n\n`stride=(1, 1)` and `padding="same"`:\n\n>>> x = np.array([[1., 2., 3.],\n...               [4., 5., 6.],\n...               [7., 8., 9.]])\n>>> x = np.reshape(x, [1, 3, 3, 1])\n>>> max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),\n...    strides=(1, 1), padding="same")\n>>> max_pool_2d(x)',
    "std_args": [
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxPooling3D": {
    "description": 'Max pooling operation for 3D data (spatial or spatio-temporal).\n\nDownsamples the input along its spatial dimensions (depth, height, and\nwidth) by taking the maximum value over an input window (of size defined by\n`pool_size`) for each channel of the input. The window is shifted by\n`strides` along each dimension.\n\nArgs:\n    pool_size: int or tuple of 3 integers, factors by which to downscale\n        (dim1, dim2, dim3). If only one integer is specified, the same\n        window length will be used for all dimensions.\n    strides: int or tuple of 3 integers, or None. Strides values. If None,\n        it will default to `pool_size`. If only one int is specified, the\n        same stride size will be used for all dimensions.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input such that output has the same\n        height/width dimension as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape\n        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while\n        `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        It defaults to the `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json`. If you never set it, then it\n        will be `"channels_last"`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    5D tensor with shape:\n    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`\n- If `data_format="channels_first"`:\n    5D tensor with shape:\n    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`\n\nExample:\n\n```python\ndepth = 30\nheight = 30\nwidth = 30\nchannels = 3\n\ninputs = keras.layers.Input(shape=(depth, height, width, channels))\nlayer = keras.layers.MaxPooling3D(pool_size=3)\noutputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)\n```',
    "std_args": [
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxUnpool1d": {
    "description": "Computes a partial inverse of :class:`MaxPool1d`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxUnpool2d": {
    "description": "Computes a partial inverse of :class:`MaxPool2d`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
    ],
    "variants": {},
  },
  "MaxUnpool3d": {
    "description": "Computes a partial inverse of :class:`MaxPool3d`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
    ],
    "variants": {},
  },
  "Maxpool": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "kernel_size", "type": "Union"},
      {"name": "padding", "type": "Union"},
      {"name": "stride", "type": "Union"},
    ],
    "variants": {},
  },
  "Mean": {
    "description": "Element-wise mean of each of the input tensors (with Numpy-style broadcasting support). All inputs and outputs must have the same data type. This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "data_0", "type": "Tensor"},
    ],
    "variants": {},
  },
  "MelSpectrogram": {
    "description": 'A preprocessing layer to convert raw audio signals to Mel spectrograms.\n\nThis layer takes `float32`/`float64` single or batched audio signal as\ninputs and computes the Mel spectrogram using Short-Time Fourier Transform\nand Mel scaling. The input should be a 1D (unbatched) or 2D (batched) tensor\nrepresenting audio signals. The output will be a 2D or 3D tensor\nrepresenting Mel spectrograms.\n\nA spectrogram is an image-like representation that shows the frequency\nspectrum of a signal over time. It uses x-axis to represent time, y-axis to\nrepresent frequency, and each pixel to represent intensity.\nMel spectrograms are a special type of spectrogram that use the mel scale,\nwhich approximates how humans perceive sound. They are commonly used in\nspeech and music processing tasks like speech recognition, speaker\nidentification, and music genre classification.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nReferences:\n- [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram),\n- [Mel scale](https://en.wikipedia.org/wiki/Mel_scale).\n\nArgs:\n    fft_length: Integer, size of the FFT window.\n    sequence_stride: Integer, number of samples between successive STFT\n        columns.\n    sequence_length: Integer, size of the window used for applying\n        `window` to each audio frame. If `None`, defaults to `fft_length`.\n    window: String, name of the window function to use. Available values\n        are `"hann"` and `"hamming"`. If `window` is a tensor, it will be\n        used directly as the window and its length must be\n        `sequence_length`. If `window` is `None`, no windowing is\n        used. Defaults to `"hann"`.\n    sampling_rate: Integer, sample rate of the input signal.\n    num_mel_bins: Integer, number of mel bins to generate.\n    min_freq: Float, minimum frequency of the mel bins.\n    max_freq: Float, maximum frequency of the mel bins.\n        If `None`, defaults to `sampling_rate / 2`.\n    power_to_db: If True, convert the power spectrogram to decibels.\n    top_db: Float, minimum negative cut-off `max(10 * log10(S)) - top_db`.\n    mag_exp: Float, exponent for the magnitude spectrogram.\n        1 for magnitude, 2 for power, etc. Default is 2.\n    ref_power: Float, the power is scaled relative to it\n        `10 * log10(S / ref_power)`.\n    min_power: Float, minimum value for power and `ref_power`.\n\nExamples:\n\n**Unbatched audio signal**\n\n>>> layer = keras.layers.MelSpectrogram(num_mel_bins=64,\n...                                     sampling_rate=8000,\n...                                     sequence_stride=256,\n...                                     fft_length=2048)\n>>> layer(keras.random.uniform(shape=(16000,))).shape\n(64, 63)\n\n**Batched audio signal**\n\n>>> layer = keras.layers.MelSpectrogram(num_mel_bins=80,\n...                                     sampling_rate=8000,\n...                                     sequence_stride=128,\n...                                     fft_length=2048)\n>>> layer(keras.random.uniform(shape=(2, 16000))).shape\n(2, 80, 125)\n\nInput shape:\n    1D (unbatched) or 2D (batched) tensor with shape:`(..., samples)`.\n\nOutput shape:\n    2D (unbatched) or 3D (batched) tensor with\n    shape:`(..., num_mel_bins, time)`.',
    "std_args": [
      {"name": "fft_length", "type": "Any"},
      {"name": "sequence_stride", "type": "Any"},
      {"name": "sequence_length", "type": "Any"},
      {"name": "window", "type": "Any"},
      {"name": "sampling_rate", "type": "Any"},
      {"name": "num_mel_bins", "type": "Any"},
      {"name": "min_freq", "type": "Any"},
      {"name": "max_freq", "type": "Any"},
      {"name": "power_to_db", "type": "Any"},
      {"name": "top_db", "type": "Any"},
      {"name": "mag_exp", "type": "Any"},
      {"name": "min_power", "type": "Any"},
      {"name": "ref_power", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Min": {
    "description": "Element-wise min of each of the input tensors (with Numpy-style broadcasting support). All inputs and outputs must have the same data type. This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "data_0", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Mish": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "MixUp": {
    "description": 'MixUp implements the MixUp data augmentation technique.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nReferences:\n    - [MixUp paper](https://arxiv.org/abs/1710.09412).\n    - [MixUp for Object Detection paper](https://arxiv.org/pdf/1902.04103).\n\nArgs:\n    alpha: Float between 0 and 1. Controls the blending strength.\n           Smaller values mean less mixing, while larger values allow\n           for more  blending between images. Defaults to 0.2,\n           recommended for ImageNet1k classification.\n    seed: Integer. Used to create a random seed.\n\nExample:\n```python\n(images, labels), _ = keras.datasets.cifar10.load_data()\nimages, labels = images[:8], labels[:8]\nlabels = keras.ops.cast(keras.ops.one_hot(labels.flatten(), 10), "float32")\nmix_up = keras.layers.MixUp(alpha=0.2)\noutput = mix_up({"images": images, "labels": labels})\n```',
    "std_args": [
      {"name": "alpha", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Mod": {
    "description": "Performs an element-wise binary modulo operation. The semantics and supported data types depend on the value of the `fmod` attribute which must be `0` (default), or `1`. If the `fmod` attribute is set to `0`, `T` is constrained to integer data types and the semantics follow that of the Python `%`-op...",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
      {"name": "fmod", "type": "int"},
    ],
    "variants": {},
  },
  "Module": {
    "description": "Auto-generated from flax_nnx_code_defs",
    "std_args": [],
    "variants": {},
  },
  "Mse": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "Mul": {
    "description": "Performs element-wise binary multiplication (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md). (Opset 14 change): Extend supported types to include uint8, int8, uint16, and i...",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "MultiSteps": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "MultiStepsState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "MultiTransformState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "MultiheadAttention": {
    "description": "Multi-head attention mechanism.",
    "std_args": [
      {"name": "embed_dim", "type": "Any"},
      {"name": "num_heads", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "add_bias_kv", "type": "Any"},
      {"name": "add_zero_attn", "type": "Any"},
      {"name": "kdim", "type": "Any"},
      {"name": "vdim", "type": "Any"},
      {"name": "batch_first", "type": "Any"},
    ],
    "variants": {},
  },
  "Multinomial": {
    "description": "Generate a tensor of samples from a multinomial distribution according to the probabilities of each of the possible outcomes.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "dtype", "type": "int"},
      {"name": "sample_size", "type": "int"},
      {"name": "seed", "type": "float"},
    ],
    "variants": {},
  },
  "NONE": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Names": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Nll": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "NonNegativeParamsState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "NonZero": {
    "description": "Returns the indices of the elements that are non-zero (in row-major order - by dimension). NonZero behaves similar to numpy.nonzero: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html, but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is differe...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Nop": {
    "description": "Auto-generated from sass_code_defs",
    "std_args": [],
    "variants": {},
  },
  "Normalization": {
    "description": "A preprocessing layer that normalizes continuous features.\n\nThis layer will shift and scale inputs into a distribution centered around\n0 with standard deviation 1. It accomplishes this by precomputing the mean\nand variance of the data, and calling `(input - mean) / sqrt(var)` at\nruntime.\n\nThe mean and variance values for the layer must be either supplied on\nconstruction or learned via `adapt()`. `adapt()` will compute the mean and\nvariance of the data and store them as the layer's weights. `adapt()` should\nbe called before `fit()`, `evaluate()`, or `predict()`.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    axis: Integer, tuple of integers, or None. The axis or axes that should\n        have a separate mean and variance for each index in the shape.\n        For example, if shape is `(None, 5)` and `axis=1`, the layer will\n        track 5 separate mean and variance values for the last axis.\n        If `axis` is set to `None`, the layer will normalize\n        all elements in the input by a scalar mean and variance.\n        When `-1`, the last axis of the input is assumed to be a\n        feature dimension and is normalized per index.\n        Note that in the specific case of batched scalar inputs where\n        the only axis is the batch axis, the default will normalize\n        each index in the batch separately.\n        In this case, consider passing `axis=None`. Defaults to `-1`.\n    mean: The mean value(s) to use during normalization. The passed value(s)\n        will be broadcast to the shape of the kept axes above;\n        if the value(s) cannot be broadcast, an error will be raised when\n        this layer's `build()` method is called.\n        `mean` and `variance` must be specified together.\n    variance: The variance value(s) to use during normalization. The passed\n        value(s) will be broadcast to the shape of the kept axes above;\n        if the value(s) cannot be broadcast, an error will be raised when\n        this layer's `build()` method is called.\n        `mean` and `variance` must be specified together.\n    invert: If `True`, this layer will apply the inverse transformation\n        to its inputs: it would turn a normalized input back into its\n        original form.\n\nExamples:\n\nCalculate a global mean and variance by analyzing the dataset in `adapt()`.\n\n>>> adapt_data = np.array([1., 2., 3., 4., 5.], dtype='float32')\n>>> input_data = np.array([1., 2., 3.], dtype='float32')\n>>> layer = keras.layers.Normalization(axis=None)\n>>> layer.adapt(adapt_data)\n>>> layer(input_data)\narray([-1.4142135, -0.70710677, 0.], dtype=float32)\n\nCalculate a mean and variance for each index on the last axis.\n\n>>> adapt_data = np.array([[0., 7., 4.],\n...                        [2., 9., 6.],\n...                        [0., 7., 4.],\n...                        [2., 9., 6.]], dtype='float32')\n>>> input_data = np.array([[0., 7., 4.]], dtype='float32')\n>>> layer = keras.layers.Normalization(axis=-1)\n>>> layer.adapt(adapt_data)\n>>> layer(input_data)\narray([-1., -1., -1.], dtype=float32)\n\nPass the mean and variance directly.\n\n>>> input_data = np.array([[1.], [2.], [3.]], dtype='float32')\n>>> layer = keras.layers.Normalization(mean=3., variance=2.)\n>>> layer(input_data)\narray([[-1.4142135 ],\n       [-0.70710677],\n       [ 0.        ]], dtype=float32)\n\nUse the layer to de-normalize inputs (after adapting the layer).\n\n>>> adapt_data = np.array([[0., 7., 4.],\n...                        [2., 9., 6.],\n...                        [0., 7., 4.],\n...                        [2., 9., 6.]], dtype='float32')\n>>> input_data = np.array([[1., 2., 3.]], dtype='float32')\n>>> layer = keras.layers.Normalization(axis=-1, invert=True)\n>>> layer.adapt(adapt_data)\n>>> layer(input_data)\narray([2., 10., 8.], dtype=float32)",
    "std_args": [
      {"name": "axis", "type": "Any"},
      {"name": "mean", "type": "Any"},
      {"name": "variance", "type": "Any"},
      {"name": "invert", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Normalize": {
    "description": "Normalize a tensor image with mean and standard deviation.",
    "std_args": [
      {"name": "mean", "type": "Any"},
      {"name": "std", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "Object": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "type", "type": "Any"},
      {"name": "start", "type": "Any"},
      {"name": "end", "type": "Any"},
      {"name": "kv_sep", "type": "Any"},
      {"name": "indent", "type": "Any"},
      {"name": "empty_repr", "type": "Any"},
      {"name": "comment", "type": "Any"},
      {"name": "same_line", "type": "Any"},
    ],
    "variants": {},
  },
  "OneHot": {
    "description": "Produces a one-hot tensor based on inputs. The locations represented by the index values in the 'indices' input tensor will have 'on_value' and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value' are specified as part of required input argument 'values', ...",
    "std_args": [
      {"name": "indices", "type": "Any"},
      {"name": "depth", "type": "Any"},
      {"name": "values", "type": "Any"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "OptState": {
    "description": "Any optimizer state",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "value", "type": "Any"},
      {"name": "is_hijax", "type": "Any"},
      {"name": "has_ref", "type": "Any"},
      {"name": "is_mutable", "type": "Any"},
      {"name": "eager_sharding", "type": "Any"},
      {"name": "metadata", "type": "Any"},
    ],
    "variants": {},
  },
  "OptaxTest": {
    "description": "Test optax can be imported correctly.",
    "std_args": [],
    "variants": {},
  },
  "Optimizer": {
    "description": "Simple train state for the common case with a single Optax optimizer.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "model", "type": "Any"},
      {"name": "tx", "type": "Any"},
      {"name": "wrt", "type": "Any"},
    ],
    "variants": {},
  },
  "Optional": {
    "description": "Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute, or a non-empty value containing the input element.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "type", "type": "Any"},
    ],
    "variants": {},
  },
  "PRelu": {
    "description": "PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`, `f(x) = x for x >= 0`., is applied to the data tensor elementwise. This operator supports **unidirectional broadcasting** (tensor slope should be un...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "slope", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Param": {
    "description": "Container for trainable parameter.",
    "std_args": [
      {"name": "value", "type": "Any"},
    ],
    "variants": {},
  },
  "Parameter": {
    "description": "A kind of Tensor that is to be considered a module parameter.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "data", "type": "Any"},
      {"name": "requires_grad", "type": "Any"},
    ],
    "variants": {},
  },
  "ParameterList": {
    "description": "Holds parameters in a list.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "values", "type": "Any"},
    ],
    "variants": {},
  },
  "PartitionState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Permute": {
    "description": "Permutes the dimensions of the input according to a given pattern.\n\nUseful e.g. connecting RNNs and convnets.\n\nArgs:\n    dims: Tuple of integers. Permutation pattern does not include the\n        batch dimension. Indexing starts at 1.\n        For instance, `(1, 3, 2)` permutes the second and third dimensions\n        of the input.\n\nInput shape:\n    Arbitrary.\n\nOutput shape:\n    Same as the input shape, but with the dimensions re-ordered according\n    to the specified pattern.\n\nExample:\n\n>>> x = keras.Input(shape=(10, 64))\n>>> y = keras.layers.Permute((2, 1))(x)\n>>> y.shape\n(None, 64, 10)",
    "std_args": [
      {"name": "dims", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Pipeline": {
    "description": "Applies a series of layers to an input.\n\nThis class is useful to build a preprocessing pipeline,\nin particular an image data augmentation pipeline.\nCompared to a `Sequential` model, `Pipeline` features\na few important differences:\n\n- It's not a `Model`, just a plain layer.\n- When the layers in the pipeline are compatible\n    with `tf.data`, the pipeline will also\n    remain `tf.data` compatible. That is to say,\n    the pipeline will not attempt to convert\n    its inputs to backend-native tensors\n    when in a tf.data context (unlike a `Sequential`\n    model).\n\nExample:\n\n```python\nfrom keras import layers\npreprocessing_pipeline = layers.Pipeline([\n    layers.AutoContrast(),\n    layers.RandomZoom(0.2),\n    layers.RandomRotation(0.2),\n])\n\n# `ds` is a tf.data.Dataset\npreprocessed_ds = ds.map(\n    preprocessing_pipeline,\n    num_parallel_calls=4,\n)\n```",
    "std_args": [
      {"name": "layers", "type": "Any"},
      {"name": "name", "type": "Any"},
    ],
    "variants": {},
  },
  "Poly": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "request", "type": "Any"},
    ],
    "variants": {},
  },
  "Pow": {
    "description": "Pow takes input data (Tensor<T>) and exponent Tensor, and produces one output data (Tensor<T>) where the function `f(x) = x^exponent`, is applied to the data tensor elementwise. This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broa...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "Y", "type": "Any"},
    ],
    "variants": {},
  },
  "RMSNormalization": {
    "description": "This is RMS normalization defined in ONNX as function as described in the paper https://arxiv.org/pdf/1910.07467. The overall computation can be split into two stages. The root mean squared norm is taken over the last D dimensions, where D is the dimension of normalized_shape. For example, if normal...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "scale", "type": "Any"},
      {"name": "axis", "type": "int"},
      {"name": "epsilon", "type": "float"},
      {"name": "stash_type", "type": "int"},
    ],
    "variants": {},
  },
  "RMSprop": {
    "description": "Root Mean Square Propagation optimizer.",
    "std_args": [
      {"name": "params", "type": "Any"},
      {"name": "lr", "type": "Any"},
      {"name": "rho", "type": "Any"},
      {"name": "eps", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
      {"name": "momentum", "type": "Any"},
      {"name": "centered", "type": "Any"},
    ],
    "variants": {},
  },
  "RNN": {
    "description": "Computes an one-layer simple RNN. This operator is usually supported via some custom implementation such as CuDNN. Notations: * `X` - input tensor * `i` - input gate * `t` - time step (t-1 means previous time step) * `Wi` - W parameter weight matrix for input gate * `Ri` - R recurrence weight matrix...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "W", "type": "Tensor"},
      {"name": "R", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
      {"name": "sequence_lens", "type": "Any"},
      {"name": "initial_h", "type": "Tensor"},
      {"name": "activation_alpha", "type": "List[float]"},
      {"name": "activation_beta", "type": "List[float]"},
      {"name": "activations", "type": "List[str]"},
      {"name": "clip", "type": "float"},
      {"name": "direction", "type": "str"},
      {"name": "hidden_size", "type": "int"},
      {"name": "layout", "type": "int"},
    ],
    "variants": {},
  },
  "RNNBase": {
    "description": "Base class for RNN modules (RNN, LSTM, GRU).",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "input_size", "type": "Any"},
      {"name": "hidden_size", "type": "Any"},
      {"name": "num_layers", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "batch_first", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "bidirectional", "type": "Any"},
      {"name": "proj_size", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "RPC_AVAILABLE": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "RandAugment": {
    "description": "RandAugment performs the Rand Augment operation on input images.\n\nThis layer can be thought of as an all-in-one image augmentation layer. The\npolicy implemented by this layer has been benchmarked extensively and is\neffective on a wide variety of datasets.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nReferences:\n    - [RandAugment](https://arxiv.org/abs/1909.13719)\n\nArgs:\n    value_range: The range of values the input image can take.\n        Default is `(0, 255)`. Typically, this would be `(0, 1)`\n        for normalized images or `(0, 255)` for raw images.\n    num_ops: The number of augmentation operations to apply sequentially\n        to each image. Default is 2.\n    factor: The strength of the augmentation as a normalized value\n        between 0 and 1. Default is 0.5.\n    interpolation: The interpolation method to use for resizing operations.\n        Options include `nearest`, `bilinear`. Default is `bilinear`.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "value_range", "type": "Any"},
      {"name": "num_ops", "type": "Any"},
      {"name": "factor", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomBrightness": {
    "description": "A preprocessing layer which randomly adjusts brightness during training.\n\nThis layer will randomly increase/reduce the brightness for the input RGB\nimages. At inference time, the output will be identical to the input.\nCall the layer with `training=True` to adjust the brightness of the input.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The\n        factor is used to determine the lower bound and upper bound of the\n        brightness adjustment. A float value will be chosen randomly between\n        the limits. When -1.0 is chosen, the output image will be black, and\n        when 1.0 is chosen, the image will be fully white.\n        When only one float is provided, eg, 0.2,\n        then -0.2 will be used for lower bound and 0.2\n        will be used for upper bound.\n    value_range: Optional list/tuple of 2 floats\n        for the lower and upper limit\n        of the values of the input data.\n        To make no change, use `[0.0, 1.0]`, e.g., if the image input\n        has been scaled before this layer. Defaults to `[0.0, 255.0]`.\n        The brightness adjustment will be scaled to this range, and the\n        output values will be clipped to this range.\n    seed: optional integer, for fixed RNG behavior.\n\nInputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel\n    values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)\n\nOutput: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the\n    `factor`. By default, the layer will output floats.\n    The output value will be clipped to the range `[0, 255]`,\n    the valid range of RGB colors, and\n    rescaled based on the `value_range` if needed.\n\nExample:\n\n```python\nrandom_bright = keras.layers.RandomBrightness(factor=0.2)\n\n# An image with shape [2, 2, 3]\nimage = [[[1, 2, 3], [4 ,5 ,6]], [[7, 8, 9], [10, 11, 12]]]\n\n# Assume we randomly select the factor to be 0.1, then it will apply\n# 0.1 * 255 to all the channel\noutput = random_bright(image, training=True)\n\n# output will be int64 with 25.5 added to each channel and round down.\n>>> array([[[26.5, 27.5, 28.5]\n            [29.5, 30.5, 31.5]]\n           [[32.5, 33.5, 34.5]\n            [35.5, 36.5, 37.5]]],\n          shape=(2, 2, 3), dtype=int64)\n```",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomColorDegeneration": {
    "description": "Randomly performs the color degeneration operation on given images.\n\nThe sharpness operation first converts an image to gray scale, then back to\ncolor. It then takes a weighted average between original image and the\ndegenerated image. This makes colors appear more dull.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    factor: A tuple of two floats or a single float.\n        `factor` controls the extent to which the\n        image sharpness is impacted. `factor=0.0` makes this layer perform a\n        no-op operation, while a value of 1.0 uses the degenerated result\n        entirely. Values between 0 and 1 result in linear interpolation\n        between the original image and the sharpened image.\n        Values should be between `0.0` and `1.0`. If a tuple is used, a\n        `factor` is sampled between the two values for every image\n        augmented. If a single float is used, a value between `0.0` and the\n        passed float is sampled. In order to ensure the value is always the\n        same, please pass a tuple with two identical floats: `(0.5, 0.5)`.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomColorJitter": {
    "description": "RandomColorJitter class randomly apply brightness, contrast, saturation\nand hue image processing operation sequentially and randomly on the\ninput.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    value_range: the range of values the incoming images will have.\n        Represented as a two number tuple written [low, high].\n        This is typically either `[0, 1]` or `[0, 255]` depending\n        on how your preprocessing pipeline is set up.\n    brightness_factor: Float or a list/tuple of 2 floats between -1.0\n        and 1.0. The factor is used to determine the lower bound and\n        upper bound of the brightness adjustment. A float value will\n        be chosen randomly between the limits. When -1.0 is chosen,\n        the output image will be black, and when 1.0 is chosen, the\n        image will be fully white. When only one float is provided,\n        eg, 0.2, then -0.2 will be used for lower bound and 0.2 will\n        be used for upper bound.\n    contrast_factor: a positive float represented as fraction of value,\n        or a tuple of size 2 representing lower and upper bound. When\n        represented as a single float, lower = upper. The contrast\n        factor will be randomly picked between `[1.0 - lower, 1.0 +\n        upper]`. For any pixel x in the channel, the output will be\n        `(x - mean) * factor + mean` where `mean` is the mean value\n        of the channel.\n    saturation_factor: A tuple of two floats or a single float. `factor`\n        controls the extent to which the image saturation is impacted.\n        `factor=0.5` makes this layer perform a no-op operation.\n        `factor=0.0` makes the image fully grayscale. `factor=1.0`\n        makes the image fully saturated. Values should be between\n        `0.0` and `1.0`. If a tuple is used, a `factor` is sampled\n        between the two values for every image augmented. If a single\n        float is used, a value between `0.0` and the passed float is\n        sampled. To ensure the value is always the same, pass a tuple\n        with two identical floats: `(0.5, 0.5)`.\n    hue_factor: A single float or a tuple of two floats. `factor`\n        controls the extent to which the image hue is impacted.\n        `factor=0.0` makes this layer perform a no-op operation,\n        while a value of `1.0` performs the most aggressive contrast\n        adjustment available. If a tuple is used, a `factor` is\n        sampled between the two values for every image augmented.\n        If a single float is used, a value between `0.0` and the\n        passed float is sampled. In order to ensure the value is\n        always the same, please pass a tuple with two identical\n        floats: `(0.5, 0.5)`.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "value_range", "type": "Any"},
      {"name": "brightness_factor", "type": "Any"},
      {"name": "contrast_factor", "type": "Any"},
      {"name": "saturation_factor", "type": "Any"},
      {"name": "hue_factor", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomContrast": {
    "description": 'A preprocessing layer which randomly adjusts contrast during training.\n\nThis layer will randomly adjust the contrast of an image or images\nby a random factor. Contrast is adjusted independently\nfor each channel of each image during training.\n\nFor each channel, this layer computes the mean of the image pixels in the\nchannel and then adjusts each component `x` of each pixel to\n`(x - mean) * contrast_factor + mean`.\n\nInput pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and\nin integer or floating point dtype.\nBy default, the layer will output floats.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format.\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format.\n\nArgs:\n    factor: a positive float represented as fraction of value, or a tuple of\n        size 2 representing lower and upper bound.\n        When represented as a single float, lower = upper.\n        The contrast factor will be randomly picked between\n        `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,\n        the output will be `(x - mean) * factor + mean`\n        where `mean` is the mean value of the channel.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomCrop": {
    "description": "Crop the given image at a random location.",
    "std_args": [
      {"name": "size", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "pad_if_needed", "type": "Any"},
      {"name": "fill", "type": "Any"},
      {"name": "padding_mode", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomElasticTransform": {
    "description": 'A preprocessing layer that applies random elastic transformations.\n\nThis layer distorts input images by applying elastic deformations,\nsimulating a physically realistic transformation. The magnitude of the\ndistortion is controlled by the `scale` parameter, while the `factor`\ndetermines the probability of applying the transformation.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    factor: A single float or a tuple of two floats.\n        `factor` controls the probability of applying the transformation.\n        - `factor=0.0` ensures no erasing is applied.\n        - `factor=1.0` means erasing is always applied.\n        - If a tuple `(min, max)` is provided, a probability value\n          is sampled between `min` and `max` for each image.\n        - If a single float is provided, a probability is sampled\n          between `0.0` and the given float.\n        Default is 1.0.\n    scale: A float or a tuple of two floats defining the magnitude of\n        the distortion applied.\n        - If a tuple `(min, max)` is provided, a random scale value is\n          sampled within this range.\n        - If a single float is provided, a random scale value is sampled\n          between `0.0` and the given float.\n        Default is 1.0.\n    interpolation: Interpolation mode. Supported values: `"nearest"`,\n        `"bilinear"`.\n    fill_mode: Points outside the boundaries of the input are filled\n        according to the given mode. Available methods are `"constant"`,\n        `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.\n        - `"reflect"`: `(d c b a | a b c d | d c b a)`\n            The input is extended by reflecting about the edge of the last\n            pixel.\n        - `"constant"`: `(k k k k | a b c d | k k k k)`\n            The input is extended by filling all values beyond\n            the edge with the same constant value k specified by\n            `fill_value`.\n        - `"wrap"`: `(a b c d | a b c d | a b c d)`\n            The input is extended by wrapping around to the opposite edge.\n        - `"nearest"`: `(a a a a | a b c d | d d d d)`\n            The input is extended by the nearest pixel.\n        Note that when using torch backend, `"reflect"` is redirected to\n        `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not\n        support `"reflect"`.\n        Note that torch backend does not support `"wrap"`.\n    fill_value: a float represents the value to be filled outside the\n        boundaries when `fill_mode="constant"`.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "scale", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "fill_mode", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomErasing": {
    "description": "Random Erasing data augmentation technique.\n\nRandom Erasing is a data augmentation method where random patches of\nan image are erased (replaced by a constant value or noise)\nduring training to improve generalization.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nReferences:\n   - [Random Erasing paper](https://arxiv.org/abs/1708.04896).\n\nArgs:\n    factor: A single float or a tuple of two floats.\n        `factor` controls the probability of applying the transformation.\n        - `factor=0.0` ensures no erasing is applied.\n        - `factor=1.0` means erasing is always applied.\n        - If a tuple `(min, max)` is provided, a probability value\n          is sampled between `min` and `max` for each image.\n        - If a single float is provided, a probability is sampled\n          between `0.0` and the given float.\n        Default is 1.0.\n    scale: A tuple of two floats representing the aspect ratio range of\n        the erased patch. This defines the width-to-height ratio of\n        the patch to be erased. It can help control the rw shape of\n        the erased region. Default is (0.02, 0.33).\n    fill_value: A value to fill the erased region with. This can be set to\n        a constant value or `None` to sample a random value\n        from a normal distribution. Default is `None`.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "scale", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomFlip": {
    "description": 'A preprocessing layer which randomly flips images during training.\n\nThis layer will flip the images horizontally and or vertically based on the\n`mode` attribute. During inference time, the output will be identical to\ninput. Call the layer with `training=True` to flip the input.\nInput pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and\nof integer or floating point dtype.\nBy default, the layer will output floats.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format.\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format.\n\nArgs:\n    mode: String indicating which flip mode to use. Can be `"horizontal"`,\n        `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a\n        left-right flip and `"vertical"` is a top-bottom flip. Defaults to\n        `"horizontal_and_vertical"`\n    seed: Integer. Used to create a random seed.\n    **kwargs: Base layer keyword arguments, such as\n        `name` and `dtype`.',
    "std_args": [
      {"name": "mode", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomGaussianBlur": {
    "description": "Applies random Gaussian blur to images for data augmentation.\n\nThis layer performs a Gaussian blur operation on input images with a\nrandomly selected degree of blurring, controlled by the `factor` and\n`sigma` arguments.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    factor: A single float or a tuple of two floats.\n        `factor` controls the extent to which the image hue is impacted.\n        `factor=0.0` makes this layer perform a no-op operation,\n        while a value of `1.0` performs the most aggressive\n        blurring available. If a tuple is used, a `factor` is\n        sampled between the two values for every image augmented. If a\n        single float is used, a value between `0.0` and the passed float is\n        sampled. Default is 1.0.\n    kernel_size: Integer. Size of the Gaussian kernel used for blurring.\n        Must be an odd integer. Default is 3.\n    sigma: Float or tuple of two floats. Standard deviation of the Gaussian\n        kernel. Controls the intensity of the blur. If a tuple is provided,\n        a value is sampled between the two for each image. Default is 1.0.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "sigma", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomGrayscale": {
    "description": 'Preprocessing layer for random conversion of RGB images to grayscale.\n\nThis layer randomly converts input images to grayscale with a specified\nfactor. When applied, it maintains the original number of channels\nbut sets all channels to the same grayscale value. This can be useful\nfor data augmentation and training models to be robust to color\nvariations.\n\nThe conversion preserves the perceived luminance of the original color\nimage using standard RGB to grayscale conversion coefficients. Images\nthat are not selected for conversion remain unchanged.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    factor: Float between 0 and 1, specifying the factor of\n        converting each image to grayscale. Defaults to 0.5. A value of\n        1.0 means all images will be converted, while 0.0 means no images\n        will be converted.\n    data_format: String, one of `"channels_last"` (default) or\n        `"channels_first"`. The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch, height, width, channels)` while `"channels_first"`\n        corresponds to inputs with shape\n        `(batch, channels, height, width)`.\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format,\n    or `(..., channels, height, width)`, in `"channels_first"` format.\n\nOutput shape:\n    Same as input shape. The output maintains the same number of channels\n    as the input, even for grayscale-converted images where all channels\n    will have the same value.',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomHorizontalFlip": {
    "description": "Horizontally flip the image randomly with a given probability.",
    "std_args": [
      {"name": "p", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomHue": {
    "description": 'Randomly adjusts the hue on given images.\n\nThis layer will randomly increase/reduce the hue for the input RGB\nimages.\n\nThe image hue is adjusted by converting the image(s) to HSV and rotating the\nhue channel (H) by delta. The image is then converted back to RGB.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    factor: A single float or a tuple of two floats.\n        `factor` controls the extent to which the\n        image hue is impacted. `factor=0.0` makes this layer perform a\n        no-op operation, while a value of `1.0` performs the most aggressive\n        contrast adjustment available. If a tuple is used, a `factor` is\n        sampled between the two values for every image augmented. If a\n        single float is used, a value between `0.0` and the passed float is\n        sampled. In order to ensure the value is always the same, please\n        pass a tuple with two identical floats: `(0.5, 0.5)`.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.\n\nExample:\n\n```python\n(images, labels), _ = keras.datasets.cifar10.load_data()\nrandom_hue = keras.layers.RandomHue(factor=0.5, value_range=[0, 1])\nimages = keras.ops.cast(images, "float32")\naugmented_images_batch = random_hue(images[:8])\n```',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomInvert": {
    "description": "Preprocessing layer for random inversion of image colors.\n\nThis layer randomly inverts the colors of input images with a specified\nprobability range. When applied, each image has a chance of having its\ncolors inverted, where the pixel values are transformed to their\ncomplementary values. Images that are not selected for inversion\nremain unchanged.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    factor: A single float or a tuple of two floats.\n        `factor` controls the probability of inverting the image colors.\n        If a tuple is provided, the value is sampled between the two values\n        for each image, where `factor[0]` is the minimum and `factor[1]` is\n        the maximum probability. If a single float is provided, a value\n        between `0.0` and the provided float is sampled.\n        Defaults to `(0, 1)`.\n    value_range: a tuple or a list of two elements. The first value\n        represents the lower bound for values in passed images, the second\n        represents the upper bound. Images passed to the layer should have\n        values within `value_range`. Defaults to `(0, 255)`.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomPerspective": {
    "description": 'A preprocessing layer that applies random perspective transformations.\n\nThis layer distorts the perspective of input images by shifting their\ncorner points, simulating a 3D-like transformation. The amount of distortion\nis controlled by the `factor` and `scale` parameters.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    factor: A float or a tuple of two floats.\n        Represents the probability of applying the perspective\n        transformation to each image in the batch.\n        - `factor=0.0` ensures no transformation is applied.\n        - `factor=1.0` means the transformation is always applied.\n        - If a tuple `(min, max)` is provided, a probability is randomly\n          sampled between `min` and `max` for each image.\n        - If a single float is given, the probability is sampled between\n          `0.0` and the provided float.\n        Default is 1.0.\n    scale: A float defining the relative amount of perspective shift.\n        Determines how much the image corners are displaced, affecting\n        the intensity of the perspective effect.\n    interpolation: Interpolation mode. Supported values: `"nearest"`,\n        `"bilinear"`.\n    fill_value: a float represents the value to be filled outside the\n        boundaries when `fill_mode="constant"`.\n    seed: Integer. Used to create a random seed.',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "scale", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomPosterization": {
    "description": "Reduces the number of bits for each color channel.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nReferences:\n- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)\n- [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)\n\nArgs:\n    value_range: a tuple or a list of two elements. The first value\n        represents the lower bound for values in passed images, the second\n        represents the upper bound. Images passed to the layer should have\n        values within `value_range`. Defaults to `(0, 255)`.\n    factor: integer, the number of bits to keep for each channel. Must be a\n        value between 1-8.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomRotation": {
    "description": 'A preprocessing layer which randomly rotates images during training.\n\nThis layer will apply random rotations to each image, filling empty space\naccording to `fill_mode`.\n\nBy default, random rotations are only applied during training.\nAt inference time, the layer does nothing. If you need to apply random\nrotations at inference time, pass `training=True` when calling the layer.\n\nInput pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and\nof integer or floating point dtype.\nBy default, the layer will output floats.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format\n\nArgs:\n    factor: a float represented as fraction of 2 Pi, or a tuple of size 2\n        representing lower and upper bound for rotating clockwise and\n        counter-clockwise. A positive values means rotating\n        counter clock-wise,\n        while a negative value means clock-wise.\n        When represented as a single\n        float, this value is used for both the upper and lower bound.\n        For instance, `factor=(-0.2, 0.3)`\n        results in an output rotation by a random\n        amount in the range `[-20% * 360, 30% * 360]`.\n        `factor=0.2` results in an\n        output rotating by a random amount\n        in the range `[-20% * 360, 20% * 360]`.\n    fill_mode: Points outside the boundaries of the input are filled\n        according to the given mode\n        (one of `{"constant", "reflect", "wrap", "nearest"}`).\n        - *reflect*: `(d c b a | a b c d | d c b a)`\n            The input is extended by reflecting about\n            the edge of the last pixel.\n        - *constant*: `(k k k k | a b c d | k k k k)`\n            The input is extended by\n            filling all values beyond the edge with\n            the same constant value k = 0.\n        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by\n            wrapping around to the opposite edge.\n        - *nearest*: `(a a a a | a b c d | d d d d)`\n            The input is extended by the nearest pixel.\n    interpolation: Interpolation mode. Supported values: `"nearest"`,\n        `"bilinear"`.\n    seed: Integer. Used to create a random seed.\n    fill_value: a float represents the value to be filled outside\n        the boundaries when `fill_mode="constant"`.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "fill_mode", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomSaturation": {
    "description": 'Randomly adjusts the saturation on given images.\n\nThis layer will randomly increase/reduce the saturation for the input RGB\nimages.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    factor: A tuple of two floats or a single float.\n        `factor` controls the extent to which the image saturation\n        is impacted. `factor=0.5` makes this layer perform a no-op\n        operation. `factor=0.0` makes the image fully grayscale.\n        `factor=1.0` makes the image fully saturated. Values should\n        be between `0.0` and `1.0`. If a tuple is used, a `factor`\n        is sampled between the two values for every image augmented.\n        If a single float is used, a value between `0.0` and the passed\n        float is sampled. To ensure the value is always the same,\n        pass a tuple with two identical floats: `(0.5, 0.5)`.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.\n\nExample:\n```python\n(images, labels), _ = keras.datasets.cifar10.load_data()\nimages = images.astype("float32")\nrandom_saturation = keras.layers.RandomSaturation(factor=0.2)\naugmented_images = random_saturation(images)\n```',
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomSharpness": {
    "description": "Randomly performs the sharpness operation on given images.\n\nThe sharpness operation first performs a blur, then blends between the\noriginal image and the processed image. This operation adjusts the clarity\nof the edges in an image, ranging from blurred to enhanced sharpness.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    factor: A tuple of two floats or a single float.\n        `factor` controls the extent to which the image sharpness\n        is impacted. `factor=0.0` results in a fully blurred image,\n        `factor=0.5` applies no operation (preserving the original image),\n        and `factor=1.0` enhances the sharpness beyond the original. Values\n        should be between `0.0` and `1.0`. If a tuple is used, a `factor`\n        is sampled between the two values for every image augmented.\n        If a single float is used, a value between `0.0` and the passed\n        float is sampled. To ensure the value is always the same,\n        pass a tuple with two identical floats: `(0.5, 0.5)`.\n    value_range: the range of values the incoming images will have.\n        Represented as a two-number tuple written `[low, high]`. This is\n        typically either `[0, 1]` or `[0, 255]` depending on how your\n        preprocessing pipeline is set up.\n    seed: Integer. Used to create a random seed.",
    "std_args": [
      {"name": "factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomShear": {
    "description": 'A preprocessing layer that randomly applies shear transformations to\nimages.\n\nThis layer shears the input images along the x-axis and/or y-axis by a\nrandomly selected factor within the specified range. The shear\ntransformation is applied to each image independently in a batch. Empty\nregions created during the transformation are filled according to the\n`fill_mode` and `fill_value` parameters.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    x_factor: A tuple of two floats. For each augmented image, a value\n        is sampled from the provided range. If a float is passed, the\n        range is interpreted as `(0, x_factor)`. Values represent a\n        percentage of the image to shear over. For example, 0.3 shears\n        pixels up to 30% of the way across the image. All provided values\n        should be positive.\n    y_factor: A tuple of two floats. For each augmented image, a value\n        is sampled from the provided range. If a float is passed, the\n        range is interpreted as `(0, y_factor)`. Values represent a\n        percentage of the image to shear over. For example, 0.3 shears\n        pixels up to 30% of the way across the image. All provided values\n        should be positive.\n    interpolation: Interpolation mode. Supported values: `"nearest"`,\n        `"bilinear"`.\n    fill_mode: Points outside the boundaries of the input are filled\n        according to the given mode. Available methods are `"constant"`,\n        `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.\n        - `"reflect"`: `(d c b a | a b c d | d c b a)`\n            The input is extended by reflecting about the edge of the\n            last pixel.\n        - `"constant"`: `(k k k k | a b c d | k k k k)`\n            The input is extended by filling all values beyond the edge\n            with the same constant value `k` specified by `fill_value`.\n        - `"wrap"`: `(a b c d | a b c d | a b c d)`\n            The input is extended by wrapping around to the opposite edge.\n        - `"nearest"`: `(a a a a | a b c d | d d d d)`\n            The input is extended by the nearest pixel.\n        Note that when using torch backend, `"reflect"` is redirected to\n        `"mirror"` `(c d c b | a b c d | c b a b)` because torch does\n        not support `"reflect"`.\n        Note that torch backend does not support `"wrap"`.\n    fill_value: A float representing the value to be filled outside the\n        boundaries when `fill_mode="constant"`.\n    seed: Integer. Used to create a random seed.',
    "std_args": [
      {"name": "x_factor", "type": "Any"},
      {"name": "y_factor", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "fill_mode", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomTranslation": {
    "description": 'A preprocessing layer which randomly translates images during training.\n\nThis layer will apply random translations to each image during training,\nfilling empty space according to `fill_mode`.\n\nInput pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and\nof integer or floating point dtype. By default, the layer will output\nfloats.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format,\n    or `(..., channels, height, width)`, in `"channels_first"` format.\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., target_height, target_width, channels)`,\n    or `(..., channels, target_height, target_width)`,\n    in `"channels_first"` format.\n\nArgs:\n    height_factor: a float represented as fraction of value, or a tuple of\n        size 2 representing lower and upper bound for shifting vertically. A\n        negative value means shifting image up, while a positive value means\n        shifting image down. When represented as a single positive float,\n        this value is used for both the upper and lower bound. For instance,\n        `height_factor=(-0.2, 0.3)` results in an output shifted by a random\n        amount in the range `[-20%, +30%]`. `height_factor=0.2` results in\n        an output height shifted by a random amount in the range\n        `[-20%, +20%]`.\n    width_factor: a float represented as fraction of value, or a tuple of\n        size 2 representing lower and upper bound for shifting horizontally.\n        A negative value means shifting image left, while a positive value\n        means shifting image right. When represented as a single positive\n        float, this value is used for both the upper and lower bound. For\n        instance, `width_factor=(-0.2, 0.3)` results in an output shifted\n        left by 20%, and shifted right by 30%. `width_factor=0.2` results\n        in an output height shifted left or right by 20%.\n    fill_mode: Points outside the boundaries of the input are filled\n        according to the given mode. Available methods are `"constant"`,\n        `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.\n        - `"reflect"`: `(d c b a | a b c d | d c b a)`\n            The input is extended by reflecting about the edge of the last\n            pixel.\n        - `"constant"`: `(k k k k | a b c d | k k k k)`\n            The input is extended by filling all values beyond\n            the edge with the same constant value k specified by\n            `fill_value`.\n        - `"wrap"`: `(a b c d | a b c d | a b c d)`\n            The input is extended by wrapping around to the opposite edge.\n        - `"nearest"`: `(a a a a | a b c d | d d d d)`\n            The input is extended by the nearest pixel.\n        Note that when using torch backend, `"reflect"` is redirected to\n        `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not\n        support `"reflect"`.\n        Note that torch backend does not support `"wrap"`.\n    interpolation: Interpolation mode. Supported values: `"nearest"`,\n        `"bilinear"`.\n    seed: Integer. Used to create a random seed.\n    fill_value: a float represents the value to be filled outside the\n        boundaries when `fill_mode="constant"`.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.',
    "std_args": [
      {"name": "height_factor", "type": "Any"},
      {"name": "width_factor", "type": "Any"},
      {"name": "fill_mode", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomVerticalFlip": {
    "description": "Vertically flip the image randomly with a given probability.",
    "std_args": [
      {"name": "p", "type": "Any"},
    ],
    "variants": {},
  },
  "RandomZoom": {
    "description": 'A preprocessing layer which randomly zooms images during training.\n\nThis layer will randomly zoom in or out on each axis of an image\nindependently, filling empty space according to `fill_mode`.\n\nInput pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and\nof integer or floating point dtype.\nBy default, the layer will output floats.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format,\n    or `(..., channels, height, width)`, in `"channels_first"` format.\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., target_height, target_width, channels)`,\n    or `(..., channels, target_height, target_width)`,\n    in `"channels_first"` format.\n\nArgs:\n    height_factor: a float represented as fraction of value, or a tuple of\n        size 2 representing lower and upper bound for zooming vertically.\n        When represented as a single float, this value is used for both the\n        upper and lower bound. A positive value means zooming out, while a\n        negative value means zooming in. For instance,\n        `height_factor=(0.2, 0.3)` result in an output zoomed out by a\n        random amount in the range `[+20%, +30%]`.\n        `height_factor=(-0.3, -0.2)` result in an output zoomed in by a\n        random amount in the range `[+20%, +30%]`.\n    width_factor: a float represented as fraction of value, or a tuple of\n        size 2 representing lower and upper bound for zooming horizontally.\n        When represented as a single float, this value is used for both the\n        upper and lower bound. For instance, `width_factor=(0.2, 0.3)`\n        result in an output zooming out between 20% to 30%.\n        `width_factor=(-0.3, -0.2)` result in an output zooming in between\n        20% to 30%. `None` means i.e., zooming vertical and horizontal\n        directions by preserving the aspect ratio. Defaults to `None`.\n    fill_mode: Points outside the boundaries of the input are filled\n        according to the given mode. Available methods are `"constant"`,\n        `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"reflect"`.\n        - `"reflect"`: `(d c b a | a b c d | d c b a)`\n            The input is extended by reflecting about the edge of the last\n            pixel.\n        - `"constant"`: `(k k k k | a b c d | k k k k)`\n            The input is extended by filling all values beyond\n            the edge with the same constant value k specified by\n            `fill_value`.\n        - `"wrap"`: `(a b c d | a b c d | a b c d)`\n            The input is extended by wrapping around to the opposite edge.\n        - `"nearest"`: `(a a a a | a b c d | d d d d)`\n            The input is extended by the nearest pixel.\n        Note that when using torch backend, `"reflect"` is redirected to\n        `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not\n        support `"reflect"`.\n        Note that torch backend does not support `"wrap"`.\n    interpolation: Interpolation mode. Supported values: `"nearest"`,\n        `"bilinear"`.\n    seed: Integer. Used to create a random seed.\n    fill_value: a float that represents the value to be filled outside\n        the boundaries when `fill_mode="constant"`.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.\n\nExample:\n\n>>> input_img = np.random.random((32, 224, 224, 3))\n>>> layer = keras.layers.RandomZoom(.5, .2)\n>>> out_img = layer(input_img)',
    "std_args": [
      {"name": "height_factor", "type": "Any"},
      {"name": "width_factor", "type": "Any"},
      {"name": "fill_mode", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ReLU": {
    "description": "Rectified Linear Unit.",
    "std_args": [],
    "variants": {},
  },
  "Reciprocal": {
    "description": "Reciprocal takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the reciprocal is, y = 1/x, is applied to the tensor elementwise.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Relu": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Relu6": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "RepeatVector": {
    "description": "Repeats the input n times.\n\nExample:\n\n>>> x = keras.Input(shape=(32,))\n>>> y = keras.layers.RepeatVector(3)(x)\n>>> y.shape\n(None, 3, 32)\n\nArgs:\n    n: Integer, repetition factor.\n\nInput shape:\n    2D tensor with shape `(batch_size, features)`.\n\nOutput shape:\n    3D tensor with shape `(batch_size, n, features)`.",
    "std_args": [
      {"name": "n", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Rescaling": {
    "description": "A preprocessing layer which rescales input values to a new range.\n\nThis layer rescales every value of an input (often an image) by multiplying\nby `scale` and adding `offset`.\n\nFor instance:\n\n1. To rescale an input in the `[0, 255]` range\nto be in the `[0, 1]` range, you would pass `scale=1./255`.\n\n2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,\nyou would pass `scale=1./127.5, offset=-1`.\n\nThe rescaling is applied both during training and inference. Inputs can be\nof integer or floating point dtype, and by default the layer will output\nfloats.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    scale: Float, the scale to apply to the inputs.\n    offset: Float, the offset to apply to the inputs.\n    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.",
    "std_args": [
      {"name": "scale", "type": "Any"},
      {"name": "offset", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Reshape": {
    "description": "Reshape the input tensor similar to numpy.reshape. First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor. At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and th...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "shape", "type": "int"},
      {"name": "allowzero", "type": "int"},
    ],
    "variants": {},
  },
  "Resize": {
    "description": "Resize the input image to the given size.",
    "std_args": [
      {"name": "size", "type": "Any"},
    ],
    "variants": {},
  },
  "Resizing": {
    "description": 'A preprocessing layer which resizes images.\n\nThis layer resizes an image input to a target height and width. The input\nshould be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`\nformat. Input pixel values can be of any range\n(e.g. `[0., 1.)` or `[0, 255]`).\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you\'re using).\n\nInput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., height, width, channels)`, in `"channels_last"` format,\n    or `(..., channels, height, width)`, in `"channels_first"` format.\n\nOutput shape:\n    3D (unbatched) or 4D (batched) tensor with shape:\n    `(..., target_height, target_width, channels)`,\n    or `(..., channels, target_height, target_width)`,\n    in `"channels_first"` format.\n\nArgs:\n    height: Integer, the height of the output shape.\n    width: Integer, the width of the output shape.\n    interpolation: String, the interpolation method.\n        Supports `"bilinear"`, `"nearest"`, `"bicubic"`,\n        `"lanczos3"`, `"lanczos5"`. Defaults to `"bilinear"`.\n    crop_to_aspect_ratio: If `True`, resize the images without aspect\n        ratio distortion. When the original aspect ratio differs\n        from the target aspect ratio, the output image will be\n        cropped so as to return the\n        largest possible window in the image (of size `(height, width)`)\n        that matches the target aspect ratio. By default\n        (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.\n    pad_to_aspect_ratio: If `True`, pad the images without aspect\n        ratio distortion. When the original aspect ratio differs\n        from the target aspect ratio, the output image will be\n        evenly padded on the short side.\n    fill_mode: When using `pad_to_aspect_ratio=True`, padded areas\n        are filled according to the given mode. Only `"constant"` is\n        supported at this time\n        (fill with constant value, equal to `fill_value`).\n    fill_value: Float. Padding value to use when `pad_to_aspect_ratio=True`.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json`. If you never set it, then it will be\n        `"channels_last"`.\n    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.',
    "std_args": [
      {"name": "height", "type": "Any"},
      {"name": "width", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "crop_to_aspect_ratio", "type": "Any"},
      {"name": "pad_to_aspect_ratio", "type": "Any"},
      {"name": "fill_mode", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
      {"name": "antialias", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ReversibleEmbedding": {
    "description": 'An embedding layer which can project backwards to the input dim.\n\nThis layer is an extension of `keras.layers.Embedding` for language models.\nThis layer can be called "in reverse" with `reverse=True`, in which case the\nlayer will linearly project from `output_dim` back to `input_dim`.\n\nBy default, the reverse projection will use the transpose of the\n`embeddings` weights to project to `input_dim` (weights are "tied"). If\n`tie_weights=False`, the model will use a separate, trainable variable for\nreverse projection.\n\nThis layer has no bias terms.\n\nArgs:\n    input_dim: Integer. Size of the vocabulary,\n        i.e. maximum integer index + 1.\n    output_dim: Integer. Dimension of the dense embedding.\n    tie_weights: Boolean, whether or not the matrix for embedding and\n        the matrix for the `reverse` projection should share the same\n        weights.\n    embeddings_initializer: Initializer for the `embeddings`\n        matrix (see `keras.initializers`).\n    embeddings_regularizer: Regularizer function applied to\n        the `embeddings` matrix (see `keras.regularizers`).\n    embeddings_constraint: Constraint function applied to\n        the `embeddings` matrix (see `keras.constraints`).\n    mask_zero: Boolean, whether or not the input value 0 is a special\n        "padding" value that should be masked out.\n    reverse_dtype: The dtype for the reverse projection computation.\n        Defaults to the `compute_dtype` of the layer.\n    logit_soft_cap: If `logit_soft_cap` is set and `reverse=True`, the\n        output logits will be scaled by\n        `tanh(logits / logit_soft_cap) * logit_soft_cap`. This narrows the\n        range of output logits and can improve training.\n    **kwargs: other keyword arguments passed to `keras.layers.Embedding`,\n        including `name`, `trainable`, `dtype` etc.\n\nCall arguments:\n    inputs: The tensor inputs to the layer.\n    reverse: Boolean. If `True` the layer will perform a linear projection\n        from `output_dim` to `input_dim`, instead of a normal embedding\n        call. Default to `False`.\n\nExample:\n```python\nbatch_size = 16\nvocab_size = 100\nhidden_dim = 32\nseq_length = 50\n\n# Generate random inputs.\ntoken_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))\n\nembedding = keras.layers.ReversibleEmbedding(vocab_size, hidden_dim)\n# Embed tokens to shape `(batch_size, seq_length, hidden_dim)`.\nhidden_states = embedding(token_ids)\n# Project hidden states to shape `(batch_size, seq_length, vocab_size)`.\nlogits = embedding(hidden_states, reverse=True)\n```\n\nReferences:\n- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)\n- [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)',
    "std_args": [
      {"name": "input_dim", "type": "Any"},
      {"name": "output_dim", "type": "Any"},
      {"name": "tie_weights", "type": "Any"},
      {"name": "embeddings_initializer", "type": "Any"},
      {"name": "embeddings_regularizer", "type": "Any"},
      {"name": "embeddings_constraint", "type": "Any"},
      {"name": "mask_zero", "type": "Any"},
      {"name": "reverse_dtype", "type": "Any"},
      {"name": "logit_soft_cap", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Rmsnorm": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "dtype", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "Rnncellbase": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Round": {
    "description": "Round takes one input Tensor and rounds the values, element-wise, meaning it finds the nearest integer for each value. In case of halves, the rule is to round them to the nearest even integer. If input x is integral, +0, -0, NaN,  or infinite, x itself is returned. The output tensor has the same sha...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "SGD": {
    "description": "Stochastic Gradient Descent optimizer.",
    "std_args": [
      {"name": "params", "type": "Any"},
      {"name": "lr", "type": "Any"},
      {"name": "momentum", "type": "Any"},
      {"name": "dampening", "type": "Any"},
      {"name": "weight_decay", "type": "Any"},
      {"name": "nesterov", "type": "Any"},
    ],
    "variants": {},
  },
  "STFT": {
    "description": "Computes the Short-time Fourier Transform of the signal.",
    "std_args": [
      {"name": "signal", "type": "Any"},
      {"name": "frame_step", "type": "Any"},
      {"name": "window", "type": "Any"},
      {"name": "frame_length", "type": "Any"},
      {"name": "onesided", "type": "int"},
    ],
    "variants": {},
  },
  "STFTSpectrogram": {
    "description": 'Layer to compute the Short-Time Fourier Transform (STFT) on a 1D signal.\n\nA layer that computes Spectrograms of the input signal to produce\na spectrogram. This layers utilizes Short-Time Fourier Transform (STFT) by\nThe layer computes Spectrograms based on STFT by utilizing convolution\nkernels, which allows parallelization on GPUs and trainable kernels for\nfine-tuning support. This layer allows different modes of output\n(e.g., log-scaled magnitude, phase, power spectral density, etc.) and\nprovides flexibility in windowing, padding, and scaling options for the\nSTFT calculation.\n\nExamples:\n\nApply it as a non-trainable preprocessing layer on 3 audio tracks of\n1 channel, 10 seconds and sampled at 16 kHz.\n\n>>> layer = keras.layers.STFTSpectrogram(\n...     mode=\'log\',\n...     frame_length=256,\n...     frame_step=128,   # 50% overlap\n...     fft_length=512,\n...     window="hann",\n...     padding="valid",\n...     trainable=False,  # non-trainable, preprocessing only\n... )\n>>> layer(keras.random.uniform(shape=(3, 160000, 1))).shape\n(3, 1249, 257)\n\nApply it as a trainable processing layer on 3 stereo audio tracks of\n2 channels, 10 seconds and sampled at 16 kHz. This is initialized as the\nnon-trainable layer, but then can be trained jointly within a model.\n\n>>> layer = keras.layers.STFTSpectrogram(\n...     mode=\'log\',\n...     frame_length=256,\n...     frame_step=128,    # 50% overlap\n...     fft_length=512,\n...     window="hamming",  # hamming windowing function\n...     padding="same",    # padding to preserve the time dimension\n...     trainable=True,    # trainable, this is the default in keras\n... )\n>>> layer(keras.random.uniform(shape=(3, 160000, 2))).shape\n(3, 1250, 514)\n\nSimilar to the last example, but add an extra dimension so the output is\nan image to be used with image models. We apply this here on a signal of\n3 input channels to output an image tensor, hence is directly applicable\nwith an image model.\n\n>>> layer = keras.layers.STFTSpectrogram(\n...     mode=\'log\',\n...     frame_length=256,\n...     frame_step=128,\n...     fft_length=512,\n...     padding="same",\n...     expand_dims=True,  # this adds the extra dimension\n... )\n>>> layer(keras.random.uniform(shape=(3, 160000, 3))).shape\n(3, 1250, 257, 3)\n\nArgs:\n    mode: String, the output type of the spectrogram. Can be one of\n        `"log"`, `"magnitude`", `"psd"`, `"real`", `"imag`", `"angle`",\n        `"stft`". Defaults to `"log`".\n    frame_length: Integer, The length of each frame (window) for STFT in\n        samples. Defaults to 256.\n    frame_step: Integer, the step size (hop length) between\n        consecutive frames. If not provided, defaults to half the\n        frame_length. Defaults to `frame_length // 2`.\n    fft_length: Integer, the size of frequency bins used in the Fast-Fourier\n        Transform (FFT) to apply to each frame. Should be greater than or\n        equal to `frame_length`.  Recommended to be a power of two. Defaults\n        to the smallest power of two that is greater than or equal\n        to `frame_length`.\n    window: (String or array_like), the windowing function to apply to each\n        frame. Can be `"hann`" (default), `"hamming`", or a custom window\n        provided as an array_like.\n    periodic: Boolean, if True, the window function will be treated as\n        periodic. Defaults to `False`.\n    scaling: String, type of scaling applied to the window. Can be\n        `"density`", `"spectrum`", or None. Default is `"density`".\n    padding: String, padding strategy. Can be `"valid`" or `"same`".\n        Defaults to `"valid"`.\n    expand_dims: Boolean, if True, will expand the output into spectrograms\n        into two dimensions to be compatible with image models.\n        Defaults to `False`.\n    data_format: String, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, weight)`. Defaults to `"channels_last"`.\n\nRaises:\n    ValueError: If an invalid value is provided for `"mode`", `"scaling`",\n        `"padding`", or other input arguments.\n    TypeError: If the input data type is not one of `"float16`",\n        `"float32`", or `"float64`".\n\nInput shape:\n    A 3D tensor of shape `(batch_size, time_length, input_channels)`, if\n    `data_format=="channels_last"`, and of shape\n    `(batch_size, input_channels, time_length)` if\n    `data_format=="channels_first"`, where `time_length` is the length of\n    the input signal, and `input_channels` is the number of input channels.\n    The same kernels are applied to each channel independently.\n\nOutput shape:\n    If `data_format=="channels_first" and not expand_dims`, a 3D tensor:\n        `(batch_size, input_channels * freq_channels, new_time_length)`\n    If `data_format=="channels_last" and not expand_dims`, a 3D tensor:\n        `(batch_size, new_time_length, input_channels * freq_channels)`\n    If `data_format=="channels_first" and expand_dims`, a 4D tensor:\n        `(batch_size, input_channels, new_time_length, freq_channels)`\n    If `data_format=="channels_last" and expand_dims`, a 4D tensor:\n        `(batch_size, new_time_length, freq_channels, input_channels)`\n\n    where `new_time_length` depends on the padding, and `freq_channels` is\n    the number of FFT bins `(fft_length // 2 + 1)`.',
    "std_args": [
      {"name": "mode", "type": "Any"},
      {"name": "frame_length", "type": "Any"},
      {"name": "frame_step", "type": "Any"},
      {"name": "fft_length", "type": "Any"},
      {"name": "window", "type": "Any"},
      {"name": "periodic", "type": "Any"},
      {"name": "scaling", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "expand_dims", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "STRING": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Save": {
    "description": "Serialize object to disk.",
    "std_args": [
      {"name": "obj", "type": "Any"},
      {"name": "f", "type": "Any"},
    ],
    "variants": {},
  },
  "ScaleByTrustRatioState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ScaleState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Scan": {
    "description": "Scan can be used to iterate over one or more scan_input tensors, constructing zero or more scan_output tensors. It combines ideas from general recurrences, functional programming constructs such as scan, fold, map, and zip, and is intended to enable generalizations of RNN-like constructs for sequenc...",
    "std_args": [
      {"name": "initial_state_and_scan_inputs", "type": "Any"},
      {"name": "body", "type": "Any"},
      {"name": "num_scan_inputs", "type": "int"},
      {"name": "scan_input_axes", "type": "List[int]"},
      {"name": "scan_input_directions", "type": "List[int]"},
      {"name": "scan_output_axes", "type": "List[int]"},
      {"name": "scan_output_directions", "type": "List[int]"},
    ],
    "variants": {},
  },
  "Scatter": {
    "description": "This operator is deprecated. Please use ScatterElements, which provides the same functionality. Scatter takes three inputs `data`, `updates`, and `indices` of the same rank r >= 1 and an optional attribute axis that identifies an axis of `data` (by default, the outer-most axis, that is axis 0). The ...",
    "std_args": [],
    "variants": {},
  },
  "Selu": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [],
    "variants": {},
  },
  "SeparableConv1D": {
    "description": '1D separable convolution layer.\n\nThis layer performs a depthwise convolution that acts separately on\nchannels, followed by a pointwise convolution that mixes channels.\nIf `use_bias` is True and a bias initializer is provided,\nit adds a bias vector to the output. It then optionally applies an\nactivation function to produce the final output.\n\nArgs:\n    filters: int, the dimensionality of the output space (i.e. the number\n        of filters in the pointwise convolution).\n    kernel_size: int or tuple/list of 1 integers, specifying the size of the\n        depthwise convolution window.\n    strides: int or tuple/list of 1 integers, specifying the stride length\n        of the depthwise convolution. If only one int is specified, the same\n        stride size will be used for all dimensions. `strides > 1` is\n        incompatible with `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 1 integers, specifying the dilation\n        rate to use for dilated convolution. If only one int is specified,\n        the same dilation rate will be used for all dimensions.\n    depth_multiplier: The number of depthwise convolution output channels\n        for each input channel. The total number of depthwise convolution\n        output channels will be equal to `input_channel * depth_multiplier`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    depthwise_initializer: An initializer for the depthwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    pointwise_initializer: An initializer for the pointwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    bias_initializer: An initializer for the bias vector. If None, the\n        default initializer (\'"zeros"\') will be used.\n    depthwise_regularizer: Optional regularizer for the depthwise\n        convolution kernel.\n    pointwise_regularizer: Optional regularizer for the pointwise\n        convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    depthwise_constraint: Optional projection function to be applied to the\n        depthwise kernel after being updated by an `Optimizer` (e.g. used\n        for norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape).\n    pointwise_constraint: Optional projection function to be applied to the\n        pointwise kernel after being updated by an `Optimizer`.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, steps, channels)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, channels, steps)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, new_steps, filters)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, filters, new_steps)`\n\nReturns:\n    A 3D tensor representing\n    `activation(separable_conv1d(inputs, kernel) + bias)`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 12)\n>>> y = keras.layers.SeparableConv1D(3, 4, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 4, 4)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "depth_multiplier", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "depthwise_initializer", "type": "Any"},
      {"name": "pointwise_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "depthwise_regularizer", "type": "Any"},
      {"name": "pointwise_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "depthwise_constraint", "type": "Any"},
      {"name": "pointwise_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "SeparableConv2D": {
    "description": '2D separable convolution layer.\n\nThis layer performs a depthwise convolution that acts separately on\nchannels, followed by a pointwise convolution that mixes channels.\nIf `use_bias` is True and a bias initializer is provided,\nit adds a bias vector to the output. It then optionally applies an\nactivation function to produce the final output.\n\nArgs:\n    filters: int, the dimensionality of the output space (i.e. the number\n        of filters in the pointwise convolution).\n    kernel_size: int or tuple/list of 2 integers, specifying the size of the\n        depthwise convolution window.\n    strides: int or tuple/list of 2 integers, specifying the stride length\n        of the depthwise convolution. If only one int is specified, the same\n        stride size will be used for all dimensions. `strides > 1` is\n        incompatible with `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file\n        at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 2 integers, specifying the dilation\n        rate to use for dilated convolution. If only one int is specified,\n        the same dilation rate will be used for all dimensions.\n    depth_multiplier: The number of depthwise convolution output channels\n        for each input channel. The total number of depthwise convolution\n        output channels will be equal to `input_channel * depth_multiplier`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    depthwise_initializer: An initializer for the depthwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    pointwise_initializer: An initializer for the pointwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    bias_initializer: An initializer for the bias vector. If None, the\n        default initializer (\'"zeros"\') will be used.\n    depthwise_regularizer: Optional regularizer for the depthwise\n        convolution kernel.\n    pointwise_regularizer: Optional regularizer for the pointwise\n        convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    depthwise_constraint: Optional projection function to be applied to the\n        depthwise kernel after being updated by an `Optimizer` (e.g. used\n        for norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape).\n    pointwise_constraint: Optional projection function to be applied to the\n        pointwise kernel after being updated by an `Optimizer`.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, height, width, channels)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`\n\nReturns:\n    A 4D tensor representing\n    `activation(separable_conv2d(inputs, kernel) + bias)`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 10, 12)\n>>> y = keras.layers.SeparableConv2D(3, 4, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 4, 4, 4)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "depth_multiplier", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "depthwise_initializer", "type": "Any"},
      {"name": "pointwise_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "depthwise_regularizer", "type": "Any"},
      {"name": "pointwise_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "depthwise_constraint", "type": "Any"},
      {"name": "pointwise_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "SeparableConvolution1D": {
    "description": '1D separable convolution layer.\n\nThis layer performs a depthwise convolution that acts separately on\nchannels, followed by a pointwise convolution that mixes channels.\nIf `use_bias` is True and a bias initializer is provided,\nit adds a bias vector to the output. It then optionally applies an\nactivation function to produce the final output.\n\nArgs:\n    filters: int, the dimensionality of the output space (i.e. the number\n        of filters in the pointwise convolution).\n    kernel_size: int or tuple/list of 1 integers, specifying the size of the\n        depthwise convolution window.\n    strides: int or tuple/list of 1 integers, specifying the stride length\n        of the depthwise convolution. If only one int is specified, the same\n        stride size will be used for all dimensions. `strides > 1` is\n        incompatible with `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, steps, features)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, features, steps)`. It defaults to the `image_data_format`\n        value found in your Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 1 integers, specifying the dilation\n        rate to use for dilated convolution. If only one int is specified,\n        the same dilation rate will be used for all dimensions.\n    depth_multiplier: The number of depthwise convolution output channels\n        for each input channel. The total number of depthwise convolution\n        output channels will be equal to `input_channel * depth_multiplier`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    depthwise_initializer: An initializer for the depthwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    pointwise_initializer: An initializer for the pointwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    bias_initializer: An initializer for the bias vector. If None, the\n        default initializer (\'"zeros"\') will be used.\n    depthwise_regularizer: Optional regularizer for the depthwise\n        convolution kernel.\n    pointwise_regularizer: Optional regularizer for the pointwise\n        convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    depthwise_constraint: Optional projection function to be applied to the\n        depthwise kernel after being updated by an `Optimizer` (e.g. used\n        for norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape).\n    pointwise_constraint: Optional projection function to be applied to the\n        pointwise kernel after being updated by an `Optimizer`.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, steps, channels)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, channels, steps)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 3D tensor with shape: `(batch_shape, new_steps, filters)`\n- If `data_format="channels_first"`:\n    A 3D tensor with shape: `(batch_shape, filters, new_steps)`\n\nReturns:\n    A 3D tensor representing\n    `activation(separable_conv1d(inputs, kernel) + bias)`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 12)\n>>> y = keras.layers.SeparableConv1D(3, 4, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 4, 4)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "depth_multiplier", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "depthwise_initializer", "type": "Any"},
      {"name": "pointwise_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "depthwise_regularizer", "type": "Any"},
      {"name": "pointwise_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "depthwise_constraint", "type": "Any"},
      {"name": "pointwise_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "SeparableConvolution2D": {
    "description": '2D separable convolution layer.\n\nThis layer performs a depthwise convolution that acts separately on\nchannels, followed by a pointwise convolution that mixes channels.\nIf `use_bias` is True and a bias initializer is provided,\nit adds a bias vector to the output. It then optionally applies an\nactivation function to produce the final output.\n\nArgs:\n    filters: int, the dimensionality of the output space (i.e. the number\n        of filters in the pointwise convolution).\n    kernel_size: int or tuple/list of 2 integers, specifying the size of the\n        depthwise convolution window.\n    strides: int or tuple/list of 2 integers, specifying the stride length\n        of the depthwise convolution. If only one int is specified, the same\n        stride size will be used for all dimensions. `strides > 1` is\n        incompatible with `dilation_rate > 1`.\n    padding: string, either `"valid"` or `"same"` (case-insensitive).\n        `"valid"` means no padding. `"same"` results in padding evenly to\n        the left/right or up/down of the input. When `padding="same"` and\n        `strides=1`, the output has the same size as the input.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        The ordering of the dimensions in the inputs. `"channels_last"`\n        corresponds to inputs with shape `(batch, height, width, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch, channels, height, width)`. It defaults to the\n        `image_data_format` value found in your Keras config file\n        at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n    dilation_rate: int or tuple/list of 2 integers, specifying the dilation\n        rate to use for dilated convolution. If only one int is specified,\n        the same dilation rate will be used for all dimensions.\n    depth_multiplier: The number of depthwise convolution output channels\n        for each input channel. The total number of depthwise convolution\n        output channels will be equal to `input_channel * depth_multiplier`.\n    activation: Activation function. If `None`, no activation is applied.\n    use_bias: bool, if `True`, bias will be added to the output.\n    depthwise_initializer: An initializer for the depthwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    pointwise_initializer: An initializer for the pointwise convolution\n        kernel. If None, then the default initializer (`"glorot_uniform"`)\n        will be used.\n    bias_initializer: An initializer for the bias vector. If None, the\n        default initializer (\'"zeros"\') will be used.\n    depthwise_regularizer: Optional regularizer for the depthwise\n        convolution kernel.\n    pointwise_regularizer: Optional regularizer for the pointwise\n        convolution kernel.\n    bias_regularizer: Optional regularizer for the bias vector.\n    activity_regularizer: Optional regularizer function for the output.\n    depthwise_constraint: Optional projection function to be applied to the\n        depthwise kernel after being updated by an `Optimizer` (e.g. used\n        for norm constraints or value constraints for layer weights). The\n        function must take as input the unprojected variable and must return\n        the projected variable (which must have the same shape).\n    pointwise_constraint: Optional projection function to be applied to the\n        pointwise kernel after being updated by an `Optimizer`.\n    bias_constraint: Optional projection function to be applied to the\n        bias after being updated by an `Optimizer`.\n\nInput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, height, width, channels)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, channels, height, width)`\n\nOutput shape:\n\n- If `data_format="channels_last"`:\n    A 4D tensor with shape: `(batch_size, new_height, new_width, filters)`\n- If `data_format="channels_first"`:\n    A 4D tensor with shape: `(batch_size, filters, new_height, new_width)`\n\nReturns:\n    A 4D tensor representing\n    `activation(separable_conv2d(inputs, kernel) + bias)`.\n\nExample:\n\n>>> x = np.random.rand(4, 10, 10, 12)\n>>> y = keras.layers.SeparableConv2D(3, 4, 3, 2, activation=\'relu\')(x)\n>>> print(y.shape)\n(4, 4, 4, 4)',
    "std_args": [
      {"name": "filters", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
      {"name": "depth_multiplier", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "depthwise_initializer", "type": "Any"},
      {"name": "pointwise_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "depthwise_regularizer", "type": "Any"},
      {"name": "pointwise_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "depthwise_constraint", "type": "Any"},
      {"name": "pointwise_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Sequential": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Shape": {
    "description": "Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor. Optional attributes start and end can be used to compute a slice of the input tensor's shape. If start axis is omitted, the slice starts from axis 0. The end axis, if specified, is exclusive (and the ret...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "end", "type": "int"},
      {"name": "start", "type": "int"},
    ],
    "variants": {},
  },
  "ShouldSkipUpdateFunction": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Sigmoid": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "Sign": {
    "description": "Calculate the sign of the given input tensor element-wise. If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Silu": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "SimpleRNN": {
    "description": 'Fully-connected RNN where the output is to be fed back as the new input.\n\nArgs:\n    units: Positive integer, dimensionality of the output space.\n    activation: Activation function to use.\n        Default: hyperbolic tangent (`tanh`).\n        If you pass None, no activation is applied\n        (ie. "linear" activation: `a(x) = x`).\n    use_bias: Boolean, (default `True`), whether the layer uses\n        a bias vector.\n    kernel_initializer: Initializer for the `kernel` weights matrix,\n        used for the linear transformation of the inputs. Default:\n        `"glorot_uniform"`.\n    recurrent_initializer: Initializer for the `recurrent_kernel`\n        weights matrix, used for the linear transformation of the recurrent\n        state.  Default: `"orthogonal"`.\n    bias_initializer: Initializer for the bias vector. Default: `"zeros"`.\n    kernel_regularizer: Regularizer function applied to the `kernel` weights\n        matrix. Default: `None`.\n    recurrent_regularizer: Regularizer function applied to the\n        `recurrent_kernel` weights matrix. Default: `None`.\n    bias_regularizer: Regularizer function applied to the bias vector.\n        Default: `None`.\n    activity_regularizer: Regularizer function applied to the output of the\n        layer (its "activation"). Default: `None`.\n    kernel_constraint: Constraint function applied to the `kernel` weights\n        matrix. Default: `None`.\n    recurrent_constraint: Constraint function applied to the\n        `recurrent_kernel` weights matrix.  Default: `None`.\n    bias_constraint: Constraint function applied to the bias vector.\n        Default: `None`.\n    dropout: Float between 0 and 1.\n        Fraction of the units to drop for the linear transformation\n        of the inputs. Default: 0.\n    recurrent_dropout: Float between 0 and 1.\n        Fraction of the units to drop for the linear transformation of the\n        recurrent state. Default: 0.\n    return_sequences: Boolean. Whether to return the last output\n        in the output sequence, or the full sequence. Default: `False`.\n    return_state: Boolean. Whether to return the last state\n        in addition to the output. Default: `False`.\n    go_backwards: Boolean (default: `False`).\n        If `True`, process the input sequence backwards and return the\n        reversed sequence.\n    stateful: Boolean (default: `False`). If `True`, the last state\n        for each sample at index i in a batch will be used as the\n        initial state for the sample of index i in the following batch.\n    unroll: Boolean (default: `False`).\n        If `True`, the network will be unrolled,\n        else a symbolic loop will be used.\n        Unrolling can speed-up an RNN,\n        although it tends to be more memory-intensive.\n        Unrolling is only suitable for short sequences.\n\nCall arguments:\n    sequence: A 3D tensor, with shape `[batch, timesteps, feature]`.\n    mask: Binary tensor of shape `[batch, timesteps]` indicating whether\n        a given timestep should be masked. An individual `True` entry\n        indicates that the corresponding timestep should be utilized,\n        while a `False` entry indicates that the corresponding timestep\n        should be ignored.\n    training: Python boolean indicating whether the layer should behave in\n        training mode or in inference mode.\n        This argument is passed to the cell when calling it.\n        This is only relevant if `dropout` or `recurrent_dropout` is used.\n    initial_state: List of initial state tensors to be passed to the first\n        call of the cell.\n\nExample:\n\n```python\ninputs = np.random.random((32, 10, 8))\nsimple_rnn = keras.layers.SimpleRNN(4)\noutput = simple_rnn(inputs)  # The output has shape `(32, 4)`.\nsimple_rnn = keras.layers.SimpleRNN(\n    4, return_sequences=True, return_state=True\n)\n# whole_sequence_output has shape `(32, 10, 4)`.\n# final_state has shape `(32, 4)`.\nwhole_sequence_output, final_state = simple_rnn(inputs)\n```',
    "std_args": [
      {"name": "units", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "recurrent_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "recurrent_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "activity_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "recurrent_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "recurrent_dropout", "type": "Any"},
      {"name": "return_sequences", "type": "Any"},
      {"name": "return_state", "type": "Any"},
      {"name": "go_backwards", "type": "Any"},
      {"name": "stateful", "type": "Any"},
      {"name": "unroll", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "SimpleRNNCell": {
    "description": 'Cell class for SimpleRNN.\n\nThis class processes one step within the whole time sequence input, whereas\n`keras.layer.SimpleRNN` processes the whole sequence.\n\nArgs:\n    units: Positive integer, dimensionality of the output space.\n    activation: Activation function to use.\n        Default: hyperbolic tangent (`tanh`).\n        If you pass `None`, no activation is applied\n        (ie. "linear" activation: `a(x) = x`).\n    use_bias: Boolean, (default `True`), whether the layer\n        should use a bias vector.\n    kernel_initializer: Initializer for the `kernel` weights matrix,\n        used for the linear transformation of the inputs. Default:\n        `"glorot_uniform"`.\n    recurrent_initializer: Initializer for the `recurrent_kernel`\n        weights matrix, used for the linear transformation\n        of the recurrent state. Default: `"orthogonal"`.\n    bias_initializer: Initializer for the bias vector. Default: `"zeros"`.\n    kernel_regularizer: Regularizer function applied to the `kernel` weights\n        matrix. Default: `None`.\n    recurrent_regularizer: Regularizer function applied to the\n        `recurrent_kernel` weights matrix. Default: `None`.\n    bias_regularizer: Regularizer function applied to the bias vector.\n        Default: `None`.\n    kernel_constraint: Constraint function applied to the `kernel` weights\n        matrix. Default: `None`.\n    recurrent_constraint: Constraint function applied to the\n        `recurrent_kernel` weights matrix. Default: `None`.\n    bias_constraint: Constraint function applied to the bias vector.\n        Default: `None`.\n    dropout: Float between 0 and 1. Fraction of the units to drop for the\n        linear transformation of the inputs. Default: 0.\n    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop\n        for the linear transformation of the recurrent state. Default: 0.\n    seed: Random seed for dropout.\n\nCall arguments:\n    sequence: A 2D tensor, with shape `(batch, features)`.\n    states: A 2D tensor with shape `(batch, units)`, which is the state\n        from the previous time step.\n    training: Python boolean indicating whether the layer should behave in\n        training mode or in inference mode. Only relevant when `dropout` or\n        `recurrent_dropout` is used.\n\nExample:\n\n```python\ninputs = np.random.random([32, 10, 8]).astype(np.float32)\nrnn = keras.layers.RNN(keras.layers.SimpleRNNCell(4))\noutput = rnn(inputs)  # The output has shape `(32, 4)`.\nrnn = keras.layers.RNN(\n    keras.layers.SimpleRNNCell(4),\n    return_sequences=True,\n    return_state=True\n)\n# whole_sequence_output has shape `(32, 10, 4)`.\n# final_state has shape `(32, 4)`.\nwhole_sequence_output, final_state = rnn(inputs)\n```',
    "std_args": [
      {"name": "units", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "use_bias", "type": "Any"},
      {"name": "kernel_initializer", "type": "Any"},
      {"name": "recurrent_initializer", "type": "Any"},
      {"name": "bias_initializer", "type": "Any"},
      {"name": "kernel_regularizer", "type": "Any"},
      {"name": "recurrent_regularizer", "type": "Any"},
      {"name": "bias_regularizer", "type": "Any"},
      {"name": "kernel_constraint", "type": "Any"},
      {"name": "recurrent_constraint", "type": "Any"},
      {"name": "bias_constraint", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "recurrent_dropout", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Sin": {
    "description": "Calculates the sine of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Sinh": {
    "description": "Calculates the hyperbolic sine of the given input tensor element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Size": {
    "description": "Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.",
    "std_args": [
      {"name": "data", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Slice": {
    "description": "Produces a slice of the input tensor along multiple axes. Similar to numpy: https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor of its input `data` tensor. An effective `starts[i...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "starts", "type": "Any"},
      {"name": "ends", "type": "Any"},
      {"name": "axes", "type": "Any"},
      {"name": "steps", "type": "Any"},
    ],
    "variants": {},
  },
  "Smoothl1": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "beta", "type": "float"},
      {"name": "reduction", "type": "str"},
    ],
    "variants": {},
  },
  "SnapshotState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Softmax": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Softmax2d": {
    "description": "Applies SoftMax over features to each spatial location.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Softmin": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Softplus": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Softshrink": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Softsign": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Solarization": {
    "description": "Applies `(max_value - pixel + min_value)` for each pixel in the image.\n\nWhen created without `threshold` parameter, the layer performs solarization\nto all values. When created with specified `threshold` the layer only\naugments pixels that are above the `threshold` value.\n\n**Note:** This layer is safe to use inside a `tf.data` or `grain` pipeline\n(independently of which backend you're using).\n\nArgs:\n    addition_factor: (Optional)  A tuple of two floats or a single float,\n        between 0 and 1.\n        For each augmented image a value is\n        sampled from the provided range. If a float is passed, the range is\n        interpreted as `(0, addition_factor)`. If specified, this value\n        (times the value range of input images, e.g. 255), is\n        added to each pixel before solarization and thresholding.\n        Defaults to 0.0.\n    threshold_factor: (Optional)  A tuple of two floats or a single float.\n        For each augmented image a value is\n        sampled from the provided range. If a float is passed, the range is\n        interpreted as `(0, threshold_factor)`. If specified, only pixel\n        values above this threshold will be solarized.\n    value_range: a tuple or a list of two elements. The first value\n        represents the lower bound for values in input images, the second\n        represents the upper bound. Images passed to the layer should have\n        values within `value_range`. Typical values to pass\n        are `(0, 255)` (RGB image) or `(0., 1.)` (scaled image).\n    seed: Integer. Used to create a random seed.\n    **kwargs: Base layer keyword arguments, such as `name` and `dtype`.\n\nExample:\n\n```python\n(images, labels), _ = keras.datasets.cifar10.load_data()\nprint(images[0, 0, 0])\n# [59 62 63]\n# Note that images are Tensor with values in the range [0, 255]\nsolarization = Solarization(value_range=(0, 255))\nimages = solarization(images)\nprint(images[0, 0, 0])\n# [196, 193, 192]\n```",
    "std_args": [
      {"name": "addition_factor", "type": "Any"},
      {"name": "threshold_factor", "type": "Any"},
      {"name": "value_range", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Sparsemax": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "labels", "type": "Any"},
      {"name": "logits", "type": "Any"},
    ],
    "variants": {},
  },
  "SpatialDropout1D": {
    "description": "Spatial 1D version of Dropout.\n\nThis layer performs the same function as Dropout, however, it drops\nentire 1D feature maps instead of individual elements. If adjacent frames\nwithin feature maps are strongly correlated (as is normally the case in\nearly convolution layers) then regular dropout will not regularize the\nactivations and will otherwise just result in an effective learning rate\ndecrease. In this case, `SpatialDropout1D` will help promote independence\nbetween feature maps and should be used instead.\n\nArgs:\n    rate: Float between 0 and 1. Fraction of the input units to drop.\n\nCall arguments:\n    inputs: A 3D tensor.\n    training: Python boolean indicating whether the layer\n        should behave in training mode (applying dropout)\n        or in inference mode (pass-through).\n\nInput shape:\n    3D tensor with shape: `(samples, timesteps, channels)`\n\nOutput shape: Same as input.\n\nReference:\n\n- [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)",
    "std_args": [
      {"name": "rate", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "SpatialDropout2D": {
    "description": 'Spatial 2D version of Dropout.\n\nThis version performs the same function as Dropout, however, it drops\nentire 2D feature maps instead of individual elements. If adjacent pixels\nwithin feature maps are strongly correlated (as is normally the case in\nearly convolution layers) then regular dropout will not regularize the\nactivations and will otherwise just result in an effective learning rate\ndecrease. In this case, `SpatialDropout2D` will help promote independence\nbetween feature maps and should be used instead.\n\nArgs:\n    rate: Float between 0 and 1. Fraction of the input units to drop.\n    data_format: `"channels_first"` or `"channels_last"`.\n        In `"channels_first"` mode, the channels dimension (the depth)\n        is at index 1, in `"channels_last"` mode is it at index 3.\n        It defaults to the `image_data_format` value found in your\n        Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n\nCall arguments:\n    inputs: A 4D tensor.\n    training: Python boolean indicating whether the layer\n        should behave in training mode (applying dropout)\n        or in inference mode (pass-through).\n\nInput shape:\n    4D tensor with shape: `(samples, channels, rows, cols)` if\n        data_format=\'channels_first\'\n    or 4D tensor with shape: `(samples, rows, cols, channels)` if\n        data_format=\'channels_last\'.\n\nOutput shape: Same as input.\n\nReference:\n\n- [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)',
    "std_args": [
      {"name": "rate", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "SpatialDropout3D": {
    "description": 'Spatial 3D version of Dropout.\n\nThis version performs the same function as Dropout, however, it drops\nentire 3D feature maps instead of individual elements. If adjacent voxels\nwithin feature maps are strongly correlated (as is normally the case in\nearly convolution layers) then regular dropout will not regularize the\nactivations and will otherwise just result in an effective learning rate\ndecrease. In this case, SpatialDropout3D will help promote independence\nbetween feature maps and should be used instead.\n\nArgs:\n    rate: Float between 0 and 1. Fraction of the input units to drop.\n    data_format: `"channels_first"` or `"channels_last"`.\n        In `"channels_first"` mode, the channels dimension (the depth)\n        is at index 1, in `"channels_last"` mode is it at index 4.\n        It defaults to the `image_data_format` value found in your\n        Keras config file at `~/.keras/keras.json`.\n        If you never set it, then it will be `"channels_last"`.\n\nCall arguments:\n    inputs: A 5D tensor.\n    training: Python boolean indicating whether the layer\n            should behave in training mode (applying dropout)\n            or in inference mode (pass-through).\n\nInput shape:\n    5D tensor with shape: `(samples, channels, dim1, dim2, dim3)` if\n        data_format=\'channels_first\'\n    or 5D tensor with shape: `(samples, dim1, dim2, dim3, channels)` if\n        data_format=\'channels_last\'.\n\nOutput shape: Same as input.\n\nReference:\n\n- [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)',
    "std_args": [
      {"name": "rate", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "seed", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "SpectralNorm": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "n_power_iterations", "type": "Any"},
      {"name": "dim", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "SpectralNormalization": {
    "description": "Performs spectral normalization on the weights of a target layer.\n\nThis wrapper controls the Lipschitz constant of the weights of a layer by\nconstraining their spectral norm, which can stabilize the training of GANs.\n\nArgs:\n    layer: A `keras.layers.Layer` instance that\n        has either a `kernel` (e.g. `Conv2D`, `Dense`...)\n        or an `embeddings` attribute (`Embedding` layer).\n    power_iterations: int, the number of iterations during normalization.\n    **kwargs: Base wrapper keyword arguments.\n\nExamples:\n\nWrap `keras.layers.Conv2D`:\n>>> x = np.random.rand(1, 10, 10, 1)\n>>> conv2d = SpectralNormalization(keras.layers.Conv2D(2, 2))\n>>> y = conv2d(x)\n>>> y.shape\n(1, 9, 9, 2)\n\nWrap `keras.layers.Dense`:\n>>> x = np.random.rand(1, 10, 10, 1)\n>>> dense = SpectralNormalization(keras.layers.Dense(10))\n>>> y = dense(x)\n>>> y.shape\n(1, 10, 10, 10)\n\nReference:\n\n- [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).",
    "std_args": [
      {"name": "layer", "type": "Any"},
      {"name": "power_iterations", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Split": {
    "description": "Split a tensor into a list of tensors, along the specified 'axis'. Either input 'split' or the attribute 'num_outputs' should be specified, but not both. If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts. If the tensor is not evenly splittable into `num_out...",
    "std_args": [
      {"name": "input", "type": "Tensor"},
      {"name": "split", "type": "int"},
      {"name": "axis", "type": "int"},
      {"name": "num_outputs", "type": "int"},
    ],
    "variants": {},
  },
  "Sqrt": {
    "description": "Square root takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where the square root is, y = x^0.5, is applied to the tensor elementwise. If x is negative, then it will return NaN.",
    "std_args": [
      {"name": "X", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Squeeze": {
    "description": "Remove single-dimensional entries from the shape of a tensor. Takes an input `axes` with a list of axes to squeeze. If `axes` is not provided, all the single dimensions will be removed from the shape. If an axis is selected with shape entry not equal to one, an error is raised.",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "axes", "type": "int"},
    ],
    "variants": {},
  },
  "StackedRNNCells": {
    "description": "Wrapper allowing a stack of RNN cells to behave as a single cell.\n\nUsed to implement efficient stacked RNNs.\n\nArgs:\n  cells: List of RNN cell instances.\n\nExample:\n\n```python\nbatch_size = 3\nsentence_length = 5\nnum_features = 2\nnew_shape = (batch_size, sentence_length, num_features)\nx = np.reshape(np.arange(30), new_shape)\n\nrnn_cells = [keras.layers.LSTMCell(128) for _ in range(2)]\nstacked_lstm = keras.layers.StackedRNNCells(rnn_cells)\nlstm_layer = keras.layers.RNN(stacked_lstm)\n\nresult = lstm_layer(x)\n```",
    "std_args": [
      {"name": "cells", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "StepLR": {
    "description": "Decays the learning rate of each parameter group by gamma every step_size epochs.",
    "std_args": [
      {"name": "optimizer", "type": "Any"},
      {"name": "step_size", "type": "Any"},
      {"name": "gamma", "type": "Any"},
    ],
    "variants": {},
  },
  "StringLookup": {
    "description": 'A preprocessing layer that maps strings to (possibly encoded) indices.\n\nThis layer translates a set of arbitrary strings into integer output via a\ntable-based vocabulary lookup. This layer will perform no splitting or\ntransformation of input strings. For a layer that can split and tokenize\nnatural language, see the `keras.layers.TextVectorization` layer.\n\nThe vocabulary for the layer must be either supplied on construction or\nlearned via `adapt()`. During `adapt()`, the layer will analyze a data set,\ndetermine the frequency of individual strings tokens, and create a\nvocabulary from them. If the vocabulary is capped in size, the most frequent\ntokens will be used to create the vocabulary and all others will be treated\nas out-of-vocabulary (OOV).\n\nThere are two possible output modes for the layer. When `output_mode` is\n`"int"`, input strings are converted to their index in the vocabulary (an\ninteger).\nWhen `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`, input strings\nare encoded into an array where each dimension corresponds to an element in\nthe vocabulary.\n\nThe vocabulary can optionally contain a mask token as well as an OOV token\n(which can optionally occupy multiple indices in the vocabulary, as set\nby `num_oov_indices`).\nThe position of these tokens in the vocabulary is fixed. When `output_mode`\nis `"int"`, the vocabulary will begin with the mask token (if set), followed\nby OOV indices, followed by the rest of the vocabulary. When `output_mode`\nis `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will begin with\nOOV indices and instances of the mask token will be dropped.\n\n**Note:** This layer uses TensorFlow internally. It cannot\nbe used as part of the compiled computation graph of a model with\nany backend other than TensorFlow.\nIt can however be used with any backend when running eagerly.\nIt can also always be used as part of an input preprocessing pipeline\nwith any backend (outside the model itself), which is how we recommend\nusing this layer.\n\n**Note:** This layer is safe to use inside a `tf.data` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    max_tokens: Maximum size of the vocabulary for this layer. This should\n        only be specified when adapting the vocabulary or when setting\n        `pad_to_max_tokens=True`. If None, there is no cap on the size of\n        the vocabulary. Note that this size includes the OOV\n        and mask tokens. Defaults to `None`.\n    num_oov_indices: The number of out-of-vocabulary tokens to use.\n        If this value is more than 1, OOV inputs are modulated to\n        determine their OOV value.\n        If this value is 0, OOV inputs will cause an error when calling\n        the layer. Defaults to `1`.\n    mask_token: A token that represents masked inputs. When `output_mode` is\n        `"int"`, the token is included in the vocabulary and mapped to index\n        0.\n        In other output modes, the token will not appear in the vocabulary\n        and instances of the mask token in the input will be dropped.\n        If set to `None`, no mask term will be added. Defaults to `None`.\n    oov_token: Only used when `invert` is True. The token to return for OOV\n        indices. Defaults to `"[UNK]"`.\n    vocabulary: Optional. Either an array of strings or a string path to a\n        text file. If passing an array, you can pass a tuple, list, 1D NumPy\n        array, or 1D tensor containing the string vocabulary terms.\n        If passing a file path, the file should contain one line per term in\n        the vocabulary. If this argument is set, there is no need to\n        `adapt()` the layer.\n    idf_weights: Only valid when `output_mode` is `"tf_idf"`.\n        A tuple, list, 1D NumPy array, or 1D tensor or the same length\n        as the vocabulary, containing the floating point inverse document\n        frequency weights, which will be multiplied by per sample term\n        counts for the final TF-IDF weight.\n        If the `vocabulary` argument is set and `output_mode` is `"tf_idf"`,\n        this argument must be supplied.\n    invert: Only valid when `output_mode` is `"int"`.\n        If `True`, this layer will map indices to vocabulary items\n        instead of mapping vocabulary items to indices.\n        Defaults to `False`.\n    output_mode: Specification for the output of the layer. Values can be\n        `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"`\n        configuring the layer as follows:\n        - `"int"`: Return the vocabulary indices of the input tokens.\n        - `"one_hot"`: Encodes each individual element in the input into an\n            array the same size as the vocabulary,\n            containing a 1 at the element index. If the last dimension\n            is size 1, will encode on that dimension.\n            If the last dimension is not size 1, will append a new\n            dimension for the encoded output.\n        - `"multi_hot"`: Encodes each sample in the input into a single\n            array the same size as the vocabulary containing a 1 for each\n            vocabulary term present in the sample.\n            Treats the last dimension as the sample dimension, if the input\n            shape is `(..., sample_length)`, the output shape will be\n            `(..., num_tokens)`.\n        - `"count"`: As `"multi_hot"`, but the int array contains\n            a count of the number of times the token at that index\n            appeared in the sample.\n        - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is\n            applied to find the value in each token slot.\n        For `"int"` output, any shape of input and output is supported.\n        For all other output modes, currently only output up to rank 2\n        is supported. Defaults to `"int"`.\n    pad_to_max_tokens: Only applicable when `output_mode` is `"multi_hot"`,\n        `"count"`, or `"tf_idf"`. If `True`, the output will have\n        its feature axis padded to `max_tokens` even if the number\n        of unique tokens in the vocabulary is less than `max_tokens`,\n        resulting in a tensor of shape `(batch_size, max_tokens)`\n        regardless of vocabulary size. Defaults to `False`.\n    sparse: Boolean. Only applicable to `"multi_hot"`, `"count"`, and\n        `"tf_idf"` output modes. Only supported with TensorFlow\n        backend. If `True`, returns a `SparseTensor`\n        instead of a dense `Tensor`. Defaults to `False`.\n    encoding: Optional. The text encoding to use to interpret the input\n        strings. Defaults to `"utf-8"`.\n\nExamples:\n\n**Creating a lookup layer with a known vocabulary**\n\nThis example creates a lookup layer with a pre-existing vocabulary.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = [["a", "c", "d"], ["d", "z", "b"]]\n>>> layer = StringLookup(vocabulary=vocab)\n>>> layer(data)\narray([[1, 3, 4],\n       [4, 0, 2]])\n\n**Creating a lookup layer with an adapted vocabulary**\n\nThis example creates a lookup layer and generates the vocabulary by\nanalyzing the dataset.\n\n>>> data = [["a", "c", "d"], ["d", "z", "b"]]\n>>> layer = StringLookup()\n>>> layer.adapt(data)\n>>> layer.get_vocabulary()\n[\'[UNK]\', \'d\', \'z\', \'c\', \'b\', \'a\']\n\nNote that the OOV token `"[UNK]"` has been added to the vocabulary.\nThe remaining tokens are sorted by frequency\n(`"d"`, which has 2 occurrences, is first) then by inverse sort order.\n\n>>> data = [["a", "c", "d"], ["d", "z", "b"]]\n>>> layer = StringLookup()\n>>> layer.adapt(data)\n>>> layer(data)\narray([[5, 3, 1],\n       [1, 2, 4]])\n\n**Lookups with multiple OOV indices**\n\nThis example demonstrates how to use a lookup layer with multiple OOV\nindices.  When a layer is created with more than one OOV index, any OOV\nvalues are hashed into the number of OOV buckets, distributing OOV values in\na deterministic fashion across the set.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = [["a", "c", "d"], ["m", "z", "b"]]\n>>> layer = StringLookup(vocabulary=vocab, num_oov_indices=2)\n>>> layer(data)\narray([[2, 4, 5],\n       [0, 1, 3]])\n\nNote that the output for OOV value \'m\' is 0, while the output for OOV value\n`"z"` is 1. The in-vocab terms have their output index increased by 1 from\nearlier examples (a maps to 2, etc) in order to make space for the extra OOV\nvalue.\n\n**One-hot output**\n\nConfigure the layer with `output_mode=\'one_hot\'`. Note that the first\n`num_oov_indices` dimensions in the ont_hot encoding represent OOV values.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = ["a", "b", "c", "d", "z"]\n>>> layer = StringLookup(vocabulary=vocab, output_mode=\'one_hot\')\n>>> layer(data)\narray([[0., 1., 0., 0., 0.],\n       [0., 0., 1., 0., 0.],\n       [0., 0., 0., 1., 0.],\n       [0., 0., 0., 0., 1.],\n       [1., 0., 0., 0., 0.]], dtype=int64)\n\n**Multi-hot output**\n\nConfigure the layer with `output_mode=\'multi_hot\'`. Note that the first\n`num_oov_indices` dimensions in the multi_hot encoding represent OOV values.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]\n>>> layer = StringLookup(vocabulary=vocab, output_mode=\'multi_hot\')\n>>> layer(data)\narray([[0., 1., 0., 1., 1.],\n       [1., 0., 1., 0., 1.]], dtype=int64)\n\n**Token count output**\n\nConfigure the layer with `output_mode=\'count\'`. As with multi_hot output,\nthe first `num_oov_indices` dimensions in the output represent OOV values.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]\n>>> layer = StringLookup(vocabulary=vocab, output_mode=\'count\')\n>>> layer(data)\narray([[0., 1., 0., 1., 2.],\n       [2., 0., 1., 0., 1.]], dtype=int64)\n\n**TF-IDF output**\n\nConfigure the layer with `output_mode="tf_idf"`. As with multi_hot output,\nthe first `num_oov_indices` dimensions in the output represent OOV values.\n\nEach token bin will output `token_count * idf_weight`, where the idf weights\nare the inverse document frequency weights per token. These should be\nprovided along with the vocabulary. Note that the `idf_weight` for OOV\nvalues will default to the average of all idf weights passed in.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> idf_weights = [0.25, 0.75, 0.6, 0.4]\n>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]\n>>> layer = StringLookup(output_mode="tf_idf")\n>>> layer.set_vocabulary(vocab, idf_weights=idf_weights)\n>>> layer(data)\narray([[0.  , 0.25, 0.  , 0.6 , 0.8 ],\n       [1.0 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)\n\nTo specify the idf weights for OOV values, you will need to pass the entire\nvocabulary including the leading OOV token.\n\n>>> vocab = ["[UNK]", "a", "b", "c", "d"]\n>>> idf_weights = [0.9, 0.25, 0.75, 0.6, 0.4]\n>>> data = [["a", "c", "d", "d"], ["d", "z", "b", "z"]]\n>>> layer = StringLookup(output_mode="tf_idf")\n>>> layer.set_vocabulary(vocab, idf_weights=idf_weights)\n>>> layer(data)\narray([[0.  , 0.25, 0.  , 0.6 , 0.8 ],\n       [1.8 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)\n\nWhen adapting the layer in `"tf_idf"` mode, each input sample will be\nconsidered a document, and IDF weight per token will be calculated as\n`log(1 + num_documents / (1 + token_document_count))`.\n\n**Inverse lookup**\n\nThis example demonstrates how to map indices to strings using this layer.\n(You can also use `adapt()` with `inverse=True`, but for simplicity we\'ll\npass the vocab in this example.)\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = [[1, 3, 4], [4, 0, 2]]\n>>> layer = StringLookup(vocabulary=vocab, invert=True)\n>>> layer(data)\narray([[b\'a\', b\'c\', b\'d\'],\n       [b\'d\', b\'[UNK]\', b\'b\']], dtype=object)\n\nNote that the first index corresponds to the OOV token by default.\n\n\n**Forward and inverse lookup pairs**\n\nThis example demonstrates how to use the vocabulary of a standard lookup\nlayer to create an inverse lookup layer.\n\n>>> vocab = ["a", "b", "c", "d"]\n>>> data = [["a", "c", "d"], ["d", "z", "b"]]\n>>> layer = StringLookup(vocabulary=vocab)\n>>> i_layer = StringLookup(vocabulary=vocab, invert=True)\n>>> int_data = layer(data)\n>>> i_layer(int_data)\narray([[b\'a\', b\'c\', b\'d\'],\n       [b\'d\', b\'[UNK]\', b\'b\']], dtype=object)\n\nIn this example, the input value `"z"` resulted in an output of `"[UNK]"`,\nsince 1000 was not in the vocabulary - it got represented as an OOV, and all\nOOV values are returned as `"[UNK]"` in the inverse layer. Also, note that\nfor the inverse to work, you must have already set the forward layer\nvocabulary either directly or via `adapt()` before calling\n`get_vocabulary()`.',
    "std_args": [
      {"name": "max_tokens", "type": "Any"},
      {"name": "num_oov_indices", "type": "Any"},
      {"name": "mask_token", "type": "Any"},
      {"name": "oov_token", "type": "Any"},
      {"name": "vocabulary", "type": "Any"},
      {"name": "idf_weights", "type": "Any"},
      {"name": "invert", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "pad_to_max_tokens", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "encoding", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Sub": {
    "description": "Performs element-wise binary subtraction (with Numpy-style broadcasting support). This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md). (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int1...",
    "std_args": [
      {"name": "A", "type": "Tensor"},
      {"name": "B", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Sum": {
    "description": "Element-wise sum of each of the input tensors (with Numpy-style broadcasting support). All inputs and outputs must have the same data type. This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).",
    "std_args": [
      {"name": "data_0", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Synchronize": {
    "description": "Execution Barrier.",
    "std_args": [],
    "variants": {},
  },
  "T": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "TFSMLayer": {
    "description": "Reload a Keras model/layer that was saved via SavedModel / ExportArchive.\n\nArguments:\n    filepath: `str` or `pathlib.Path` object. The path to the SavedModel.\n    call_endpoint: Name of the endpoint to use as the `call()` method\n        of the reloaded layer. If the SavedModel was created\n        via `model.export()`,\n        then the default endpoint name is `'serve'`. In other cases\n        it may be named `'serving_default'`.\n\nExample:\n\n```python\nmodel.export(\"path/to/artifact\")\nreloaded_layer = TFSMLayer(\"path/to/artifact\")\noutputs = reloaded_layer(inputs)\n```\n\nThe reloaded object can be used like a regular Keras layer, and supports\ntraining/fine-tuning of its trainable weights. Note that the reloaded\nobject retains none of the internal structure or custom methods of the\noriginal object -- it's a brand new layer created around the saved\nfunction.\n\n**Limitations:**\n\n* Only call endpoints with a single `inputs` tensor argument\n(which may optionally be a dict/tuple/list of tensors) are supported.\nFor endpoints with multiple separate input tensor arguments, consider\nsubclassing `TFSMLayer` and implementing a `call()` method with a\ncustom signature.\n* If you need training-time behavior to differ from inference-time behavior\n(i.e. if you need the reloaded object to support a `training=True` argument\nin `__call__()`), make sure that the training-time call function is\nsaved as a standalone endpoint in the artifact, and provide its name\nto the `TFSMLayer` via the `call_training_endpoint` argument.",
    "std_args": [
      {"name": "filepath", "type": "Any"},
      {"name": "call_endpoint", "type": "Any"},
      {"name": "call_training_endpoint", "type": "Any"},
      {"name": "trainable", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "TUPLE": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "T_module": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Tan": {
    "description": "Calculates the tangent of the given input tensor, element-wise.",
    "std_args": [
      {"name": "input", "type": "Tensor"},
    ],
    "variants": {},
  },
  "Tanh": {
    "description": "Auto-discovered via Consensus (Score: 3.0)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "Tanhshrink": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [],
    "variants": {},
  },
  "Tensor": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "TextVectorization": {
    "description": 'A preprocessing layer which maps text features to integer sequences.\n\nThis layer has basic options for managing text in a Keras model. It\ntransforms a batch of strings (one example = one string) into either a list\nof token indices (one example = 1D tensor of integer token indices) or a\ndense representation (one example = 1D tensor of float values representing\ndata about the example\'s tokens). This layer is meant to handle natural\nlanguage inputs. To handle simple string inputs (categorical strings or\npre-tokenized strings) see `kers_core.layers.StringLookup`.\n\nThe vocabulary for the layer must be either supplied on construction or\nlearned via `adapt()`. When this layer is adapted, it will analyze the\ndataset, determine the frequency of individual string values, and create a\nvocabulary from them. This vocabulary can have unlimited size or be capped,\ndepending on the configuration options for this layer; if there are more\nunique values in the input than the maximum vocabulary size, the most\nfrequent terms will be used to create the vocabulary.\n\nThe processing of each example contains the following steps:\n\n1. Standardize each example (usually lowercasing + punctuation stripping)\n2. Split each example into substrings (usually words)\n3. Recombine substrings into tokens (usually ngrams)\n4. Index tokens (associate a unique int value with each token)\n5. Transform each example using this index, either into a vector of ints or\n   a dense float vector.\n\nSome notes on passing callables to customize splitting and normalization for\nthis layer:\n\n1. Any callable can be passed to this Layer, but if you want to serialize\n   this object you should only pass functions that are registered Keras\n   serializables (see `keras.saving.register_keras_serializable`\n   for more details).\n2. When using a custom callable for `standardize`, the data received\n   by the callable will be exactly as passed to this layer. The callable\n   should return a tensor of the same shape as the input.\n3. When using a custom callable for `split`, the data received by the\n   callable will have the 1st dimension squeezed out - instead of\n   `[["string to split"], ["another string to split"]]`, the Callable will\n   see `["string to split", "another string to split"]`.\n   The callable should return a `tf.Tensor` of dtype `string`\n   with the first dimension containing the split tokens -\n   in this example, we should see something like `[["string", "to",\n   "split"], ["another", "string", "to", "split"]]`.\n\n**Note:** This layer uses TensorFlow internally. It cannot\nbe used as part of the compiled computation graph of a model with\nany backend other than TensorFlow.\nIt can however be used with any backend when running eagerly.\nIt can also always be used as part of an input preprocessing pipeline\nwith any backend (outside the model itself), which is how we recommend\nto use this layer.\n\n**Note:** This layer is safe to use inside a `tf.data` pipeline\n(independently of which backend you\'re using).\n\nArgs:\n    max_tokens: Maximum size of the vocabulary for this layer. This should\n        only be specified when adapting a vocabulary or when setting\n        `pad_to_max_tokens=True`. Note that this vocabulary\n        contains 1 OOV token, so the effective number of tokens is\n        `(max_tokens - 1 - (1 if output_mode == "int" else 0))`.\n    standardize: Optional specification for standardization to apply to the\n        input text. Values can be:\n        - `None`: No standardization.\n        - `"lower_and_strip_punctuation"`: Text will be lowercased and all\n            punctuation removed.\n        - `"lower"`: Text will be lowercased.\n        - `"strip_punctuation"`: All punctuation will be removed.\n        - Callable: Inputs will passed to the callable function,\n            which should be standardized and returned.\n    split: Optional specification for splitting the input text.\n        Values can be:\n        - `None`: No splitting.\n        - `"whitespace"`: Split on whitespace.\n        - `"character"`: Split on each unicode character.\n        - Callable: Standardized inputs will passed to the callable\n            function, which should be split and returned.\n    ngrams: Optional specification for ngrams to create from the\n        possibly-split input text. Values can be `None`, an integer\n        or tuple of integers; passing an integer will create ngrams\n        up to that integer, and passing a tuple of integers will\n        create ngrams for the specified values in the tuple.\n        Passing `None` means that no ngrams will be created.\n    output_mode: Optional specification for the output of the layer.\n        Values can be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`,\n        configuring the layer as follows:\n        - `"int"`: Outputs integer indices, one integer index per split\n            string token. When `output_mode == "int"`,\n            0 is reserved for masked locations;\n            this reduces the vocab size to `max_tokens - 2`\n            instead of `max_tokens - 1`.\n        - `"multi_hot"`: Outputs a single int array per batch, of either\n            vocab_size or max_tokens size, containing 1s in all elements\n            where the token mapped to that index exists at least\n            once in the batch item.\n        - `"count"`: Like `"multi_hot"`, but the int array contains\n            a count of the number of times the token at that index\n            appeared in the batch item.\n        - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm\n            is applied to find the value in each token slot.\n        For `"int"` output, any shape of input and output is supported.\n        For all other output modes, currently only rank 1 inputs\n        (and rank 2 outputs after splitting) are supported.\n    output_sequence_length: Only valid in INT mode. If set, the output will\n        have its time dimension padded or truncated to exactly\n        `output_sequence_length` values, resulting in a tensor of shape\n        `(batch_size, output_sequence_length)` regardless of how many tokens\n        resulted from the splitting step. Defaults to `None`. If `ragged`\n        is `True` then `output_sequence_length` may still truncate the\n        output.\n    pad_to_max_tokens: Only valid in  `"multi_hot"`, `"count"`,\n        and `"tf_idf"` modes. If `True`, the output will have\n        its feature axis padded to `max_tokens` even if the number\n        of unique tokens in the vocabulary is less than `max_tokens`,\n        resulting in a tensor of shape `(batch_size, max_tokens)`\n        regardless of vocabulary size. Defaults to `False`.\n    vocabulary: Optional. Either an array of strings or a string path to a\n        text file. If passing an array, can pass a tuple, list,\n        1D NumPy array, or 1D tensor containing the string vocabulary terms.\n        If passing a file path, the file should contain one line per term\n        in the vocabulary. If this argument is set,\n        there is no need to `adapt()` the layer.\n    idf_weights: Only valid when `output_mode` is `"tf_idf"`. A tuple, list,\n        1D NumPy array, or 1D tensor of the same length as the vocabulary,\n        containing the floating point inverse document frequency weights,\n        which will be multiplied by per sample term counts for\n        the final `tf_idf` weight. If the `vocabulary` argument is set,\n        and `output_mode` is `"tf_idf"`, this argument must be supplied.\n    ragged: Boolean. Only applicable to `"int"` output mode.\n        Only supported with TensorFlow backend.\n        If `True`, returns a `RaggedTensor` instead of a dense `Tensor`,\n        where each sequence may have a different length\n        after string splitting. Defaults to `False`.\n    sparse: Boolean. Only applicable to `"multi_hot"`, `"count"`, and\n        `"tf_idf"` output modes. Only supported with TensorFlow\n        backend. If `True`, returns a `SparseTensor`\n        instead of a dense `Tensor`. Defaults to `False`.\n    encoding: Optional. The text encoding to use to interpret the input\n        strings. Defaults to `"utf-8"`.\n\nExamples:\n\nThis example instantiates a `TextVectorization` layer that lowercases text,\nsplits on whitespace, strips punctuation, and outputs integer vocab indices.\n\n>>> max_tokens = 5000  # Maximum vocab size.\n>>> max_len = 4  # Sequence length to pad the outputs to.\n>>> # Create the layer.\n>>> vectorize_layer = TextVectorization(\n...     max_tokens=max_tokens,\n...     output_mode=\'int\',\n...     output_sequence_length=max_len)\n\n>>> # Now that the vocab layer has been created, call `adapt` on the\n>>> # list of strings to create the vocabulary.\n>>> vectorize_layer.adapt(["foo bar", "bar baz", "baz bada boom"])\n\n>>> # Now, the layer can map strings to integers -- you can use an\n>>> # embedding layer to map these integers to learned embeddings.\n>>> input_data = [["foo qux bar"], ["qux baz"]]\n>>> vectorize_layer(input_data)\narray([[4, 1, 3, 0],\n       [1, 2, 0, 0]])\n\nThis example instantiates a `TextVectorization` layer by passing a list\nof vocabulary terms to the layer\'s `__init__()` method.\n\n>>> vocab_data = ["earth", "wind", "and", "fire"]\n>>> max_len = 4  # Sequence length to pad the outputs to.\n>>> # Create the layer, passing the vocab directly. You can also pass the\n>>> # vocabulary arg a path to a file containing one vocabulary word per\n>>> # line.\n>>> vectorize_layer = keras.layers.TextVectorization(\n...     max_tokens=max_tokens,\n...     output_mode=\'int\',\n...     output_sequence_length=max_len,\n...     vocabulary=vocab_data)\n\n>>> # Because we\'ve passed the vocabulary directly, we don\'t need to adapt\n>>> # the layer - the vocabulary is already set. The vocabulary contains the\n>>> # padding token (\'\') and OOV token (\'[UNK]\')\n>>> # as well as the passed tokens.\n>>> vectorize_layer.get_vocabulary()\n[\'\', \'[UNK]\', \'earth\', \'wind\', \'and\', \'fire\']',
    "std_args": [
      {"name": "max_tokens", "type": "Any"},
      {"name": "standardize", "type": "Any"},
      {"name": "split", "type": "Any"},
      {"name": "ngrams", "type": "Any"},
      {"name": "output_mode", "type": "Any"},
      {"name": "output_sequence_length", "type": "Any"},
      {"name": "pad_to_max_tokens", "type": "Any"},
      {"name": "vocabulary", "type": "Any"},
      {"name": "idf_weights", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "ragged", "type": "Any"},
      {"name": "encoding", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Threshold": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "threshold", "type": "Any"},
    ],
    "variants": {},
  },
  "Tile": {
    "description": "Constructs a tensor by tiling a given tensor. This is the same as function `tile` in Numpy, but no broadcast. For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]",
    "std_args": [
      {"name": "input", "type": "Tensor"},
      {"name": "repeats", "type": "Any"},
    ],
    "variants": {},
  },
  "TimeDistributed": {
    "description": "This wrapper allows to apply a layer to every temporal slice of an input.\n\nEvery input should be at least 3D, and the dimension of index one of the\nfirst input will be considered to be the temporal dimension.\n\nConsider a batch of 32 video samples, where each sample is a 128x128 RGB\nimage with `channels_last` data format, across 10 timesteps.\nThe batch input shape is `(32, 10, 128, 128, 3)`.\n\nYou can then use `TimeDistributed` to apply the same `Conv2D` layer to each\nof the 10 timesteps, independently:\n\n>>> inputs = layers.Input(shape=(10, 128, 128, 3), batch_size=32)\n>>> conv_2d_layer = layers.Conv2D(64, (3, 3))\n>>> outputs = layers.TimeDistributed(conv_2d_layer)(inputs)\n>>> outputs.shape\n(32, 10, 126, 126, 64)\n\nBecause `TimeDistributed` applies the same instance of `Conv2D` to each of\nthe timestamps, the same set of weights are used at each timestamp.\n\nArgs:\n    layer: a `keras.layers.Layer` instance.\n\nCall arguments:\n    inputs: Input tensor of shape (batch, time, ...) or nested tensors,\n        and each of which has shape (batch, time, ...).\n    training: Python boolean indicating whether the layer should behave in\n        training mode or in inference mode. This argument is passed to the\n        wrapped layer (only if the layer supports this argument).\n    mask: Binary tensor of shape `(samples, timesteps)` indicating whether\n        a given timestep should be masked. This argument is passed to the\n        wrapped layer (only if the layer supports this argument).",
    "std_args": [
      {"name": "layer", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ToTensor": {
    "description": "Convert a PIL Image or numpy.ndarray to tensor.",
    "std_args": [],
    "variants": {},
  },
  "TopK": {
    "description": "Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs: * Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which contains the values of the top k elements ...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "K", "type": "int"},
      {"name": "axis", "type": "int"},
      {"name": "largest", "type": "int"},
      {"name": "sorted", "type": "int"},
    ],
    "variants": {},
  },
  "TorchFunctional": {
    "description": "Abstract Functional Namespace (e.g. F in torch.nn.functional)",
    "std_args": [],
    "variants": {},
  },
  "TorchModuleWrapper": {
    "description": 'Torch module wrapper layer.\n\n`TorchModuleWrapper` is a wrapper class that can turn any\n`torch.nn.Module` into a Keras layer, in particular by making its\nparameters trackable by Keras.\n\n`TorchModuleWrapper` is only compatible with the PyTorch backend and\ncannot be used with the TensorFlow or JAX backends.\n\nArgs:\n    module: `torch.nn.Module` instance. If it\'s a `LazyModule`\n        instance, then its parameters must be initialized before\n        passing the instance to `TorchModuleWrapper` (e.g. by calling\n        it once).\n    output_shape :The shape of the output of this layer. It helps Keras\n        perform automatic shape inference.\n    name: The name of the layer (string).\n\nExample:\n\nHere\'s an example of how the `TorchModuleWrapper` can be used with vanilla\nPyTorch modules.\n\n```python\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nimport keras\nfrom keras.layers import TorchModuleWrapper\n\nclass Classifier(keras.Model):\n    def __init__(self, **kwargs):\n        super().__init__(**kwargs)\n        # Wrap `torch.nn.Module`s with `TorchModuleWrapper`\n        # if they contain parameters\n        self.conv1 = TorchModuleWrapper(\n            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))\n        )\n        self.conv2 = TorchModuleWrapper(\n            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))\n        )\n        self.pool = nn.MaxPool2d(kernel_size=(2, 2))\n        self.flatten = nn.Flatten()\n        self.dropout = nn.Dropout(p=0.5)\n        self.fc = TorchModuleWrapper(nn.Linear(1600, 10))\n\n    def call(self, inputs):\n        x = F.relu(self.conv1(inputs))\n        x = self.pool(x)\n        x = F.relu(self.conv2(x))\n        x = self.pool(x)\n        x = self.flatten(x)\n        x = self.dropout(x)\n        x = self.fc(x)\n        return F.softmax(x, dim=1)\n\n\nmodel = Classifier()\nmodel.build((1, 28, 28))\nprint("Output shape:", model(torch.ones(1, 1, 28, 28).to("cuda")).shape)\n\nmodel.compile(\n    loss="sparse_categorical_crossentropy",\n    optimizer="adam",\n    metrics=["accuracy"]\n)\nmodel.fit(train_loader, epochs=5)\n```',
    "std_args": [
      {"name": "module", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "output_shape", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "TraceState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Transformer": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "activation", "type": "Union"},
      {"name": "custom_decoder", "type": "Optional"},
      {"name": "custom_encoder", "type": "Optional"},
      {"name": "dropout", "type": "float"},
      {"name": "norm_first", "type": "bool"},
      {"name": "num_decoder_layers", "type": "int"},
      {"name": "num_encoder_layers", "type": "int"},
    ],
    "variants": {},
  },
  "TransformerDecoderLayer": {
    "description": "TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "d_model", "type": "Any"},
      {"name": "nhead", "type": "Any"},
      {"name": "dim_feedforward", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "layer_norm_eps", "type": "Any"},
      {"name": "batch_first", "type": "Any"},
      {"name": "norm_first", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "TransformerEncoderLayer": {
    "description": "TransformerEncoderLayer is made up of self-attn and feedforward network.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "d_model", "type": "Any"},
      {"name": "nhead", "type": "Any"},
      {"name": "dim_feedforward", "type": "Any"},
      {"name": "dropout", "type": "Any"},
      {"name": "activation", "type": "Any"},
      {"name": "layer_norm_eps", "type": "Any"},
      {"name": "batch_first", "type": "Any"},
      {"name": "norm_first", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "Transformerdecoder": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "activation", "type": "Union"},
      {"name": "dropout", "type": "float"},
      {"name": "norm_first", "type": "bool"},
    ],
    "variants": {},
  },
  "Transformerencoder": {
    "description": "Auto-discovered via Consensus (Score: 2.0)",
    "std_args": [
      {"name": "activation", "type": "Union"},
      {"name": "dropout", "type": "float"},
      {"name": "norm_first", "type": "bool"},
    ],
    "variants": {},
  },
  "Transpose": {
    "description": "Returns a transpose of the input tensor. (Similar to `numpy.transpose`). The optional attribute `perm` must be a permutation of the dimensions of the input tensor. Axis `i` of the output tensor corresponds to the axis `perm[i]` of the input tensor. For example, when perm=(1, 0, 2), given an input te...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "perm", "type": "List[int]"},
    ],
    "variants": {},
  },
  "UInt8": {
    "description": "8-bit unsigned integer type (Byte).",
    "std_args": [],
    "variants": {},
  },
  "Unflatten": {
    "description": "Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "dim", "type": "Any"},
      {"name": "unflattened_size", "type": "Any"},
    ],
    "variants": {},
  },
  "Unfold": {
    "description": "Extracts sliding local blocks from a batched input tensor.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "stride", "type": "Any"},
    ],
    "variants": {},
  },
  "Unique": {
    "description": "Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned. Otherwise the input tensor is flattened and unique values of the flattened tensor are returned. This operator returns the unique values or sliced unique subten...",
    "std_args": [
      {"name": "X", "type": "Tensor"},
      {"name": "axis", "type": "int"},
      {"name": "sorted", "type": "int"},
    ],
    "variants": {},
  },
  "UnitNormalization": {
    "description": "Unit normalization layer.\n\nNormalize a batch of inputs so that each input in the batch has a L2 norm\nequal to 1 (across the axes specified in `axis`).\n\nExample:\n\n>>> data = np.arange(6).reshape(2, 3)\n>>> normalized_data = keras.layers.UnitNormalization()(data)\n>>> np.sum(normalized_data[0, :] ** 2)\n1.0\n\nArgs:\n    axis: Integer or list/tuple. The axis or axes to normalize across.\n        Typically, this is the features axis or axes. The left-out axes are\n        typically the batch axis or axes. `-1` is the last dimension\n        in the input. Defaults to `-1`.",
    "std_args": [
      {"name": "axis", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "Unsqueeze": {
    "description": "Insert single-dimensional entries to the shape of an input tensor (`data`). Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`). For example, given an input ...",
    "std_args": [
      {"name": "data", "type": "Tensor"},
      {"name": "axes", "type": "int"},
    ],
    "variants": {},
  },
  "UpSampling1D": {
    "description": "Upsampling layer for 1D inputs.\n\nRepeats each temporal step `size` times along the time axis.\n\nExample:\n\n>>> input_shape = (2, 2, 3)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> x\n[[[ 0  1  2]\n  [ 3  4  5]]\n [[ 6  7  8]\n  [ 9 10 11]]]\n>>> y = keras.layers.UpSampling1D(size=2)(x)\n>>> y\n[[[ 0.  1.  2.]\n  [ 0.  1.  2.]\n  [ 3.  4.  5.]\n  [ 3.  4.  5.]]\n [[ 6.  7.  8.]\n  [ 6.  7.  8.]\n  [ 9. 10. 11.]\n  [ 9. 10. 11.]]]\n\nArgs:\n    size: Integer. Upsampling factor.\n\nInput shape:\n    3D tensor with shape: `(batch_size, steps, features)`.\n\nOutput shape:\n    3D tensor with shape: `(batch_size, upsampled_steps, features)`.",
    "std_args": [
      {"name": "size", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "UpSampling2D": {
    "description": 'Upsampling layer for 2D inputs.\n\nThe implementation uses interpolative resizing, given the resize method\n(specified by the `interpolation` argument). Use `interpolation=nearest`\nto repeat the rows and columns of the data.\n\nExample:\n\n>>> input_shape = (2, 2, 1, 3)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> print(x)\n[[[[ 0  1  2]]\n  [[ 3  4  5]]]\n [[[ 6  7  8]]\n  [[ 9 10 11]]]]\n>>> y = keras.layers.UpSampling2D(size=(1, 2))(x)\n>>> print(y)\n[[[[ 0  1  2]\n   [ 0  1  2]]\n  [[ 3  4  5]\n   [ 3  4  5]]]\n [[[ 6  7  8]\n   [ 6  7  8]]\n  [[ 9 10 11]\n   [ 9 10 11]]]]\n\nArgs:\n    size: Int, or tuple of 2 integers.\n        The upsampling factors for rows and columns.\n    data_format: A string,\n        one of `"channels_last"` (default) or `"channels_first"`.\n        The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, height, width, channels)` while `"channels_first"`\n        corresponds to inputs with shape\n        `(batch_size, channels, height, width)`.\n        When unspecified, uses\n        `image_data_format` value found in your Keras config file at\n        `~/.keras/keras.json` (if exists) else `"channels_last"`.\n        Defaults to `"channels_last"`.\n    interpolation: A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,\n        `"lanczos5"`, `"nearest"`.\n\nInput shape:\n    4D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n        `(batch_size, rows, cols, channels)`\n    - If `data_format` is `"channels_first"`:\n        `(batch_size, channels, rows, cols)`\n\nOutput shape:\n    4D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n        `(batch_size, upsampled_rows, upsampled_cols, channels)`\n    - If `data_format` is `"channels_first"`:\n        `(batch_size, channels, upsampled_rows, upsampled_cols)`',
    "std_args": [
      {"name": "size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "interpolation", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "UpSampling3D": {
    "description": 'Upsampling layer for 3D inputs.\n\nRepeats the 1st, 2nd and 3rd dimensions\nof the data by `size[0]`, `size[1]` and `size[2]` respectively.\n\nExample:\n\n>>> input_shape = (2, 1, 2, 1, 3)\n>>> x = np.ones(input_shape)\n>>> y = keras.layers.UpSampling3D(size=(2, 2, 2))(x)\n>>> y.shape\n(2, 2, 4, 2, 3)\n\nArgs:\n    size: Int, or tuple of 3 integers.\n        The upsampling factors for dim1, dim2 and dim3.\n    data_format: A string,\n        one of `"channels_last"` (default) or `"channels_first"`.\n        The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        When unspecified, uses\n        `image_data_format` value found in your Keras config file at\n         `~/.keras/keras.json` (if exists) else `"channels_last"`.\n        Defaults to `"channels_last"`.\n\nInput shape:\n    5D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n        `(batch_size, dim1, dim2, dim3, channels)`\n    - If `data_format` is `"channels_first"`:\n        `(batch_size, channels, dim1, dim2, dim3)`\n\nOutput shape:\n    5D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,\n        channels)`\n    - If `data_format` is `"channels_first"`:\n        `(batch_size, channels, upsampled_dim1, upsampled_dim2,\n        upsampled_dim3)`',
    "std_args": [
      {"name": "size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "UpdateCvState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Upsample": {
    "description": "Upsample the input tensor. Each dimension value of the output tensor is: output_dimension = floor(input_dimension * scale).",
    "std_args": [],
    "variants": {},
  },
  "V": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Variable": {
    "description": "Generic state container.",
    "std_args": [
      {"name": "value", "type": "Any"},
    ],
    "variants": {},
  },
  "View": {
    "description": "Returns a new tensor with the same data and elements but different shape.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "shape", "type": "Any"},
    ],
    "variants": {},
  },
  "WeightNorm": {
    "description": "The class representing a Python class.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "dim", "type": "Any"},
    ],
    "variants": {},
  },
  "Where": {
    "description": "Return elements, either from X or Y, depending on condition. Where behaves like [numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html) with three parameters. This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the...",
    "std_args": [
      {"name": "condition", "type": "Any"},
      {"name": "X", "type": "Tensor"},
      {"name": "Y", "type": "Tensor"},
    ],
    "variants": {},
  },
  "WrappedSchedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "Wrapper": {
    "description": "Abstract wrapper base class.\n\nWrappers take another layer and augment it in various ways.\nDo not use this class as a layer, it is only an abstract base class.\nTwo usable wrappers are the `TimeDistributed` and `Bidirectional` layers.\n\nArgs:\n    layer: The layer to be wrapped.",
    "std_args": [
      {"name": "layer", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ZeroNansState": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ZeroPadding1D": {
    "description": 'Zero-padding layer for 1D input (e.g. temporal sequence).\n\nExample:\n\n>>> input_shape = (2, 2, 3)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> x\n[[[ 0  1  2]\n  [ 3  4  5]]\n [[ 6  7  8]\n  [ 9 10 11]]]\n>>> y = keras.layers.ZeroPadding1D(padding=2)(x)\n>>> y\n[[[ 0  0  0]\n  [ 0  0  0]\n  [ 0  1  2]\n  [ 3  4  5]\n  [ 0  0  0]\n  [ 0  0  0]]\n [[ 0  0  0]\n  [ 0  0  0]\n  [ 6  7  8]\n  [ 9 10 11]\n  [ 0  0  0]\n  [ 0  0  0]]]\n\nArgs:\n    padding: Int, or tuple of int (length 2), or dictionary.\n        - If int: how many zeros to add at the beginning and end of\n          the padding dimension (axis 1).\n        - If tuple of 2 ints: how many zeros to add at the beginning and the\n          end of the padding dimension (`(left_pad, right_pad)`).\n    data_format: A string, one of `"channels_last"` (default) or\n        `"channels_first"`. The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, axis_to_pad, channels)` while `"channels_first"`\n        corresponds to inputs with shape\n        `(batch_size, channels, axis_to_pad)`.\n        When unspecified, uses `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json` (if exists). Defaults to\n        `"channels_last"`.\n\nInput shape:\n    3D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, axis_to_pad, features)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, features, axis_to_pad)`\n\nOutput shape:\n    3D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, padded_axis, features)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, features, padded_axis)`',
    "std_args": [
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ZeroPadding2D": {
    "description": 'Zero-padding layer for 2D input (e.g. picture).\n\nThis layer can add rows and columns of zeros at the top, bottom, left and\nright side of an image tensor.\n\nExample:\n\n>>> input_shape = (1, 1, 2, 2)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> x\n[[[[0 1]\n   [2 3]]]]\n>>> y = keras.layers.ZeroPadding2D(padding=1)(x)\n>>> y\n[[[[0 0]\n   [0 0]\n   [0 0]\n   [0 0]]\n  [[0 0]\n   [0 1]\n   [2 3]\n   [0 0]]\n  [[0 0]\n   [0 0]\n   [0 0]\n   [0 0]]]]\n\nArgs:\n    padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.\n        - If int: the same symmetric padding is applied to height and width.\n        - If tuple of 2 ints: interpreted as two different symmetric padding\n          values for height and width:\n          `(symmetric_height_pad, symmetric_width_pad)`.\n        - If tuple of 2 tuples of 2 ints: interpreted as\n         `((top_pad, bottom_pad), (left_pad, right_pad))`.\n    data_format: A string, one of `"channels_last"` (default) or\n        `"channels_first"`. The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, height, width, channels)` while `"channels_first"`\n        corresponds to inputs with shape\n        `(batch_size, channels, height, width)`.\n        When unspecified, uses `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json` (if exists). Defaults to\n        `"channels_last"`.\n\nInput shape:\n    4D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, height, width, channels)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, channels, height, width)`\n\nOutput shape:\n    4D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, padded_height, padded_width, channels)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, channels, padded_height, padded_width)`',
    "std_args": [
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "ZeroPadding3D": {
    "description": 'Zero-padding layer for 3D data (spatial or spatio-temporal).\n\nExample:\n\n>>> input_shape = (1, 1, 2, 2, 3)\n>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)\n>>> y = keras.layers.ZeroPadding3D(padding=2)(x)\n>>> y.shape\n(1, 5, 6, 6, 3)\n\nArgs:\n    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.\n        - If int: the same symmetric padding is applied to depth, height,\n          and width.\n        - If tuple of 3 ints: interpreted as three different symmetric\n          padding values for depth, height, and width:\n          `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.\n        - If tuple of 3 tuples of 2 ints: interpreted as\n          `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,\n          right_dim2_pad), (left_dim3_pad, right_dim3_pad))`.\n    data_format: A string, one of `"channels_last"` (default) or\n        `"channels_first"`. The ordering of the dimensions in the inputs.\n        `"channels_last"` corresponds to inputs with shape\n        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`\n        while `"channels_first"` corresponds to inputs with shape\n        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.\n        When unspecified, uses `image_data_format` value found in your Keras\n        config file at `~/.keras/keras.json` (if exists). Defaults to\n        `"channels_last"`.\n\nInput shape:\n    5D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, first_axis_to_pad, second_axis_to_pad,\n      third_axis_to_pad, depth)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,\n      third_axis_to_pad)`\n\nOutput shape:\n    5D tensor with shape:\n    - If `data_format` is `"channels_last"`:\n      `(batch_size, first_padded_axis, second_padded_axis,\n      third_axis_to_pad, depth)`\n    - If `data_format` is `"channels_first"`:\n      `(batch_size, depth, first_padded_axis, second_padded_axis,\n      third_axis_to_pad)`',
    "std_args": [
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "__array_namespace_info__": {
    "description": "Returns a namespace with Array API namespace inspection utilities.",
    "std_args": [],
    "variants": {},
  },
  "abs": {
    "description": "Calculates the absolute value for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "absolute": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "acos": {
    "description": "Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "acosh": {
    "description": "Calculates an implementation-dependent approximation to the inverse hyperbolic cosine for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "activation": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "activation_relu_or_gelu": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "adaptive_average_pool": {
    "description": 'Adaptive average pooling operation.\n\nApplies an adaptive average pooling operation that automatically\ncomputes the kernel size and stride to pool the input to the\nspecified `output_size`. This operation is useful when you want a\nfixed output size regardless of input size, commonly used in models\nlike ResNet for global feature extraction.\n\nArgs:\n    inputs: Tensor of rank 4. Input tensor of shape:\n        - If `data_format="channels_last"`:\n            `(batch_size, height, width, channels)`.\n        - If `data_format="channels_first"`:\n            `(batch_size, channels, height, width)`.\n    output_size: Integer or tuple/list of 2 integers, specifying the target\n        output spatial dimensions `(output_height, output_width)`. If a\n        single\n        integer is provided, the same value is used for both dimensions.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, defaults to `"channels_last"`.\n\nReturns:\n    A tensor of rank 4 representing the adaptive average pooled result.\n\nExample:\n\n>>> x = np.random.rand(2, 64, 64, 3)\n>>> y = keras.ops.adaptive_average_pool(x, output_size=(32, 32))\n>>> y.shape\n(2, 32, 32, 3)\n\n>>> # Works with any input size\n>>> x = np.random.rand(2, 100, 80, 3)\n>>> y = keras.ops.adaptive_average_pool(x, output_size=7)\n>>> y.shape\n(2, 7, 7, 3)',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
    ],
    "variants": {},
  },
  "adaptive_grad_clip": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "adaptive_max_pool": {
    "description": 'Adaptive max pooling operation.\n\nApplies an adaptive max pooling operation that automatically computes the\nkernel size and stride to pool the input to the specified `output_size`.\nThis operation is useful when you want a fixed output size regardless of\ninput size, commonly used in models like ResNet for global feature\nextraction.\nArgs:\n    inputs: Tensor of rank 4. Input tensor of shape:\n        - If `data_format="channels_last"`:\n            `(batch_size, height, width, channels)`.\n        - If `data_format="channels_first"`:\n            `(batch_size, channels, height, width)`.\n    output_size: Integer or tuple/list of 2 integers, specifying the target\n        output spatial dimensions `(output_height, output_width)`. If a\n        single\n        integer is provided, the same value is used for both dimensions.\n    data_format: string, either `"channels_last"` or `"channels_first"`.\n        Defaults to the value found in your Keras config file at\n        `~/.keras/keras.json`. If never set, defaults to `"channels_last"`.\n\nReturns:\n    A tensor of rank 4 representing the adaptive max pooled result.\n\nExample:\n\n>>> x = np.random.rand(2, 64, 64, 3)\n>>> y = keras.ops.adaptive_max_pool(x, output_size=(32, 32))\n>>> y.shape\n(2, 32, 32, 3)\n\n>>> # Works with any input size\n>>> x = np.random.rand(2, 100, 80, 3)\n>>> y = keras.ops.adaptive_max_pool(x, output_size=7)\n>>> y.shape\n(2, 7, 7, 3)',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "output_size", "type": "Any"},
      {"name": "data_format", "type": "Any"},
    ],
    "variants": {},
  },
  "add": {
    "description": "Calculates the sum for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex]"},
      {"name": "x2", "type": "Union[array, int, float, complex]"},
    ],
    "variants": {},
  },
  "add_decayed_weights": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "add_noise": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "add_scale": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "addressable_data": {
    "description": "Return an array of the addressable data at a particular index.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "index", "type": "Any"},
    ],
    "variants": {},
  },
  "align_corners": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "all": {
    "description": "Tests whether all input array elements evaluate to ``True`` along a specified axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "allclose": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "alpha": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "amax": {
    "description": "Alias of :func:`jax.numpy.max`.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "initial", "type": "Any"},
      {"name": "where", "type": "Any"},
    ],
    "variants": {},
  },
  "amin": {
    "description": "Alias of :func:`jax.numpy.min`.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "initial", "type": "Any"},
      {"name": "where", "type": "Any"},
    ],
    "variants": {},
  },
  "angle": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "any": {
    "description": "Tests whether any input array element evaluates to ``True`` along a specified axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "append": {
    "description": "Append a given value at the end of the list.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "value", "type": "Any"},
    ],
    "variants": {},
  },
  "apply": {
    "description": "Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "fn", "type": "Any"},
    ],
    "variants": {},
  },
  "apply_along_axis": {
    "description": "Apply a function to 1D array slices along an axis.\n\nJAX implementation of :func:`numpy.apply_along_axis`. While NumPy implements\nthis iteratively, JAX implements this via :func:`jax.vmap`, and so ``func1d``\nmust be compatible with ``vmap``.\n\nArgs:\n  func1d: a callable function with signature ``func1d(arr, /, *args, **kwargs)``\n    where ``*args`` and ``**kwargs`` are the additional positional and keyword\n    arguments passed to :func:`apply_along_axis`.\n  axis: integer axis along which to apply the function.\n  arr: the array over which to apply the function.\n  args, kwargs: additional positional and keyword arguments are passed through\n    to ``func1d``.\n\nReturns:\n  The result of ``func1d`` applied along the specified axis.\n\nSee also:\n  - :func:`jax.vmap`: a more direct way to create a vectorized version of a function.\n  - :func:`jax.numpy.apply_over_axes`: repeatedly apply a function over multiple axes.\n  - :func:`jax.numpy.vectorize`: create a vectorized version of a function.\n\nExamples:\n  A simple example in two dimensions, where the function is applied either row-wise\n  or column-wise:\n\n  >>> x = jnp.array([[1, 2, 3],\n  ...                [4, 5, 6]])\n  >>> def func1d(x):\n  ...   return jnp.sum(x ** 2)\n  >>> jnp.apply_along_axis(func1d, 0, x)\n  Array([17, 29, 45], dtype=int32)\n  >>> jnp.apply_along_axis(func1d, 1, x)\n  Array([14, 77], dtype=int32)\n\n  For 2D inputs, this can be equivalently expressed using :func:`jax.vmap`,\n  though note that `vmap` specifies the mapped axis rather than the applied axis:\n\n  >>> jax.vmap(func1d, in_axes=1)(x)  # same as applying along axis 0\n  Array([17, 29, 45], dtype=int32)\n  >>> jax.vmap(func1d, in_axes=0)(x)  # same as applying along axis 1\n  Array([14, 77], dtype=int32)\n\n  For 3D inputs, :func:`apply_along_axis` is equivalent to mapping over two\n  dimensions:\n\n  >>> x_3d = jnp.arange(24).reshape(2, 3, 4)\n  >>> jnp.apply_along_axis(func1d, 2, x_3d)\n  Array([[  14,  126,  366],\n         [ 734, 1230, 1854]], dtype=int32)\n  >>> jax.vmap(jax.vmap(func1d))(x_3d)\n  Array([[  14,  126,  366],\n         [ 734, 1230, 1854]], dtype=int32)\n\n  The applied function may also take arbitrary positional or keyword arguments,\n  which should be passed directly as additional arguments to :func:`apply_along_axis`:\n\n  >>> def func1d(x, exponent):\n  ...   return jnp.sum(x ** exponent)\n  >>> jnp.apply_along_axis(func1d, 0, x, exponent=3)\n  Array([ 65, 133, 243], dtype=int32)",
    "std_args": [
      {"name": "func1d", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "arr", "type": "Any"},
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "apply_gradients": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "grads", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "apply_if_finite": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "apply_mask": {
    "description": "Simply handles the multiplication between the parameter being pruned and the generated mask.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "module", "type": "Any"},
    ],
    "variants": {},
  },
  "apply_over_axes": {
    "description": "Apply a function repeatedly over specified axes.\n\nJAX implementation of :func:`numpy.apply_over_axes`.\n\nArgs:\n  func: the function to apply, with signature ``func(Array, int) -> Array``, and\n    where ``y = func(x, axis)`` must satisfy ``y.ndim in [x.ndim, x.ndim - 1]``.\n  a: N-dimensional array over which to apply the function.\n  axes: the sequence of axes over which to apply the function.\n\nReturns:\n  An N-dimensional array containing the result of the repeated function application.\n\nSee also:\n  - :func:`jax.numpy.apply_along_axis`: apply a 1D function along a single axis.\n\nExamples:\n  This function is designed to have similar semantics to typical associative\n  :mod:`jax.numpy` reductions over one or more axes with ``keepdims=True``.\n  For example:\n\n  >>> x = jnp.array([[1, 2, 3],\n  ...                [4, 5, 6]])\n\n  >>> jnp.apply_over_axes(jnp.sum, x, [0])\n  Array([[5, 7, 9]], dtype=int32)\n  >>> jnp.sum(x, [0], keepdims=True)\n  Array([[5, 7, 9]], dtype=int32)\n\n  >>> jnp.apply_over_axes(jnp.min, x, [1])\n  Array([[1],\n         [4]], dtype=int32)\n  >>> jnp.min(x, [1], keepdims=True)\n  Array([[1],\n         [4]], dtype=int32)\n\n  >>> jnp.apply_over_axes(jnp.prod, x, [0, 1])\n  Array([[720]], dtype=int32)\n  >>> jnp.prod(x, [0, 1], keepdims=True)\n  Array([[720]], dtype=int32)",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "a", "type": "Any"},
      {"name": "axes", "type": "Any"},
    ],
    "variants": {},
  },
  "approximate": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arange": {
    "description": "Returns evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array.",
    "std_args": [
      {"name": "start", "type": "Union[int, float]"},
      {"name": "stop", "type": "Optional[Union[int, float]]"},
      {"name": "step", "type": "Union[int, float]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "arccos": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arccosh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arcsin": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arcsinh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arctan": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arctan2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "arctanh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "argmax": {
    "description": "Returns the indices of the maximum values along a specified axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[int]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "argmin": {
    "description": "Returns the indices of the minimum values along a specified axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[int]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "argpartition": {
    "description": "Return the indices that partially sort the array.\n\nRefer to :func:`jax.numpy.argpartition` for the full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "kth", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "args": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "argsort": {
    "description": "Returns the indices that sort an array ``x`` along a specified axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "int"},
      {"name": "descending", "type": "bool"},
      {"name": "stable", "type": "bool"},
    ],
    "variants": {},
  },
  "argwhere": {
    "description": "Find the indices of nonzero array elements\n\nJAX implementation of :func:`numpy.argwhere`.\n\n``jnp.argwhere(x)`` is essentially equivalent to ``jnp.column_stack(jnp.nonzero(x))``\nwith special handling for zero-dimensional (i.e. scalar) inputs.\n\nBecause the size of the output of ``argwhere`` is data-dependent, the function is not\ntypically compatible with JIT. The JAX version adds the optional ``size`` argument, which\nspecifies the size of the leading dimension of the output - it must be specified statically\nfor ``jnp.argwhere`` to be compiled with non-static operands. See :func:`jax.numpy.nonzero`\nfor a full discussion of ``size`` and its semantics.\n\nArgs:\n  a: array for which to find nonzero elements\n  size: optional integer specifying statically the number of expected nonzero elements.\n    This must be specified in order to use ``argwhere`` within JAX transformations like\n    :func:`jax.jit`. See :func:`jax.numpy.nonzero` for more information.\n  fill_value: optional array specifying the fill value when ``size`` is specified.\n    See :func:`jax.numpy.nonzero` for more information.\n\nReturns:\n  a two-dimensional array of shape ``[size, x.ndim]``. If ``size`` is not specified as\n  an argument, it is equal to the number of nonzero elements in ``x``.\n\nSee Also:\n  - :func:`jax.numpy.where`\n  - :func:`jax.numpy.nonzero`\n\nExamples:\n  Two-dimensional array:\n\n  >>> x = jnp.array([[1, 0, 2],\n  ...                [0, 3, 0]])\n  >>> jnp.argwhere(x)\n  Array([[0, 0],\n         [0, 2],\n         [1, 1]], dtype=int32)\n\n  Equivalent computation using :func:`jax.numpy.column_stack` and :func:`jax.numpy.nonzero`:\n\n  >>> jnp.column_stack(jnp.nonzero(x))\n  Array([[0, 0],\n         [0, 2],\n         [1, 1]], dtype=int32)\n\n  Special case for zero-dimensional (i.e. scalar) inputs:\n\n  >>> jnp.argwhere(1)\n  Array([], shape=(1, 0), dtype=int32)\n  >>> jnp.argwhere(0)\n  Array([], shape=(0, 0), dtype=int32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "array": {
    "description": "Constant: array",
    "std_args": [],
    "variants": {},
  },
  "array_equal": {
    "description": "Check if two arrays are element-wise equal.\n\nJAX implementation of :func:`numpy.array_equal`.\n\nArgs:\n  a1: first input array to compare.\n  a2: second input array to compare.\n  equal_nan: Boolean. If ``True``, NaNs in ``a1`` will be considered\n    equal to NaNs in ``a2``. Default is ``False``.\n\nReturns:\n  Boolean scalar array indicating whether the input arrays are element-wise equal.\n\nSee Also:\n  - :func:`jax.numpy.allclose`\n  - :func:`jax.numpy.array_equiv`\n\nExamples:\n  >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))\n  Array(True, dtype=bool)\n  >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2]))\n  Array(False, dtype=bool)\n  >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))\n  Array(False, dtype=bool)\n  >>> jnp.array_equal(jnp.array([1, 2, float('nan')]),\n  ...                 jnp.array([1, 2, float('nan')]))\n  Array(False, dtype=bool)\n  >>> jnp.array_equal(jnp.array([1, 2, float('nan')]),\n  ...                 jnp.array([1, 2, float('nan')]), equal_nan=True)\n  Array(True, dtype=bool)",
    "std_args": [
      {"name": "a1", "type": "Any"},
      {"name": "a2", "type": "Any"},
      {"name": "equal_nan", "type": "Any"},
    ],
    "variants": {},
  },
  "array_equiv": {
    "description": "Check if two arrays are element-wise equal.\n\nJAX implementation of :func:`numpy.array_equiv`.\n\nThis function will return ``False`` if the input arrays cannot be broadcasted\nto the same shape.\n\nArgs:\n  a1: first input array to compare.\n  a2: second input array to compare.\n\nReturns:\n  Boolean scalar array indicating whether the input arrays are\n  element-wise equal after broadcasting.\n\nSee Also:\n  - :func:`jax.numpy.allclose`\n  - :func:`jax.numpy.array_equal`\n\nExamples:\n  >>> jnp.array_equiv(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))\n  Array(True, dtype=bool)\n  >>> jnp.array_equiv(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))\n  Array(False, dtype=bool)\n  >>> jnp.array_equiv(jnp.array([[1, 2, 3], [1, 2, 3]]),\n  ...                 jnp.array([1, 2, 3]))\n  Array(True, dtype=bool)",
    "std_args": [
      {"name": "a1", "type": "Any"},
      {"name": "a2", "type": "Any"},
    ],
    "variants": {},
  },
  "array_split": {
    "description": "Split an array into sub-arrays.\n\nJAX implementation of :func:`numpy.array_split`.\n\nRefer to the documentation of :func:`jax.numpy.split` for details; ``array_split``\nis equivalent to ``split``, but allows integer ``indices_or_sections`` which does\nnot evenly divide the split axis.\n\nExamples:\n  >>> x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])\n  >>> chunks = jnp.array_split(x, 4)\n  >>> print(*chunks)\n  [1 2 3] [4 5] [6 7] [8 9]\n\nSee also:\n  - :func:`jax.numpy.split`: split an array along any axis.\n  - :func:`jax.numpy.vsplit`: split vertically, i.e. along axis=0\n  - :func:`jax.numpy.hsplit`: split horizontally, i.e. along axis=1\n  - :func:`jax.numpy.dsplit`: split depth-wise, i.e. along axis=2",
    "std_args": [
      {"name": "ary", "type": "Any"},
      {"name": "indices_or_sections", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "as_string": {
    "description": "Return object as STRING expression (string literal constant).",
    "std_args": [
      {"name": "obj", "type": "Any"},
      {"name": "kind", "type": "Any"},
    ],
    "variants": {},
  },
  "asarray": {
    "description": "Convert the input to an array.",
    "std_args": [
      {"name": "obj", "type": "Union[array, bool, int, float, complex, NestedSequence, SupportsBufferProtocol]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
      {"name": "copy", "type": "Optional[bool]"},
    ],
    "variants": {},
  },
  "asin": {
    "description": "Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "asinh": {
    "description": "Calculates an implementation-dependent approximation to the inverse hyperbolic sine for each element ``x_i`` in the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "assert_equal": {
    "description": "Asserts that two items are equal.",
    "std_args": [
      {"name": "actual", "type": "Any"},
      {"name": "desired", "type": "Any"},
      {"name": "err_msg", "type": "Any"},
    ],
    "variants": {},
  },
  "associative_scan": {
    "description": "Performs a scan with an associative binary operation, in parallel.\n\nThis operation his similar to `scan`, with the key difference that\n`associative_scan` is a parallel implementation with\npotentially significant performance benefits, especially when jit compiled.\nThe catch is that it can only be used when `f` is a binary associative\noperation (i.e. it must verify `f(a, f(b, c)) == f(f(a, b), c)`).\n\nFor an introduction to associative scans, refer to this paper:\nBlelloch, Guy E. 1990.\n[Prefix Sums and Their Applications](\n    https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).\n\nArgs:\n    f: A Python callable implementing an associative binary operation with\n        signature `r = f(a, b)`. Function `f` must be associative, i.e.,\n        it must satisfy the equation\n        `f(a, f(b, c)) == f(f(a, b), c)`.\n        The inputs and result are (possibly nested Python tree structures\n        of) array(s) matching `elems`. Each array has a dimension in place\n        of the `axis` dimension. `f` should be applied elementwise over\n        the `axis` dimension.\n        The result `r` has the same shape (and structure) as the\n        two inputs `a` and `b`.\n    elems: A (possibly nested Python tree structure of) array(s), each with\n        an `axis` dimension of size `num_elems`.\n    reverse: A boolean stating if the scan should be reversed with respect\n        to the `axis` dimension.\n    axis: an integer identifying the axis over which the scan should occur.\n\nReturns:\n    A (possibly nested Python tree structure of) array(s) of the same shape\n    and structure as `elems`, in which the `k`'th element of `axis` is\n    the result of recursively applying `f` to combine the first `k`\n    elements of `elems` along `axis`. For example, given\n    `elems = [a, b, c, ...]`, the result would be\n    `[a, f(a, b), f(f(a, b), c), ...]`.\n\nExamples:\n\n>>> sum_fn = lambda x, y: x + y\n>>> xs = keras.ops.arange(5)\n>>> ys = keras.ops.associative_scan(sum_fn, xs, axis=0)\n>>> ys\n[0, 1, 3, 6, 10]\n\n>>> sum_fn = lambda x, y: [x[0] + y[0], x[1] + y[1], x[2] + y[2]]\n>>> xs = [keras.ops.array([[1, 2]]) for _ in range(3)]\n>>> ys = keras.ops.associative_scan(sum_fn, xs, axis=0)\n>>> ys\n[[1, 3], [1, 3], [1, 3]]",
    "std_args": [
      {"name": "f", "type": "Any"},
      {"name": "elems", "type": "Any"},
      {"name": "reverse", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "astype": {
    "description": "Copies an array to a specified data type irrespective of :ref:`type-promotion` rules.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "dtype", "type": "dtype"},
      {"name": "copy", "type": "bool"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "at": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "a", "type": "Any"},
      {"name": "indices", "type": "Any"},
      {"name": "b", "type": "Any"},
    ],
    "variants": {},
  },
  "atan": {
    "description": "Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "atan2": {
    "description": "Calculates an implementation-dependent approximation of the inverse tangent of the quotient ``x1/x2``, having domain ``[-infinity, +infinity] x [-infinity, +infinity]`` (where the ``x`` notation denotes the set of ordered pairs of elements ``(x1_i, x2_i)``) and codomain ``[-\u03c0, +\u03c0]``, for each pair of elements ``(x1_i, x2_i)`` of the input arrays ``x1`` and ``x2``, respectively. Each element-wise result is expressed in radians.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "atanh": {
    "description": "Calculates an implementation-dependent approximation to the inverse hyperbolic tangent for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "atleast_1d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "atleast_2d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "atleast_3d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "attr": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "average": {
    "description": "Compute the weighed average.\n\nJAX Implementation of :func:`numpy.average`.\n\nArgs:\n  a: array to be averaged\n  axis: an optional integer or sequence of integers specifying the axis along which\n    the mean to be computed. If not specified, mean is computed along all the axes.\n  weights: an optional array of weights for a weighted average. This must either exactly\n    match the shape of `a`, or if `axis` is specified, it must have shape ``a.shape[axis]``\n    for a single axis, or shape ``tuple(a.shape[ax] for ax in axis)`` for multiple axes.\n  returned: If False (default) then return only the average. If True then return both\n    the average and the normalization factor (i.e. the sum of weights).\n  keepdims: If True, reduced axes are left in the result with size 1. If False (default)\n    then reduced axes are squeezed out.\n\nReturns:\n  An array ``average`` or tuple of arrays ``(average, normalization)`` if\n  ``returned`` is True.\n\nSee also:\n  - :func:`jax.numpy.mean`: unweighted mean.\n\nExamples:\n  Simple average:\n\n  >>> x = jnp.array([1, 2, 3, 2, 4])\n  >>> jnp.average(x)\n  Array(2.4, dtype=float32)\n\n  Weighted average:\n\n  >>> weights = jnp.array([2, 1, 3, 2, 2])\n  >>> jnp.average(x, weights=weights)\n  Array(2.5, dtype=float32)\n\n  Use ``returned=True`` to optionally return the normalization, i.e. the\n  sum of weights:\n\n  >>> jnp.average(x, returned=True)\n  (Array(2.4, dtype=float32), Array(5., dtype=float32))\n  >>> jnp.average(x, weights=weights, returned=True)\n  (Array(2.5, dtype=float32), Array(10., dtype=float32))\n\n  Weighted average along a specified axis:\n\n  >>> x = jnp.array([[8, 2, 7],\n  ...                [3, 6, 4]])\n  >>> weights = jnp.array([1, 2, 3])\n  >>> jnp.average(x, weights=weights, axis=1)\n  Array([5.5, 4.5], dtype=float32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "weights", "type": "Any"},
      {"name": "returned", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
    ],
    "variants": {},
  },
  "average_pool": {
    "description": 'Average pooling operation.\n\nArgs:\n    inputs: Tensor of rank N+2. `inputs` has shape\n        `(batch_size,) + inputs_spatial_shape + (num_channels,)` if\n        `data_format="channels_last"`, or\n        `(batch_size, num_channels) + inputs_spatial_shape` if\n        `data_format="channels_first"`. Pooling happens over the spatial\n        dimensions only.\n    pool_size: int or tuple/list of integers of size\n        `len(inputs_spatial_shape)`, specifying the size of the pooling\n        window for each spatial dimension of the input tensor. If\n        `pool_size` is int, then every spatial dimension shares the same\n        `pool_size`.\n    strides: int or tuple/list of integers of size\n        `len(inputs_spatial_shape)`. The stride of the sliding window for\n        each spatial dimension of the input tensor. If `strides` is int,\n        then every spatial dimension shares the same `strides`.\n    padding: string, either `"valid"` or `"same"`. `"valid"` means no\n        padding is applied, and `"same"` results in padding evenly to the\n        left/right or up/down of the input such that output has the\n        same height/width dimension as the input when `strides=1`.\n    data_format: A string, either `"channels_last"` or `"channels_first"`.\n        `data_format` determines the ordering of the dimensions in the\n        inputs. If `data_format="channels_last"`, `inputs` is of shape\n        `(batch_size, ..., channels)` while if\n        `data_format="channels_first"`, `inputs` is of shape\n        `(batch_size, channels, ...)`.\n\nReturns:\n    A tensor of rank N+2, the result of the average pooling operation.',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
    ],
    "variants": {},
  },
  "axis": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bartlett": {
    "description": "Return a Bartlett window of size M.\n\nJAX implementation of :func:`numpy.bartlett`.\n\nArgs:\n  M: The window size.\n\nReturns:\n  An array of size M containing the Bartlett window.\n\nExamples:\n  >>> with jnp.printoptions(precision=2, suppress=True):\n  ...   print(jnp.bartlett(4))\n  [0.   0.67 0.67 0.  ]\n\nSee also:\n  - :func:`jax.numpy.blackman`: return a Blackman window of size M.\n  - :func:`jax.numpy.hamming`: return a Hamming window of size M.\n  - :func:`jax.numpy.hanning`: return a Hanning window of size M.\n  - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.",
    "std_args": [
      {"name": "M", "type": "Any"},
    ],
    "variants": {},
  },
  "base": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "batch_normalization": {
    "description": "Normalizes `x` by `mean` and `variance`.\n\nThis op is typically used by the batch normalization step in a neural\nnetwork. It normalizes the input tensor along the given axis.\n\nArgs:\n    x: Input tensor.\n    mean: A mean vector of the same length as the `axis` dimension of the\n        input thensor.\n    variance: A variance vector of the same length as the `axis` dimension\n        of the input tensor.\n    axis: Integer, the axis that should be normalized.\n    offset: An offset vector of the same length as the `axis` dimension of\n        the input tensor. If not `None`, `offset` is added to the normalized\n        tensor. Defaults to `None`.\n    scale: A scale vector of the same length as the `axis` dimension of the\n        input tensor. If not `None`, the normalized tensor is multiplied by\n        `scale`. Defaults to `None`.\n    epsilon: Small float added to variance to avoid dividing by zero.\n        Defaults to 1e-3.\n\nReturns:\n    The normalized tensor.\n\nExample:\n\n>>> x = keras.ops.convert_to_tensor(\n...     [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]\n... )\n>>> keras.ops.batch_normalization(\n...     x,\n...     mean=[0.4, 0.5, 0.6],\n...     variance=[0.67, 0.67, 0.67],\n...     axis=-1\n... )\narray([[-3.6624e-01, -3.6624e-01, -3.6624e-01],\n       [-4.6445e-09,  0.0000e+00, -1.8578e-08],\n       [ 3.6624e-01,  3.6624e-01,  3.6624e-01]])",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "mean", "type": "Any"},
      {"name": "variance", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "offset", "type": "Any"},
      {"name": "scale", "type": "Any"},
      {"name": "epsilon", "type": "Any"},
    ],
    "variants": {},
  },
  "batch_shape": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "beta": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bfloat16": {
    "description": "Casts all floating point parameters and buffers to ``bfloat16`` datatype.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "bias": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bias_correction": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bias_hh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bias_ih": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bias_k": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bias_v": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "binary_cross_entropy": {
    "description": "Compute Binary Cross Entropy between the target and input probabilities.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
    ],
    "variants": {},
  },
  "bincount": {
    "description": "Count the number of occurrences of each value in an integer array.\n\nJAX implementation of :func:`numpy.bincount`.\n\nFor an array of non-negative integers ``x``, this function returns an array ``counts``\nof size ``x.max() + 1``, such that ``counts[i]`` contains the number of occurrences\nof the value ``i`` in ``x``.\n\nThe JAX version has a few differences from the NumPy version:\n\n- In NumPy, passing an array ``x`` with negative entries will result in an error.\n  In JAX, negative values are clipped to zero.\n- JAX adds an optional ``length`` parameter which can be used to statically specify\n  the length of the output array so that this function can be used with transformations\n  like :func:`jax.jit`. In this case, items larger than `length + 1` will be dropped.\n\nArgs:\n  x : 1-dimensional array of non-negative integers\n  weights: optional array of weights associated with ``x``. If not specified, the\n    weight for each entry will be ``1``.\n  minlength: the minimum length of the output counts array.\n  length: the length of the output counts array. Must be specified statically for\n    ``bincount`` to be used with :func:`jax.jit` and other JAX transformations.\n\nReturns:\n  An array of counts or summed weights reflecting the number of occurrences of values\n  in ``x``.\n\nSee Also:\n  - :func:`jax.numpy.histogram`\n  - :func:`jax.numpy.digitize`\n  - :func:`jax.numpy.unique_counts`\n\nExamples:\n  Basic bincount:\n\n  >>> x = jnp.array([1, 1, 2, 3, 3, 3])\n  >>> jnp.bincount(x)\n  Array([0, 2, 1, 3], dtype=int32)\n\n  Weighted bincount:\n\n  >>> weights = jnp.array([1, 2, 3, 4, 5, 6])\n  >>> jnp.bincount(x, weights)\n  Array([ 0,  3,  3, 15], dtype=int32)\n\n  Specifying a static ``length`` makes this jit-compatible:\n\n  >>> jit_bincount = jax.jit(jnp.bincount, static_argnames=['length'])\n  >>> jit_bincount(x, length=5)\n  Array([0, 2, 1, 3, 0], dtype=int32)\n\n  Any negative numbers are clipped to the first bin, and numbers beyond the\n  specified ``length`` are dropped:\n\n  >>> x = jnp.array([-1, -1, 1, 3, 10])\n  >>> jnp.bincount(x, length=5)\n  Array([2, 1, 0, 1, 0], dtype=int32)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "weights", "type": "Any"},
      {"name": "minlength", "type": "Any"},
      {"name": "length", "type": "Any"},
    ],
    "variants": {},
  },
  "binomial": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bits": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "bitwise_and": {
    "description": "Computes the bitwise AND of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, bool]"},
      {"name": "x2", "type": "Union[array, int, bool]"},
    ],
    "variants": {},
  },
  "bitwise_invert": {
    "description": "Inverts (flips) each bit for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "bitwise_left_shift": {
    "description": "Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the left by appending ``x2_i`` (i.e., the respective element in the input array ``x2``) zeros to the right of ``x1_i``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int]"},
      {"name": "x2", "type": "Union[array, int]"},
    ],
    "variants": {},
  },
  "bitwise_not": {
    "description": "Compute bit-wise inversion, or bit-wise NOT, element-wise.\n\nComputes the bit-wise NOT of the underlying binary representation of the\nintegers in the input arrays. This ufunc implements the C/Python operator\n`~`.\n\nArgs:\n    x: Input integer tensor.\n\nReturns:\n    Result tensor.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "bitwise_or": {
    "description": "Computes the bitwise OR of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, bool]"},
      {"name": "x2", "type": "Union[array, int, bool]"},
    ],
    "variants": {},
  },
  "bitwise_right_shift": {
    "description": "Shifts the bits of each element ``x1_i`` of the input array ``x1`` to the right according to the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int]"},
      {"name": "x2", "type": "Union[array, int]"},
    ],
    "variants": {},
  },
  "bitwise_xor": {
    "description": "Computes the bitwise XOR of the underlying binary representation of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, bool]"},
      {"name": "x2", "type": "Union[array, int, bool]"},
    ],
    "variants": {},
  },
  "blackman": {
    "description": "Return a Blackman window of size M.\n\nJAX implementation of :func:`numpy.blackman`.\n\nArgs:\n  M: The window size.\n\nReturns:\n  An array of size M containing the Blackman window.\n\nExamples:\n  >>> with jnp.printoptions(precision=2, suppress=True):\n  ...   print(jnp.blackman(4))\n  [-0.    0.63  0.63 -0.  ]\n\nSee also:\n  - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.\n  - :func:`jax.numpy.hamming`: return a Hamming window of size M.\n  - :func:`jax.numpy.hanning`: return a Hanning window of size M.\n  - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.",
    "std_args": [
      {"name": "M", "type": "Any"},
    ],
    "variants": {},
  },
  "bool_": {
    "description": "A JAX scalar constructor of type bool.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "broadcast": {
    "description": "Broadcasts a tensor to specified GPU devices.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
      {"name": "devices", "type": "Any"},
      {"name": "out", "type": "Any"},
    ],
    "variants": {},
  },
  "broadcast_arrays": {
    "description": "Broadcasts one or more arrays against one another.",
    "std_args": [],
    "variants": {},
  },
  "broadcast_shapes": {
    "description": "Broadcast input shapes to a common output shape.\n\nJAX implementation of :func:`numpy.broadcast_shapes`. JAX uses NumPy-style\nbroadcasting rules, which you can read more about at `NumPy broadcasting`_.\n\nArgs:\n  shapes: 0 or more shapes specified as sequences of integers\n\nReturns:\n  The broadcasted shape as a tuple of integers.\n\nSee Also:\n  - :func:`jax.numpy.broadcast_arrays`: broadcast arrays to a common shape.\n  - :func:`jax.numpy.broadcast_to`: broadcast an array to a specified shape.\n\nExamples:\n  Some compatible shapes:\n\n  >>> jnp.broadcast_shapes((1,), (4,))\n  (4,)\n  >>> jnp.broadcast_shapes((3, 1), (4,))\n  (3, 4)\n  >>> jnp.broadcast_shapes((3, 1), (1, 4), (5, 1, 1))\n  (5, 3, 4)\n\n  Incompatible shapes:\n\n  >>> jnp.broadcast_shapes((3, 1), (4, 1))  # doctest: +IGNORE_EXCEPTION_DETAIL\n  Traceback (most recent call last):\n  ValueError: Incompatible shapes for broadcasting: shapes=[(3, 1), (4, 1)]\n\n.. _NumPy broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html",
    "std_args": [
      {"name": "shapes", "type": "Any"},
    ],
    "variants": {},
  },
  "broadcast_to": {
    "description": "Broadcasts an array to a specified shape.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "shape", "type": "Tuple[int, Ellipsis]"},
    ],
    "variants": {},
  },
  "byte": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "cached": {
    "description": "Context manager that enables the caching system within parametrizations registered with :func:`register_parametrization`.",
    "std_args": [],
    "variants": {},
  },
  "can_cast": {
    "description": "Determines if one data type can be cast to another data type according to type promotion rules (see :ref:`type-promotion`).",
    "std_args": [
      {"name": "from_", "type": "Union[dtype, array]"},
      {"name": "to", "type": "dtype"},
    ],
    "variants": {},
  },
  "cast_like": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "categorical": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "categorical_crossentropy": {
    "description": "Computes categorical cross-entropy loss between target and output tensor.\n\nThe categorical cross-entropy loss is commonly used in multi-class\nclassification tasks where each input sample can belong to one of\nmultiple classes. It measures the dissimilarity\nbetween the target and output probabilities or logits.\n\nArgs:\n    target: The target tensor representing the true categorical labels.\n        Its shape should match the shape of the `output` tensor\n        except for the last dimension.\n    output: The output tensor representing the predicted probabilities\n        or logits. Its shape should match the shape of the `target`\n        tensor except for the last dimension.\n    from_logits: (optional) Whether `output` is a tensor of logits or\n        probabilities.\n        Set it to `True` if `output` represents logits; otherwise,\n        set it to `False` if `output` represents probabilities.\n        Defaults to `False`.\n    axis: (optional) The axis along which the categorical cross-entropy\n        is computed.\n        Defaults to `-1`, which corresponds to the last dimension of\n        the tensors.\n\nReturns:\n    Integer tensor: The computed categorical cross-entropy loss between\n    `target` and `output`.\n\nExample:\n\n>>> target = keras.ops.convert_to_tensor(\n... [[1, 0, 0],\n...  [0, 1, 0],\n...  [0, 0, 1]])\n>>> output = keras.ops.convert_to_tensor(\n... [[0.9, 0.05, 0.05],\n...  [0.1, 0.8, 0.1],\n...  [0.2, 0.3, 0.5]])\n>>> categorical_crossentropy(target, output)\narray([0.10536054 0.22314355 0.6931472 ], shape=(3,), dtype=float32)",
    "std_args": [
      {"name": "target", "type": "Any"},
      {"name": "output", "type": "Any"},
      {"name": "from_logits", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "cbrt": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "cdouble": {
    "description": "A JAX scalar constructor of type complex128.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "ceil": {
    "description": "Rounds each element ``x_i`` of the input array ``x`` to the smallest (i.e., closest to ``-infinity``) integer-valued number that is not less than ``x_i``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "ceil_mode": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "chain": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "char": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "character": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "children": {
    "description": "Return an iterator over immediate children modules.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "chisquare": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "choice": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "cholesky": {
    "description": "Returns the lower (upper) Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "upper", "type": "bool"},
    ],
    "variants": {},
  },
  "cholesky_inverse": {
    "description": "Computes the inverse of a symmetric positive-definite matrix.\n\nArgs:\n    x: Input tensor of shape `(..., M, M)`.\n    upper (bool): Determines whether to use the upper- or lower-triangular\n        factor for the internal computation. Defaults to False.\n\nReturns:\n    A tensor of shape `(..., M, M)` representing the inverse of `x`.\n\nRaises:\n    ValueError: If `x` is not a symmetric positive-definite matrix.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "upper", "type": "Any"},
    ],
    "variants": {},
  },
  "choose": {
    "description": "Construct an array choosing from elements of multiple arrays.\n\nRefer to :func:`jax.numpy.choose` for the full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "choices", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "mode", "type": "Any"},
    ],
    "variants": {},
  },
  "clear": {
    "description": "Remove all items from the ParameterDict.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "clip": {
    "description": "Clamps each element ``x_i`` of the input array ``x`` to the range ``[min, max]``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "min", "type": "Optional[Union[int, float, array]]"},
      {"name": "max", "type": "Optional[Union[int, float, array]]"},
    ],
    "variants": {},
  },
  "clip_by_block_rms": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "clip_by_global_norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "clip_grad_norm": {
    "description": "Clip the gradient norm of an iterable of parameters.",
    "std_args": [
      {"name": "parameters", "type": "Any"},
      {"name": "max_norm", "type": "Any"},
      {"name": "norm_type", "type": "Any"},
      {"name": "error_if_nonfinite", "type": "Any"},
      {"name": "foreach", "type": "Any"},
    ],
    "variants": {},
  },
  "close": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "column_stack": {
    "description": "Stack arrays column-wise.\n\nJAX implementation of :func:`numpy.column_stack`.\n\nFor arrays of two or more dimensions, this is equivalent to\n:func:`jax.numpy.concatenate` with ``axis=1``.\n\nArgs:\n  tup: a sequence of arrays to stack; each must have the same leading dimension.\n    Input arrays will be promoted to at least rank 2. If a single array is given\n    it will be treated equivalently to `tup = unstack(tup)`, but the implementation\n    will avoid explicit unstacking.\n  dtype: optional dtype of the resulting array. If not specified, the dtype\n    will be determined via type promotion rules described in :ref:`type-promotion`.\n\nReturns:\n  the stacked result.\n\nSee also:\n  - :func:`jax.numpy.stack`: stack along arbitrary axes\n  - :func:`jax.numpy.concatenate`: concatenation along existing axes.\n  - :func:`jax.numpy.vstack`: stack vertically, i.e. along axis 0.\n  - :func:`jax.numpy.hstack`: stack horizontally, i.e. along axis 1.\n  - :func:`jax.numpy.dstack`: stack depth-wise, i.e. along axis 2.\n\nExamples:\n  Scalar values:\n\n  >>> jnp.column_stack([1, 2, 3])\n  Array([[1, 2, 3]], dtype=int32, weak_type=True)\n\n  1D arrays:\n\n  >>> x = jnp.arange(3)\n  >>> y = jnp.ones(3)\n  >>> jnp.column_stack([x, y])\n  Array([[0., 1.],\n         [1., 1.],\n         [2., 1.]], dtype=float32)\n\n  2D arrays:\n\n  >>> x = x.reshape(3, 1)\n  >>> y = y.reshape(3, 1)\n  >>> jnp.column_stack([x, y])\n  Array([[0., 1.],\n         [1., 1.],\n         [2., 1.]], dtype=float32)",
    "std_args": [
      {"name": "tup", "type": "Any"},
    ],
    "variants": {},
  },
  "complex128": {
    "description": "A JAX scalar constructor of type complex128.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "complex64": {
    "description": "A JAX scalar constructor of type complex64.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "complex_": {
    "description": "A JAX scalar constructor of type complex128.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "complexfloating": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "compute_mask": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "t", "type": "Any"},
      {"name": "default_mask", "type": "Any"},
    ],
    "variants": {},
  },
  "concat": {
    "description": "Joins a sequence of arrays along an existing axis.",
    "std_args": [
      {"name": "arrays", "type": "Union[Tuple[array, Ellipsis], List[array]]"},
      {"name": "axis", "type": "Optional[int]"},
    ],
    "variants": {},
  },
  "cond": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conditionally_mask": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conditionally_transform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conj": {
    "description": "Returns the complex conjugate for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "conjugate": {
    "description": "Return the complex conjugate of the array.\n\nRefer to :func:`jax.numpy.conjugate` for the full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "constant_": {
    "description": "Fill the input Tensor with the value :math:`\\text{val}`.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
      {"name": "val", "type": "Any"},
    ],
    "variants": {},
  },
  "constant_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "control_delta_method": {
    "description": "The control delta covariant method.",
    "std_args": [
      {"name": "function", "type": "Any"},
    ],
    "variants": {},
  },
  "control_variates_jacobians": {
    "description": "Obtain jacobians using control variates.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "control_variate_from_function", "type": "Any"},
      {"name": "grad_estimator", "type": "Any"},
      {"name": "params", "type": "Any"},
      {"name": "dist_builder", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
      {"name": "control_variate_state", "type": "Any"},
      {"name": "estimate_cv_coeffs", "type": "Any"},
      {"name": "estimate_cv_coeffs_num_samples", "type": "Any"},
    ],
    "variants": {},
  },
  "conv1d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conv1d_input": {
    "description": "Compute the gradient of conv1d with respect to the input of the convolution.",
    "std_args": [
      {"name": "input_size", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "grad_output", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
    ],
    "variants": {},
  },
  "conv1d_weight": {
    "description": "Compute the gradient of conv1d with respect to the weight of the convolution.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "weight_size", "type": "Any"},
      {"name": "grad_output", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
    ],
    "variants": {},
  },
  "conv2d_input": {
    "description": "Compute the gradient of conv2d with respect to the input of the convolution.",
    "std_args": [
      {"name": "input_size", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "grad_output", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
    ],
    "variants": {},
  },
  "conv2d_weight": {
    "description": "Compute the gradient of conv2d with respect to the weight of the convolution.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "weight_size", "type": "Any"},
      {"name": "grad_output", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
    ],
    "variants": {},
  },
  "conv3d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conv3d_input": {
    "description": "Compute the gradient of conv3d with respect to the input of the convolution.",
    "std_args": [
      {"name": "input_size", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "grad_output", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
    ],
    "variants": {},
  },
  "conv3d_weight": {
    "description": "Compute the gradient of conv3d with respect to the weight of the convolution.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "weight_size", "type": "Any"},
      {"name": "grad_output", "type": "Any"},
      {"name": "stride", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "groups", "type": "Any"},
    ],
    "variants": {},
  },
  "conv_transpose": {
    "description": 'General N-D convolution transpose.\n\nAlso known as de-convolution. This ops supports 1D, 2D and 3D convolution.\n\nArgs:\n    inputs: Tensor of rank N+2. `inputs` has shape\n        `(batch_size,) + inputs_spatial_shape + (num_channels,)` if\n        `data_format="channels_last"`, or\n        `(batch_size, num_channels) + inputs_spatial_shape` if\n        `data_format="channels_first"`.\n    kernel: Tensor of rank N+2. `kernel` has shape\n        [kernel_spatial_shape, num_output_channels, num_input_channels],\n        `num_input_channels` should match the number of channels in\n        `inputs`.\n    strides: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the strides of the convolution along each spatial\n        dimension. If `strides` is int, then every spatial dimension shares\n        the same `strides`.\n    padding: string, either `"valid"` or `"same"`. `"valid"` means no\n        padding is applied, and `"same"` results in padding evenly to the\n        left/right or up/down of the input such that output has the\n        same height/width dimension as the input when `strides=1`.\n    output_padding: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the amount of padding along the height and width of\n        the output tensor. Can be a single integer to specify the same\n        value for all spatial dimensions. The amount of output padding\n        along a given dimension must be lower than the stride along that\n        same dimension. If set to `None` (default), the output shape is\n        inferred.\n    data_format: A string, either `"channels_last"` or `"channels_first"`.\n        `data_format` determines the ordering of the dimensions in the\n        inputs. If `data_format="channels_last"`, `inputs` is of shape\n        `(batch_size, ..., channels)` while if\n        `data_format="channels_first"`, `inputs` is of shape\n        `(batch_size, channels, ...)`.\n    dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the dilation rate to use for dilated convolution. If\n        `dilation_rate` is int, then every spatial dimension shares\n        the same `dilation_rate`.\n\nReturns:\n    A tensor of rank N+2, the result of the conv operation.',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "kernel", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "output_padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
    ],
    "variants": {},
  },
  "conv_transpose1d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conv_transpose2d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "conv_transpose3d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "convert_to_numpy": {
    "description": "Convert a tensor to a NumPy array.\n\nArgs:\n    x: A tensor.\n\nReturns:\n    A NumPy array.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "convert_to_tensor": {
    "description": "Convert a NumPy array or Python array to a tensor.\n\nNative tensors for the current backend or left unchanged unless the `dtype`,\n`sparse` or `ragged` arguments are set.\n\nArgs:\n    x: A NumPy array, Python array (can be nested) or a backend tensor.\n    dtype: The target type. If `None`, the type of `x` is used.\n    sparse: Whether to keep sparse tensors. `False` will cause sparse\n        tensors to be densified. The default value of `None` means that\n        sparse tensors are kept only if the backend supports them.\n    ragged: Whether to keep ragged tensors. `False` will cause ragged\n        tensors to be densified. The default value of `None` means that\n        ragged tensors are kept only if the backend supports them.\n\nReturns:\n    A backend tensor of the specified `dtype` and sparseness.\n\nExample:\n\n>>> x = np.array([1, 2, 3])\n>>> y = keras.ops.convert_to_tensor(x)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "sparse", "type": "Any"},
      {"name": "ragged", "type": "Any"},
    ],
    "variants": {},
  },
  "convex_kl_divergence": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "convolve": {
    "description": "Returns the discrete, linear convolution of two one-dimensional sequences.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "v", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "propagate_mask", "type": "Any"},
    ],
    "variants": {},
  },
  "copy": {
    "description": "Return a copy of this :class:`~torch.nn.ParameterDict` instance.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "copy_to_host_async": {
    "description": "Copies an ``Array`` to the host asynchronously.\n\nFor arrays that live an an accelerator, such as a GPU or a TPU, JAX may\ncache the value of the array on the host. Normally this happens\nbehind the scenes when the value of an on-device array is requested by the\nuser, but waiting to initiate a device-to-host copy until the value is\nrequested requires that JAX block the caller while waiting for the copy to\ncomplete.\n\n``copy_to_host_async`` requests that JAX populate its on-host cache of an\narray, but does not wait for the copy to complete. This may speed up a\nfuture on-host access to the array's contents.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "copysign": {
    "description": "Composes a floating-point value with the magnitude of ``x1_i`` and the sign of ``x2_i`` for each element of the input array ``x1``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "corrcoef": {
    "description": "Return Pearson product-moment correlation coefficients.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "y", "type": "Any"},
      {"name": "rowvar", "type": "Any"},
      {"name": "allow_masked", "type": "Any"},
    ],
    "variants": {},
  },
  "correlate": {
    "description": "Cross-correlation of two 1-dimensional sequences.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "v", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "propagate_mask", "type": "Any"},
    ],
    "variants": {},
  },
  "cos": {
    "description": "Calculates an implementation-dependent approximation to the cosine for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "cosh": {
    "description": "Calculates an implementation-dependent approximation to the hyperbolic cosine for each element ``x_i`` in the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "cosine_decay_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "cosine_distance": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "cosine_onecycle_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "cosine_similarity": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "count": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "count_nonzero": {
    "description": "Counts the number of array elements which are non-zero.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "cov": {
    "description": "Estimate the covariance matrix.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "y", "type": "Any"},
      {"name": "rowvar", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "allow_masked", "type": "Any"},
      {"name": "ddof", "type": "Any"},
    ],
    "variants": {},
  },
  "cpu": {
    "description": "Move all model parameters and buffers to the CPU.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "create_mask": {
    "description": "This function creates a mask tensor from a mod_fn function.",
    "std_args": [
      {"name": "mod_fn", "type": "Any"},
      {"name": "B", "type": "Any"},
      {"name": "H", "type": "Any"},
      {"name": "Q_LEN", "type": "Any"},
      {"name": "KV_LEN", "type": "Any"},
      {"name": "device", "type": "Any"},
    ],
    "variants": {},
  },
  "cross": {
    "description": "Returns the cross product of 3-element vectors.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "cross_entropy": {
    "description": "Compute the cross entropy loss between input logits and target.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "ignore_index", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "label_smoothing", "type": "Any"},
    ],
    "variants": {},
  },
  "csingle": {
    "description": "A JAX scalar constructor of type complex64.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "ctc_decode": {
    "description": 'Decodes the output of a CTC model.\n\nArgs:\n    inputs: A tensor of shape `(batch_size, max_length, num_classes)`\n        containing the logits (the output of the model).\n        They should *not* be normalized via softmax.\n    sequence_lengths: A tensor of shape `(batch_size,)` containing the\n        sequence lengths for the batch.\n    strategy: A string for the decoding strategy. Supported values are\n        `"greedy"` and `"beam_search"`.\n    beam_width: An integer scalar beam width used in beam search.\n        Defaults to 100.\n    top_paths: An integer scalar, the number of top paths to return.\n        Defaults to 1.\n    merge_repeated: A boolean scalar, whether to merge repeated\n        labels in the output. Defaults to `True`.\n    mask_index: An integer scalar, the index of the mask character in\n        the vocabulary. Defaults to `0`.\n\nReturns:\n    A tuple containing:\n    - The tensor representing the list of decoded sequences. If\n        `strategy="greedy"`, the shape is `(1, batch_size, max_length)`. If\n        `strategy="beam_search"`, the shape is\n        `(top_paths, batch_size, max_length)`. Note that: `-1` indicates the\n        blank label.\n    - If `strategy="greedy"`, a tensor of shape `(batch_size, 1)`\n        representing the negative of the sum of the probability logits for\n        each sequence. If `strategy="beam_seatch"`, a tensor of shape\n        `(batch_size, top_paths)` representing the log probability for each\n        sequence.',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "sequence_lengths", "type": "Any"},
      {"name": "strategy", "type": "Any"},
      {"name": "beam_width", "type": "Any"},
      {"name": "top_paths", "type": "Any"},
      {"name": "merge_repeated", "type": "Any"},
      {"name": "mask_index", "type": "Any"},
    ],
    "variants": {},
  },
  "ctc_loss": {
    "description": "Compute the Connectionist Temporal Classification loss.",
    "std_args": [
      {"name": "log_probs", "type": "Any"},
      {"name": "targets", "type": "Any"},
      {"name": "input_lengths", "type": "Any"},
      {"name": "target_lengths", "type": "Any"},
      {"name": "blank", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "zero_infinity", "type": "Any"},
    ],
    "variants": {},
  },
  "ctc_loss_with_forward_probs": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "cumprod": {
    "description": "Return the cumulative product of the array.\n\nRefer to :func:`jax.numpy.cumprod` for the full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "out", "type": "Any"},
    ],
    "variants": {},
  },
  "cumulative_prod": {
    "description": "Calculates the cumulative product of elements in the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[int]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "include_initial", "type": "bool"},
    ],
    "variants": {},
  },
  "cumulative_sum": {
    "description": "Calculates the cumulative sum of elements in the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[int]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "include_initial", "type": "bool"},
    ],
    "variants": {},
  },
  "custom_gradient": {
    "description": "Decorator to define a function with a custom gradient.\n\nThis decorator allows fine grained control over the gradients of a sequence\nfor operations. This may be useful for multiple reasons, including providing\na more efficient or numerically stable gradient for a sequence of\noperations.\n\nArgs:\n    f: Function `f(*args)` that returns a tuple\n        `(output, grad_fn)`, where:\n        - `args` is a sequence of (nested structures of) tensor inputs to\n            the function.\n        - `output` is a (nested structure of) tensor outputs of applying\n            operations in `forward_fn` to `args`.\n        - `grad_fn` is a function with the signature `grad_fn(*args,\n            upstream)` which returns a tuple of tensors the same size as\n            (flattened) `args`: the derivatives of tensors in `output` with\n            respect to the tensors in `args`. `upstream` is a tensor or\n            sequence of tensors holding the initial value gradients for each\n            tensor in `output`.\n\nReturns:\n    A function `h(*args)` which returns the same value as\n    `f(*args)[0]` and whose gradient is determined by\n    `f(*args)[1]`.\n\n\nExamples:\n\n1. Backend-agnostic example.\n\n```python\n@ops.custom_gradient\ndef log1pexp(x):\n    e = ops.exp(x)\n\n    def grad(*args, upstream=None):\n        if upstream is None:\n            (upstream,) = args\n        return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))\n\n    return ops.log(1 + e), grad\n```\n\nNote that the grad function that returns gradient computation\nrequires `args` as well as an `upstream` keyword argument, depending\non the backend being set. With the JAX and TensorFlow backends,\nit requires only one argument, whereas it might use the `upstream`\nargument in the case of the PyTorch backend.\n\nWhen working with TensorFlow/JAX backend, `grad(upstream)`\nis sufficient. With PyTorch, the `grad` function requires\n`*args` as well as `upstream`, e.g. `def grad(*args, upstream)`.\nFollow the previous example to use `@ops.custom_gradient` in\na way that is compatible with all backends.\n\n2. Here's JAX & TensorFlow-specific example:\n\n```python\n@ops.custom_gradient\ndef log1pexp(x):\n    e = ops.exp(x)\n    def grad(upstream):\n        return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))\n    return ops.log(1 + e), grad\n```\n\n3. Lastly, here's a PyTorch-specific example,\nusing `*args` & `upstream`:\n\n```python\n@ops.custom_gradient\ndef log1pexp(x):\n    e = ops.exp(x)\n    def grad(*args, upstream):\n        return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))\n    return ops.log(1 + e), grad\n```",
    "std_args": [
      {"name": "f", "type": "Any"},
    ],
    "variants": {},
  },
  "d_model": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "data": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "data_ptr": {
    "description": "Get valid data pointer or buffer access.",
    "std_args": [],
    "variants": {},
  },
  "decoder": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "deg2rad": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "degrees": {
    "description": "Alias of :func:`jax.numpy.rad2deg`",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "delete": {
    "description": "Delete entry or entries from an array.\n\nJAX implementation of :func:`numpy.delete`.\n\nArgs:\n  arr: array from which entries will be deleted.\n  obj: index, indices, or slice to be deleted.\n  axis: axis along which entries will be deleted.\n  assume_unique_indices: In case of array-like integer (not boolean) indices,\n    assume the indices are unique, and perform the deletion in a way that is\n    compatible with JIT and other JAX transformations.\n\nReturns:\n  Copy of ``arr`` with specified indices deleted.\n\nNote:\n  ``delete()`` usually requires the index specification to be static. If the\n  index is an integer array that is guaranteed to contain unique entries, you\n  may specify ``assume_unique_indices=True`` to perform the operation in a\n  manner that does not require static indices.\n\nSee also:\n  - :func:`jax.numpy.insert`: insert entries into an array.\n\nExamples:\n  Delete entries from a 1D array:\n\n  >>> a = jnp.array([4, 5, 6, 7, 8, 9])\n  >>> jnp.delete(a, 2)\n  Array([4, 5, 7, 8, 9], dtype=int32)\n  >>> jnp.delete(a, slice(1, 4))  # delete a[1:4]\n  Array([4, 8, 9], dtype=int32)\n  >>> jnp.delete(a, slice(None, None, 2))  # delete a[::2]\n  Array([5, 7, 9], dtype=int32)\n\n  Delete entries from a 2D array along a specified axis:\n\n  >>> a2 = jnp.array([[4, 5, 6],\n  ...                 [7, 8, 9]])\n  >>> jnp.delete(a2, 1, axis=1)\n  Array([[4, 6],\n         [7, 9]], dtype=int32)\n\n  Delete multiple entries via a sequence of indices:\n\n  >>> indices = jnp.array([0, 1, 3])\n  >>> jnp.delete(a, indices)\n  Array([6, 8, 9], dtype=int32)\n\n  This will fail under :func:`~jax.jit` and other transformations, because\n  the output shape cannot be known with the possibility of duplicate indices:\n\n  >>> jax.jit(jnp.delete)(a, indices)  # doctest: +IGNORE_EXCEPTION_DETAIL\n  Traceback (most recent call last):\n    ...\n  ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[3].\n\n  If you can ensure that the indices are unique, pass ``assume_unique_indices``\n  to allow this to be executed under JIT:\n\n  >>> jit_delete = jax.jit(jnp.delete, static_argnames=['assume_unique_indices'])\n  >>> jit_delete(a, indices, assume_unique_indices=True)\n  Array([6, 8, 9], dtype=int32)",
    "std_args": [
      {"name": "arr", "type": "Any"},
      {"name": "obj", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "assume_unique_indices", "type": "Any"},
    ],
    "variants": {},
  },
  "depthwise_conv": {
    "description": 'General N-D depthwise convolution.\n\nThis ops supports 1D and 2D depthwise convolution.\n\nArgs:\n    inputs: Tensor of rank N+2. `inputs` has shape\n        `(batch_size,) + inputs_spatial_shape + (num_channels,)` if\n        `data_format="channels_last"`, or\n        `(batch_size, num_channels) + inputs_spatial_shape` if\n        `data_format="channels_first"`.\n    kernel: Tensor of rank N+2. `kernel` has shape\n        [kernel_spatial_shape, num_input_channels, num_channels_multiplier],\n        `num_input_channels` should match the number of channels in\n        `inputs`.\n    strides: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the strides of the convolution along each spatial\n        dimension. If `strides` is int, then every spatial dimension shares\n        the same `strides`.\n    padding: string, either `"valid"` or `"same"`. `"valid"` means no\n        padding is applied, and `"same"` results in padding evenly to the\n        left/right or up/down of the input such that output has the\n        same height/width dimension as the input when `strides=1`.\n    data_format: A string, either `"channels_last"` or `"channels_first"`.\n        `data_format` determines the ordering of the dimensions in the\n        inputs. If `data_format="channels_last"`, `inputs` is of shape\n        `(batch_size, ..., channels)` while if\n        `data_format="channels_first"`, `inputs` is of shape\n        `(batch_size, channels, ...)`.\n    dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the dilation rate to use for dilated convolution. If\n        `dilation_rate` is int, then every spatial dimension shares\n        the same `dilation_rate`.\n\nReturns:\n    A tensor of rank N+2, the result of the depthwise conv operation.',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "kernel", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
    ],
    "variants": {},
  },
  "deserialize": {
    "description": "Return a Keras activation function via its config.",
    "std_args": [
      {"name": "config", "type": "Any"},
      {"name": "custom_objects", "type": "Any"},
    ],
    "variants": {},
  },
  "det": {
    "description": "Returns the determinant of a square matrix (or a stack of square matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "device": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "device_ids": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "device_mesh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "device_type": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "diag": {
    "description": "Returns the specified diagonal or constructs a diagonal array.\n\nJAX implementation of :func:`numpy.diag`.\n\nThe JAX version always returns a copy of the input, although if this is used\nwithin a JIT compilation, the compiler may avoid the copy.\n\nArgs:\n  v: Input array. Can be a 1-D array to create a diagonal matrix or a\n    2-D array to extract a diagonal.\n  k: optional, default=0. Diagonal offset. Positive values place the diagonal\n    above the main diagonal, negative values place it below the main diagonal.\n\nReturns:\n  If `v` is a 2-D array, a 1-D array containing the diagonal elements.\n  If `v` is a 1-D array, a 2-D array with the input elements placed along the\n  specified diagonal.\n\nSee also:\n  - :func:`jax.numpy.diagflat`\n  - :func:`jax.numpy.diagonal`\n\nExamples:\n  Creating a diagonal matrix from a 1-D array:\n\n  >>> jnp.diag(jnp.array([1, 2, 3]))\n  Array([[1, 0, 0],\n         [0, 2, 0],\n         [0, 0, 3]], dtype=int32)\n\n  Specifying a diagonal offset:\n\n  >>> jnp.diag(jnp.array([1, 2, 3]), k=1)\n  Array([[0, 1, 0, 0],\n         [0, 0, 2, 0],\n         [0, 0, 0, 3],\n         [0, 0, 0, 0]], dtype=int32)\n\n  Extracting a diagonal from a 2-D array:\n\n  >>> x = jnp.array([[1, 2, 3],\n  ...                [4, 5, 6],\n  ...                [7, 8, 9]])\n  >>> jnp.diag(x)\n  Array([1, 5, 9], dtype=int32)",
    "std_args": [
      {"name": "v", "type": "Any"},
      {"name": "k", "type": "Any"},
    ],
    "variants": {},
  },
  "diag_indices": {
    "description": "Return indices for accessing the main diagonal of a multidimensional array.\n\nJAX implementation of :func:`numpy.diag_indices`.\n\nArgs:\n  n: int. The size of each dimension of the square array.\n  ndim: optional, int, default=2. The number of dimensions of the array.\n\nReturns:\n  A tuple of arrays, each of length `n`, containing the indices to access\n  the main diagonal.\n\nSee also:\n  - :func:`jax.numpy.diag_indices_from`\n  - :func:`jax.numpy.diagonal`\n\nExamples:\n  >>> jnp.diag_indices(3)\n  (Array([0, 1, 2], dtype=int32), Array([0, 1, 2], dtype=int32))\n  >>> jnp.diag_indices(4, ndim=3)\n  (Array([0, 1, 2, 3], dtype=int32),\n  Array([0, 1, 2, 3], dtype=int32),\n  Array([0, 1, 2, 3], dtype=int32))",
    "std_args": [
      {"name": "n", "type": "Any"},
      {"name": "ndim", "type": "Any"},
    ],
    "variants": {},
  },
  "diag_indices_from": {
    "description": "Return indices for accessing the main diagonal of a given array.\n\nJAX implementation of :func:`numpy.diag_indices_from`.\n\nArgs:\n  arr: Input array. Must be at least 2-dimensional and have equal length along\n    all dimensions.\n\nReturns:\n  A tuple of arrays containing the indices to access the main diagonal of\n  the input array.\n\nSee also:\n  - :func:`jax.numpy.diag_indices`\n  - :func:`jax.numpy.diagonal`\n\nExamples:\n  >>> arr = jnp.array([[1, 2, 3],\n  ...                  [4, 5, 6],\n  ...                  [7, 8, 9]])\n  >>> jnp.diag_indices_from(arr)\n  (Array([0, 1, 2], dtype=int32), Array([0, 1, 2], dtype=int32))\n  >>> arr = jnp.array([[[1, 2], [3, 4]],\n  ...                  [[5, 6], [7, 8]]])\n  >>> jnp.diag_indices_from(arr)\n  (Array([0, 1], dtype=int32),\n  Array([0, 1], dtype=int32),\n  Array([0, 1], dtype=int32))",
    "std_args": [
      {"name": "arr", "type": "Any"},
    ],
    "variants": {},
  },
  "diagflat": {
    "description": "Return a 2-D array with the flattened input array laid out on the diagonal.\n\nJAX implementation of :func:`numpy.diagflat`.\n\nThis differs from `np.diagflat` for some scalar values of `v`. JAX always returns\na two-dimensional array, whereas NumPy may return a scalar depending on the type\nof `v`.\n\nArgs:\n  v: Input array. Can be N-dimensional but is flattened to 1D.\n  k: optional, default=0. Diagonal offset. Positive values place the diagonal\n    above the main diagonal, negative values place it below the main diagonal.\n\nReturns:\n  A 2D array with the input elements placed along the diagonal with the\n  specified offset (k). The remaining entries are filled with zeros.\n\nSee also:\n  - :func:`jax.numpy.diag`\n  - :func:`jax.numpy.diagonal`\n\nExamples:\n  >>> jnp.diagflat(jnp.array([1, 2, 3]))\n  Array([[1, 0, 0],\n         [0, 2, 0],\n         [0, 0, 3]], dtype=int32)\n  >>> jnp.diagflat(jnp.array([1, 2, 3]), k=1)\n  Array([[0, 1, 0, 0],\n         [0, 0, 2, 0],\n         [0, 0, 0, 3],\n         [0, 0, 0, 0]], dtype=int32)\n  >>> a = jnp.array([[1, 2],\n  ...                [3, 4]])\n  >>> jnp.diagflat(a)\n  Array([[1, 0, 0, 0],\n         [0, 2, 0, 0],\n         [0, 0, 3, 0],\n         [0, 0, 0, 4]], dtype=int32)",
    "std_args": [
      {"name": "v", "type": "Any"},
      {"name": "k", "type": "Any"},
    ],
    "variants": {},
  },
  "diagonal": {
    "description": "Returns the specified diagonals of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "offset", "type": "int"},
    ],
    "variants": {},
  },
  "diff": {
    "description": "Calculates the n-th discrete forward difference along a specified axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "int"},
      {"name": "n", "type": "int"},
      {"name": "prepend", "type": "Optional[array]"},
      {"name": "append", "type": "Optional[array]"},
    ],
    "variants": {},
  },
  "digamma": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "digitize": {
    "description": "Returns the indices of the bins to which each value in `x` belongs.\n\nArgs:\n    x: Input array to be binned.\n    bins: Array of bins. It has to be one-dimensional and monotonically\n        increasing.\n\nReturns:\n    Output array of indices, of same shape as `x`.\n\nExample:\n>>> x = np.array([0.0, 1.0, 3.0, 1.6])\n>>> bins = np.array([0.0, 3.0, 4.5, 7.0])\n>>> keras.ops.digitize(x, bins)\narray([1, 1, 2, 1])",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "bins", "type": "Any"},
    ],
    "variants": {},
  },
  "dilation": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dim": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dims": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dirichlet": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "div_value": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "divide": {
    "description": "Calculates the division of each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex]"},
      {"name": "x2", "type": "Union[array, int, float, complex]"},
    ],
    "variants": {},
  },
  "divide_no_nan": {
    "description": "Safe element-wise division which returns 0 where the denominator is 0.\n\nArgs:\n    x1: First input tensor.\n    x2: Second input tensor.\n\nReturns:\n    The quotient `x1/x2`, element-wise, with zero where x2 is zero.",
    "std_args": [
      {"name": "x1", "type": "Any"},
      {"name": "x2", "type": "Any"},
    ],
    "variants": {},
  },
  "divmod": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dot": {
    "description": "Compute the dot product of two arrays.\n\nRefer to :func:`jax.numpy.dot` for the full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "b", "type": "Any"},
      {"name": "precision", "type": "Any"},
      {"name": "preferred_element_type", "type": "Any"},
    ],
    "variants": {},
  },
  "dot_product_attention": {
    "description": "Computes dot-product attention given query, key, and value.",
    "std_args": [
      {"name": "query", "type": "Any"},
      {"name": "key", "type": "Any"},
      {"name": "value", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "mask", "type": "Any"},
      {"name": "broadcast_dropout", "type": "Any"},
      {"name": "dropout_rng", "type": "Any"},
      {"name": "dropout_rate", "type": "Any"},
      {"name": "deterministic", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "precision", "type": "Any"},
      {"name": "module", "type": "Any"},
      {"name": "promote_dtype", "type": "Any"},
      {"name": "is_causal", "type": "Any"},
    ],
    "variants": {},
  },
  "double": {
    "description": "Casts all floating point parameters and buffers to ``double`` datatype.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "downscale_factor": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dropout1": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dropout2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dropout2d": {
    "description": "Randomly zero out entire channels (a channel is a 2D feature map).",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "p", "type": "Any"},
      {"name": "training", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "dropout3": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "dropout3d": {
    "description": "Randomly zero out entire channels (a channel is a 3D feature map).",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "p", "type": "Any"},
      {"name": "training", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "dsplit": {
    "description": "Split an array into sub-arrays depth-wise.\n\nJAX implementation of :func:`numpy.dsplit`.\n\nRefer to the documentation of :func:`jax.numpy.split` for details. ``dsplit`` is\nequivalent to ``split`` with ``axis=2``.\n\nExamples:\n\n  >>> x = jnp.arange(12).reshape(3, 1, 4)\n  >>> print(x)\n  [[[ 0  1  2  3]]\n  <BLANKLINE>\n   [[ 4  5  6  7]]\n  <BLANKLINE>\n   [[ 8  9 10 11]]]\n  >>> x1, x2 = jnp.dsplit(x, 2)\n  >>> print(x1)\n  [[[0 1]]\n  <BLANKLINE>\n   [[4 5]]\n  <BLANKLINE>\n   [[8 9]]]\n  >>> print(x2)\n  [[[ 2  3]]\n  <BLANKLINE>\n   [[ 6  7]]\n  <BLANKLINE>\n   [[10 11]]]\n\nSee also:\n  - :func:`jax.numpy.split`: split an array along any axis.\n  - :func:`jax.numpy.vsplit`: split vertically, i.e. along axis=0\n  - :func:`jax.numpy.hsplit`: split horizontally, i.e. along axis=1\n  - :func:`jax.numpy.array_split`: like ``split``, but allows ``indices_or_sections``\n    to be an integer that does not evenly divide the size of the array.",
    "std_args": [
      {"name": "ary", "type": "Any"},
      {"name": "indices_or_sections", "type": "Any"},
    ],
    "variants": {},
  },
  "dstack": {
    "description": "Stack arrays depth-wise.\n\nJAX implementation of :func:`numpy.dstack`.\n\nFor arrays of three or more dimensions, this is equivalent to\n:func:`jax.numpy.concatenate` with ``axis=2``.\n\nArgs:\n  tup: a sequence of arrays to stack; each must have the same shape along all\n    but the third axis. Input arrays will be promoted to at least rank 3. If a\n    single array is given it will be treated equivalently to `tup = unstack(tup)`,\n    but the implementation will avoid explicit unstacking.\n  dtype: optional dtype of the resulting array. If not specified, the dtype\n    will be determined via type promotion rules described in :ref:`type-promotion`.\n\nReturns:\n  the stacked result.\n\nSee also:\n  - :func:`jax.numpy.stack`: stack along arbitrary axes\n  - :func:`jax.numpy.concatenate`: concatenation along existing axes.\n  - :func:`jax.numpy.vstack`: stack vertically, i.e. along axis 0.\n  - :func:`jax.numpy.hstack`: stack horizontally, i.e. along axis 1.\n\nExamples:\n  Scalar values:\n\n  >>> jnp.dstack([1, 2, 3])\n  Array([[[1, 2, 3]]], dtype=int32, weak_type=True)\n\n  1D arrays:\n\n  >>> x = jnp.arange(3)\n  >>> y = jnp.ones(3)\n  >>> jnp.dstack([x, y])\n  Array([[[0., 1.],\n          [1., 1.],\n          [2., 1.]]], dtype=float32)\n\n  2D arrays:\n\n  >>> x = x.reshape(1, 3)\n  >>> y = y.reshape(1, 3)\n  >>> jnp.dstack([x, y])\n  Array([[[0., 1.],\n          [1., 1.],\n          [2., 1.]]], dtype=float32)",
    "std_args": [
      {"name": "tup", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "dtype": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "e": {
    "description": "IEEE 754 floating-point representation of Euler's constant.",
    "std_args": [],
    "variants": {},
  },
  "ediff1d": {
    "description": "Compute the differences between consecutive elements of an array.",
    "std_args": [
      {"name": "arr", "type": "Any"},
      {"name": "to_end", "type": "Any"},
      {"name": "to_begin", "type": "Any"},
    ],
    "variants": {},
  },
  "eig": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "eigh": {
    "description": "Returns an eigenvalue decomposition of a complex Hermitian or real symmetric matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "eigvals": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "eigvalsh": {
    "description": "Returns the eigenvalues of a complex Hermitian or real symmetric matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "einsum_path": {
    "description": 'Evaluates the optimal contraction path without evaluating the einsum.\n\nJAX implementation of :func:`numpy.einsum_path`. This function calls into\nthe opt_einsum_ package, and makes use of its optimization routines.\n\nArgs:\n  subscripts: string containing axes names separated by commas.\n  *operands: sequence of one or more arrays corresponding to the subscripts.\n  optimize: specify how to optimize the order of computation. In JAX this defaults\n    to ``"auto"``. Other options are ``True`` (same as ``"optimize"``), ``False``\n    (unoptimized), or any string supported by ``opt_einsum``, which\n    includes ``"optimize"``,, ``"greedy"``, ``"eager"``, and others.\n\nReturns:\n  A tuple containing the path that may be passed to :func:`~jax.numpy.einsum`, and a\n  printable object representing this optimal path.\n\nExamples:\n  >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)\n  >>> x = jax.random.randint(key1, minval=-5, maxval=5, shape=(2, 3))\n  >>> y = jax.random.randint(key2, minval=-5, maxval=5, shape=(3, 100))\n  >>> z = jax.random.randint(key3, minval=-5, maxval=5, shape=(100, 5))\n  >>> path, path_info = jnp.einsum_path("ij,jk,kl", x, y, z, optimize="optimal")\n  >>> print(path)\n  [(1, 2), (0, 1)]\n  >>> print(path_info)\n        Complete contraction:  ij,jk,kl->il\n              Naive scaling:  4\n          Optimized scaling:  3\n            Naive FLOP count:  9.000e+3\n        Optimized FLOP count:  3.060e+3\n        Theoretical speedup:  2.941e+0\n        Largest intermediate:  1.500e+1 elements\n      --------------------------------------------------------------------------------\n      scaling        BLAS                current                             remaining\n      --------------------------------------------------------------------------------\n        3           GEMM              kl,jk->lj                             ij,lj->il\n        3           GEMM              lj,ij->il                                il->il\n\n  Use the computed path in :func:`~jax.numpy.einsum`:\n\n  >>> jnp.einsum("ij,jk,kl", x, y, z, optimize=path)\n  Array([[-754,  324, -142,   82,   50],\n         [ 408,  -50,   87,  -29,    7]], dtype=int32)\n\n.. _opt_einsum: https://github.com/dgasmith/opt_einsum',
    "std_args": [
      {"name": "subscripts", "type": "Any"},
      {"name": "operands", "type": "Any"},
      {"name": "optimize", "type": "Any"},
    ],
    "variants": {},
  },
  "ema": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "embedding_dim": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "empty": {
    "description": "Returns an uninitialized array having a specified `shape`.",
    "std_args": [
      {"name": "shape", "type": "Union[int, Tuple[int, Ellipsis]]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "empty_like": {
    "description": "Returns an uninitialized array with the same ``shape`` as an input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "enable_grad": {
    "description": "Context-manager that enables gradient calculation.",
    "std_args": [],
    "variants": {},
  },
  "encoder": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "end_dim": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "eps": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "equal": {
    "description": "Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex, bool]"},
      {"name": "x2", "type": "Union[array, int, float, complex, bool]"},
    ],
    "variants": {},
  },
  "erfc": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "erfcx": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "erfinv": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "estimate_control_variate_coefficients": {
    "description": "Estimates the control variate coefficients for the given parameters.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "control_variate_from_function", "type": "Any"},
      {"name": "grad_estimator", "type": "Any"},
      {"name": "params", "type": "Any"},
      {"name": "dist_builder", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
      {"name": "control_variate_state", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "euler_gamma": {
    "description": "Constant: 0.5772156649015329",
    "std_args": [],
    "variants": {},
  },
  "eval": {
    "description": "Set the module in evaluation mode.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "exp": {
    "description": "Exponential.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "exp2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "expand_dims": {
    "description": "Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "expit": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "expm1": {
    "description": "Calculates an implementation-dependent approximation to ``exp(x)-1`` for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "exponential": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "exponential_decay": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "extend": {
    "description": "Append values from a Python iterable to the end of the list.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "values", "type": "Any"},
    ],
    "variants": {},
  },
  "extract": {
    "description": "Return the elements of an array that satisfy a condition.\n\nJAX implementation of :func:`numpy.extract`.\n\nArgs:\n  condition: array of conditions. Will be converted to boolean and flattened to 1D.\n  arr: array of values to extract. Will be flattened to 1D.\n  size: optional static size for output. Must be specified in order for ``extract``\n    to be compatible with JAX transformations like :func:`~jax.jit` or :func:`~jax.vmap`.\n  fill_value: if ``size`` is specified, fill padded entries with this value (default: 0).\n\nReturns:\n  1D array of extracted entries . If ``size`` is specified, the result will have shape\n  ``(size,)`` and be right-padded with ``fill_value``. If ``size`` is not specified,\n  the output shape will depend on the number of True entries in ``condition``.\n\nNotes:\n  This function does not require strict shape agreement between ``condition`` and ``arr``.\n  If ``condition.size > arr.size``, then ``condition`` will be truncated, and if\n  ``arr.size > condition.size``, then ``arr`` will be truncated.\n\nSee also:\n  :func:`jax.numpy.compress`: multi-dimensional version of ``extract``.\n\nExamples:\n   Extract values from a 1D array:\n\n   >>> x = jnp.array([1, 2, 3, 4, 5, 6])\n   >>> mask = (x % 2 == 0)\n   >>> jnp.extract(mask, x)\n   Array([2, 4, 6], dtype=int32)\n\n   In the simplest case, this is equivalent to boolean indexing:\n\n   >>> x[mask]\n   Array([2, 4, 6], dtype=int32)\n\n   For use with JAX transformations, you can pass the ``size`` argument to\n   specify a static shape for the output, along with an optional ``fill_value``\n   that defaults to zero:\n\n   >>> jnp.extract(mask, x, size=len(x), fill_value=0)\n   Array([2, 4, 6, 0, 0, 0], dtype=int32)\n\n   Notice that unlike with boolean indexing, ``extract`` does not require strict\n   agreement between the sizes of the array and condition, and will effectively\n   truncate both to the minimum size:\n\n   >>> short_mask = jnp.array([False, True])\n   >>> jnp.extract(short_mask, x)\n   Array([2], dtype=int32)\n   >>> long_mask = jnp.array([True, False, True, False, False, False, False, False])\n   >>> jnp.extract(long_mask, x)\n   Array([1, 3], dtype=int32)",
    "std_args": [
      {"name": "condition", "type": "Any"},
      {"name": "arr", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "extract_sequences": {
    "description": "Expands the dimension of last axis into sequences of `sequence_length`.\n\nSlides a window of size `sequence_length` over the last axis of the input\nwith a stride of `sequence_stride`, replacing the last axis with\n`[num_sequences, sequence_length]` sequences.\n\nIf the dimension along the last axis is N, the number of sequences can be\ncomputed by:\n\n`num_sequences = 1 + (N - sequence_length) // sequence_stride`\n\nArgs:\n    x: Input tensor.\n    sequence_length: An integer representing the sequences length.\n    sequence_stride: An integer representing the sequences hop size.\n\nReturns:\n    A tensor of sequences with shape [..., num_sequences, sequence_length].\n\nExample:\n\n>>> x = keras.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])\n>>> extract_sequences(x, 3, 2)\narray([[1, 2, 3],\n   [3, 4, 5]])",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "sequence_length", "type": "Any"},
      {"name": "sequence_stride", "type": "Any"},
    ],
    "variants": {},
  },
  "eye": {
    "description": "Returns a two-dimensional array with ones on the ``k``\\th diagonal and zeros elsewhere.",
    "std_args": [
      {"name": "n_rows", "type": "int"},
      {"name": "n_cols", "type": "Optional[int]"},
      {"name": "k", "type": "int"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "eye_": {
    "description": "Fill the 2-dimensional input `Tensor` with the identity matrix.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
    ],
    "variants": {},
  },
  "f": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "fft": {
    "description": "Computes the one-dimensional discrete Fourier transform.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "Optional[int]"},
      {"name": "axis", "type": "int"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "fft2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "fftfreq": {
    "description": "Computes the discrete Fourier transform sample frequencies.",
    "std_args": [
      {"name": "n", "type": "int"},
      {"name": "d", "type": "float"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "fftn": {
    "description": "Computes the n-dimensional discrete Fourier transform.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "s", "type": "Optional[Sequence[int]]"},
      {"name": "axes", "type": "Optional[Sequence[int]]"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "fftshift": {
    "description": "Shifts the zero-frequency component to the center of the spectrum.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axes", "type": "Optional[Union[int, Sequence[int]]]"},
    ],
    "variants": {},
  },
  "fill_diagonal": {
    "description": "Return a copy of the array with the diagonal overwritten.\n\nJAX implementation of :func:`numpy.fill_diagonal`.\n\nThe semantics of :func:`numpy.fill_diagonal` are to modify arrays in-place, which\nis not possible for JAX's immutable arrays. The JAX version returns a modified\ncopy of the input, and adds the ``inplace`` parameter which must be set to\n`False`` by the user as a reminder of this API difference.\n\nArgs:\n  a: input array. Must have ``a.ndim >= 2``. If ``a.ndim >= 3``, then all\n    dimensions must be the same size.\n  val: scalar or array with which to fill the diagonal. If an array, it will\n    be flattened and repeated to fill the diagonal entries.\n  wrap: Not implemented by JAX. Only the default value of ``False`` is supported.\n  inplace: must be set to False to indicate that the input is not modified\n    in-place, but rather a modified copy is returned.\n\nReturns:\n  A copy of ``a`` with the diagonal set to ``val``.\n\nExamples:\n  >>> x = jnp.zeros((3, 3), dtype=int)\n  >>> jnp.fill_diagonal(x, jnp.array([1, 2, 3]), inplace=False)\n  Array([[1, 0, 0],\n         [0, 2, 0],\n         [0, 0, 3]], dtype=int32)\n\n  Unlike :func:`numpy.fill_diagonal`, the input ``x`` is not modified.\n\n  If the diagonal value has too many entries, it will be truncated\n\n  >>> jnp.fill_diagonal(x, jnp.arange(100, 200), inplace=False)\n  Array([[100,   0,   0],\n         [  0, 101,   0],\n         [  0,   0, 102]], dtype=int32)\n\n  If the diagonal has too few entries, it will be repeated:\n\n  >>> x = jnp.zeros((4, 4), dtype=int)\n  >>> jnp.fill_diagonal(x, jnp.array([3, 4]), inplace=False)\n  Array([[3, 0, 0, 0],\n         [0, 4, 0, 0],\n         [0, 0, 3, 0],\n         [0, 0, 0, 4]], dtype=int32)\n\n  For non-square arrays, the diagonal of the leading square slice is filled:\n\n  >>> x = jnp.zeros((3, 5), dtype=int)\n  >>> jnp.fill_diagonal(x, 1, inplace=False)\n  Array([[1, 0, 0, 0, 0],\n         [0, 1, 0, 0, 0],\n         [0, 0, 1, 0, 0]], dtype=int32)\n\n  And for square N-dimensional arrays, the N-dimensional diagonal is filled:\n\n  >>> y = jnp.zeros((2, 2, 2))\n  >>> jnp.fill_diagonal(y, 1, inplace=False)\n  Array([[[1., 0.],\n          [0., 0.]],\n  <BLANKLINE>\n         [[0., 0.],\n          [0., 1.]]], dtype=float32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "val", "type": "Any"},
      {"name": "wrap", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "filters": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "find_duplicates": {
    "description": "Finds duplicate nodes or node leaves in the given node.",
    "std_args": [
      {"name": "node", "type": "Any"},
      {"name": "only", "type": "Any"},
    ],
    "variants": {},
  },
  "finfo": {
    "description": "Machine limits for floating-point data types.",
    "std_args": [
      {"name": "type", "type": "Union[dtype, array]"},
    ],
    "variants": {},
  },
  "flatnonzero": {
    "description": "Return indices of nonzero elements in a flattened array\n\nJAX implementation of :func:`numpy.flatnonzero`.\n\n``jnp.flatnonzero(x)`` is equivalent to ``nonzero(ravel(a))[0]``. For a full\ndiscussion of the parameters to this function, refer to :func:`jax.numpy.nonzero`.\n\nArgs:\n  a: N-dimensional array.\n  size: optional static integer specifying the number of nonzero entries to\n    return. See :func:`jax.numpy.nonzero` for more discussion of this parameter.\n  fill_value: optional padding value when ``size`` is specified. Defaults to 0.\n    See :func:`jax.numpy.nonzero` for more discussion of this parameter.\n\nReturns:\n  Array containing the indices of each nonzero value in the flattened array.\n\nSee Also:\n  - :func:`jax.numpy.nonzero`\n  - :func:`jax.numpy.where`\n\nExamples:\n  >>> x = jnp.array([[0, 5, 0],\n  ...                [6, 0, 8]])\n  >>> jnp.flatnonzero(x)\n  Array([1, 3, 5], dtype=int32)\n\n  This is equivalent to calling :func:`~jax.numpy.nonzero` on the flattened\n  array, and extracting the first entry in the resulting tuple:\n\n  >>> jnp.nonzero(x.ravel())[0]\n  Array([1, 3, 5], dtype=int32)\n\n  The returned indices can be used to extract nonzero entries from the\n  flattened array:\n\n  >>> indices = jnp.flatnonzero(x)\n  >>> x.ravel()[indices]\n  Array([5, 6, 8], dtype=int32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "flatten_parameters": {
    "description": "Reset parameter data pointer so that they can use faster code paths.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "flexible": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "flip": {
    "description": "Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
    ],
    "variants": {},
  },
  "fliplr": {
    "description": "Reverse the order of elements of an array along axis 1.\n\nJAX implementation of :func:`numpy.fliplr`.\n\nArgs:\n  m: Array with at least two dimensions.\n\nReturns:\n  An array with the elements in reverse order along axis 1.\n\nSee Also:\n  - :func:`jax.numpy.flip`: reverse the order along the given axis\n  - :func:`jax.numpy.flipud`: reverse the order along axis 0\n\nExamples:\n  >>> x = jnp.array([[1, 2],\n  ...                [3, 4]])\n  >>> jnp.fliplr(x)\n  Array([[2, 1],\n         [4, 3]], dtype=int32)",
    "std_args": [
      {"name": "m", "type": "Any"},
    ],
    "variants": {},
  },
  "flipud": {
    "description": "Reverse the order of elements of an array along axis 0.\n\nJAX implementation of :func:`numpy.flipud`.\n\nArgs:\n  m: Array with at least one dimension.\n\nReturns:\n  An array with the elements in reverse order along axis 0.\n\nSee Also:\n  - :func:`jax.numpy.flip`: reverse the order along the given axis\n  - :func:`jax.numpy.fliplr`: reverse the order along axis 1\n\nExamples:\n  >>> x = jnp.array([[1, 2],\n  ...                [3, 4]])\n  >>> jnp.flipud(x)\n  Array([[3, 4],\n         [1, 2]], dtype=int32)",
    "std_args": [
      {"name": "m", "type": "Any"},
    ],
    "variants": {},
  },
  "float": {
    "description": "Casts all floating point parameters and buffers to ``float`` datatype.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "float4_e2m1fn": {
    "description": "A JAX scalar constructor of type float4_e2m1fn.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e3m4": {
    "description": "A JAX scalar constructor of type float8_e3m4.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e4m3": {
    "description": "A JAX scalar constructor of type float8_e4m3.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e4m3b11fnuz": {
    "description": "A JAX scalar constructor of type float8_e4m3b11fnuz.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e4m3fn": {
    "description": "A JAX scalar constructor of type float8_e4m3fn.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e4m3fnuz": {
    "description": "A JAX scalar constructor of type float8_e4m3fnuz.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e5m2": {
    "description": "A JAX scalar constructor of type float8_e5m2.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e5m2fnuz": {
    "description": "A JAX scalar constructor of type float8_e5m2fnuz.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float8_e8m0fnu": {
    "description": "A JAX scalar constructor of type float8_e8m0fnu.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "float_": {
    "description": "A JAX scalar constructor of type float64.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "floating": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "floor": {
    "description": "Rounds each element ``x_i`` of the input array ``x`` to the greatest (i.e., closest to ``+infinity``) integer-valued number that is not greater than ``x_i``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "floor_divide": {
    "description": "Rounds the result of dividing each element ``x1_i`` of the input array ``x1`` by the respective element ``x2_i`` of the input array ``x2`` to the greatest (i.e., closest to `+infinity`) integer-value number that is not greater than the division result.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "fn": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "fori_loop": {
    "description": "A Flax NNX transformation of `jax.lax.fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html>`_.",
    "std_args": [
      {"name": "lower", "type": "Any"},
      {"name": "upper", "type": "Any"},
      {"name": "body_fun", "type": "Any"},
      {"name": "init_val", "type": "Any"},
      {"name": "unroll", "type": "Any"},
    ],
    "variants": {},
  },
  "forward": {
    "description": "Auto-generated from flax_nnx_dynamic_wiring",
    "std_args": [],
    "variants": {},
  },
  "freeze": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "from_dlpack": {
    "description": "Returns a new array containing the data from another (array) object with a ``__dlpack__`` method.",
    "std_args": [
      {"name": "x", "type": "object"},
      {"name": "device", "type": "Optional[device]"},
      {"name": "copy", "type": "Optional[bool]"},
    ],
    "variants": {},
  },
  "frombuffer": {
    "description": "Convert a buffer into a 1-D JAX array.\n\nJAX implementation of :func:`numpy.frombuffer`.\n\nArgs:\n  buffer: an object containing the data. It must be either a bytes object with\n    a length that is an integer multiple of the dtype element size, or\n    it must be an object exporting the `Python buffer interface`_.\n  dtype: optional. Desired data type for the array. Default is ``float64``.\n    This specifies the dtype used to parse the buffer, but note that after parsing,\n    64-bit values will be cast to 32-bit JAX arrays if the ``jax_enable_x64``\n    flag is set to ``False``.\n  count: optional integer specifying the number of items to read from the buffer.\n    If -1 (default), all items from the buffer are read.\n  offset: optional integer specifying the number of bytes to skip at the beginning\n    of the buffer. Default is 0.\n\nReturns:\n  A 1-D JAX array representing the interpreted data from the buffer.\n\nSee also:\n  - :func:`jax.numpy.fromstring`: convert a string of text into 1-D JAX array.\n\nExamples:\n  Using a bytes buffer:\n\n  >>> buf = b\"\\x00\\x01\\x02\\x03\\x04\"\n  >>> jnp.frombuffer(buf, dtype=jnp.uint8)\n  Array([0, 1, 2, 3, 4], dtype=uint8)\n  >>> jnp.frombuffer(buf, dtype=jnp.uint8, offset=1)\n  Array([1, 2, 3, 4], dtype=uint8)\n\n  Constructing a JAX array via the Python buffer interface, using Python's\n  built-in :mod:`array` module.\n\n  >>> from array import array\n  >>> pybuffer = array('i', [0, 1, 2, 3, 4])\n  >>> jnp.frombuffer(pybuffer, dtype=jnp.int32)\n  Array([0, 1, 2, 3, 4], dtype=int32)\n\n.. _Python buffer interface: https://docs.python.org/3/c-api/buffer.html",
    "std_args": [
      {"name": "buffer", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "count", "type": "Any"},
      {"name": "offset", "type": "Any"},
    ],
    "variants": {},
  },
  "fromfile": {
    "description": "Unimplemented JAX wrapper for jnp.fromfile.\n\nThis function is left deliberately unimplemented because it may be non-pure and thus\nunsafe for use with JIT and other JAX transformations. Consider using\n``jnp.asarray(np.fromfile(...))`` instead, although care should be taken if ``np.fromfile``\nis used within jax transformations because of its potential side-effect of consuming the\nfile object; for more information see `Common Gotchas: Pure Functions\n<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_.",
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "fromfunction": {
    "description": 'Create an array from a function applied over indices.\n\nJAX implementation of :func:`numpy.fromfunction`. The JAX implementation\ndiffers in that it dispatches via :func:`jax.vmap`, and so unlike in NumPy\nthe function logically operates on scalar inputs, and need not explicitly\nhandle broadcasted inputs (See *Examples* below).\n\nArgs:\n  function: a function that takes *N* dynamic scalars and outputs a scalar.\n  shape: a length-*N* tuple of integers specifying the output shape.\n  dtype: optionally specify the dtype of the inputs. Defaults to floating-point.\n  kwargs: additional keyword arguments are passed statically to ``function``.\n\nReturns:\n  An array of shape ``shape`` if ``function`` returns a scalar, or in general\n  a pytree of arrays with leading dimensions ``shape``, as determined by the\n  output of ``function``.\n\nSee also:\n  - :func:`jax.vmap`: the core transformation that the :func:`fromfunction`\n    API is built on.\n\nExamples:\n  Generate a multiplication table of a given shape:\n\n  >>> jnp.fromfunction(jnp.multiply, shape=(3, 6), dtype=int)\n  Array([[ 0,  0,  0,  0,  0,  0],\n         [ 0,  1,  2,  3,  4,  5],\n         [ 0,  2,  4,  6,  8, 10]], dtype=int32)\n\n  When ``function`` returns a non-scalar the output will have leading\n  dimension of ``shape``:\n\n  >>> def f(x):\n  ...   return (x + 1) * jnp.arange(3)\n  >>> jnp.fromfunction(f, shape=(2,))\n  Array([[0., 1., 2.],\n         [0., 2., 4.]], dtype=float32)\n\n  ``function`` may return multiple results, in which case each is mapped\n  independently:\n\n  >>> def f(x, y):\n  ...   return x + y, x * y\n  >>> x_plus_y, x_times_y = jnp.fromfunction(f, shape=(3, 5))\n  >>> print(x_plus_y)\n  [[0. 1. 2. 3. 4.]\n   [1. 2. 3. 4. 5.]\n   [2. 3. 4. 5. 6.]]\n  >>> print(x_times_y)\n  [[0. 0. 0. 0. 0.]\n   [0. 1. 2. 3. 4.]\n   [0. 2. 4. 6. 8.]]\n\n  The JAX implementation differs slightly from NumPy\'s implementation. In\n  :func:`numpy.fromfunction`, the function is expected to explicitly operate\n  element-wise on the full grid of input values:\n\n  >>> def f(x, y):\n  ...   print(f"{x.shape = }\\n{y.shape = }")\n  ...   return x + y\n  ...\n  >>> np.fromfunction(f, (2, 3))\n  x.shape = (2, 3)\n  y.shape = (2, 3)\n  array([[0., 1., 2.],\n         [1., 2., 3.]])\n\n  In :func:`jax.numpy.fromfunction`, the function is vectorized via\n  :func:`jax.vmap`, and so is expected to operate on scalar values:\n\n  >>> jnp.fromfunction(f, (2, 3))\n  x.shape = ()\n  y.shape = ()\n  Array([[0., 1., 2.],\n         [1., 2., 3.]], dtype=float32)',
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "shape", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "fromiter": {
    "description": "Unimplemented JAX wrapper for jnp.fromiter.\n\nThis function is left deliberately unimplemented because it may be non-pure and thus\nunsafe for use with JIT and other JAX transformations. Consider using\n``jnp.asarray(np.fromiter(...))`` instead, although care should be taken if ``np.fromiter``\nis used within jax transformations because of its potential side-effect of consuming the\niterable object; for more information see `Common Gotchas: Pure Functions\n<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions>`_.",
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "frompyfunc": {
    "description": "Create a JAX ufunc from an arbitrary JAX-compatible scalar function.\n\nArgs:\n  func : a callable that takes `nin` scalar arguments and returns `nout` outputs.\n  nin: integer specifying the number of scalar inputs\n  nout: integer specifying the number of scalar outputs\n  identity: (optional) a scalar specifying the identity of the operation, if any.\n\nReturns:\n  wrapped : jax.numpy.ufunc wrapper of func.\n\nExamples:\n  Here is an example of creating a ufunc similar to :obj:`jax.numpy.add`:\n\n  >>> import operator\n  >>> add = frompyfunc(operator.add, nin=2, nout=1, identity=0)\n\n  Now all the standard :class:`jax.numpy.ufunc` methods are available:\n\n  >>> x = jnp.arange(4)\n  >>> add(x, 10)\n  Array([10, 11, 12, 13], dtype=int32)\n  >>> add.outer(x, x)\n  Array([[0, 1, 2, 3],\n         [1, 2, 3, 4],\n         [2, 3, 4, 5],\n         [3, 4, 5, 6]], dtype=int32)\n  >>> add.reduce(x)\n  Array(6, dtype=int32)\n  >>> add.accumulate(x)\n  Array([0, 1, 3, 6], dtype=int32)\n  >>> add.at(x, 1, 10, inplace=False)\n  Array([ 0, 11,  2,  3], dtype=int32)",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "nin", "type": "Any"},
      {"name": "nout", "type": "Any"},
      {"name": "identity", "type": "Any"},
    ],
    "variants": {},
  },
  "fromstring": {
    "description": 'Convert a string of text into 1-D JAX array.\n\nJAX implementation of :func:`numpy.fromstring`.\n\nArgs:\n  string: input string containing the data.\n  dtype: optional. Desired data type for the array. Default is ``float``.\n  count: optional integer specifying the number of items to read from the string.\n    If -1 (default), all items are read.\n  sep: the string used to separate values in the input string.\n\nReturns:\n  A 1-D JAX array containing the parsed data from the input string.\n\nSee also:\n  - :func:`jax.numpy.frombuffer`: construct a JAX array from an object\n    that implements the buffer interface.\n\nExamples:\n  >>> jnp.fromstring("1 2 3", dtype=int, sep=" ")\n  Array([1, 2, 3], dtype=int32)\n  >>> jnp.fromstring("0.1, 0.2, 0.3", dtype=float, count=2, sep=",")\n  Array([0.1, 0.2], dtype=float32)',
    "std_args": [
      {"name": "string", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "count", "type": "Any"},
      {"name": "sep", "type": "Any"},
    ],
    "variants": {},
  },
  "full": {
    "description": "Returns a new array having a specified ``shape`` and filled with ``fill_value``.",
    "std_args": [
      {"name": "shape", "type": "Union[int, Tuple[int, Ellipsis]]"},
      {"name": "fill_value", "type": "Union[bool, int, float, complex]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "full_like": {
    "description": "Returns a new array filled with ``fill_value`` and having the same ``shape`` as an input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "fill_value", "type": "Union[bool, int, float, complex]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "gamma": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "gammainc": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "gammaincc": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "gammaln": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "gaussian_nll_loss": {
    "description": "Compute the Gaussian negative log likelihood loss.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "var", "type": "Any"},
      {"name": "full", "type": "Any"},
      {"name": "eps", "type": "Any"},
      {"name": "reduction", "type": "Any"},
    ],
    "variants": {},
  },
  "gcd": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "generic": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "geometric": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "geomspace": {
    "description": "Generate geometrically-spaced values.\n\nJAX implementation of :func:`numpy.geomspace`.\n\nArgs:\n  start: scalar or array. Specifies the starting values.\n  stop: scalar or array. Specifies the stop values.\n  num: int, optional, default=50. Number of values to generate.\n  endpoint: bool, optional, default=True. If True, then include the ``stop`` value\n    in the result. If False, then exclude the ``stop`` value.\n  dtype: optional. Specifies the dtype of the output.\n  axis: int, optional, default=0. Axis along which to generate the geomspace.\n\nReturns:\n  An array containing the geometrically-spaced values.\n\nSee also:\n  - :func:`jax.numpy.arange`: Generate ``N`` evenly-spaced values given a starting\n    point and a step value.\n  - :func:`jax.numpy.linspace`: Generate evenly-spaced values.\n  - :func:`jax.numpy.logspace`: Generate logarithmically-spaced values.\n\nExamples:\n  List 5 geometrically-spaced values between 1 and 16:\n\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.geomspace(1, 16, 5)\n  Array([ 1.,  2.,  4.,  8., 16.], dtype=float32)\n\n  List 4 geomtrically-spaced values between 1 and 16, with ``endpoint=False``:\n\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.geomspace(1, 16, 4, endpoint=False)\n  Array([1., 2., 4., 8.], dtype=float32)\n\n  Multi-dimensional geomspace:\n\n  >>> start = jnp.array([1, 1000])\n  >>> stop = jnp.array([27, 1])\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.geomspace(start, stop, 4)\n  Array([[   1., 1000.],\n         [   3.,  100.],\n         [   9.,   10.],\n         [  27.,    1.]], dtype=float32)",
    "std_args": [
      {"name": "start", "type": "Any"},
      {"name": "stop", "type": "Any"},
      {"name": "num", "type": "Any"},
      {"name": "endpoint", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "get": {
    "description": "Return the parameter associated with key if present. Otherwise return default if provided, None if not.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "key", "type": "Any"},
      {"name": "default", "type": "Any"},
    ],
    "variants": {},
  },
  "get_all_with_path": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "get_item": {
    "description": "Return `x[key]`.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "key", "type": "Any"},
    ],
    "variants": {},
  },
  "get_parameter": {
    "description": "Return the parameter given by ``target`` if it exists, otherwise throw an error.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "target", "type": "Any"},
    ],
    "variants": {},
  },
  "get_printoptions": {
    "description": "Alias of :func:`numpy.get_printoptions`.\n\nJAX arrays are printed via NumPy, so NumPy's `printoptions`\nconfigurations will apply to printed JAX arrays.\n\nSee the :func:`numpy.set_printoptions` documentation for details\non the available options and their meanings.",
    "std_args": [],
    "variants": {},
  },
  "glorot_normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "glorot_uniform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "grad": {
    "description": "Evaluates gradient.",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "argnums", "type": "Any"},
      {"name": "has_aux", "type": "Any"},
    ],
    "variants": {},
  },
  "greater": {
    "description": "Computes the truth value of ``x1_i > x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "greater_equal": {
    "description": "Computes the truth value of ``x1_i >= x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "group": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "group_size": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "groups": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "gumbel": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "half": {
    "description": "Casts all floating point parameters and buffers to ``half`` datatype.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "hamming": {
    "description": "Return a Hamming window of size M.\n\nJAX implementation of :func:`numpy.hamming`.\n\nArgs:\n  M: The window size.\n\nReturns:\n  An array of size M containing the Hamming window.\n\nExamples:\n  >>> with jnp.printoptions(precision=2, suppress=True):\n  ...   print(jnp.hamming(4))\n  [0.08 0.77 0.77 0.08]\n\nSee also:\n  - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.\n  - :func:`jax.numpy.blackman`: return a Blackman window of size M.\n  - :func:`jax.numpy.hanning`: return a Hanning window of size M.\n  - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.",
    "std_args": [
      {"name": "M", "type": "Any"},
    ],
    "variants": {},
  },
  "hanning": {
    "description": "Return a Hanning window of size M.\n\nJAX implementation of :func:`numpy.hanning`.\n\nArgs:\n  M: The window size.\n\nReturns:\n  An array of size M containing the Hanning window.\n\nExamples:\n  >>> with jnp.printoptions(precision=2, suppress=True):\n  ...   print(jnp.hanning(4))\n  [0.   0.75 0.75 0.  ]\n\nSee also:\n  - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.\n  - :func:`jax.numpy.blackman`: return a Blackman window of size M.\n  - :func:`jax.numpy.hamming`: return a Hamming window of size M.\n  - :func:`jax.numpy.kaiser`: return a Kaiser window of size M.",
    "std_args": [
      {"name": "M", "type": "Any"},
    ],
    "variants": {},
  },
  "hard_shrink": {
    "description": "Hard Shrink activation function.\n\nIt is defined as:\n\n`hard_shrink(x) = x` if `|x| > threshold`,\n`hard_shrink(x) = 0` otherwise.\n\nArgs:\n    x: Input tensor.\n    threshold: Threshold value. Defaults to 0.5.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "threshold", "type": "Any"},
    ],
    "variants": {},
  },
  "hard_sigmoid": {
    "description": 'Hard sigmoid activation function.\n\nThe hard sigmoid activation is defined as:\n\n- `0` if `if x <= -3`\n- `1` if `x >= 3`\n- `(x/6) + 0.5` if `-3 < x < 3`\n\nIt\'s a faster, piecewise linear approximation\nof the sigmoid activation.\n\nArgs:\n    x: Input tensor.\n\nReference:\n\n- [Wikipedia "Hard sigmoid"](https://en.wikipedia.org/wiki/Hard_sigmoid)',
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "hard_silu": {
    "description": "Hard SiLU activation function, also known as Hard Swish.\n\nIt is defined as:\n\n- `0` if `if x < -3`\n- `x` if `x > 3`\n- `x * (x + 3) / 6` if `-3 <= x <= 3`\n\nIt's a faster, piecewise linear approximation of the silu activation.\n\nArgs:\n    x: Input tensor.\n\nReference:\n\n- [A Howard, 2019](https://arxiv.org/abs/1905.02244)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "hard_swish": {
    "description": "Hard SiLU activation function, also known as Hard Swish.\n\nIt is defined as:\n\n- `0` if `if x < -3`\n- `x` if `x > 3`\n- `x * (x + 3) / 6` if `-3 <= x <= 3`\n\nIt's a faster, piecewise linear approximation of the silu activation.\n\nArgs:\n    x: Input tensor.\n\nReference:\n\n- [A Howard, 2019](https://arxiv.org/abs/1905.02244)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "hard_tanh": {
    "description": "HardTanh activation function.\n\nIt is defined as:\n`hard_tanh(x) = -1 for x < -1`,\n`hard_tanh(x) = x for -1 <= x <= 1`,\n`hard_tanh(x) = 1 for x > 1`.\n\nArgs:\n    x: Input tensor.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "he_normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "he_uniform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "head": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "head_dim": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "heaviside": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "hfft": {
    "description": "Computes the one-dimensional discrete Fourier transform of a signal with Hermitian symmetry.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "Optional[int]"},
      {"name": "axis", "type": "int"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "hidden_size": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "hinge_loss": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "histogram": {
    "description": "Compute a 1-dimensional histogram.\n\nJAX implementation of :func:`numpy.histogram`.\n\nArgs:\n  a: array of values to be binned. May be any size or dimension.\n  bins: Specify the number of bins in the histogram (default: 10). ``bins``\n    may also be an array specifying the locations of the bin edges.\n  range: tuple of scalars. Specifies the range of the data. If not specified,\n    the range is inferred from the data.\n  weights: An optional array specifying the weights of the data points.\n    Should be broadcast-compatible with ``a``. If not specified, each\n    data point is weighted equally.\n  density: If True, return the normalized histogram in units of counts\n    per unit length. If False (default) return the (weighted) counts per bin.\n\nReturns:\n  A tuple of arrays ``(histogram, bin_edges)``, where ``histogram`` contains\n  the aggregated data, and ``bin_edges`` specifies the boundaries of the bins.\n\nSee Also:\n  - :func:`jax.numpy.bincount`: Count the number of occurrences of each value in an array.\n  - :func:`jax.numpy.histogram2d`: Compute the histogram of a 2D array.\n  - :func:`jax.numpy.histogramdd`: Compute the histogram of an N-dimensional array.\n  - :func:`jax.numpy.histogram_bin_edges`: Compute the bin edges for a histogram.\n\nExamples:\n  >>> a = jnp.array([1, 2, 3, 10, 11, 15, 19, 25])\n  >>> counts, bin_edges = jnp.histogram(a, bins=8)\n  >>> print(counts)\n  [3. 0. 0. 2. 1. 0. 1. 1.]\n  >>> print(bin_edges)\n  [ 1.  4.  7. 10. 13. 16. 19. 22. 25.]\n\n  Specifying the bin range:\n\n  >>> counts, bin_edges = jnp.histogram(a, range=(0, 25), bins=5)\n  >>> print(counts)\n  [3. 0. 2. 2. 1.]\n  >>> print(bin_edges)\n  [ 0.  5. 10. 15. 20. 25.]\n\n  Specifying the bin edges explicitly:\n\n  >>> bin_edges = jnp.array([0, 10, 20, 30])\n  >>> counts, _ = jnp.histogram(a, bins=bin_edges)\n  >>> print(counts)\n  [3. 4. 1.]\n\n  Using ``density=True`` returns a normalized histogram:\n\n  >>> density, bin_edges = jnp.histogram(a, density=True)\n  >>> dx = jnp.diff(bin_edges)\n  >>> normed_sum = jnp.sum(density * dx)\n  >>> jnp.allclose(normed_sum, 1.0)\n  Array(True, dtype=bool)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "bins", "type": "Any"},
      {"name": "range", "type": "Any"},
      {"name": "weights", "type": "Any"},
      {"name": "density", "type": "Any"},
    ],
    "variants": {},
  },
  "histogram2d": {
    "description": "Compute a 2-dimensional histogram.\n\nJAX implementation of :func:`numpy.histogram2d`.\n\nArgs:\n  x: one-dimensional array of x-values for points to be binned.\n  y: one-dimensional array of y-values for points to be binned.\n  bins: Specify the number of bins in the histogram (default: 10). ``bins``\n    may also be an array specifying the locations of the bin edges, or a pair\n    of integers or pair of arrays specifying the number of bins in each\n    dimension.\n  range: Pair of arrays or lists of the form ``[[xmin, xmax], [ymin, ymax]]``\n    specifying the range of the data in each dimension. If not specified, the\n    range is inferred from the data.\n  weights: An optional array specifying the weights of the data points.\n    Should be the same shape as ``x`` and ``y``. If not specified, each\n    data point is weighted equally.\n  density: If True, return the normalized histogram in units of counts\n    per unit area. If False (default) return the (weighted) counts per bin.\n\nReturns:\n  A tuple of arrays ``(histogram, x_edges, y_edges)``, where ``histogram``\n  contains the aggregated data, and ``x_edges`` and ``y_edges`` specify the\n  boundaries of the bins.\n\nSee Also:\n  - :func:`jax.numpy.histogram`: Compute the histogram of a 1D array.\n  - :func:`jax.numpy.histogramdd`: Compute the histogram of an N-dimensional array.\n  - :func:`jax.numpy.histogram_bin_edges`: Compute the bin edges for a histogram.\n\nExamples:\n  >>> x = jnp.array([1, 2, 3, 10, 11, 15, 19, 25])\n  >>> y = jnp.array([2, 5, 6, 8, 13, 16, 17, 18])\n  >>> counts, x_edges, y_edges = jnp.histogram2d(x, y, bins=8)\n  >>> counts.shape\n  (8, 8)\n  >>> x_edges\n  Array([ 1.,  4.,  7., 10., 13., 16., 19., 22., 25.], dtype=float32)\n  >>> y_edges\n  Array([ 2.,  4.,  6.,  8., 10., 12., 14., 16., 18.], dtype=float32)\n\n  Specifying the bin range:\n\n  >>> counts, x_edges, y_edges = jnp.histogram2d(x, y, range=[(0, 25), (0, 25)], bins=5)\n  >>> counts.shape\n  (5, 5)\n  >>> x_edges\n  Array([ 0.,  5., 10., 15., 20., 25.], dtype=float32)\n  >>> y_edges\n  Array([ 0.,  5., 10., 15., 20., 25.], dtype=float32)\n\n  Specifying the bin edges explicitly:\n\n  >>> x_edges = jnp.array([0, 10, 20, 30])\n  >>> y_edges = jnp.array([0, 10, 20, 30])\n  >>> counts, _, _ = jnp.histogram2d(x, y, bins=[x_edges, y_edges])\n  >>> counts\n  Array([[3, 0, 0],\n         [1, 3, 0],\n         [0, 1, 0]], dtype=int32)\n\n  Using ``density=True`` returns a normalized histogram:\n\n  >>> density, x_edges, y_edges = jnp.histogram2d(x, y, density=True)\n  >>> dx = jnp.diff(x_edges)\n  >>> dy = jnp.diff(y_edges)\n  >>> normed_sum = jnp.sum(density * dx[:, None] * dy[None, :])\n  >>> jnp.allclose(normed_sum, 1.0)\n  Array(True, dtype=bool)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "y", "type": "Any"},
      {"name": "bins", "type": "Any"},
      {"name": "range", "type": "Any"},
      {"name": "weights", "type": "Any"},
      {"name": "density", "type": "Any"},
    ],
    "variants": {},
  },
  "histogram_bin_edges": {
    "description": "Compute the bin edges for a histogram.\n\nJAX implementation of :func:`numpy.histogram_bin_edges`.\n\nArgs:\n  a: array of values to be binned\n  bins: Specify the number of bins in the histogram (default: 10).\n  range: tuple of scalars. Specifies the range of the data. If not specified,\n    the range is inferred from the data.\n  weights: unused by JAX.\n\nReturns:\n  An array of bin edges for the histogram.\n\nSee also:\n  - :func:`jax.numpy.histogram`: compute a 1D histogram.\n  - :func:`jax.numpy.histogram2d`: compute a 2D histogram.\n  - :func:`jax.numpy.histogramdd`: compute an N-dimensional histogram.\n\nExamples:\n  >>> a = jnp.array([2, 5, 3, 6, 4, 1])\n  >>> jnp.histogram_bin_edges(a, bins=5)\n  Array([1., 2., 3., 4., 5., 6.], dtype=float32)\n  >>> jnp.histogram_bin_edges(a, bins=5, range=(-10, 10))  # doctest: +SKIP\n  Array([-10.,  -6.,  -2.,   2.,   6.,  10.], dtype=float32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "bins", "type": "Any"},
      {"name": "range", "type": "Any"},
      {"name": "weights", "type": "Any"},
    ],
    "variants": {},
  },
  "histogramdd": {
    "description": "Compute an N-dimensional histogram.\n\nJAX implementation of :func:`numpy.histogramdd`.\n\nArgs:\n  sample: input array of shape ``(N, D)`` representing ``N`` points in\n    ``D`` dimensions.\n  bins: Specify the number of bins in each dimension of the histogram.\n    (default: 10). May also be a length-D sequence of integers or arrays\n    of bin edges.\n  range: Length-D sequence of pairs specifying the range for each dimension.\n    If not specified, the range is inferred from the data.\n  weights: An optional shape ``(N,)`` array specifying the weights of the\n    data points.\n    Should be the same shape as ``sample``. If not specified, each\n    data point is weighted equally.\n  density: If True, return the normalized histogram in units of counts\n    per unit volume. If False (default) return the (weighted) counts per bin.\n\nReturns:\n  A tuple of arrays ``(histogram, bin_edges)``, where ``histogram`` contains\n  the aggregated data, and ``bin_edges`` specifies the boundaries of the bins.\n\nSee Also:\n  - :func:`jax.numpy.histogram`: Compute the histogram of a 1D array.\n  - :func:`jax.numpy.histogram2d`: Compute the histogram of a 2D array.\n  - :func:`jax.numpy.histogram_bin_edges`: Compute the bin edges for a histogram.\n\nExamples:\n  A histogram over 100 points in three dimensions\n\n  >>> key = jax.random.key(42)\n  >>> a = jax.random.normal(key, (100, 3))\n  >>> counts, bin_edges = jnp.histogramdd(a, bins=6,\n  ...                                     range=[(-3, 3), (-3, 3), (-3, 3)])\n  >>> counts.shape\n  (6, 6, 6)\n  >>> bin_edges  # doctest: +SKIP\n  [Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32),\n   Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32),\n   Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32)]\n\n  Using ``density=True`` returns a normalized histogram:\n\n  >>> density, bin_edges = jnp.histogramdd(a, density=True)\n  >>> bin_widths = map(jnp.diff, bin_edges)\n  >>> dx, dy, dz = jnp.meshgrid(*bin_widths, indexing='ij')\n  >>> normed = jnp.sum(density * dx * dy * dz)\n  >>> jnp.allclose(normed, 1.0)\n  Array(True, dtype=bool)",
    "std_args": [
      {"name": "sample", "type": "Any"},
      {"name": "bins", "type": "Any"},
      {"name": "range", "type": "Any"},
      {"name": "weights", "type": "Any"},
      {"name": "density", "type": "Any"},
    ],
    "variants": {},
  },
  "hsplit": {
    "description": "Split an array into sub-arrays horizontally.\n\nJAX implementation of :func:`numpy.hsplit`.\n\nRefer to the documentation of :func:`jax.numpy.split` for details. ``hsplit`` is\nequivalent to ``split`` with ``axis=1``, or ``axis=0`` for one-dimensional arrays.\n\nExamples:\n  1D array:\n\n  >>> x = jnp.array([1, 2, 3, 4, 5, 6])\n  >>> x1, x2 = jnp.hsplit(x, 2)\n  >>> print(x1, x2)\n  [1 2 3] [4 5 6]\n\n  2D array:\n\n  >>> x = jnp.array([[1, 2, 3, 4],\n  ...                [5, 6, 7, 8]])\n  >>> x1, x2 = jnp.hsplit(x, 2)\n  >>> print(x1)\n  [[1 2]\n   [5 6]]\n  >>> print(x2)\n  [[3 4]\n   [7 8]]\n\nSee also:\n  - :func:`jax.numpy.split`: split an array along any axis.\n  - :func:`jax.numpy.vsplit`: split vertically, i.e. along axis=0\n  - :func:`jax.numpy.dsplit`: split depth-wise, i.e. along axis=2\n  - :func:`jax.numpy.array_split`: like ``split``, but allows ``indices_or_sections``\n    to be an integer that does not evenly divide the size of the array.",
    "std_args": [
      {"name": "ary", "type": "Any"},
      {"name": "indices_or_sections", "type": "Any"},
    ],
    "variants": {},
  },
  "hstack": {
    "description": "Horizontally stack arrays.\n\nJAX implementation of :func:`numpy.hstack`.\n\nFor arrays of one or more dimensions, this is equivalent to\n:func:`jax.numpy.concatenate` with ``axis=1``.\n\nArgs:\n  tup: a sequence of arrays to stack; each must have the same shape along all\n    but the second axis. Input arrays will be promoted to at least rank 1.\n    If a single array is given it will be treated equivalently to\n    `tup = unstack(tup)`, but the implementation will avoid explicit unstacking.\n  dtype: optional dtype of the resulting array. If not specified, the dtype\n    will be determined via type promotion rules described in :ref:`type-promotion`.\n\nReturns:\n  the stacked result.\n\nSee also:\n  - :func:`jax.numpy.stack`: stack along arbitrary axes\n  - :func:`jax.numpy.concatenate`: concatenation along existing axes.\n  - :func:`jax.numpy.vstack`: stack vertically, i.e. along axis 0.\n  - :func:`jax.numpy.dstack`: stack depth-wise, i.e. along axis 2.\n\nExamples:\n  Scalar values:\n\n  >>> jnp.hstack([1, 2, 3])\n  Array([1, 2, 3], dtype=int32, weak_type=True)\n\n  1D arrays:\n\n  >>> x = jnp.arange(3)\n  >>> y = jnp.ones(3)\n  >>> jnp.hstack([x, y])\n  Array([0., 1., 2., 1., 1., 1.], dtype=float32)\n\n  2D arrays:\n\n  >>> x = x.reshape(3, 1)\n  >>> y = y.reshape(3, 1)\n  >>> jnp.hstack([x, y])\n  Array([[0., 1.],\n         [1., 1.],\n         [2., 1.]], dtype=float32)",
    "std_args": [
      {"name": "tup", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "huber_loss": {
    "description": "Compute the Huber loss, with optional weighting.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "delta", "type": "Any"},
      {"name": "weight", "type": "Any"},
    ],
    "variants": {},
  },
  "hypot": {
    "description": "Computes the square root of the sum of squares for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "ifft": {
    "description": "Computes the one-dimensional inverse discrete Fourier transform.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "Optional[int]"},
      {"name": "axis", "type": "int"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "ifft2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ifftn": {
    "description": "Computes the n-dimensional inverse discrete Fourier transform.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "s", "type": "Optional[Sequence[int]]"},
      {"name": "axes", "type": "Optional[Sequence[int]]"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "ifftshift": {
    "description": "Inverse of ``fftshift``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axes", "type": "Optional[Union[int, Sequence[int]]]"},
    ],
    "variants": {},
  },
  "ihfft": {
    "description": "Computes the one-dimensional inverse discrete Fourier transform of a signal with Hermitian symmetry.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "Optional[int]"},
      {"name": "axis", "type": "int"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "iinfo": {
    "description": "Machine limits for integer data types.",
    "std_args": [
      {"name": "type", "type": "Union[dtype, array]"},
    ],
    "variants": {},
  },
  "imag": {
    "description": "Returns the imaginary component of a complex number for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "in1_features": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "in2_features": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "in_features": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "in_top_k": {
    "description": "Checks if the targets are in the top-k predictions.\n\nArgs:\n    targets: A tensor of true labels.\n    predictions: A tensor of predicted labels.\n    k: An integer representing the number of predictions to consider.\n\nReturns:\n    A boolean tensor of the same shape as `targets`, where each element\n    indicates whether the corresponding target is in the top-k predictions.\n\nExample:\n\n>>> targets = keras.ops.convert_to_tensor([2, 5, 3])\n>>> predictions = keras.ops.convert_to_tensor(\n... [[0.1, 0.4, 0.6, 0.9, 0.5],\n...  [0.1, 0.7, 0.9, 0.8, 0.3],\n...  [0.1, 0.6, 0.9, 0.9, 0.5]])\n>>> in_top_k(targets, predictions, k=3)\narray([ True False  True], shape=(3,), dtype=bool)",
    "std_args": [
      {"name": "targets", "type": "Any"},
      {"name": "predictions", "type": "Any"},
      {"name": "k", "type": "Any"},
    ],
    "variants": {},
  },
  "index": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "indices": {
    "description": "Generate arrays of grid indices.\n\nJAX implementation of :func:`numpy.indices`.\n\nArgs:\n  dimensions: the shape of the grid.\n  dtype: the dtype of the indices (defaults to integer).\n  sparse: if True, then return sparse indices. Default is False, which\n    returns dense indices.\n\nReturns:\n  An array of shape ``(len(dimensions), *dimensions)`` If ``sparse`` is False,\n  or a sequence of arrays of the same length as ``dimensions`` if ``sparse`` is True.\n\nSee also:\n  - :func:`jax.numpy.meshgrid`: generate a grid from arbitrary input arrays.\n  - :obj:`jax.numpy.mgrid`: generate dense indices using a slicing syntax.\n  - :obj:`jax.numpy.ogrid`: generate sparse indices using a slicing syntax.\n\nExamples:\n  >>> jnp.indices((2, 3))\n  Array([[[0, 0, 0],\n          [1, 1, 1]],\n  <BLANKLINE>\n         [[0, 1, 2],\n          [0, 1, 2]]], dtype=int32)\n  >>> jnp.indices((2, 3), sparse=True)\n  (Array([[0],\n         [1]], dtype=int32), Array([[0, 1, 2]], dtype=int32))",
    "std_args": [
      {"name": "dimensions", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "sparse", "type": "Any"},
    ],
    "variants": {},
  },
  "inexact": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "inf": {
    "description": "IEEE 754 floating-point representation of (positive) infinity.",
    "std_args": [],
    "variants": {},
  },
  "init": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "inject_hyperparams": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "inject_stateful_hyperparams": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "inner": {
    "description": "Returns the inner product of a and b for arrays of floating point types.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "b", "type": "Any"},
    ],
    "variants": {},
  },
  "inplace": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "insert": {
    "description": "Insert a given module before a given index in the list.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "index", "type": "Any"},
      {"name": "module", "type": "Any"},
    ],
    "variants": {},
  },
  "int": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "int2": {
    "description": "A JAX scalar constructor of type int2.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "int4": {
    "description": "A JAX scalar constructor of type int4.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "int8": {
    "description": "A JAX scalar constructor of type int8.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "int_": {
    "description": "A JAX scalar constructor of type int64.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "integer": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "interp": {
    "description": 'One-dimensional linear interpolation.\n\nJAX implementation of :func:`numpy.interp`.\n\nArgs:\n  x: N-dimensional array of x coordinates at which to evaluate the interpolation.\n  xp: one-dimensional sorted array of points to be interpolated.\n  fp: array of shape ``xp.shape`` containing the function values associated with ``xp``.\n  left: specify how to handle points ``x < xp[0]``. Default is to return ``fp[0]``.\n    If ``left`` is a scalar value, it will return this value. if ``left`` is the string\n    ``"extrapolate"``, then the value will be determined by linear extrapolation.\n    ``left`` is ignored if ``period`` is specified.\n  right: specify how to handle points ``x > xp[-1]``. Default is to return ``fp[-1]``.\n    If ``right`` is a scalar value, it will return this value. if ``right`` is the string\n    ``"extrapolate"``, then the value will be determined by linear extrapolation.\n    ``right`` is ignored if ``period`` is specified.\n  period: optionally specify the period for the *x* coordinates, for e.g. interpolation\n    in angular space.\n\nReturns:\n  an array of shape ``x.shape`` containing the interpolated function at values ``x``.\n\nExamples:\n  >>> xp = jnp.arange(10)\n  >>> fp = 2 * xp\n  >>> x = jnp.array([0.5, 2.0, 3.5])\n  >>> interp(x, xp, fp)\n  Array([1., 4., 7.], dtype=float32)\n\n  Unless otherwise specified, extrapolation will be constant:\n\n  >>> x = jnp.array([-10., 10.])\n  >>> interp(x, xp, fp)\n  Array([ 0., 18.], dtype=float32)\n\n  Use ``"extrapolate"`` mode for linear extrapolation:\n\n  >>> interp(x, xp, fp, left=\'extrapolate\', right=\'extrapolate\')\n  Array([-20.,  20.], dtype=float32)\n\n  For periodic interpolation, specify the ``period``:\n\n  >>> xp = jnp.array([0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])\n  >>> fp = jnp.sin(xp)\n  >>> x = 2 * jnp.pi  # note: not in input array\n  >>> jnp.interp(x, xp, fp, period=2 * jnp.pi)\n  Array(0., dtype=float32)',
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "xp", "type": "Any"},
      {"name": "fp", "type": "Any"},
      {"name": "left", "type": "Any"},
      {"name": "right", "type": "Any"},
      {"name": "period", "type": "Any"},
    ],
    "variants": {},
  },
  "interpolate": {
    "description": "Down/up samples the input.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "scale_factor", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "align_corners", "type": "Any"},
      {"name": "recompute_scale_factor", "type": "Any"},
      {"name": "antialias", "type": "Any"},
    ],
    "variants": {},
  },
  "intersect1d": {
    "description": "Compute the set intersection of two 1D arrays.\n\nJAX implementation of :func:`numpy.intersect1d`.\n\nBecause the size of the output of ``intersect1d`` is data-dependent, the function\nis not typically compatible with :func:`~jax.jit` and other JAX transformations.\nThe JAX version adds the optional ``size`` argument which must be specified\nstatically for ``jnp.intersect1d`` to be used in such contexts.\n\nArgs:\n  ar1: first array of values to intersect.\n  ar2: second array of values to intersect.\n  assume_unique: if True, assume the input arrays contain unique values. This allows\n    a more efficient implementation, but if ``assume_unique`` is True and the input\n    arrays contain duplicates, the behavior is undefined. default: False.\n  return_indices: If True, return arrays of indices specifying where the intersected\n    values first appear in the input arrays.\n  size: if specified, return only the first ``size`` sorted elements. If there are fewer\n    elements than ``size`` indicates, the return value will be padded with ``fill_value``,\n    and returned indices will be padded with an out-of-bound index.\n  fill_value: when ``size`` is specified and there are fewer than the indicated number of\n    elements, fill the remaining entries ``fill_value``. Defaults to the smallest value\n    in the intersection.\n\nReturns:\n  An array ``intersection``, or if ``return_indices=True``, a tuple of arrays\n  ``(intersection, ar1_indices, ar2_indices)``. Returned values are\n\n  - ``intersection``:\n    A 1D array containing each value that appears in both ``ar1`` and ``ar2``.\n  - ``ar1_indices``:\n    *(returned if return_indices=True)* an array of shape ``intersection.shape`` containing\n    the indices in flattened ``ar1`` of values in ``intersection``. For 1D inputs,\n    ``intersection`` is equivalent to ``ar1[ar1_indices]``.\n  - ``ar2_indices``:\n    *(returned if return_indices=True)* an array of shape ``intersection.shape`` containing\n    the indices in flattened ``ar2`` of values in ``intersection``. For 1D inputs,\n    ``intersection`` is equivalent to ``ar2[ar2_indices]``.\n\nSee also:\n  - :func:`jax.numpy.union1d`: the set union of two 1D arrays.\n  - :func:`jax.numpy.setxor1d`: the set XOR of two 1D arrays.\n  - :func:`jax.numpy.setdiff1d`: the set difference of two 1D arrays.\n\nExamples:\n  >>> ar1 = jnp.array([1, 2, 3, 4])\n  >>> ar2 = jnp.array([3, 4, 5, 6])\n  >>> jnp.intersect1d(ar1, ar2)\n  Array([3, 4], dtype=int32)\n\n  Computing intersection with indices:\n\n  >>> intersection, ar1_indices, ar2_indices = jnp.intersect1d(ar1, ar2, return_indices=True)\n  >>> intersection\n  Array([3, 4], dtype=int32)\n\n  ``ar1_indices`` gives the indices of the intersected values within ``ar1``:\n\n   >>> ar1_indices\n   Array([2, 3], dtype=int32)\n   >>> jnp.all(intersection == ar1[ar1_indices])\n   Array(True, dtype=bool)\n\n  ``ar2_indices`` gives the indices of the intersected values within ``ar2``:\n\n   >>> ar2_indices\n   Array([0, 1], dtype=int32)\n   >>> jnp.all(intersection == ar2[ar2_indices])\n   Array(True, dtype=bool)",
    "std_args": [
      {"name": "ar1", "type": "Any"},
      {"name": "ar2", "type": "Any"},
      {"name": "assume_unique", "type": "Any"},
      {"name": "return_indices", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "inv": {
    "description": "Returns the multiplicative inverse of a square matrix (or a stack of square matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "invert_permutation": {
    "description": "Returns the inverse of ``permutation``.",
    "std_args": [
      {"name": "permutation", "type": "Any"},
    ],
    "variants": {},
  },
  "irfft": {
    "description": "Computes the one-dimensional inverse of ``rfft`` for complex-valued input.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "Optional[int]"},
      {"name": "axis", "type": "int"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "irfft2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "irfftn": {
    "description": "Computes the n-dimensional inverse of ``rfftn`` for complex-valued input.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "s", "type": "Optional[Sequence[int]]"},
      {"name": "axes", "type": "Optional[Sequence[int]]"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "is_tensor": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "isclose": {
    "description": "Return whether two tensors are element-wise almost equal.\n\nArgs:\n    x1: First input tensor.\n    x2: Second input tensor.\n    rtol: Relative tolerance.\n    atol: Absolute tolerance.\n    equal_nan: If `True`, element-wise NaNs are considered equal.\n\nReturns:\n    Output boolean tensor.",
    "std_args": [
      {"name": "x1", "type": "Any"},
      {"name": "x2", "type": "Any"},
      {"name": "rtol", "type": "Any"},
      {"name": "atol", "type": "Any"},
      {"name": "equal_nan", "type": "Any"},
    ],
    "variants": {},
  },
  "iscomplex": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "var", "type": "Any"},
    ],
    "variants": {},
  },
  "iscomplexobj": {
    "description": "Check if the input is a complex number or an array containing complex elements.\n\nJAX implementation of :func:`numpy.iscomplexobj`.\n\nThe function evaluates based on input type rather than value.\nInputs with zero imaginary parts are still considered complex.\n\nArgs:\n  x: input object to check.\n\nReturns:\n  True if ``x`` is a complex number or an array containing at least one complex element,\n  False otherwise.\n\nSee Also:\n  - :func:`jax.numpy.isrealobj`\n  - :func:`jax.numpy.iscomplex`\n\nExamples:\n  >>> jnp.iscomplexobj(True)\n  False\n  >>> jnp.iscomplexobj(0)\n  False\n  >>> jnp.iscomplexobj(jnp.array([1, 2]))\n  False\n  >>> jnp.iscomplexobj(1+2j)\n  True\n  >>> jnp.iscomplexobj(jnp.array([0, 1+2j]))\n  True",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "isdtype": {
    "description": 'Returns a boolean indicating whether a provided dtype is of a specified data type "kind".',
    "std_args": [
      {"name": "dtype", "type": "dtype"},
      {"name": "kind", "type": "Union[dtype, str, Tuple[Union[dtype, str], Ellipsis]]"},
    ],
    "variants": {},
  },
  "isfinite": {
    "description": "Tests each element ``x_i`` of the input array ``x`` to determine if finite.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "isin": {
    "description": "Determine whether elements in ``element`` appear in ``test_elements``.\n\nJAX implementation of :func:`numpy.isin`.\n\nArgs:\n  element: input array of elements for which membership will be checked.\n  test_elements: N-dimensional array of test values to check for the presence of\n    each element.\n  invert: If True, return ``~isin(element, test_elements)``. Default is False.\n  assume_unique: if true, input arrays are assumed to be unique, which can\n    lead to more efficient computation. If the input arrays are not unique\n    and assume_unique is set to True, the results are undefined.\n  method: string specifying the method used to compute the result. Supported\n    options are 'compare_all', 'binary_search', 'sort', and 'auto' (default).\n\nReturns:\n  A boolean array of shape ``element.shape`` that specifies whether each element\n  appears in ``test_elements``.\n\nExamples:\n  >>> elements = jnp.array([1, 2, 3, 4])\n  >>> test_elements = jnp.array([[1, 5, 6, 3, 7, 1]])\n  >>> jnp.isin(elements, test_elements)\n  Array([ True, False,  True, False], dtype=bool)",
    "std_args": [
      {"name": "element", "type": "Any"},
      {"name": "test_elements", "type": "Any"},
      {"name": "assume_unique", "type": "Any"},
      {"name": "invert", "type": "Any"},
      {"name": "method", "type": "Any"},
    ],
    "variants": {},
  },
  "isinf": {
    "description": "Tests each element ``x_i`` of the input array ``x`` to determine if equal to positive or negative infinity.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "isnan": {
    "description": "Tests each element ``x_i`` of the input array ``x`` to determine whether the element is ``NaN``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "isneginf": {
    "description": "Return boolean array indicating whether each element of input is negative infinite.\n\nJAX implementation of :obj:`numpy.isneginf`.\n\nArgs:\n  x: input array or scalar. ``complex`` dtype are not supported.\n\nReturns:\n  A boolean array of same shape as ``x`` containing ``True`` where ``x`` is\n  ``-inf``, and ``False`` otherwise.\n\nSee also:\n  - :func:`jax.numpy.isinf`: Returns a boolean array indicating whether each\n    element of input is either positive or negative infinity.\n  - :func:`jax.numpy.isposinf`: Returns a boolean array indicating whether each\n    element of input is positive infinity.\n  - :func:`jax.numpy.isfinite`: Returns a boolean array indicating whether each\n    element of input is finite.\n  - :func:`jax.numpy.isnan`: Returns a boolean array indicating whether each\n    element of input is not a number (``NaN``).\n\nExamples:\n  >>> jnp.isneginf(jnp.inf)\n  Array(False, dtype=bool)\n  >>> x = jnp.array([-jnp.inf, 5, jnp.inf, jnp.nan, 1])\n  >>> jnp.isneginf(x)\n  Array([ True, False, False, False, False], dtype=bool)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "out", "type": "Any"},
    ],
    "variants": {},
  },
  "isposinf": {
    "description": "Return boolean array indicating whether each element of input is positive infinite.\n\nJAX implementation of :obj:`numpy.isposinf`.\n\nArgs:\n  x: input array or scalar. ``complex`` dtype are not supported.\n\nReturns:\n  A boolean array of same shape as ``x`` containing ``True`` where ``x`` is\n  ``inf``, and ``False`` otherwise.\n\nSee also:\n  - :func:`jax.numpy.isinf`: Returns a boolean array indicating whether each\n    element of input is either positive or negative infinity.\n  - :func:`jax.numpy.isneginf`: Returns a boolean array indicating whether each\n    element of input is negative infinity.\n  - :func:`jax.numpy.isfinite`: Returns a boolean array indicating whether each\n    element of input is finite.\n  - :func:`jax.numpy.isnan`: Returns a boolean array indicating whether each\n    element of input is not a number (``NaN``).\n\nExamples:\n  >>> jnp.isposinf(5)\n  Array(False, dtype=bool)\n  >>> x = jnp.array([-jnp.inf, 5, jnp.inf, jnp.nan, 1])\n  >>> jnp.isposinf(x)\n  Array([False, False,  True, False, False], dtype=bool)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "out", "type": "Any"},
    ],
    "variants": {},
  },
  "isreal": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "var", "type": "Any"},
    ],
    "variants": {},
  },
  "isrealobj": {
    "description": "Check if the input is not a complex number or an array containing complex elements.\n\nJAX implementation of :func:`numpy.isrealobj`.\n\nThe function evaluates based on input type rather than value.\nInputs with zero imaginary parts are still considered complex.\n\nArgs:\n  x: input object to check.\n\nReturns:\n  False if ``x`` is a complex number or an array containing at least one complex element,\n  True otherwise.\n\nSee Also:\n  - :func:`jax.numpy.iscomplexobj`\n  - :func:`jax.numpy.isreal`\n\nExamples:\n  >>> jnp.isrealobj(0)\n  True\n  >>> jnp.isrealobj(1.2)\n  True\n  >>> jnp.isrealobj(jnp.array([1, 2]))\n  True\n  >>> jnp.isrealobj(1+2j)\n  False\n  >>> jnp.isrealobj(jnp.array([0, 1+2j]))\n  False",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "isscalar": {
    "description": "Return True if the input is a scalar.\n\nJAX implementation of :func:`numpy.isscalar`. JAX's implementation differs\nfrom NumPy's in that it considers zero-dimensional arrays to be scalars; see\nthe *Note* below for more details.\n\nArgs:\n  element: input object to check; any type is valid input.\n\nReturns:\n  True if ``element`` is a scalar value or an array-like object with zero\n  dimensions, False otherwise.\n\nNote:\n  JAX and NumPy differ in their representation of scalar values. NumPy has\n  special scalar objects (e.g. ``np.int32(0)``) which are distinct from\n  zero-dimensional arrays (e.g. ``np.array(0)``), and :func:`numpy.isscalar`\n  returns ``True`` for the former and ``False`` for the latter.\n\n  JAX does not define special scalar objects, but rather represents scalars as\n  zero-dimensional arrays. As such, :func:`jax.numpy.isscalar` returns ``True``\n  for both scalar objects (e.g. ``0.0`` or ``np.float32(0.0)``) and array-like\n  objects with zero dimensions (e.g. ``jnp.array(0.0)``, ``np.array(0.0)``).\n\n  One reason for the different conventions in ``isscalar`` is to maintain\n  JIT-invariance: i.e. the property that the result of a function should not\n  change when it is JIT-compiled. Because scalar inputs are cast to\n  zero-dimensional JAX arrays at JIT boundaries, the semantics of\n  :func:`numpy.isscalar` are such that the result changes under JIT:\n\n  >>> np.isscalar(1.0)\n  True\n  >>> jax.jit(np.isscalar)(1.0)\n  Array(False, dtype=bool)\n\n  By treating zero-dimensional arrays as scalars, :func:`jax.numpy.isscalar`\n  avoids this issue:\n\n  >>> jnp.isscalar(1.0)\n  True\n  >>> jax.jit(jnp.isscalar)(1.0)\n  Array(True, dtype=bool)\n\nExamples:\n  In JAX, both scalars and zero-dimensional array-like objects are considered\n  scalars:\n\n  >>> jnp.isscalar(1.0)\n  True\n  >>> jnp.isscalar(1 + 1j)\n  True\n  >>> jnp.isscalar(jnp.array(1))  # zero-dimensional JAX array\n  True\n  >>> jnp.isscalar(jnp.int32(1))  # JAX scalar constructor\n  True\n  >>> jnp.isscalar(np.array(1.0))  # zero-dimensional NumPy array\n  True\n  >>> jnp.isscalar(np.int32(1))  # NumPy scalar type\n  True\n\n  Arrays with one or more dimension are not considered scalars:\n\n  >>> jnp.isscalar(jnp.array([1]))\n  False\n  >>> jnp.isscalar(np.array([1]))\n  False\n\n  Compare this to :func:`numpy.isscalar`, which returns ``True`` for\n  scalar-typed objects, and ``False`` for *all* arrays, even those with\n  zero dimensions:\n\n  >>> np.isscalar(np.int32(1))  # scalar object\n  True\n  >>> np.isscalar(np.array(1))  # zero-dimensional array\n  False\n\n  In JAX, as in NumPy, objects which are not array-like are not considered\n  scalars:\n\n  >>> jnp.isscalar(None)\n  False\n  >>> jnp.isscalar([1])\n  False\n  >>> jnp.isscalar(())\n  False\n  >>> jnp.isscalar(slice(10))\n  False",
    "std_args": [
      {"name": "element", "type": "Any"},
    ],
    "variants": {},
  },
  "issubdtype": {
    "description": "Return True if arg1 is equal or lower than arg2 in the type hierarchy.\n\nJAX implementation of :func:`numpy.issubdtype`.\n\nThe main difference in JAX's implementation is that it properly handles\ndtype extensions such as :code:`bfloat16`.\n\nArgs:\n  arg1: dtype-like object. In typical usage, this will be a dtype specifier,\n    such as ``\"float32\"`` (i.e. a string), ``np.dtype('int32')`` (i.e. an\n    instance of :class:`numpy.dtype`), ``jnp.complex64`` (i.e. a JAX scalar\n    constructor), or ``np.uint8`` (i.e. a NumPy scalar type).\n  arg2: dtype-like object. In typical usage, this will be a generic scalar\n    type, such as ``jnp.integer``, ``jnp.floating``, or ``jnp.complexfloating``.\n\nReturns:\n  True if arg1 represents a dtype that is equal or lower in the type\n  hierarchy than arg2.\n\nSee also:\n  - :func:`jax.numpy.isdtype`: similar function aligning with the array API standard.\n\nExamples:\n  >>> jnp.issubdtype('uint32', jnp.unsignedinteger)\n  True\n  >>> jnp.issubdtype(np.int32, jnp.integer)\n  True\n  >>> jnp.issubdtype(jnp.bfloat16, jnp.floating)\n  True\n  >>> jnp.issubdtype(np.dtype('complex64'), jnp.complexfloating)\n  True\n  >>> jnp.issubdtype('complex64', jnp.integer)\n  False\n\n  Be aware that while this is very similar to :func:`numpy.issubdtype`, the\n  results of these differ in the case of JAX's custom floating point types:\n\n  >>> np.issubdtype('bfloat16', np.floating)\n  False\n  >>> jnp.issubdtype('bfloat16', jnp.floating)\n  True",
    "std_args": [
      {"name": "arg1", "type": "Any"},
      {"name": "arg2", "type": "Any"},
    ],
    "variants": {},
  },
  "istft": {
    "description": 'Inverse Short-Time Fourier Transform along the last axis of the input.\n\nTo reconstruct an original waveform, the parameters should be the same in\n`stft`.\n\nArgs:\n    x: Tuple of the real and imaginary parts of the input tensor. Both\n        tensors in the tuple should be of floating type.\n    sequence_length: An integer representing the sequence length.\n    sequence_stride: An integer representing the sequence hop size.\n    fft_length: An integer representing the size of the FFT that produced\n        `stft`. Should be of type `int32`.\n    length: An integer representing the output is clipped to exactly length.\n        If not specified, no padding or clipping take place. Defaults to\n        `None`.\n    window: A string, a tensor of the window or `None`. If `window` is a\n        string, available values are `"hann"` and `"hamming"`. If `window`\n        is a tensor, it will be used directly as the window and its length\n        must be `sequence_length`. If `window` is `None`, no windowing is\n        used. Defaults to `"hann"`.\n    center: Whether `x` was padded on both sides so that the t-th sequence\n        is centered at time `t * sequence_stride`. Defaults to `True`.\n\nReturns:\n    A tensor containing the inverse Short-Time Fourier Transform along the\n    last axis of `x`.\n\nExample:\n\n>>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])\n>>> istft(stft(x, 1, 1, 1), 1, 1, 1)\narray([0.0, 1.0, 2.0, 3.0, 4.0])',
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "sequence_length", "type": "Any"},
      {"name": "sequence_stride", "type": "Any"},
      {"name": "fft_length", "type": "Any"},
      {"name": "length", "type": "Any"},
      {"name": "window", "type": "Any"},
      {"name": "center", "type": "Any"},
    ],
    "variants": {},
  },
  "item": {
    "description": "Copy an element of an array to a standard Python scalar and return it.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "args", "type": "Any"},
    ],
    "variants": {},
  },
  "items": {
    "description": "Return an iterable of the ParameterDict key/value pairs.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "itemsize": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ix_": {
    "description": "Return a multi-dimensional grid (open mesh) from N one-dimensional sequences.\n\nJAX implementation of :func:`numpy.ix_`.\n\nArgs:\n  *args: N one-dimensional arrays\n\nReturns:\n  Tuple of Jax arrays forming an open mesh, each with N dimensions.\n\nSee Also:\n  - :obj:`jax.numpy.ogrid`\n  - :obj:`jax.numpy.mgrid`\n  - :func:`jax.numpy.meshgrid`\n\nExamples:\n  >>> rows = jnp.array([0, 2])\n  >>> cols = jnp.array([1, 3])\n  >>> open_mesh = jnp.ix_(rows, cols)\n  >>> open_mesh\n  (Array([[0],\n        [2]], dtype=int32), Array([[1, 3]], dtype=int32))\n  >>> [grid.shape for grid in open_mesh]\n  [(2, 1), (1, 2)]\n  >>> x = jnp.array([[10, 20, 30, 40],\n  ...                [50, 60, 70, 80],\n  ...                [90, 100, 110, 120],\n  ...                [130, 140, 150, 160]])\n  >>> x[open_mesh]\n  Array([[ 20,  40],\n         [100, 120]], dtype=int32)",
    "std_args": [
      {"name": "args", "type": "Any"},
    ],
    "variants": {},
  },
  "jit": {
    "description": "JIT Compilation.",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "static_argnums", "type": "Any"},
    ],
    "variants": {},
  },
  "join": {
    "description": "Context manager for training with uneven inputs across processes in DDP.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "divide_by_initial_world_size", "type": "Any"},
      {"name": "enable", "type": "Any"},
      {"name": "throw_on_early_termination", "type": "Any"},
    ],
    "variants": {},
  },
  "join_schedules": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "jvp": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "_", "type": "Any"},
      {"name": "primals", "type": "Any"},
      {"name": "tangents", "type": "Any"},
      {"name": "avals", "type": "Any"},
    ],
    "variants": {},
  },
  "k": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "kaiming_normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "kaiming_uniform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "kaiser": {
    "description": "Return a Kaiser window of size M.\n\nJAX implementation of :func:`numpy.kaiser`.\n\nArgs:\n  M: The window size.\n  beta: The Kaiser window parameter.\n\nReturns:\n  An array of size M containing the Kaiser window.\n\nExamples:\n  >>> with jnp.printoptions(precision=2, suppress=True):\n  ...   print(jnp.kaiser(4, 1.5))\n  [0.61 0.95 0.95 0.61]\n\nSee also:\n  - :func:`jax.numpy.bartlett`: return a Bartlett window of size M.\n  - :func:`jax.numpy.blackman`: return a Blackman window of size M.\n  - :func:`jax.numpy.hamming`: return a Hamming window of size M.\n  - :func:`jax.numpy.hanning`: return a Hanning window of size M.",
    "std_args": [
      {"name": "M", "type": "Any"},
      {"name": "beta", "type": "Any"},
    ],
    "variants": {},
  },
  "keep_params_nonnegative": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "kernel_size": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "key": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "keys": {
    "description": "Return an iterable of the ParameterDict keys.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "kl_divergence": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "kron": {
    "description": "Kronecker product of `x1` and `x2`.\n\nComputes the Kronecker product of two input tensors. If `x1` has shape\n`(a0, a1, ..., an)` and `x2` has shape `(b0, b1, ..., bn)`, then the\noutput will have shape `(a0*b0, a1*b1, ..., an*bn)`.\n\nArgs:\n    x1: First input tensor.\n    x2: Second input tensor.\n\nReturns:\n    A tensor representing the Kronecker product of `x1` and `x2`.",
    "std_args": [
      {"name": "x1", "type": "Any"},
      {"name": "x2", "type": "Any"},
    ],
    "variants": {},
  },
  "l1_loss": {
    "description": "Compute the L1 loss, with optional weighting.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "weight", "type": "Any"},
    ],
    "variants": {},
  },
  "l2_loss": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "label_smoothing": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "lambd": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "laplace": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "layer_norm": {
    "description": "Apply Layer Normalization for last certain number of dimensions.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "normalized_shape", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "bias", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "layers": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "lcm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ldexp": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "leaky_relu": {
    "description": "leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "negative_slope", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "left_shift": {
    "description": "Shift the bits of an integer to the left.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "n", "type": "Any"},
    ],
    "variants": {},
  },
  "less": {
    "description": "Computes the truth value of ``x1_i < x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "less_equal": {
    "description": "Computes the truth value of ``x1_i <= x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "linear1": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "linear2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "linear_onecycle_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "linear_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "linear_to_mel_weight_matrix": {
    "description": 'Returns a matrix to warp linear scale spectrograms to the mel scale.\n\nReturns a weight matrix that can be used to re-weight a tensor\ncontaining `num_spectrogram_bins` linearly sampled frequency information\nfrom `[0, sampling_rate / 2]` into `num_mel_bins` frequency information\nfrom `[lower_edge_hertz, upper_edge_hertz]` on the mel scale.\n\nThis function follows the [Hidden Markov Model Toolkit (HTK)](\nhttp://htk.eng.cam.ac.uk/) convention, defining the mel scale in\nterms of a frequency in hertz according to the following formula:\n\n```mel(f) = 2595 * log10( 1 + f/700)```\n\nIn the returned matrix, all the triangles (filterbanks) have a peak\nvalue of 1.0.\n\nFor example, the returned matrix `A` can be used to right-multiply a\nspectrogram `S` of shape `[frames, num_spectrogram_bins]` of linear\nscale spectrum values (e.g. STFT magnitudes) to generate a\n"mel spectrogram" `M` of shape `[frames, num_mel_bins]`.\n\n```\n# `S` has shape [frames, num_spectrogram_bins]\n# `M` has shape [frames, num_mel_bins]\nM = keras.ops.matmul(S, A)\n```\n\nThe matrix can be used with `keras.ops.tensordot` to convert an\narbitrary rank `Tensor` of linear-scale spectral bins into the\nmel scale.\n\n```\n# S has shape [..., num_spectrogram_bins].\n# M has shape [..., num_mel_bins].\nM = keras.ops.tensordot(S, A, 1)\n```\n\nReferences:\n- [Mel scale (Wikipedia)](https://en.wikipedia.org/wiki/Mel_scale)\n\nArgs:\n    num_mel_bins: Python int. How many bands in the resulting\n        mel spectrum.\n    num_spectrogram_bins: An integer `Tensor`. How many bins there are\n        in the source spectrogram data, which is understood to be\n        `fft_size // 2 + 1`, i.e. the spectrogram only contains the\n        nonredundant FFT bins.\n    sampling_rate: An integer or float `Tensor`. Samples per second of\n        the input signal used to create the spectrogram. Used to figure\n        out the frequencies corresponding to each spectrogram bin,\n        which dictates how they are mapped into the mel scale.\n    lower_edge_hertz: Python float. Lower bound on the frequencies to be\n        included in the mel spectrum. This corresponds to the lower\n        edge of the lowest triangular band.\n    upper_edge_hertz: Python float. The desired top edge of the highest\n        frequency band.\n    dtype: The `DType` of the result matrix. Must be a floating point\n        type.\n\nReturns:\n    A tensor of shape `[num_spectrogram_bins, num_mel_bins]`.',
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "num_mel_bins", "type": "Any"},
      {"name": "num_spectrogram_bins", "type": "Any"},
      {"name": "sampling_rate", "type": "Any"},
      {"name": "lower_edge_hertz", "type": "Any"},
      {"name": "upper_edge_hertz", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "linspace": {
    "description": "Returns evenly spaced numbers over a specified interval.",
    "std_args": [
      {"name": "start", "type": "Union[int, float, complex]"},
      {"name": "stop", "type": "Union[int, float, complex]"},
      {"name": "num", "type": "int"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
      {"name": "endpoint", "type": "bool"},
    ],
    "variants": {},
  },
  "load_state_dict": {
    "description": "Copies parameters and buffers from state_dict into this module.",
    "std_args": [
      {"name": "state_dict", "type": "Any"},
      {"name": "strict", "type": "Any"},
    ],
    "variants": {},
  },
  "log": {
    "description": "Logarithm.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "log10": {
    "description": "Calculates an implementation-dependent approximation to the base ``10`` logarithm for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "log1p": {
    "description": "Calculates an implementation-dependent approximation to ``log(1+x)``, where ``log`` refers to the natural (base ``e``) logarithm, for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "log2": {
    "description": "Calculates an implementation-dependent approximation to the base ``2`` logarithm for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "log_cosh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "log_ndtr": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "log_prob": {
    "description": "Compute log probabilities for all :math:`\\texttt{n\\_classes}`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "input", "type": "Any"},
    ],
    "variants": {},
  },
  "log_sigmoid": {
    "description": "Logarithm of the sigmoid activation function.\n\nIt is defined as `f(x) = log(1 / (1 + exp(-x)))`.\n\nArgs:\n    x: Input tensor.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "log_softmax": {
    "description": "Applies the LogSoftmax function to an n-dimensional input Tensor.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "dim", "type": "Any"},
    ],
    "variants": {},
  },
  "logaddexp": {
    "description": "Calculates the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "logaddexp2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "logdet": {
    "description": "Computes log of the determinant of a hermitian positive definite matrix.\n\nArgs:\n    x: Input matrix. It must 2D and square.\n\nReturns:\n    The natural log of the determinant of matrix.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "logger": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "logical_and": {
    "description": "Computes the logical AND for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, bool]"},
      {"name": "x2", "type": "Union[array, bool]"},
    ],
    "variants": {},
  },
  "logical_not": {
    "description": "Computes the logical NOT for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "logical_or": {
    "description": "Computes the logical OR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, bool]"},
      {"name": "x2", "type": "Union[array, bool]"},
    ],
    "variants": {},
  },
  "logical_xor": {
    "description": "Computes the logical XOR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, bool]"},
      {"name": "x2", "type": "Union[array, bool]"},
    ],
    "variants": {},
  },
  "logistic": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "logit": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "lognormal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "logspace": {
    "description": "Generate logarithmically-spaced values.\n\nJAX implementation of :func:`numpy.logspace`.\n\nArgs:\n  start: scalar or array. Used to specify the start value. The start value is\n    ``base ** start``.\n  stop: scalar or array. Used to specify the stop value. The end value is\n    ``base ** stop``.\n  num: int, optional, default=50. Number of values to generate.\n  endpoint: bool, optional, default=True. If True, then include the ``stop`` value\n    in the result. If False, then exclude the ``stop`` value.\n  base: scalar or array, optional, default=10. Specifies the base of the logarithm.\n  dtype: optional. Specifies the dtype of the output.\n  axis: int, optional, default=0. Axis along which to generate the logspace.\n\nReturns:\n  An array of logarithm.\n\nSee also:\n  - :func:`jax.numpy.arange`: Generate ``N`` evenly-spaced values given a starting\n    point and a step value.\n  - :func:`jax.numpy.linspace`: Generate evenly-spaced values.\n  - :func:`jax.numpy.geomspace`: Generate geometrically-spaced values.\n\nExamples:\n  List 5 logarithmically spaced values between 1 (``10 ** 0``) and 100\n  (``10 ** 2``):\n\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.logspace(0, 2, 5)\n  Array([  1.   ,   3.162,  10.   ,  31.623, 100.   ], dtype=float32)\n\n  List 5 logarithmically-spaced values between 1(``10 ** 0``) and 100\n  (``10 ** 2``), excluding endpoint:\n\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.logspace(0, 2, 5, endpoint=False)\n  Array([ 1.   ,  2.512,  6.31 , 15.849, 39.811], dtype=float32)\n\n  List 7 logarithmically-spaced values between 1 (``2 ** 0``) and 4 (``2 ** 2``)\n  with base 2:\n\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.logspace(0, 2, 7, base=2)\n  Array([1.   , 1.26 , 1.587, 2.   , 2.52 , 3.175, 4.   ], dtype=float32)\n\n  Multi-dimensional logspace:\n\n  >>> start = jnp.array([0, 5])\n  >>> stop = jnp.array([5, 0])\n  >>> base = jnp.array([2, 3])\n  >>> with jnp.printoptions(precision=3, suppress=True):\n  ...   jnp.logspace(start, stop, 5, base=base)\n  Array([[  1.   , 243.   ],\n         [  2.378,  61.547],\n         [  5.657,  15.588],\n         [ 13.454,   3.948],\n         [ 32.   ,   1.   ]], dtype=float32)",
    "std_args": [
      {"name": "start", "type": "Any"},
      {"name": "stop", "type": "Any"},
      {"name": "num", "type": "Any"},
      {"name": "endpoint", "type": "Any"},
      {"name": "base", "type": "Any"},
      {"name": "dtype", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "long": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "lower": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "lstsq": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "lu": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "lu_factor": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "map": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "f", "type": "Any"},
    ],
    "variants": {},
  },
  "map_params": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "margin": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "margin_ranking_loss": {
    "description": "Compute the margin ranking loss.",
    "std_args": [
      {"name": "input1", "type": "Any"},
      {"name": "input2", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "margin", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
    ],
    "variants": {},
  },
  "mask": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "mask_check": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "mask_indices": {
    "description": "Return indices of a mask of an (n, n) array.\n\nArgs:\n  n: static integer array dimension.\n  mask_func: a function that takes a shape ``(n, n)`` array and\n    an optional offset ``k``, and returns a shape ``(n, n)`` mask.\n    Examples of functions with this signature are\n    :func:`~jax.numpy.triu` and :func:`~jax.numpy.tril`.\n  k: a scalar value passed to ``mask_func``.\n  size: optional argument specifying the static size of the output arrays.\n    This is passed to :func:`~jax.numpy.nonzero` when generating the indices\n    from the mask.\n\nReturns:\n  a tuple of indices where ``mask_func`` is nonzero.\n\nSee also:\n  - :func:`jax.numpy.triu_indices`: compute ``mask_indices`` for :func:`~jax.numpy.triu`.\n  - :func:`jax.numpy.tril_indices`: compute ``mask_indices`` for :func:`~jax.numpy.tril`.\n\nExamples:\n  Calling ``mask_indices`` on built-in masking functions:\n\n  >>> jnp.mask_indices(3, jnp.triu)\n  (Array([0, 0, 0, 1, 1, 2], dtype=int32), Array([0, 1, 2, 1, 2, 2], dtype=int32))\n\n  >>> jnp.mask_indices(3, jnp.tril)\n  (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))\n\n  Calling ``mask_indices`` on a custom masking function:\n\n  >>> def mask_func(x, k=0):\n  ...   i = jnp.arange(x.shape[0])[:, None]\n  ...   j = jnp.arange(x.shape[1])\n  ...   return (i + 1) % (j + 1 + k) == 0\n  >>> mask_func(jnp.ones((3, 3)))\n  Array([[ True, False, False],\n         [ True,  True, False],\n         [ True, False,  True]], dtype=bool)\n  >>> jnp.mask_indices(3, mask_func)\n  (Array([0, 1, 1, 2, 2], dtype=int32), Array([0, 0, 1, 0, 2], dtype=int32))",
    "std_args": [
      {"name": "n", "type": "Any"},
      {"name": "mask_func", "type": "Any"},
      {"name": "k", "type": "Any"},
      {"name": "size", "type": "Any"},
    ],
    "variants": {},
  },
  "mask_mod": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "masked": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "matmul": {
    "description": "Computes the matrix product.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
    ],
    "variants": {},
  },
  "matrix_norm": {
    "description": "Computes the matrix norm of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "keepdims", "type": "bool"},
      {"name": "ord", "type": "Optional[Union[int, float, Literal[inf, Any, fro, nuc]]]"},
    ],
    "variants": {},
  },
  "matrix_power": {
    "description": "Raises a square matrix (or a stack of square matrices) ``x`` to an integer power ``n``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "int"},
    ],
    "variants": {},
  },
  "matrix_rank": {
    "description": "Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices).",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "rtol", "type": "Optional[Union[float, array]]"},
    ],
    "variants": {},
  },
  "matrix_transpose": {
    "description": "Transposes a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "max": {
    "description": "Element-wise maximum or reduction.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "max_norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "max_pool": {
    "description": 'Max pooling operation.\n\nArgs:\n    inputs: Tensor of rank N+2. `inputs` has shape\n        `(batch_size,) + inputs_spatial_shape + (num_channels,)` if\n        `data_format="channels_last"`, or\n        `(batch_size, num_channels) + inputs_spatial_shape` if\n        `data_format="channels_first"`. Pooling happens over the spatial\n        dimensions only.\n    pool_size: int or tuple/list of integers of size\n        `len(inputs_spatial_shape)`, specifying the size of the pooling\n        window for each spatial dimension of the input tensor. If\n        `pool_size` is int, then every spatial dimension shares the same\n        `pool_size`.\n    strides: int or tuple/list of integers of size\n        `len(inputs_spatial_shape)`. The stride of the sliding window for\n        each spatial dimension of the input tensor. If `strides` is int,\n        then every spatial dimension shares the same `strides`.\n    padding: string, either `"valid"` or `"same"`. `"valid"` means no\n        padding is applied, and `"same"` results in padding evenly to the\n        left/right or up/down of the input such that output has the\n        same height/width dimension as the input when `strides=1`.\n    data_format: A string, either `"channels_last"` or `"channels_first"`.\n        `data_format` determines the ordering of the dimensions in the\n        inputs. If `data_format="channels_last"`, `inputs` is of shape\n        `(batch_size, ..., channels)` while if\n        `data_format="channels_first"`, `inputs` is of shape\n        `(batch_size, channels, ...)`.\n\nReturns:\n    A tensor of rank N+2, the result of the max pooling operation.',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "pool_size", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
    ],
    "variants": {},
  },
  "max_pool1d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "max_pool2d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "max_pool3d": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "max_val": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "maximum": {
    "description": "Computes the maximum value for each element ``x1_i`` of the input array ``x1`` relative to the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "mean": {
    "description": "Calculates the arithmetic mean of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "measure_valued_estimation_mean": {
    "description": "Measure valued grads of a Gaussian expectation of `function` wrt the mean.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "dist", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
      {"name": "coupling", "type": "Any"},
    ],
    "variants": {},
  },
  "measure_valued_estimation_std": {
    "description": "Measure valued grads of a Gaussian expectation of `function` wrt the std.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "dist", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
      {"name": "coupling", "type": "Any"},
    ],
    "variants": {},
  },
  "measure_valued_jacobians": {
    "description": "Measure valued gradient estimation.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "params", "type": "Any"},
      {"name": "dist_builder", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
      {"name": "coupling", "type": "Any"},
    ],
    "variants": {},
  },
  "median": {
    "description": "Compute the median along the specified axis.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "overwrite_input", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
    ],
    "variants": {},
  },
  "merge_masks": {
    "description": "Determine mask type and combine masks if necessary.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "attn_mask", "type": "Any"},
      {"name": "key_padding_mask", "type": "Any"},
      {"name": "query", "type": "Any"},
    ],
    "variants": {},
  },
  "meshgrid": {
    "description": "Returns coordinate matrices from coordinate vectors.",
    "std_args": [
      {"name": "indexing", "type": "str"},
    ],
    "variants": {},
  },
  "metadata": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "min": {
    "description": "Element-wise minimum or reduction.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "min_val": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "minimum": {
    "description": "Computes the minimum value for each element ``x1_i`` of the input array ``x1`` relative to the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "mode": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "modules": {
    "description": "Return an iterator over all modules in the network.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "moments": {
    "description": 'Calculates the mean and variance of `x`.\n\nThe mean and variance are calculated by aggregating the contents of `x`\nacross `axes`. If `x` is 1-D and `axes = [0]` this is just the mean and\nvariance of a vector.\n\nArgs:\n    x: Input tensor.\n    axes: A list of axes which to compute mean and variance.\n    keepdims: If this is set to `True`, the axes which are reduced are left\n        in the result as dimensions with size one.\n    synchronized: Only applicable with the TensorFlow backend.\n        If `True`, synchronizes the global batch statistics (mean and\n        variance) across all devices at each training step in a\n        distributed training strategy. If `False`, each replica uses its own\n        local batch statistics.\n\nReturns:\n    A tuple containing two tensors - mean and variance.\n\nExample:\n\n>>> x = keras.ops.convert_to_tensor([0, 1, 2, 3, 100], dtype="float32")\n>>> keras.ops.moments(x, axes=[0])\n(array(21.2, dtype=float32), array(1553.3601, dtype=float32))',
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "axes", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
      {"name": "synchronized", "type": "Any"},
    ],
    "variants": {},
  },
  "momentum": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "moveaxis": {
    "description": "Moves array axes (dimensions) to new positions, while leaving other axes in their original positions.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "source", "type": "Union[int, Tuple[int, Ellipsis]]"},
      {"name": "destination", "type": "Union[int, Tuple[int, Ellipsis]]"},
    ],
    "variants": {},
  },
  "moving_avg_baseline": {
    "description": "A moving average baseline.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "decay", "type": "Any"},
      {"name": "zero_debias", "type": "Any"},
      {"name": "use_decay_early_training_heuristic", "type": "Any"},
    ],
    "variants": {},
  },
  "mse_loss": {
    "description": "Compute the element-wise mean squared error, with optional weighting.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "weight", "type": "Any"},
    ],
    "variants": {},
  },
  "multi_dot": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "multi_transform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "multiply": {
    "description": "Calculates the product for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex]"},
      {"name": "x2", "type": "Union[array, int, float, complex]"},
    ],
    "variants": {},
  },
  "multivariate_normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "n": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "name": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "named_chain": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "named_modules": {
    "description": "Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "memo", "type": "Any"},
      {"name": "prefix", "type": "Any"},
      {"name": "remove_duplicate", "type": "Any"},
    ],
    "variants": {},
  },
  "named_parameters": {
    "description": "Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "prefix", "type": "Any"},
      {"name": "recurse", "type": "Any"},
      {"name": "remove_duplicate", "type": "Any"},
    ],
    "variants": {},
  },
  "nan": {
    "description": "IEEE 754 floating-point representation of Not a Number (``NaN``).",
    "std_args": [],
    "variants": {},
  },
  "nan_to_num": {
    "description": "Replace NaN with zero and infinity with large finite numbers.\n\nArgs:\n    x: Input data.\n    nan: Optional float or int. Value to replace `NaN` entries with.\n    posinf: Optional float or int.\n        Value to replace positive infinity with.\n    neginf: Optional float or int.\n        Value to replace negative infinity with.\n\nReturns:\n    `x`, with non-finite values replaced.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "nan", "type": "Any"},
      {"name": "posinf", "type": "Any"},
      {"name": "neginf", "type": "Any"},
    ],
    "variants": {},
  },
  "nanargmax": {
    "description": "Return the index of the maximum value of an array, ignoring NaNs.\n\nJAX implementation of :func:`numpy.nanargmax`.\n\nArgs:\n  a: input array\n  axis: optional integer specifying the axis along which to find the maximum\n    value. If ``axis`` is not specified, ``a`` will be flattened.\n  out: unused by JAX\n  keepdims: if True, then return an array with the same number of dimensions\n    as ``a``.\n\nReturns:\n  an array containing the index of the maximum value along the specified axis.\n\nNote:\n  In the case of an axis with all-NaN values, the returned index will be -1.\n  This differs from the behavior of :func:`numpy.nanargmax`, which raises an error.\n\nSee also:\n  - :func:`jax.numpy.argmax`: return the index of the maximum value.\n  - :func:`jax.numpy.nanargmin`: compute ``argmin`` while ignoring NaN values.\n\nExamples:\n  >>> x = jnp.array([1, 3, 5, 4, jnp.nan])\n\n  Using a standard :func:`~jax.numpy.argmax` leads to potentially unexpected results:\n\n  >>> jnp.argmax(x)\n  Array(4, dtype=int32)\n\n  Using ``nanargmax`` returns the index of the maximum non-NaN value.\n\n  >>> jnp.nanargmax(x)\n  Array(2, dtype=int32)\n\n  >>> x = jnp.array([[1, 3, jnp.nan],\n  ...                [5, 4, jnp.nan]])\n  >>> jnp.nanargmax(x, axis=1)\n  Array([1, 0], dtype=int32)\n\n  >>> jnp.nanargmax(x, axis=1, keepdims=True)\n  Array([[1],\n         [0]], dtype=int32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
    ],
    "variants": {},
  },
  "nanargmin": {
    "description": "Return the index of the minimum value of an array, ignoring NaNs.\n\nJAX implementation of :func:`numpy.nanargmin`.\n\nArgs:\n  a: input array\n  axis: optional integer specifying the axis along which to find the maximum\n    value. If ``axis`` is not specified, ``a`` will be flattened.\n  out: unused by JAX\n  keepdims: if True, then return an array with the same number of dimensions\n    as ``a``.\n\nReturns:\n  an array containing the index of the minimum value along the specified axis.\n\nNote:\n  In the case of an axis with all-NaN values, the returned index will be -1.\n  This differs from the behavior of :func:`numpy.nanargmin`, which raises an error.\n\nSee also:\n  - :func:`jax.numpy.argmin`: return the index of the minimum value.\n  - :func:`jax.numpy.nanargmax`: compute ``argmax`` while ignoring NaN values.\n\nExamples:\n  >>> x = jnp.array([jnp.nan, 3, 5, 4, 2])\n  >>> jnp.nanargmin(x)\n  Array(4, dtype=int32)\n\n  >>> x = jnp.array([[1, 3, jnp.nan],\n  ...                [5, 4, jnp.nan]])\n  >>> jnp.nanargmin(x, axis=1)\n  Array([0, 1], dtype=int32)\n\n  >>> jnp.nanargmin(x, axis=1, keepdims=True)\n  Array([[0],\n         [1]], dtype=int32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
    ],
    "variants": {},
  },
  "nbytes": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ndarray": {
    "description": "Array base class for JAX\n\n``jax.Array`` is the public interface for instance checks and type annotation\nof JAX arrays and tracers. Its main applications are in instance checks and\ntype annotations; for example::\n\n  x = jnp.arange(5)\n  isinstance(x, jax.Array)  # returns True both inside and outside traced functions.\n\n  def f(x: Array) -> Array:  # type annotations are valid for traced and non-traced types.\n    return x\n\n``jax.Array`` should not be used directly for creation of arrays; instead you\nshould use array creation routines offered in :mod:`jax.numpy`, such as\n:func:`jax.numpy.array`, :func:`jax.numpy.zeros`, :func:`jax.numpy.ones`,\n:func:`jax.numpy.full`, :func:`jax.numpy.arange`, etc.",
    "std_args": [],
    "variants": {},
  },
  "ndim": {
    "description": "Return the number of dimensions of an array.\n\nJAX implementation of :func:`numpy.ndim`. Unlike ``np.ndim``, this function\nraises a :class:`TypeError` if the input is a collection such as a list or\ntuple.\n\nArgs:\n  a: array-like object, or any object with an ``ndim`` attribute.\n\nReturns:\n  An integer specifying the number of dimensions of ``a``.\n\nExamples:\n  Number of dimensions for arrays:\n\n  >>> x = jnp.arange(10)\n  >>> jnp.ndim(x)\n  1\n  >>> y = jnp.ones((2, 3))\n  >>> jnp.ndim(y)\n  2\n\n  This also works for scalars:\n\n  >>> jnp.ndim(3.14)\n  0\n\n  For arrays, this can also be accessed via the :attr:`jax.Array.ndim` property:\n\n  >>> x.ndim\n  1",
    "std_args": [
      {"name": "a", "type": "Any"},
    ],
    "variants": {},
  },
  "negative": {
    "description": "Computes the numerical negative of each element ``x_i`` (i.e., ``y_i = -x_i``) of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "negative_slope": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "nextafter": {
    "description": "Returns the next representable floating-point value for each element ``x1_i`` of the input array ``x1`` in the direction of the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "nll_loss": {
    "description": "Compute the negative log likelihood loss.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "ignore_index", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
    ],
    "variants": {},
  },
  "no_grad": {
    "description": "Context-manager that disabled gradient calculation.",
    "std_args": [],
    "variants": {},
  },
  "nonlinearity": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "nonzero": {
    "description": "Returns the indices of the array elements which are non-zero.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "norm1": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "norm2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "norm3": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "norm_first": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "norm_type": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "not_equal": {
    "description": "Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex, bool]"},
      {"name": "x2", "type": "Union[array, int, float, complex, bool]"},
    ],
    "variants": {},
  },
  "ntxent": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "num_embeddings": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "num_features": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "num_groups": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "num_heads": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "num_layers": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "number": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "object_": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "one_hot": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "ones": {
    "description": "Returns a new array having a specified ``shape`` and filled with ones.",
    "std_args": [
      {"name": "shape", "type": "Union[int, Tuple[int, Ellipsis]]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "ones_": {
    "description": "Fill the input Tensor with the scalar value `1`.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
    ],
    "variants": {},
  },
  "ones_like": {
    "description": "Returns a new array filled with ones and having the same ``shape`` as an input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "orthogonal": {
    "description": "Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices.",
    "std_args": [
      {"name": "module", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "orthogonal_map", "type": "Any"},
      {"name": "use_trivialization", "type": "Any"},
    ],
    "variants": {},
  },
  "orthogonal_": {
    "description": "Fill the input `Tensor` with a (semi) orthogonal matrix.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
      {"name": "gain", "type": "Any"},
      {"name": "generator", "type": "Any"},
    ],
    "variants": {},
  },
  "out_features": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "out_proj": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "outer": {
    "description": "Returns the outer product of two vectors ``x1`` and ``x2``.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
    ],
    "variants": {},
  },
  "output_device": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "output_ratio": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "output_size": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "p": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "padding": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "padding_idx": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "parameters": {
    "description": "Returns an iterator over module parameters.",
    "std_args": [
      {"name": "recurse", "type": "Any"},
    ],
    "variants": {},
  },
  "parameters_to_vector": {
    "description": "Flatten an iterable of parameters into a single vector.",
    "std_args": [
      {"name": "parameters", "type": "Any"},
    ],
    "variants": {},
  },
  "params": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "pareto": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "partition": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "pathwise_jacobians": {
    "description": "Pathwise gradient estimation.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "params", "type": "Any"},
      {"name": "dist_builder", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
    ],
    "variants": {},
  },
  "per_example_global_norm_clip": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "per_example_layer_norm_clip": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "permutation": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "permute_dims": {
    "description": "Permutes tensor dimensions.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "dims", "type": "int", "is_variadic": True},
    ],
    "variants": {},
  },
  "pi": {
    "description": "IEEE 754 floating-point representation of the mathematical constant ``\u03c0``.",
    "std_args": [],
    "variants": {},
  },
  "piecewise": {
    "description": "Evaluate a function defined piecewise across the domain.\n\nJAX implementation of :func:`numpy.piecewise`, in terms of :func:`jax.lax.switch`.\n\nNote:\n  Unlike :func:`numpy.piecewise`, :func:`jax.numpy.piecewise` requires functions\n  in ``funclist`` to be traceable by JAX, as it is implemented via\n  :func:`jax.lax.switch`.\n\nArgs:\n  x: array of input values.\n  condlist: boolean array or sequence of boolean arrays corresponding to the\n    functions in ``funclist``. If a sequence of arrays, the length of each\n    array must match the length of ``x``\n  funclist: list of arrays or functions; must either be the same length as\n    ``condlist``, or have length ``len(condlist) + 1``, in which case the\n    last entry is the default applied when none of the conditions are True.\n    Alternatively, entries of ``funclist`` may be numerical values, in which\n    case they indicate a constant function.\n  args, kwargs: additional arguments are passed to each function in\n    ``funclist``.\n\nReturns:\n  An array which is the result of evaluating the functions on ``x`` at\n  the specified conditions.\n\nSee also:\n  - :func:`jax.lax.switch`: choose between *N* functions based on an index.\n  - :func:`jax.lax.cond`: choose between two functions based on a boolean condition.\n  - :func:`jax.numpy.where`: choose between two results based on a boolean mask.\n  - :func:`jax.lax.select`: choose between two results based on a boolean mask.\n  - :func:`jax.lax.select_n`: choose between *N* results based on a boolean mask.\n\nExamples:\n  Here's an example of a function which is zero for negative values, and linear\n  for positive values:\n\n  >>> x = jnp.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])\n\n  >>> condlist = [x < 0, x >= 0]\n  >>> funclist = [lambda x: 0 * x, lambda x: x]\n  >>> jnp.piecewise(x, condlist, funclist)\n  Array([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=int32)\n\n  ``funclist`` can also contain a simple scalar value for constant functions:\n\n  >>> condlist = [x < 0, x >= 0]\n  >>> funclist = [0, lambda x: x]\n  >>> jnp.piecewise(x, condlist, funclist)\n  Array([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=int32)\n\n  You can specify a default value by appending an extra condition to ``funclist``:\n\n  >>> condlist = [x < -1, x > 1]\n  >>> funclist = [lambda x: 1 + x, lambda x: x - 1, 0]\n  >>> jnp.piecewise(x, condlist, funclist)\n  Array([-3, -2,  -1,  0,  0,  0,  1,  2, 3], dtype=int32)\n\n  ``condlist`` may also be a simple array of scalar conditions, in which case\n  the associated function applies to the whole range\n\n  >>> condlist = jnp.array([False, True, False])\n  >>> funclist = [lambda x: x * 0, lambda x: x * 10, lambda x: x * 100]\n  >>> jnp.piecewise(x, condlist, funclist)\n  Array([-40, -30, -20, -10,   0,  10,  20,  30,  40], dtype=int32)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "condlist", "type": "Any"},
      {"name": "funclist", "type": "Any"},
      {"name": "args", "type": "Any"},
      {"name": "kw", "type": "Any"},
    ],
    "variants": {},
  },
  "piecewise_constant_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "piecewise_interpolate_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "pinv": {
    "description": "Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "rtol", "type": "Optional[Union[float, array]]"},
    ],
    "variants": {},
  },
  "place": {
    "description": "Update array elements based on a mask.\n\nJAX implementation of :func:`numpy.place`.\n\nThe semantics of :func:`numpy.place` are to modify arrays in-place, which\nis not possible for JAX's immutable arrays. The JAX version returns a modified\ncopy of the input, and adds the ``inplace`` parameter which must be set to\n`False`` by the user as a reminder of this API difference.\n\nArgs:\n  arr: array into which values will be placed.\n  mask: boolean mask with the same size as ``arr``.\n  vals: values to be inserted into ``arr`` at the locations indicated\n    by mask. If too many values are supplied, they will be truncated.\n    If not enough values are supplied, they will be repeated.\n  inplace: must be set to False to indicate that the input is not modified\n    in-place, but rather a modified copy is returned.\n\nReturns:\n  A copy of ``arr`` with masked values set to entries from `vals`.\n\nSee Also:\n  - :func:`jax.numpy.put`: put elements into an array at numerical indices.\n  - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing\n\nExamples:\n  >>> x = jnp.zeros((3, 5), dtype=int)\n  >>> mask = (jnp.arange(x.size) % 3 == 0).reshape(x.shape)\n  >>> mask\n  Array([[ True, False, False,  True, False],\n         [False,  True, False, False,  True],\n         [False, False,  True, False, False]], dtype=bool)\n\n  Placing a scalar value:\n\n  >>> jnp.place(x, mask, 1, inplace=False)\n  Array([[1, 0, 0, 1, 0],\n         [0, 1, 0, 0, 1],\n         [0, 0, 1, 0, 0]], dtype=int32)\n\n  In this case, ``jnp.place`` is similar to the masked array update syntax:\n\n  >>> x.at[mask].set(1)\n  Array([[1, 0, 0, 1, 0],\n         [0, 1, 0, 0, 1],\n         [0, 0, 1, 0, 0]], dtype=int32)\n\n  ``place`` differs when placing values from an array. The array is repeated\n  to fill the masked entries:\n\n  >>> vals = jnp.array([1, 3, 5])\n  >>> jnp.place(x, mask, vals, inplace=False)\n  Array([[1, 0, 0, 3, 0],\n         [0, 5, 0, 0, 1],\n         [0, 0, 3, 0, 0]], dtype=int32)",
    "std_args": [
      {"name": "arr", "type": "Any"},
      {"name": "mask", "type": "Any"},
      {"name": "vals", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "poisson": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "polar": {
    "description": "Constructs a complex tensor whose elements are Cartesian\ncoordinates corresponding to the polar coordinates\nwith absolute value `abs` and angle `angle`.\n\nThe operation is numerically equivalent to `torch.polar()`.\nIt is not equivalent to `scipy.lingalg.polar()` which performs\nSingular Value Decomposition.\n\nGiven the magnitude (`abs_`) and angle (`angle`), this function computes the\ncorresponding complex number in the form of `real + imaginary * 1j`, where:\n- `real = abs_ * cos(angle)`\n- `imaginary = abs_ * sin(angle)`\n\nArgs:\n    abs_: The magnitude (absolute value) of the complex number.\n    angle: The angle (in radians) of the complex number.\n\nReturns:\n    A complex number (or array of complex numbers) with the same shape as\n    `abs_` and `angle`.\n\nExample:\n\n>>> abs_ = keras.random.normal((1, 2))\n>>> angle = keras.random.normal((1, 2))\n>>> keras.ops.nn.polar(abs_, angle).shape\n(1, 2)\n>>> keras.ops.nn.polar(abs_, angle)\nArray([[0.63185346-0.59370506j, 0.48960376-0.31677645j]], dtype=complex64)",
    "std_args": [
      {"name": "abs_", "type": "Any"},
      {"name": "angle", "type": "Any"},
    ],
    "variants": {},
  },
  "polydiv": {
    "description": "Returns the quotient and remainder of polynomial division.\n\nJAX implementation of :func:`numpy.polydiv`.\n\nArgs:\n  u: Array of dividend polynomial coefficients.\n  v: Array of divisor polynomial coefficients.\n  trim_leading_zeros: Default is ``False``. If ``True`` removes the leading\n    zeros in the return value to match the result of numpy. But prevents the\n    function from being able to be used in compiled code. Due to differences\n    in accumulation of floating point arithmetic errors, the cutoff for values\n    to be considered zero may lead to inconsistent results between NumPy and\n    JAX, and even between different JAX backends. The result may lead to\n    inconsistent output shapes when ``trim_leading_zeros=True``.\n\nReturns:\n  A tuple of quotient and remainder arrays. The dtype of the output is always\n  promoted to inexact.\n\nNote:\n  :func:`jax.numpy.polydiv` only accepts arrays as input unlike\n  :func:`numpy.polydiv` which accepts scalar inputs as well.\n\nSee also:\n  - :func:`jax.numpy.polyadd`: Computes the sum of two polynomials.\n  - :func:`jax.numpy.polysub`: Computes the difference of two polynomials.\n  - :func:`jax.numpy.polymul`: Computes the product of two polynomials.\n\nExamples:\n  >>> x1 = jnp.array([5, 7, 9])\n  >>> x2 = jnp.array([4, 1])\n  >>> np.polydiv(x1, x2)\n  (array([1.25  , 1.4375]), array([7.5625]))\n  >>> jnp.polydiv(x1, x2)\n  (Array([1.25  , 1.4375], dtype=float32), Array([0.    , 0.    , 7.5625], dtype=float32))\n\n  If ``trim_leading_zeros=True``, the result matches with ``np.polydiv``'s.\n\n  >>> jnp.polydiv(x1, x2, trim_leading_zeros=True)\n  (Array([1.25  , 1.4375], dtype=float32), Array([7.5625], dtype=float32))",
    "std_args": [
      {"name": "u", "type": "Any"},
      {"name": "v", "type": "Any"},
      {"name": "trim_leading_zeros", "type": "Any"},
    ],
    "variants": {},
  },
  "polymul": {
    "description": "Returns the product of two polynomials.\n\nJAX implementation of :func:`numpy.polymul`.\n\nArgs:\n  a1: 1D array of polynomial coefficients.\n  a2: 1D array of polynomial coefficients.\n  trim_leading_zeros: Default is ``False``. If ``True`` removes the leading\n    zeros in the return value to match the result of numpy. But prevents the\n    function from being able to be used in compiled code. Due to differences\n    in accumulation of floating point arithmetic errors, the cutoff for values\n    to be considered zero may lead to inconsistent results between NumPy and\n    JAX, and even between different JAX backends. The result may lead to\n    inconsistent output shapes when ``trim_leading_zeros=True``.\n\nReturns:\n  An array of the coefficients of the product of the two polynomials. The dtype\n  of the output is always promoted to inexact.\n\nNote:\n  :func:`jax.numpy.polymul` only accepts arrays as input unlike\n  :func:`numpy.polymul` which accepts scalar inputs as well.\n\nSee also:\n  - :func:`jax.numpy.polyadd`: Computes the sum of two polynomials.\n  - :func:`jax.numpy.polysub`: Computes the difference of two polynomials.\n  - :func:`jax.numpy.polydiv`: Computes the quotient and remainder of polynomial\n    division.\n\nExamples:\n  >>> x1 = np.array([2, 1, 0])\n  >>> x2 = np.array([0, 5, 0, 3])\n  >>> np.polymul(x1, x2)\n  array([10,  5,  6,  3,  0])\n  >>> jnp.polymul(x1, x2)\n  Array([ 0., 10.,  5.,  6.,  3.,  0.], dtype=float32)\n\n  If ``trim_leading_zeros=True``, the result matches with ``np.polymul``'s.\n\n  >>> jnp.polymul(x1, x2, trim_leading_zeros=True)\n  Array([10.,  5.,  6.,  3.,  0.], dtype=float32)\n\n  For input arrays of dtype ``complex``:\n\n  >>> x3 = np.array([2., 1+2j, 1-2j])\n  >>> x4 = np.array([0, 5, 0, 3])\n  >>> np.polymul(x3, x4)\n  array([10. +0.j,  5.+10.j, 11.-10.j,  3. +6.j,  3. -6.j])\n  >>> jnp.polymul(x3, x4)\n  Array([ 0. +0.j, 10. +0.j,  5.+10.j, 11.-10.j,  3. +6.j,  3. -6.j],      dtype=complex64)\n  >>> jnp.polymul(x3, x4, trim_leading_zeros=True)\n  Array([10. +0.j,  5.+10.j, 11.-10.j,  3. +6.j,  3. -6.j], dtype=complex64)",
    "std_args": [
      {"name": "a1", "type": "Any"},
      {"name": "a2", "type": "Any"},
      {"name": "trim_leading_zeros", "type": "Any"},
    ],
    "variants": {},
  },
  "polynomial_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "pop": {
    "description": "Remove key from the ParameterDict and return its parameter.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "key", "type": "Any"},
    ],
    "variants": {},
  },
  "popitem": {
    "description": "Remove and return the last inserted `(key, parameter)` pair from the ParameterDict.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "positive": {
    "description": "Computes the numerical positive of each element ``x_i`` (i.e., ``y_i = +x_i``) of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "pow": {
    "description": "Calculates an implementation-dependent approximation of exponentiation by raising each element ``x1_i`` (the base) of the input array ``x1`` to the power of ``x2_i`` (the exponent), where ``x2_i`` is the corresponding element of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex]"},
      {"name": "x2", "type": "Union[array, int, float, complex]"},
    ],
    "variants": {},
  },
  "power": {
    "description": "Calculate element-wise base ``x1`` exponential of ``x2``.\n\nJAX implementation of :obj:`numpy.power`.\n\nArgs:\n  x1: scalar or array. Specifies the bases.\n  x2: scalar or array. Specifies the exponent. ``x1`` and ``x2`` should either\n    have same shape or be broadcast compatible.\n\nReturns:\n  An array containing the base ``x1`` exponentials of ``x2`` with same dtype\n  as input.\n\nNote:\n  - When ``x2`` is a concrete integer scalar, ``jnp.power`` lowers to\n    :func:`jax.lax.integer_pow`.\n  - When ``x2`` is a traced scalar or an array, ``jnp.power`` lowers to\n    :func:`jax.lax.pow`.\n  - ``jnp.power`` raises a ``TypeError`` for integer type raised to a concrete\n    negative integer power. For a non-concrete power, the operation is invalid\n    and the returned value is implementation-defined.\n  - ``jnp.power`` returns ``nan`` for negative value raised to the power of\n    non-integer values.\n\nSee also:\n  - :func:`jax.lax.pow`: Computes element-wise power, :math:`x^y`.\n  - :func:`jax.lax.integer_pow`: Computes element-wise power :math:`x^y`, where\n    :math:`y` is a fixed integer.\n  - :func:`jax.numpy.float_power`: Computes the first array raised to the power\n    of second array, element-wise, by promoting to the inexact dtype.\n  - :func:`jax.numpy.pow`: Computes the first array raised to the power of second\n    array, element-wise.\n\nExamples:\n  Inputs with scalar integers:\n\n  >>> jnp.power(4, 3)\n  Array(64, dtype=int32, weak_type=True)\n\n  Inputs with same shape:\n\n  >>> x1 = jnp.array([2, 4, 5])\n  >>> x2 = jnp.array([3, 0.5, 2])\n  >>> jnp.power(x1, x2)\n  Array([ 8.,  2., 25.], dtype=float32)\n\n  Inputs with broadcast compatibility:\n\n  >>> x3 = jnp.array([-2, 3, 1])\n  >>> x4 = jnp.array([[4, 1, 6],\n  ...                 [1.3, 3, 5]])\n  >>> jnp.power(x3, x4)\n  Array([[16.,  3.,  1.],\n         [nan, 27.,  1.]], dtype=float32)",
    "std_args": [
      {"name": "x1", "type": "Any"},
      {"name": "x2", "type": "Any"},
    ],
    "variants": {},
  },
  "predict": {
    "description": "Return the class with the highest probability for each example in the input minibatch.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "input", "type": "Any"},
    ],
    "variants": {},
  },
  "printoptions": {
    "description": "Alias of :func:`numpy.printoptions`.\n\nJAX arrays are printed via NumPy, so NumPy's `printoptions`\nconfigurations will apply to printed JAX arrays.\n\nSee the :func:`numpy.set_printoptions` documentation for details\non the available options and their meanings.",
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "prod": {
    "description": "Calculates the product of input array ``x`` elements.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "promote_types": {
    "description": "Returns the type to which a binary operation should cast its arguments.\n\nJAX implementation of :func:`numpy.promote_types`. For details of JAX's\ntype promotion semantics, see :ref:`type-promotion`.\n\nArgs:\n  a: a :class:`numpy.dtype` or a dtype specifier.\n  b: a :class:`numpy.dtype` or a dtype specifier.\n\nReturns:\n  A :class:`numpy.dtype` object.\n\nExamples:\n  Type specifiers may be strings, dtypes, or scalar types, and the return\n  value is always a dtype:\n\n  >>> jnp.promote_types('int32', 'float32')  # strings\n  dtype('float32')\n  >>> jnp.promote_types(jnp.dtype('int32'), jnp.dtype('float32'))  # dtypes\n  dtype('float32')\n  >>> jnp.promote_types(jnp.int32, jnp.float32)  # scalar types\n  dtype('float32')\n\n  Built-in scalar types (:type:`int`, :type:`float`, or :type:`complex`) are\n  treated as weakly-typed and will not change the bit width of a strongly-typed\n  counterpart (see discussion in :ref:`type-promotion`):\n\n  >>> jnp.promote_types('uint8', int)\n  dtype('uint8')\n  >>> jnp.promote_types('float16', float)\n  dtype('float16')\n\n  This differs from the NumPy version of this function, which treats built-in scalar\n  types as equivalent to 64-bit types:\n\n  >>> import numpy\n  >>> numpy.promote_types('uint8', int)\n  dtype('int64')\n  >>> numpy.promote_types('float16', float)\n  dtype('float64')",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "b", "type": "Any"},
    ],
    "variants": {},
  },
  "psi": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "psnr": {
    "description": "Peak Signal-to-Noise Ratio (PSNR) function.\n\nThis function computes the Peak Signal-to-Noise Ratio between two signals,\n`x1` and `x2`. PSNR is a measure of the quality of a reconstructed signal.\nThe higher the PSNR, the closer the reconstructed signal is to the original\nsignal. Note that it can become negative when the signal power is\nsmaller that the noise power.\n\nArgs:\n    x1: The first input signal.\n    x2: The second input signal. Must have the same shape as `x1`.\n    max_val: The maximum possible value in the signals.\n\nReturns:\n    float: The PSNR value between `x1` and `x2`.\n\nExamples:\n\n>>> x1 = keras.random.normal((2, 4, 4, 3))\n>>> x2 = keras.random.normal((2, 4, 4, 3))\n>>> max_val = 1.0\n>>> keras.ops.nn.psnr(x1, x2, max_val)\n-3.1697404",
    "std_args": [
      {"name": "x1", "type": "Any"},
      {"name": "x2", "type": "Any"},
      {"name": "max_val", "type": "Any"},
    ],
    "variants": {},
  },
  "ptp": {
    "description": "Return the peak-to-peak range along a given axis.\n\nJAX implementation of :func:`numpy.ptp`.\n\nArgs:\n  a: input array.\n  axis: optional, int or sequence of ints, default=None. Axis along which the\n    range is computed. If None, the range is computed on the flattened array.\n  keepdims: bool, default=False. If true, reduced axes are left in the result\n    with size 1.\n  out: Unused by JAX.\n\nReturns:\n  An array with the range of elements along specified axis of input.\n\nExamples:\n  By default, ``jnp.ptp`` computes the range along all axes.\n\n  >>> x = jnp.array([[1, 3, 5, 2],\n  ...                [4, 6, 8, 1],\n  ...                [7, 9, 3, 4]])\n  >>> jnp.ptp(x)\n  Array(8, dtype=int32)\n\n  If ``axis=1``, computes the range along axis 1.\n\n  >>> jnp.ptp(x, axis=1)\n  Array([4, 7, 6], dtype=int32)\n\n  To preserve the dimensions of input, you can set ``keepdims=True``.\n\n  >>> jnp.ptp(x, axis=1, keepdims=True)\n  Array([[4],\n         [7],\n         [6]], dtype=int32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "out", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
    ],
    "variants": {},
  },
  "put": {
    "description": "Put elements into an array at given indices.\n\nJAX implementation of :func:`numpy.put`.\n\nThe semantics of :func:`numpy.put` are to modify arrays in-place, which\nis not possible for JAX's immutable arrays. The JAX version returns a modified\ncopy of the input, and adds the ``inplace`` parameter which must be set to\n`False`` by the user as a reminder of this API difference.\n\nArgs:\n  a: array into which values will be placed.\n  ind: array of indices over the flattened array at which to put values.\n  v: array of values to put into the array.\n  mode: string specifying how to handle out-of-bound indices. Supported values:\n\n    - ``\"clip\"`` (default): clip out-of-bound indices to the final index.\n    - ``\"wrap\"``: wrap out-of-bound indices to the beginning of the array.\n\n  inplace: must be set to False to indicate that the input is not modified\n    in-place, but rather a modified copy is returned.\n\nReturns:\n  A copy of ``a`` with specified entries updated.\n\nSee Also:\n  - :func:`jax.numpy.place`: place elements into an array via boolean mask.\n  - :func:`jax.numpy.ndarray.at`: array updates using NumPy-style indexing.\n  - :func:`jax.numpy.take`: extract values from an array at given indices.\n\nExamples:\n  >>> x = jnp.zeros(5, dtype=int)\n  >>> indices = jnp.array([0, 2, 4])\n  >>> values = jnp.array([10, 20, 30])\n  >>> jnp.put(x, indices, values, inplace=False)\n  Array([10,  0, 20,  0, 30], dtype=int32)\n\n  This is equivalent to the following :attr:`jax.numpy.ndarray.at` indexing syntax:\n\n  >>> x.at[indices].set(values)\n  Array([10,  0, 20,  0, 30], dtype=int32)\n\n  There are two modes for handling out-of-bound indices. By default they are\n  clipped:\n\n  >>> indices = jnp.array([0, 2, 6])\n  >>> jnp.put(x, indices, values, inplace=False, mode='clip')\n  Array([10,  0, 20,  0, 30], dtype=int32)\n\n  Alternatively, they can be wrapped to the beginning of the array:\n\n  >>> jnp.put(x, indices, values, inplace=False, mode='wrap')\n  Array([10,  30, 20,  0, 0], dtype=int32)\n\n  For N-dimensional inputs, the indices refer to the flattened array:\n\n  >>> x = jnp.zeros((3, 5), dtype=int)\n  >>> indices = jnp.array([0, 7, 14])\n  >>> jnp.put(x, indices, values, inplace=False)\n  Array([[10,  0,  0,  0,  0],\n         [ 0,  0, 20,  0,  0],\n         [ 0,  0,  0,  0, 30]], dtype=int32)",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "ind", "type": "Any"},
      {"name": "v", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "inplace", "type": "Any"},
    ],
    "variants": {},
  },
  "qr": {
    "description": "Returns the QR decomposition of a full column rank matrix (or a stack of matrices).",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "mode", "type": "Literal[reduced, complete]"},
    ],
    "variants": {},
  },
  "quantile": {
    "description": 'Compute the q-th quantile(s) of the data along the specified axis.\n\nArgs:\n    x: Input tensor.\n    q: Probability or sequence of probabilities for the quantiles to\n        compute. Values must be between 0 and 1 inclusive.\n    axis: Axis or axes along which the quantiles are computed. Defaults to\n        `axis=None` which is to compute the quantile(s) along a flattened\n        version of the array.\n    method: A string specifies the method to use for estimating the\n        quantile. Available methods are `"linear"`, `"lower"`, `"higher"`,\n        `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.\n        If the desired quantile lies between two data points `i < j`:\n        - `"linear"`: `i + (j - i) * fraction`, where fraction is the\n            fractional part of the index surrounded by `i` and `j`.\n        - `"lower"`: `i`.\n        - `"higher"`: `j`.\n        - `"midpoint"`: `(i + j) / 2`\n        - `"nearest"`: `i` or `j`, whichever is nearest.\n    keepdims: If this is set to `True`, the axes which are reduce\n        are left in the result as dimensions with size one.\n\nReturns:\n    The quantile(s). If `q` is a single probability and `axis=None`, then\n    the result is a scalar. If multiple probabilities levels are given,\n    first axis of the result corresponds to the quantiles. The other axes\n    are the axes that remain after the reduction of `x`.',
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "q", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "method", "type": "Any"},
      {"name": "keepdims", "type": "Any"},
    ],
    "variants": {},
  },
  "quantize": {
    "description": "",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "type_check", "type": "Any"},
      {"name": "config", "type": "Any"},
    ],
    "variants": {},
  },
  "radians": {
    "description": "Alias of :func:`jax.numpy.deg2rad`",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "randn": {
    "description": "Returns a tensor filled with random numbers from a normal distribution.",
    "std_args": [
      {"name": "shape", "type": "Any"},
    ],
    "variants": {},
  },
  "random_like": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "rank": {
    "description": "Get the rank of this process",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "ravel": {
    "description": "Flatten array into a 1-dimensional shape.\n\nRefer to :func:`jax.numpy.ravel` for the full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "order", "type": "Any"},
      {"name": "out_sharding", "type": "Any"},
    ],
    "variants": {},
  },
  "ravel_multi_index": {
    "description": 'Convert multi-dimensional indices into flat indices.\n\nJAX implementation of :func:`numpy.ravel_multi_index`\n\nArgs:\n  multi_index: sequence of integer arrays containing indices in each dimension.\n  dims: sequence of integer sizes; must have ``len(dims) == len(multi_index)``\n  mode: how to handle out-of bound indices. Options are\n\n    - ``"raise"`` (default): raise a ValueError. This mode is incompatible\n      with :func:`~jax.jit` or other JAX transformations.\n    - ``"clip"``: clip out-of-bound indices to valid range.\n    - ``"wrap"``: wrap out-of-bound indices to valid range.\n\n  order: ``"C"`` (default) or ``"F"``, specify whether to assume C-style\n    row-major order or Fortran-style column-major order.\n\nReturns:\n  array of flattened indices\n\nSee also:\n  :func:`jax.numpy.unravel_index`: inverse of this function.\n\nExamples:\n  Define a 2-dimensional array and a sequence of indices of even values:\n\n  >>> x = jnp.array([[2., 3., 4.],\n  ...                [5., 6., 7.]])\n  >>> indices = jnp.where(x % 2 == 0)\n  >>> indices\n  (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))\n  >>> x[indices]\n  Array([2., 4., 6.], dtype=float32)\n\n  Compute the flattened indices:\n\n  >>> indices_flat = jnp.ravel_multi_index(indices, x.shape)\n  >>> indices_flat\n  Array([0, 2, 4], dtype=int32)\n\n  These flattened indices can be used to extract the same values from the\n  flattened ``x`` array:\n\n  >>> x_flat = x.ravel()\n  >>> x_flat\n  Array([2., 3., 4., 5., 6., 7.], dtype=float32)\n  >>> x_flat[indices_flat]\n  Array([2., 4., 6.], dtype=float32)\n\n  The original indices can be recovered with :func:`~jax.numpy.unravel_index`:\n\n  >>> jnp.unravel_index(indices_flat, x.shape)\n  (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))',
    "std_args": [
      {"name": "multi_index", "type": "Any"},
      {"name": "dims", "type": "Any"},
      {"name": "mode", "type": "Any"},
      {"name": "order", "type": "Any"},
    ],
    "variants": {},
  },
  "rayleigh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "real": {
    "description": "Returns the real component of a complex number for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "reciprocal": {
    "description": "Returns the reciprocal for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "register_buffer": {
    "description": "Registers a persistent buffer.",
    "std_args": [
      {"name": "name", "type": "Any"},
      {"name": "tensor", "type": "Any"},
      {"name": "persistent", "type": "Any"},
    ],
    "variants": {},
  },
  "register_parameter": {
    "description": "Registers a learnable parameter.",
    "std_args": [
      {"name": "name", "type": "Any"},
      {"name": "param", "type": "Any"},
    ],
    "variants": {},
  },
  "relu": {
    "description": "Rectified Linear Unit.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "remainder": {
    "description": "Returns the remainder of division for each element ``x1_i`` of the input array ``x1`` and the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float]"},
      {"name": "x2", "type": "Union[array, int, float]"},
    ],
    "variants": {},
  },
  "remove": {
    "description": "Remove the pruning reparameterization from a module and the pruning method from the forward hook.",
    "std_args": [
      {"name": "module", "type": "Any"},
      {"name": "name", "type": "Any"},
    ],
    "variants": {},
  },
  "remove_axis": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "index", "type": "Any"},
      {"name": "params", "type": "Any"},
    ],
    "variants": {},
  },
  "repeat": {
    "description": "Repeats each element of an array a specified number of times on a per-element basis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "repeats", "type": "Union[int, array]"},
      {"name": "axis", "type": "Optional[int]"},
    ],
    "variants": {},
  },
  "replace": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "reset": {
    "description": "Reset all underlying ``Metric``'s.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "reshape": {
    "description": "Reshapes an array without changing its data.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "shape", "type": "Tuple[int, Ellipsis]"},
      {"name": "copy", "type": "Optional[bool]"},
    ],
    "variants": {},
  },
  "reshape_weight_to_matrix": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "weight", "type": "Any"},
    ],
    "variants": {},
  },
  "result_type": {
    "description": "Returns the dtype that results from applying type promotion rules (see :ref:`type-promotion`) to the arguments.",
    "std_args": [],
    "variants": {},
  },
  "rfft": {
    "description": "Computes the one-dimensional discrete Fourier transform for real-valued input.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "n", "type": "Optional[int]"},
      {"name": "axis", "type": "int"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "rfft2": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "rfftfreq": {
    "description": "Computes the discrete Fourier transform sample frequencies (for ``rfft`` and ``irfft``).",
    "std_args": [
      {"name": "n", "type": "int"},
      {"name": "d", "type": "float"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "rfftn": {
    "description": "Computes the n-dimensional discrete Fourier transform for real-valued input.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "s", "type": "Optional[Sequence[int]]"},
      {"name": "axes", "type": "Optional[Sequence[int]]"},
      {"name": "norm", "type": "Literal[backward, ortho, forward]"},
    ],
    "variants": {},
  },
  "right_shift": {
    "description": "Shift the bits of an integer to the right.",
    "std_args": [
      {"name": "a", "type": "Any"},
      {"name": "n", "type": "Any"},
    ],
    "variants": {},
  },
  "rms_norm": {
    "description": "Apply Root Mean Square Layer Normalization.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "normalized_shape", "type": "Any"},
      {"name": "weight", "type": "Any"},
      {"name": "eps", "type": "Any"},
    ],
    "variants": {},
  },
  "rms_normalization": {
    "description": "Performs Root Mean Square (RMS) normalization on `x`.\n\nThe Keras operation implements the operation as described in\n[Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)\nby Biao Zhang et al.\n\nThe operation is different from LayerNormalization with RMS scaling.\n\nIt is defined as `rms_normalization(x) = x * rsqrt(mean(square(x))) * scale`\n\nArgs:\n    x: Input tensor.\n    scale: Optional scaling factor for the normalization.\n    axis: The axis or axes along which to perform normalization. Defaults\n        to `-1`.\n    epsilon: A lower bound value for the norm. Defaults to\n        `backend.epsilon()`.\n\nReturns:\n    The normalized array.\n\nExample:\n\n>>> x = keras.random.normal((1, 10))\n>>> keras.ops.rms_normalization(x)\narray([[0.69384296, 0.94444374, 0.16551171, 0.05749961, 1.11008865,\n        0.52475186, 1.57686807, 1.69893307, 1.27292764, 0.30819128]])",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "scale", "type": "Any"},
      {"name": "axis", "type": "Any"},
      {"name": "epsilon", "type": "Any"},
    ],
    "variants": {},
  },
  "roll": {
    "description": "Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position. Array elements that roll beyond the first position are re-introduced at the last position.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "shift", "type": "Union[int, Tuple[int, Ellipsis]]"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
    ],
    "variants": {},
  },
  "roots": {
    "description": "Returns the roots of a polynomial given the coefficients ``p``.\n\nJAX implementations of :func:`numpy.roots`.\n\nArgs:\n  p: Array of polynomial coefficients having rank-1.\n  strip_zeros : bool, default=True. If True, then leading zeros in the\n    coefficients will be stripped, similar to :func:`numpy.roots`. If set to\n    False, leading zeros will not be stripped, and undefined roots will be\n    represented by NaN values in the function output. ``strip_zeros`` must be\n    set to ``False`` for the function to be compatible with :func:`jax.jit` and\n    other JAX transformations.\n\nReturns:\n  An array containing the roots of the polynomial.\n\nNote:\n  Unlike ``np.roots`` of this function, the ``jnp.roots`` returns the roots\n  in a complex array regardless of the values of the roots.\n\nSee Also:\n  - :func:`jax.numpy.poly`: Finds the polynomial coefficients of the given\n    sequence of roots.\n  - :func:`jax.numpy.polyfit`: Least squares polynomial fit to data.\n  - :func:`jax.numpy.polyval`: Evaluate a polynomial at specific values.\n\nExamples:\n  >>> coeffs = jnp.array([0, 1, 2])\n\n  The default behavior matches numpy and strips leading zeros:\n\n  >>> jnp.roots(coeffs)\n  Array([-2.+0.j], dtype=complex64)\n\n  With ``strip_zeros=False``, extra roots are set to NaN:\n\n  >>> jnp.roots(coeffs, strip_zeros=False)\n  Array([-2. +0.j, nan+nanj], dtype=complex64)",
    "std_args": [
      {"name": "p", "type": "Any"},
      {"name": "strip_zeros", "type": "Any"},
    ],
    "variants": {},
  },
  "rot90": {
    "description": "Rotate an array by 90 degrees in the plane specified by axes.\n\nThis function rotates an array counterclockwise\nby 90 degrees `k` times in the plane specified by `axes`.\nSupports arrays of two or more dimensions.\n\nArgs:\n    array: Input array to rotate.\n    k: Number of times the array is rotated by 90 degrees.\n    axes: A tuple of two integers specifying the\n        plane of rotation (defaults to `(0, 1)`).\n\nReturns:\n    Rotated array.\n\nExamples:\n\n>>> import numpy as np\n>>> from keras import ops\n>>> m = np.array([[1, 2], [3, 4]])\n>>> rotated = ops.rot90(m)\n>>> rotated\narray([[2, 4],\n       [1, 3]])\n\n>>> m = np.arange(8).reshape((2, 2, 2))\n>>> rotated = ops.rot90(m, k=1, axes=(1, 2))\n>>> rotated\narray([[[1, 3],\n        [0, 2]],\n       [[5, 7],\n        [4, 6]]])",
    "std_args": [
      {"name": "array", "type": "Any"},
      {"name": "k", "type": "Any"},
      {"name": "axes", "type": "Any"},
    ],
    "variants": {},
  },
  "round": {
    "description": "Rounds each element ``x_i`` of the input array ``x`` to the nearest integer-valued number.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "rsqrt": {
    "description": "Computes reciprocal of square root of x element-wise.\n\nArgs:\n    x: input tensor\n\nReturns:\n    A tensor with the same dtype as `x`.\n\nExample:\n\n>>> x = keras.ops.convert_to_tensor([1.0, 10.0, 100.0])\n>>> keras.ops.rsqrt(x)\narray([1.0, 0.31622776, 0.1], dtype=float32)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "safe_softmax_cross_entropy": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "saturate_cast": {
    "description": 'Performs a safe saturating cast to the desired dtype.\n\nSaturating cast prevents data type overflow when casting to `dtype` with\nsmaller values range. E.g.\n`ops.cast(ops.cast([-1, 256], "float32"), "uint8")` returns `[255, 0]`,\nbut `ops.saturate_cast(ops.cast([-1, 256], "float32"), "uint8")` returns\n`[0, 255]`.\n\nArgs:\n    x: A tensor or variable.\n    dtype: The target type.\n\nReturns:\n    A safely casted tensor of the specified `dtype`.\n\nExample:\n\nImage resizing with bicubic interpolation may produce values outside\noriginal range.\n>>> image2x2 = np.array([0, 1, 254, 255], dtype="uint8").reshape(1, 2, 2, 1)\n>>> image4x4 = tf.image.resize(image2x2, (4, 4), method="bicubic")\n>>> print(image4x4.numpy().squeeze())\n>>> # [[-22.500004 -22.204624 -21.618908 -21.32353 ]\n>>> #  [ 52.526054  52.82143   53.407146  53.70253 ]\n>>> #  [201.29752  201.59288  202.17859  202.47395 ]\n>>> #  [276.32355  276.61893  277.20465  277.50006 ]]\n\nCasting this resized image back to `uint8` will cause overflow.\n>>> image4x4_casted = ops.cast(image4x4, "uint8")\n>>> print(image4x4_casted.numpy().squeeze())\n>>> # [[234 234 235 235]\n>>> #  [ 52  52  53  53]\n>>> #  [201 201 202 202]\n>>> #  [ 20  20  21  21]]\n\nSaturate casting to `uint8` will clip values to `uint8` range before\ncasting and will not cause overflow.\n>>> image4x4_saturate_casted = ops.saturate_cast(image4x4, "uint8")\n>>> print(image4x4_saturate_casted.numpy().squeeze())\n>>> # [[  0   0   0   0]\n>>> #  [ 52  52  53  53]\n>>> #  [201 201 202 202]\n>>> #  [255 255 255 255]]',
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "savez": {
    "description": "savez(file: Union[file, str, pathlib.Path], *args, **kwargs)",
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "savez_compressed": {
    "description": "savez_compressed(file: Union[file, str, pathlib.Path], *args, **kwargs)",
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "scale": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "scale_factor": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "scale_grad_by_freq": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "scaled_dot_product_attention": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "scatter_update": {
    "description": "Update inputs via updates at scattered (sparse) indices.\n\nAt a high level, this operation does `inputs[indices] = updates`.\nAssume `inputs` is a tensor of shape `(D0, D1, ..., Dn)`, there are 2 main\nusages of `scatter_update`.\n\n1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`\n    is the number of updates to perform, and `updates` is a 1D tensor of\n    shape `(num_updates,)`. For example, if `inputs` is `zeros((4, 4, 4))`,\n    and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then\n    we can use:\n\n```python\ninputs = np.zeros((4, 4, 4))\nindices = [[1, 2, 3], [0, 1, 3]]\nupdates = np.array([1., 1.])\ninputs = keras.ops.scatter_update(inputs, indices, updates)\n```\n\n2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`\n    is the number of updates to perform, and `k` (`k < n`) is the size of\n    each index in `indices`. `updates` is a `n - k`-D tensor of shape\n    `(num_updates, inputs.shape[k:])`. For example, if\n    `inputs = np.zeros((4, 4, 4))`, and we want to update `inputs[1, 2, :]`\n    and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape\n    `(num_updates, 2)` (`k = 2`), and `updates` would have shape\n    `(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:\n\n```python\ninputs = np.zeros((4, 4, 4))\nindices = [[1, 2], [2, 3]]\nupdates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])\ninputs = keras.ops.scatter_update(inputs, indices, updates)\n```\n\nArgs:\n    inputs: A tensor, the tensor to be updated.\n    indices: A tensor or list/tuple of shape `(N, inputs.ndim)`, specifying\n        indices to update. `N` is the number of indices to update, must be\n        equal to the first dimension of `updates`.\n    updates: A tensor, the new values to be put to `inputs` at `indices`.\n\nReturns:\n    A tensor, has the same shape and dtype as `inputs`.",
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "indices", "type": "Any"},
      {"name": "updates", "type": "Any"},
    ],
    "variants": {},
  },
  "score_function_jacobians": {
    "description": "Score function gradient estimation.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "params", "type": "Any"},
      {"name": "dist_builder", "type": "Any"},
      {"name": "rng", "type": "Any"},
      {"name": "num_samples", "type": "Any"},
    ],
    "variants": {},
  },
  "searchsorted": {
    "description": "Finds the indices into ``x1`` such that, if the corresponding elements in ``x2`` were inserted before the indices, the order of ``x1``, when sorted in ascending order, would be preserved.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
      {"name": "side", "type": "Literal[left, right]"},
      {"name": "sorter", "type": "Optional[array]"},
    ],
    "variants": {},
  },
  "seed": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "segment_max": {
    "description": "Computes the max of segments in a tensor.\n\nArgs:\n    data: Input tensor.\n    segment_ids: A N-D tensor containing segment indices for each\n        element in `data`. data.shape[:len(segment_ids.shape)] should match.\n    num_segments: An integer representing the total number of\n        segments. If not specified, it is inferred from the maximum\n        value in `segment_ids`.\n    sorted: A boolean indicating whether `segment_ids` is sorted.\n        Defaults to `False`.\n\nReturns:\n    A tensor containing the max of segments, where each element\n    represents the max of the corresponding segment in `data`.\n\nExample:\n\n>>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])\n>>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])\n>>> num_segments = 3\n>>> keras.ops.segment_max(data, segment_ids, num_segments)\narray([2, 20, 200], dtype=int32)",
    "std_args": [
      {"name": "data", "type": "Any"},
      {"name": "segment_ids", "type": "Any"},
      {"name": "num_segments", "type": "Any"},
      {"name": "sorted", "type": "Any"},
    ],
    "variants": {},
  },
  "segment_sum": {
    "description": "Computes the sum of segments in a tensor.\n\nArgs:\n    data: Input tensor.\n    segment_ids: A N-D tensor containing segment indices for each\n        element in `data`. Num dims for segment ids should be strictly\n        smaller or equal to number of dims in data.\n    num_segments: An integer representing the total number of\n        segments. If not specified, it is inferred from the maximum\n        value in `segment_ids`.\n    sorted: A boolean indicating whether `segment_ids` is sorted.\n        Defaults to `False`.\n\nReturns:\n    A tensor containing the sum of segments, where each element\n    represents the sum of the corresponding segment in `data`.\n\nExample:\n\n>>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])\n>>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])\n>>> num_segments = 3\n>>> keras.ops.segment_sum(data, segment_ids,num_segments)\narray([3, 30, 300], dtype=int32)",
    "std_args": [
      {"name": "data", "type": "Any"},
      {"name": "segment_ids", "type": "Any"},
      {"name": "num_segments", "type": "Any"},
      {"name": "sorted", "type": "Any"},
    ],
    "variants": {},
  },
  "select": {
    "description": "Select values based on a series of conditions.\n\nJAX implementation of :func:`numpy.select`, implemented in terms\nof :func:`jax.lax.select_n`\n\nArgs:\n  condlist: sequence of array-like conditions. All entries must be mutually\n    broadcast-compatible.\n  choicelist: sequence of array-like values to choose. Must have the same length\n    as ``condlist``, and all entries must be broadcast-compatible with entries\n    of ``condlist``.\n  default: value to return when every condition is False (default: 0).\n\nReturns:\n  Array of selected values from ``choicelist`` corresponding to the first\n  ``True`` entry in ``condlist`` at each location.\n\nSee also:\n  - :func:`jax.numpy.where`: select between two values based on a single condition.\n  - :func:`jax.lax.select_n`: select between *N* values based on an index.\n\nExamples:\n  >>> condlist = [\n  ...    jnp.array([False, True, False, False]),\n  ...    jnp.array([True, False, False, False]),\n  ...    jnp.array([False, True, True, False]),\n  ... ]\n  >>> choicelist = [\n  ...    jnp.array([1, 2, 3, 4]),\n  ...    jnp.array([10, 20, 30, 40]),\n  ...    jnp.array([100, 200, 300, 400]),\n  ... ]\n  >>> jnp.select(condlist, choicelist, default=0)\n  Array([ 10,   2, 300,   0], dtype=int32)\n\n  This is logically equivalent to the following nested ``where`` statement:\n\n  >>> default = 0\n  >>> jnp.where(condlist[0],\n  ...   choicelist[0],\n  ...   jnp.where(condlist[1],\n  ...     choicelist[1],\n  ...     jnp.where(condlist[2],\n  ...       choicelist[2],\n  ...       default)))\n  Array([ 10,   2, 300,   0], dtype=int32)\n\n  However, for efficiency it is implemented in terms of :func:`jax.lax.select_n`.",
    "std_args": [
      {"name": "condlist", "type": "Any"},
      {"name": "choicelist", "type": "Any"},
      {"name": "default", "type": "Any"},
    ],
    "variants": {},
  },
  "selective_transform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "separable_conv": {
    "description": 'General N-D separable convolution.\n\nThis ops supports 1D and 2D separable convolution. `separable_conv` is\na depthwise conv followed by a pointwise conv.\n\nArgs:\n    inputs: Tensor of rank N+2. `inputs` has shape\n        `(batch_size,) + inputs_spatial_shape + (num_channels,)` if\n        `data_format="channels_last"`, or\n        `(batch_size, num_channels) + inputs_spatial_shape` if\n        `data_format="channels_first"`.\n    depthwise_kernel: Tensor of rank N+2. `depthwise_kernel` has shape\n        [kernel_spatial_shape, num_input_channels, num_channels_multiplier],\n        `num_input_channels` should match the number of channels in\n        `inputs`.\n    pointwise_kernel: Tensor of rank N+2. `pointwise_kernel` has shape\n        `(*ones_like(kernel_spatial_shape),\n        num_input_channels * num_channels_multiplier, num_output_channels)`.\n    strides: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the strides of the convolution along each spatial\n        dimension. If `strides` is int, then every spatial dimension shares\n        the same `strides`.\n    padding: string, either `"valid"` or `"same"`. `"valid"` means no\n        padding is applied, and `"same"` results in padding evenly to the\n        left/right or up/down of the input such that output has the\n        same height/width dimension as the input when `strides=1`.\n    data_format: A string, either `"channels_last"` or `"channels_first"`.\n        `data_format` determines the ordering of the dimensions in the\n        inputs. If `data_format="channels_last"`, `inputs` is of shape\n        `(batch_size, ..., channels)` while if\n        `data_format="channels_first"`, `inputs` is of shape\n        `(batch_size, channels, ...)`.\n    dilation_rate: int or int tuple/list of `len(inputs_spatial_shape)`,\n        specifying the dilation rate to use for dilated convolution. If\n        `dilation_rate` is int, then every spatial dimension shares\n        the same `dilation_rate`.\n\nReturns:\n    A tensor of rank N+2, the result of the depthwise conv operation.',
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "depthwise_kernel", "type": "Any"},
      {"name": "pointwise_kernel", "type": "Any"},
      {"name": "strides", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "data_format", "type": "Any"},
      {"name": "dilation_rate", "type": "Any"},
    ],
    "variants": {},
  },
  "serialize": {
    "description": "",
    "std_args": [
      {"name": "activation", "type": "Any"},
    ],
    "variants": {},
  },
  "set": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "set_printoptions": {
    "description": "Alias of :func:`numpy.set_printoptions`.\n\nJAX arrays are printed via NumPy, so NumPy's `printoptions`\nconfigurations will apply to printed JAX arrays.\n\nSee the :func:`numpy.set_printoptions` documentation for details\non the available options and their meanings.",
    "std_args": [
      {"name": "args", "type": "Any"},
      {"name": "kwargs", "type": "Any"},
    ],
    "variants": {},
  },
  "setdiff1d": {
    "description": "Compute the set difference of two 1D arrays.\n\nJAX implementation of :func:`numpy.setdiff1d`.\n\nBecause the size of the output of ``setdiff1d`` is data-dependent, the function\nis not typically compatible with :func:`~jax.jit` and other JAX transformations.\nThe JAX version adds the optional ``size`` argument which must be specified statically\nfor ``jnp.setdiff1d`` to be used in such contexts.\n\nArgs:\n  ar1: first array of elements to be differenced.\n  ar2: second array of elements to be differenced.\n  assume_unique: if True, assume the input arrays contain unique values. This allows\n    a more efficient implementation, but if ``assume_unique`` is True and the input\n    arrays contain duplicates, the behavior is undefined. default: False.\n  size: if specified, return only the first ``size`` sorted elements. If there are fewer\n    elements than ``size`` indicates, the return value will be padded with ``fill_value``.\n  fill_value: when ``size`` is specified and there are fewer than the indicated number of\n    elements, fill the remaining entries ``fill_value``. Defaults to the minimum value.\n\nReturns:\n  an array containing the set difference of elements in the input array: i.e. the elements\n  in ``ar1`` that are not contained in ``ar2``.\n\nSee also:\n  - :func:`jax.numpy.intersect1d`: the set intersection of two 1D arrays.\n  - :func:`jax.numpy.setxor1d`: the set XOR of two 1D arrays.\n  - :func:`jax.numpy.union1d`: the set union of two 1D arrays.\n\nExamples:\n  Computing the set difference of two arrays:\n\n  >>> ar1 = jnp.array([1, 2, 3, 4])\n  >>> ar2 = jnp.array([3, 4, 5, 6])\n  >>> jnp.setdiff1d(ar1, ar2)\n  Array([1, 2], dtype=int32)\n\n  Because the output shape is dynamic, this will fail under :func:`~jax.jit` and other\n  transformations:\n\n  >>> jax.jit(jnp.setdiff1d)(ar1, ar2)  # doctest: +IGNORE_EXCEPTION_DETAIL\n  Traceback (most recent call last):\n     ...\n  ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[4].\n  The error occurred while tracing the function setdiff1d at /Users/vanderplas/github/jax-ml/jax/jax/_src/numpy/setops.py:64 for jit. This concrete value was not available in Python because it depends on the value of the argument ar1.\n\n  In order to ensure statically-known output shapes, you can pass a static ``size``\n  argument:\n\n  >>> jit_setdiff1d = jax.jit(jnp.setdiff1d, static_argnames=['size'])\n  >>> jit_setdiff1d(ar1, ar2, size=2)\n  Array([1, 2], dtype=int32)\n\n  If ``size`` is too small, the difference is truncated:\n\n  >>> jit_setdiff1d(ar1, ar2, size=1)\n  Array([1], dtype=int32)\n\n  If ``size`` is too large, then the output is padded with ``fill_value``:\n\n  >>> jit_setdiff1d(ar1, ar2, size=4, fill_value=0)\n  Array([1, 2, 0, 0], dtype=int32)",
    "std_args": [
      {"name": "ar1", "type": "Any"},
      {"name": "ar2", "type": "Any"},
      {"name": "assume_unique", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "setxor1d": {
    "description": "Compute the set-wise xor of elements in two arrays.\n\nJAX implementation of :func:`numpy.setxor1d`.\n\nBecause the size of the output of ``setxor1d`` is data-dependent, the function is not\ncompatible with JIT or other JAX transformations.\n\nArgs:\n  ar1: first array of values to intersect.\n  ar2: second array of values to intersect.\n  assume_unique: if True, assume the input arrays contain unique values. This allows\n    a more efficient implementation, but if ``assume_unique`` is True and the input\n    arrays contain duplicates, the behavior is undefined. default: False.\n  size: if specified, return only the first ``size`` sorted elements. If there are fewer\n    elements than ``size`` indicates, the return value will be padded with ``fill_value``,\n    and returned indices will be padded with an out-of-bound index.\n  fill_value: when ``size`` is specified and there are fewer than the indicated number of\n    elements, fill the remaining entries ``fill_value``. Defaults to the smallest value\n    in the xor result.\n\nReturns:\n  An array of values that are found in exactly one of the input arrays.\n\nSee also:\n  - :func:`jax.numpy.intersect1d`: the set intersection of two 1D arrays.\n  - :func:`jax.numpy.union1d`: the set union of two 1D arrays.\n  - :func:`jax.numpy.setdiff1d`: the set difference of two 1D arrays.\n\nExamples:\n  >>> ar1 = jnp.array([1, 2, 3, 4])\n  >>> ar2 = jnp.array([3, 4, 5, 6])\n  >>> jnp.setxor1d(ar1, ar2)\n  Array([1, 2, 5, 6], dtype=int32)",
    "std_args": [
      {"name": "ar1", "type": "Any"},
      {"name": "ar2", "type": "Any"},
      {"name": "assume_unique", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "sgdr_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "short": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "shuffle": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "sigmoid_binary_cross_entropy": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "sigmoid_focal_loss": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "sign": {
    "description": "Returns an indication of the sign of a number for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "signbit": {
    "description": "Determines whether the sign bit is set for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "signedinteger": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "sin": {
    "description": "Calculates an implementation-dependent approximation to the sine for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "sinc": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "single": {
    "description": "A JAX scalar constructor of type float32.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "sinh": {
    "description": "Calculates an implementation-dependent approximation to the hyperbolic sine for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "size": {
    "description": "Get tensor shape",
    "std_args": [],
    "variants": {},
  },
  "skip_large_updates": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "skip_not_finite": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "slice_update": {
    "description": "Update an input by slicing in a tensor of updated values.\n\nAt a high level, this operation does\n`inputs[start_indices: start_indices + updates.shape] = updates`.\nAssume inputs is a tensor of shape `(D0, D1, ..., Dn)`,\n`start_indices` must be a list/tuple of n integers, specifying the starting\nindices. `updates` must have the same rank as `inputs`, and the size of each\ndim must not exceed `Di - start_indices[i]`. For example, if we have 2D\ninputs `inputs = np.zeros((5, 5))`, and we want to update the intersection\nof last 2 rows and last 2 columns as 1, i.e.,\n`inputs[3:, 3:] = np.ones((2, 2))`, then we can use the code below:\n\n```python\ninputs = np.zeros((5, 5))\nstart_indices = [3, 3]\nupdates = np.ones((2, 2))\ninputs = keras.ops.slice_update(inputs, start_indices, updates)\n```\n\nArgs:\n    inputs: A tensor, the tensor to be updated.\n    start_indices: A list/tuple of shape `(inputs.ndim,)`, specifying\n        the starting indices for updating.\n    updates: A tensor, the new values to be put to `inputs` at `indices`.\n        `updates` must have the same rank as `inputs`.\n\nReturns:\n    A tensor, has the same shape and dtype as `inputs`.",
    "std_args": [
      {"name": "inputs", "type": "Any"},
      {"name": "start_indices", "type": "Any"},
      {"name": "updates", "type": "Any"},
    ],
    "variants": {},
  },
  "slogdet": {
    "description": "Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "smooth_l1_loss": {
    "description": "Compute the Smooth L1 loss.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "target", "type": "Any"},
      {"name": "size_average", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "reduction", "type": "Any"},
      {"name": "beta", "type": "Any"},
    ],
    "variants": {},
  },
  "smooth_labels": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "snapshot": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "soft_shrink": {
    "description": "Soft Shrink activation function.\n\nIt is defined as:\n\n`soft_shrink(x) = x - threshold` if `x > threshold`,\n`soft_shrink(x) = x + threshold` if `x < -threshold`,\n`soft_shrink(x) = 0` otherwise.\n\nArgs:\n    x: Input tensor.\n    threshold: Threshold value. Defaults to 0.5.",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "threshold", "type": "Any"},
    ],
    "variants": {},
  },
  "softmax": {
    "description": "Applies the Softmax function to an n-dimensional input Tensor.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "dim", "type": "Any"},
    ],
    "variants": {},
  },
  "softmax_cross_entropy": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "softmax_cross_entropy_with_integer_labels": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "solve": {
    "description": "Returns the solution of a square system of linear equations with a unique solution.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
    ],
    "variants": {},
  },
  "solve_triangular": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "sort": {
    "description": "Returns a sorted copy of an input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "int"},
      {"name": "descending", "type": "bool"},
      {"name": "stable", "type": "bool"},
    ],
    "variants": {},
  },
  "sparse": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "sparse_": {
    "description": "Fill the 2D input `Tensor` as a sparse matrix.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
      {"name": "sparsity", "type": "Any"},
      {"name": "std", "type": "Any"},
      {"name": "generator", "type": "Any"},
    ],
    "variants": {},
  },
  "sparse_categorical_crossentropy": {
    "description": "Computes sparse categorical cross-entropy loss.\n\nThe sparse categorical cross-entropy loss is similar to categorical\ncross-entropy, but it is used when the target tensor contains integer\nclass labels instead of one-hot encoded vectors. It measures the\ndissimilarity between the target and output probabilities or logits.\n\nArgs:\n    target: The target tensor representing the true class labels as\n        integers. Its shape should match the shape of the `output`\n        tensor except for the last dimension.\n    output: The output tensor representing the predicted probabilities\n        or logits.\n        Its shape should match the shape of the `target` tensor except\n        for the last dimension.\n    from_logits: (optional) Whether `output` is a tensor of logits\n        or probabilities.\n        Set it to `True` if `output` represents logits; otherwise,\n        set it to `False` if `output` represents probabilities.\n        Defaults to `False`.\n    axis: (optional) The axis along which the sparse categorical\n        cross-entropy is computed.\n        Defaults to `-1`, which corresponds to the last dimension\n        of the tensors.\n\nReturns:\n    Integer tensor: The computed sparse categorical cross-entropy\n    loss between `target` and `output`.\n\nExample:\n\n>>> target = keras.ops.convert_to_tensor([0, 1, 2], dtype=int32)\n>>> output = keras.ops.convert_to_tensor(\n... [[0.9, 0.05, 0.05],\n...  [0.1, 0.8, 0.1],\n...  [0.2, 0.3, 0.5]])\n>>> sparse_categorical_crossentropy(target, output)\narray([0.10536056 0.22314355 0.6931472 ], shape=(3,), dtype=float32)",
    "std_args": [
      {"name": "target", "type": "Any"},
      {"name": "output", "type": "Any"},
      {"name": "from_logits", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "sparse_plus": {
    "description": "SparsePlus activation function.\n\nSparsePlus is defined as:\n\n`sparse_plus(x) = 0` for `x <= -1`.\n`sparse_plus(x) = (1/4) * (x + 1)^2` for `-1 < x < 1`.\n`sparse_plus(x) = x` for `x >= 1`.\n\nArgs:\n    x: Input tensor.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "sparse_sigmoid": {
    "description": "Sparse sigmoid activation function.\n\nIt is defined as\n\n`f(x) = 0` for `x <= -1`,\n`f(x) = 0.5 * (x + 1)` for `-1 < x < 1`,\n`f(x) = 1` for `x >= 1`.\n\nArgs:\n    x: Input tensor.\n\nReference:\n\n- [M. Blondel, A. F. T. Martins, V. Niculae, 2019](https://arxiv.org/pdf/1901.02324)",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "split_key_like": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "sqrt": {
    "description": "Square root.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "square": {
    "description": "Square.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "squared_error": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "squareplus": {
    "description": "Squareplus activation function.\n\nThe Squareplus activation function is defined as:\n\n`f(x) = (x + sqrt(x^2 + b)) / 2`\n\nWhere `b` is a smoothness parameter.\n\nArgs:\n    x: Input tensor.\n    b: Smoothness parameter. Defaults to 4.\n\nReference:\n\n- [Ramachandran et al., 2021](https://arxiv.org/abs/2112.11687)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "b", "type": "Any"},
    ],
    "variants": {},
  },
  "squeeze": {
    "description": "Removes singleton dimensions (axes) from ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Union[int, Tuple[int, Ellipsis]]"},
    ],
    "variants": {},
  },
  "stack": {
    "description": "Joins a sequence of arrays along a new axis.",
    "std_args": [
      {"name": "arrays", "type": "Union[Tuple[array, Ellipsis], List[array]]"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "start_dim": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "state": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "state_dict": {
    "description": "Returns a dictionary containing a whole state of the module.",
    "std_args": [
      {"name": "destination", "type": "Any"},
      {"name": "prefix", "type": "Any"},
      {"name": "keep_vars", "type": "Any"},
    ],
    "variants": {},
  },
  "static_graph": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "std": {
    "description": "Calculates the standard deviation of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "correction", "type": "Union[int, float]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "step": {
    "description": "Performs a single optimization step.",
    "std_args": [],
    "variants": {},
  },
  "stop_gradient": {
    "description": 'Stops gradient computation.\n\nArgs:\n    variable: A tensor variable for which the gradient\n        computation is to be disabled.\n\nReturns:\n    The variable with gradient computation disabled.\n\nExamples:\n\n>>> var = keras.backend.convert_to_tensor(\n...     [1., 2., 3.],\n...     dtype="float32"\n... )\n>>> var = keras.ops.stop_gradient(var)',
    "std_args": [
      {"name": "variable", "type": "Any"},
    ],
    "variants": {},
  },
  "stride": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "strides": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "subtract": {
    "description": "Calculates the difference for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.",
    "std_args": [
      {"name": "x1", "type": "Union[array, int, float, complex]"},
      {"name": "x2", "type": "Union[array, int, float, complex]"},
    ],
    "variants": {},
  },
  "sum": {
    "description": "Calculates the sum of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "svd": {
    "description": "Returns a singular value decomposition (SVD) of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "full_matrices", "type": "bool"},
    ],
    "variants": {},
  },
  "svdvals": {
    "description": "Returns the singular values of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "swap": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "swapaxes": {
    "description": "Swap two axes of an array.\n\nRefer to :func:`jax.numpy.swapaxes` for full documentation.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "axis1", "type": "Any"},
      {"name": "axis2", "type": "Any"},
    ],
    "variants": {},
  },
  "switch": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "index", "type": "Any"},
      {"name": "branches", "type": "Any"},
      {"name": "operands", "type": "Any"},
    ],
    "variants": {},
  },
  "take": {
    "description": "Returns elements of an array along an axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "indices", "type": "array"},
      {"name": "axis", "type": "Optional[int]"},
    ],
    "variants": {},
  },
  "take_along_axis": {
    "description": "Returns elements from an array at the one-dimensional indices specified by ``indices`` along a provided ``axis``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "indices", "type": "array"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "tan": {
    "description": "Calculates an implementation-dependent approximation to the tangent for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "tanh": {
    "description": "Calculates an implementation-dependent approximation to the hyperbolic tangent for each element ``x_i`` of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "tanh_shrink": {
    "description": "Tanh shrink activation function.\n\nIt is defined as:\n\n`f(x) = x - tanh(x)`.\n\nArgs:\n    x: Input tensor.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "tensordot": {
    "description": "Returns a tensor contraction of ``x1`` and ``x2`` over specific axes.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
      {"name": "axes", "type": "Union[int, Tuple[Sequence[int], Sequence[int]]]"},
    ],
    "variants": {},
  },
  "tensorinv": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tensorsolve": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "test_import": {
    "description": "The class representing a Python function.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "threshold_": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tile": {
    "description": "Constructs an array by tiling an input array.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "repetitions", "type": "Tuple[int, Ellipsis]"},
    ],
    "variants": {},
  },
  "to_dense": {
    "description": "Returns a dense block that is equivalent to the block mask.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "to_device": {
    "description": "Return a copy of the array on the specified device\n\nArgs:\n  device: :class:`~jax.Device` or :class:`~jax.sharding.Sharding`\n    to which the created array will be committed.\n  stream: not implemented, passing a non-None value will lead to an error.\nReturns:\n  copy of array placed on the specified device or devices.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "device", "type": "Any"},
      {"name": "stream", "type": "Any"},
    ],
    "variants": {},
  },
  "tolist": {
    "description": "Return the matrix as a (possibly nested) list.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "top_k": {
    "description": "Finds the top-k values and their indices in a tensor.\n\nArgs:\n    x: Input tensor.\n    k: An integer representing the number of top elements to retrieve.\n    sorted: A boolean indicating whether to sort the output in\n    descending order. Defaults to `True`.\n\nReturns:\n    A tuple containing two tensors. The first tensor contains the\n    top-k values, and the second tensor contains the indices of the\n    top-k values in the input tensor.\n\nExample:\n\n>>> x = keras.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])\n>>> values, indices = top_k(x, k=3)\n>>> print(values)\narray([9 7 5], shape=(3,), dtype=int32)\n>>> print(indices)\narray([4 2 0], shape=(3,), dtype=int32)",
    "std_args": [
      {"name": "x", "type": "Any"},
      {"name": "k", "type": "Any"},
      {"name": "sorted", "type": "Any"},
    ],
    "variants": {},
  },
  "trace": {
    "description": "Returns the sum along the specified diagonals of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "offset", "type": "int"},
      {"name": "dtype", "type": "Optional[dtype]"},
    ],
    "variants": {},
  },
  "train": {
    "description": "Set the module in training mode.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "mode", "type": "Any"},
    ],
    "variants": {},
  },
  "training": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "trapezoid": {
    "description": "Integrate along the given axis using the composite trapezoidal rule.\n\nArgs:\n    y: Input tensor.\n    x: Optional tensor specifying sample points corresponding to `y`.\n       If `None`, spacing is assumed to be `dx`.\n    dx: Spacing between sample points when `x` is `None`.\n    axis: Axis along which to integrate. Default is the last axis.\n\nReturns:\n    The approximate integral of `y` along the given axis.\n\nExample:\n>>> y = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]])\n>>> keras.ops.trapezoid(y, axis=1)\narray([ 4., 10.], dtype=float32)",
    "std_args": [
      {"name": "y", "type": "Any"},
      {"name": "x", "type": "Any"},
      {"name": "dx", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "tree_add_scalar_mul": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tree_l1_norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tree_l2_norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tree_linf_norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tree_map_params": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tree_scalar_mul": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tri": {
    "description": "Return an array with ones on and below the diagonal and zeros elsewhere.\n\nJAX implementation of :func:`numpy.tri`\n\nArgs:\n  N: int. Dimension of the rows of the returned array.\n  M: optional, int. Dimension of the columns of the returned array. If not\n    specified, then ``M = N``.\n  k: optional, int, default=0. Specifies the sub-diagonal on and below which\n    the array is filled with ones. ``k=0`` refers to main diagonal, ``k<0``\n    refers to sub-diagonal below the main diagonal and ``k>0`` refers to\n    sub-diagonal above the main diagonal.\n  dtype: optional, data type of the returned array. The default type is float.\n\nReturns:\n  An array of shape ``(N, M)`` containing the lower triangle with elements\n  below the sub-diagonal specified by ``k`` are set to one and zero elsewhere.\n\nSee also:\n  - :func:`jax.numpy.tril`: Returns a lower triangle of an array.\n  - :func:`jax.numpy.triu`: Returns an upper triangle of an array.\n\nExamples:\n  >>> jnp.tri(3)\n  Array([[1., 0., 0.],\n         [1., 1., 0.],\n         [1., 1., 1.]], dtype=float32)\n\n  When ``M`` is not equal to ``N``:\n\n  >>> jnp.tri(3, 4)\n  Array([[1., 0., 0., 0.],\n         [1., 1., 0., 0.],\n         [1., 1., 1., 0.]], dtype=float32)\n\n  when ``k>0``:\n\n  >>> jnp.tri(3, k=1)\n  Array([[1., 1., 0.],\n         [1., 1., 1.],\n         [1., 1., 1.]], dtype=float32)\n\n  When ``k<0``:\n\n  >>> jnp.tri(3, 4, k=-1)\n  Array([[0., 0., 0., 0.],\n         [1., 0., 0., 0.],\n         [1., 1., 0., 0.]], dtype=float32)",
    "std_args": [
      {"name": "N", "type": "Any"},
      {"name": "M", "type": "Any"},
      {"name": "k", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "triangular": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "tril": {
    "description": "Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "k", "type": "int"},
    ],
    "variants": {},
  },
  "tril_indices": {
    "description": "Return the indices of lower triangle of an array of size ``(n, m)``.\n\nJAX implementation of :func:`numpy.tril_indices`.\n\nArgs:\n  n: int. Number of rows of the array for which the indices are returned.\n  k: optional, int, default=0. Specifies the sub-diagonal on and below which\n    the indices of lower triangle are returned. ``k=0`` refers to main diagonal,\n    ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers\n    to sub-diagonal above the main diagonal.\n  m: optional, int. Number of columns of the array for which the indices are\n    returned. If not specified, then ``m = n``.\n\nReturns:\n  A tuple of two arrays containing the indices of the lower triangle, one along\n  each axis.\n\nSee also:\n  - :func:`jax.numpy.triu_indices`: Returns the indices of upper triangle of an\n    array of size ``(n, m)``.\n  - :func:`jax.numpy.triu_indices_from`: Returns the indices of upper triangle\n    of a given array.\n  - :func:`jax.numpy.tril_indices_from`: Returns the indices of lower triangle\n    of a given array.\n\nExamples:\n  If only ``n`` is provided in input, the indices of lower triangle of an array\n  of size ``(n, n)`` array are returned.\n\n  >>> jnp.tril_indices(3)\n  (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))\n\n  If both ``n`` and ``m`` are provided in input, the indices of lower triangle\n  of an ``(n, m)`` array are returned.\n\n  >>> jnp.tril_indices(3, m=2)\n  (Array([0, 1, 1, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1], dtype=int32))\n\n  If ``k = 1``, the indices on and below the first sub-diagonal above the main\n  diagonal are returned.\n\n  >>> jnp.tril_indices(3, k=1)\n  (Array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int32), Array([0, 1, 0, 1, 2, 0, 1, 2], dtype=int32))\n\n  If ``k = -1``, the indices on and below the first sub-diagonal below the main\n  diagonal are returned.\n\n  >>> jnp.tril_indices(3, k=-1)\n  (Array([1, 2, 2], dtype=int32), Array([0, 0, 1], dtype=int32))",
    "std_args": [
      {"name": "n", "type": "Any"},
      {"name": "k", "type": "Any"},
      {"name": "m", "type": "Any"},
    ],
    "variants": {},
  },
  "tril_indices_from": {
    "description": "Return the indices of lower triangle of a given array.\n\nJAX implementation of :func:`numpy.tril_indices_from`.\n\nArgs:\n  arr: input array. Must have ``arr.ndim == 2``.\n  k: optional, int, default=0. Specifies the sub-diagonal on and below which\n    the indices of upper triangle are returned. ``k=0`` refers to main diagonal,\n    ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers\n    to sub-diagonal above the main diagonal.\n\nReturns:\n  A tuple of two arrays containing the indices of the lower triangle, one along\n  each axis.\n\nSee also:\n  - :func:`jax.numpy.triu_indices_from`: Returns the indices of upper triangle\n    of a given array.\n  - :func:`jax.numpy.tril_indices`: Returns the indices of lower triangle of an\n    array of size ``(n, m)``.\n  - :func:`jax.numpy.tril`: Returns a lower triangle of an array\n\nExamples:\n  >>> arr = jnp.array([[1, 2, 3],\n  ...                  [4, 5, 6],\n  ...                  [7, 8, 9]])\n  >>> jnp.tril_indices_from(arr)\n  (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))\n\n  Elements indexed by ``jnp.tril_indices_from`` correspond to those in the\n  output of ``jnp.tril``.\n\n  >>> ind = jnp.tril_indices_from(arr)\n  >>> arr[ind]\n  Array([1, 4, 5, 7, 8, 9], dtype=int32)\n  >>> jnp.tril(arr)\n  Array([[1, 0, 0],\n         [4, 5, 0],\n         [7, 8, 9]], dtype=int32)\n\n  When ``k > 0``:\n\n  >>> jnp.tril_indices_from(arr, k=1)\n  (Array([0, 0, 1, 1, 1, 2, 2, 2], dtype=int32), Array([0, 1, 0, 1, 2, 0, 1, 2], dtype=int32))\n\n  When ``k < 0``:\n\n  >>> jnp.tril_indices_from(arr, k=-1)\n  (Array([1, 2, 2], dtype=int32), Array([0, 0, 1], dtype=int32))",
    "std_args": [
      {"name": "arr", "type": "Any"},
      {"name": "k", "type": "Any"},
    ],
    "variants": {},
  },
  "trim_zeros": {
    "description": "Trim leading and/or trailing zeros of the input array.\n\nJAX implementation of :func:`numpy.trim_zeros`.\n\nArgs:\n  filt: N-dimensional input array.\n  trim: string, optional, default = ``fb``. Specifies from which end the input\n    is trimmed.\n\n    - ``f`` - trims only the leading zeros.\n    - ``b`` - trims only the trailing zeros.\n    - ``fb`` - trims both leading and trailing zeros.\n\n  axis: optional axis or axes along which to trim. If not specified, trim along\n    all axes of the array.\n\nReturns:\n  An array containing the trimmed input with same dtype as ``filt``.\n\nExamples:\n  One-dimensional input:\n\n  >>> x = jnp.array([0, 0, 2, 0, 1, 4, 3, 0, 0, 0])\n  >>> jnp.trim_zeros(x)\n  Array([2, 0, 1, 4, 3], dtype=int32)\n  >>> jnp.trim_zeros(x, trim='f')\n  Array([2, 0, 1, 4, 3, 0, 0, 0], dtype=int32)\n  >>> jnp.trim_zeros(x, trim='b')\n  Array([0, 0, 2, 0, 1, 4, 3], dtype=int32)\n\n  Two-dimensional input:\n\n  >>> x = jnp.zeros((4, 5)).at[1:3, 1:4].set(1)\n  >>> x\n  Array([[0., 0., 0., 0., 0.],\n         [0., 1., 1., 1., 0.],\n         [0., 1., 1., 1., 0.],\n         [0., 0., 0., 0., 0.]], dtype=float32)\n  >>> jnp.trim_zeros(x)\n  Array([[1., 1., 1.],\n         [1., 1., 1.]], dtype=float32)\n  >>> jnp.trim_zeros(x, trim='f')\n  Array([[1., 1., 1., 0.],\n         [1., 1., 1., 0.],\n         [0., 0., 0., 0.]], dtype=float32)\n  >>> jnp.trim_zeros(x, axis=0)\n  Array([[0., 1., 1., 1., 0.],\n         [0., 1., 1., 1., 0.]], dtype=float32)\n  >>> jnp.trim_zeros(x, axis=1)\n  Array([[0., 0., 0.],\n         [1., 1., 1.],\n         [1., 1., 1.],\n         [0., 0., 0.]], dtype=float32)",
    "std_args": [
      {"name": "filt", "type": "Any"},
      {"name": "trim", "type": "Any"},
      {"name": "axis", "type": "Any"},
    ],
    "variants": {},
  },
  "triu": {
    "description": "Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "k", "type": "int"},
    ],
    "variants": {},
  },
  "triu_indices": {
    "description": "Return the indices of upper triangle of an array of size ``(n, m)``.\n\nJAX implementation of :func:`numpy.triu_indices`.\n\nArgs:\n  n: int. Number of rows of the array for which the indices are returned.\n  k: optional, int, default=0. Specifies the sub-diagonal on and above which\n    the indices of upper triangle are returned. ``k=0`` refers to main diagonal,\n    ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers\n    to sub-diagonal above the main diagonal.\n  m: optional, int. Number of columns of the array for which the indices are\n    returned. If not specified, then ``m = n``.\n\nReturns:\n  A tuple of two arrays containing the indices of the upper triangle, one along\n  each axis.\n\nSee also:\n  - :func:`jax.numpy.tril_indices`: Returns the indices of lower triangle of an\n    array of size ``(n, m)``.\n  - :func:`jax.numpy.triu_indices_from`: Returns the indices of upper triangle\n    of a given array.\n  - :func:`jax.numpy.tril_indices_from`: Returns the indices of lower triangle\n    of a given array.\n\nExamples:\n  If only ``n`` is provided in input, the indices of upper triangle of an array\n  of size ``(n, n)`` array are returned.\n\n  >>> jnp.triu_indices(3)\n  (Array([0, 0, 0, 1, 1, 2], dtype=int32), Array([0, 1, 2, 1, 2, 2], dtype=int32))\n\n  If both ``n`` and ``m`` are provided in input, the indices of upper triangle\n  of an ``(n, m)`` array are returned.\n\n  >>> jnp.triu_indices(3, m=2)\n  (Array([0, 0, 1], dtype=int32), Array([0, 1, 1], dtype=int32))\n\n  If ``k = 1``, the indices on and above the first sub-diagonal above the main\n  diagonal are returned.\n\n  >>> jnp.triu_indices(3, k=1)\n  (Array([0, 0, 1], dtype=int32), Array([1, 2, 2], dtype=int32))\n\n  If ``k = -1``, the indices on and above the first sub-diagonal below the main\n  diagonal are returned.\n\n  >>> jnp.triu_indices(3, k=-1)\n  (Array([0, 0, 0, 1, 1, 1, 2, 2], dtype=int32), Array([0, 1, 2, 0, 1, 2, 1, 2], dtype=int32))",
    "std_args": [
      {"name": "n", "type": "Any"},
      {"name": "k", "type": "Any"},
      {"name": "m", "type": "Any"},
    ],
    "variants": {},
  },
  "triu_indices_from": {
    "description": "Return the indices of upper triangle of a given array.\n\nJAX implementation of :func:`numpy.triu_indices_from`.\n\nArgs:\n  arr: input array. Must have ``arr.ndim == 2``.\n  k: optional, int, default=0. Specifies the sub-diagonal on and above which\n    the indices of upper triangle are returned. ``k=0`` refers to main diagonal,\n    ``k<0`` refers to sub-diagonal below the main diagonal and ``k>0`` refers\n    to sub-diagonal above the main diagonal.\n\nReturns:\n  A tuple of two arrays containing the indices of the upper triangle, one along\n  each axis.\n\nSee also:\n  - :func:`jax.numpy.tril_indices_from`: Returns the indices of lower triangle\n    of a given array.\n  - :func:`jax.numpy.triu_indices`: Returns the indices of upper triangle of an\n    array of size ``(n, m)``.\n  - :func:`jax.numpy.triu`: Return an upper triangle of an array.\n\nExamples:\n  >>> arr = jnp.array([[1, 2, 3],\n  ...                  [4, 5, 6],\n  ...                  [7, 8, 9]])\n  >>> jnp.triu_indices_from(arr)\n  (Array([0, 0, 0, 1, 1, 2], dtype=int32), Array([0, 1, 2, 1, 2, 2], dtype=int32))\n\n  Elements indexed by ``jnp.triu_indices_from`` correspond to those in the\n  output of ``jnp.triu``.\n\n  >>> ind = jnp.triu_indices_from(arr)\n  >>> arr[ind]\n  Array([1, 2, 3, 5, 6, 9], dtype=int32)\n  >>> jnp.triu(arr)\n  Array([[1, 2, 3],\n         [0, 5, 6],\n         [0, 0, 9]], dtype=int32)\n\n  When ``k > 0``:\n\n  >>> jnp.triu_indices_from(arr, k=1)\n  (Array([0, 0, 1], dtype=int32), Array([1, 2, 2], dtype=int32))\n\n  When ``k < 0``:\n\n  >>> jnp.triu_indices_from(arr, k=-1)\n  (Array([0, 0, 0, 1, 1, 1, 2, 2], dtype=int32), Array([0, 1, 2, 0, 1, 2, 1, 2], dtype=int32))",
    "std_args": [
      {"name": "arr", "type": "Any"},
      {"name": "k", "type": "Any"},
    ],
    "variants": {},
  },
  "true_divide": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "trunc": {
    "description": "Rounds each element ``x_i`` of the input array ``x`` to the nearest integer-valued number that is closer to zero than ``x_i``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "trunc_normal_": {
    "description": "Fill the input Tensor with values drawn from a truncated normal distribution.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
      {"name": "mean", "type": "Any"},
      {"name": "std", "type": "Any"},
      {"name": "a", "type": "Any"},
      {"name": "b", "type": "Any"},
      {"name": "generator", "type": "Any"},
    ],
    "variants": {},
  },
  "truncated_normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "type": {
    "description": "Casts all parameters and buffers to :attr:`dst_type`.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "dst_type", "type": "Any"},
    ],
    "variants": {},
  },
  "ufunc": {
    "description": "Universal functions which operation element-by-element on arrays.\n\nJAX implementation of :class:`numpy.ufunc`.\n\nThis is a class for JAX-backed implementations of NumPy's ufunc APIs.\nMost users will never need to instantiate :class:`ufunc`, but rather\nwill use the pre-defined ufuncs in :mod:`jax.numpy`.\n\nFor constructing your own ufuncs, see :func:`jax.numpy.frompyfunc`.\n\nExamples:\n  Universal functions are functions that apply element-wise to broadcasted\n  arrays, but they also come with a number of extra attributes and methods.\n\n  As an example, consider the function :obj:`jax.numpy.add`. The object\n  acts as a function that applies addition to broadcasted arrays in an\n  element-wise manner:\n\n  >>> x = jnp.array([1, 2, 3, 4, 5])\n  >>> jnp.add(x, 1)\n  Array([2, 3, 4, 5, 6], dtype=int32)\n\n  Each :class:`ufunc` object includes a number of attributes that describe\n  its behavior:\n\n  >>> jnp.add.nin  # number of inputs\n  2\n  >>> jnp.add.nout  # number of outputs\n  1\n  >>> jnp.add.identity  # identity value, or None if no identity exists\n  0\n\n  Binary ufuncs like :obj:`jax.numpy.add` include  number of methods to\n  apply the function to arrays in different manners.\n\n  The :meth:`~ufunc.outer` method applies the function to the\n  pair-wise outer-product of the input array values:\n\n  >>> jnp.add.outer(x, x)\n  Array([[ 2,  3,  4,  5,  6],\n         [ 3,  4,  5,  6,  7],\n         [ 4,  5,  6,  7,  8],\n         [ 5,  6,  7,  8,  9],\n         [ 6,  7,  8,  9, 10]], dtype=int32)\n\n  The :meth:`ufunc.reduce` method performs a reduction over the array.\n  For example, :meth:`jnp.add.reduce` is equivalent to ``jnp.sum``:\n\n  >>> jnp.add.reduce(x)\n  Array(15, dtype=int32)\n\n  The :meth:`ufunc.accumulate` method performs a cumulative reduction\n  over the array. For example, :meth:`jnp.add.accumulate` is equivalent\n  to :func:`jax.numpy.cumulative_sum`:\n\n  >>> jnp.add.accumulate(x)\n  Array([ 1,  3,  6, 10, 15], dtype=int32)\n\n  The :meth:`ufunc.at` method applies the function at particular indices in the\n  array; for ``jnp.add`` the computation is similar to :func:`jax.lax.scatter_add`:\n\n  >>> jnp.add.at(x, 0, 100, inplace=False)\n  Array([101,   2,   3,   4,   5], dtype=int32)\n\n  And the :meth:`ufunc.reduceat` method performs a number of ``reduce``\n  operations between specified indices of an array; for ``jnp.add`` the\n  operation is similar to :func:`jax.ops.segment_sum`:\n\n  >>> jnp.add.reduceat(x, jnp.array([0, 2]))\n  Array([ 3, 12], dtype=int32)\n\n  In this case, the first element is ``x[0:2].sum()``, and the second element\n  is ``x[2:].sum()``.",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "nin", "type": "Any"},
      {"name": "nout", "type": "Any"},
      {"name": "name", "type": "Any"},
      {"name": "nargs", "type": "Any"},
      {"name": "identity", "type": "Any"},
      {"name": "call", "type": "Any"},
      {"name": "reduce", "type": "Any"},
      {"name": "accumulate", "type": "Any"},
      {"name": "at", "type": "Any"},
      {"name": "reduceat", "type": "Any"},
    ],
    "variants": {},
  },
  "uint": {
    "description": "A JAX scalar constructor of type uint64.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "uint16": {
    "description": "A JAX scalar constructor of type uint16.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "uint2": {
    "description": "A JAX scalar constructor of type uint2.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "uint32": {
    "description": "A JAX scalar constructor of type uint32.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "uint4": {
    "description": "A JAX scalar constructor of type uint4.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "uint64": {
    "description": "A JAX scalar constructor of type uint64.\n\nWhile NumPy defines scalar types for each data type, JAX represents\nscalars as zero-dimensional arrays.",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "unflattened_size": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "unfold": {
    "description": "Extract sliding local blocks from a batched input tensor.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "kernel_size", "type": "Any"},
      {"name": "dilation", "type": "Any"},
      {"name": "padding", "type": "Any"},
      {"name": "stride", "type": "Any"},
    ],
    "variants": {},
  },
  "uniform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "union1d": {
    "description": "Compute the set union of two 1D arrays.\n\nJAX implementation of :func:`numpy.union1d`.\n\nBecause the size of the output of ``union1d`` is data-dependent, the function\nis not typically compatible with :func:`~jax.jit` and other JAX transformations.\nThe JAX version adds the optional ``size`` argument which must be specified\nstatically for ``jnp.union1d`` to be used in such contexts.\n\nArgs:\n  ar1: first array of elements to be unioned.\n  ar2: second array of elements to be unioned\n  size: if specified, return only the first ``size`` sorted elements. If there are fewer\n    elements than ``size`` indicates, the return value will be padded with ``fill_value``.\n  fill_value: when ``size`` is specified and there are fewer than the indicated number of\n    elements, fill the remaining entries ``fill_value``. Defaults to the minimum value.\n\nReturns:\n  an array containing the union of elements in the input array.\n\nSee also:\n  - :func:`jax.numpy.intersect1d`: the set intersection of two 1D arrays.\n  - :func:`jax.numpy.setxor1d`: the set XOR of two 1D arrays.\n  - :func:`jax.numpy.setdiff1d`: the set difference of two 1D arrays.\n\nExamples:\n  Computing the union of two arrays:\n\n  >>> ar1 = jnp.array([1, 2, 3, 4])\n  >>> ar2 = jnp.array([3, 4, 5, 6])\n  >>> jnp.union1d(ar1, ar2)\n  Array([1, 2, 3, 4, 5, 6], dtype=int32)\n\n  Because the output shape is dynamic, this will fail under :func:`~jax.jit` and other\n  transformations:\n\n  >>> jax.jit(jnp.union1d)(ar1, ar2)  # doctest: +IGNORE_EXCEPTION_DETAIL\n  Traceback (most recent call last):\n     ...\n  ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[4].\n  The error occurred while tracing the function union1d at /Users/vanderplas/github/jax-ml/jax/jax/_src/numpy/setops.py:101 for jit. This concrete value was not available in Python because it depends on the value of the argument ar1.\n\n  In order to ensure statically-known output shapes, you can pass a static ``size``\n  argument:\n\n  >>> jit_union1d = jax.jit(jnp.union1d, static_argnames=['size'])\n  >>> jit_union1d(ar1, ar2, size=6)\n  Array([1, 2, 3, 4, 5, 6], dtype=int32)\n\n  If ``size`` is too small, the union is truncated:\n\n  >>> jit_union1d(ar1, ar2, size=4)\n  Array([1, 2, 3, 4], dtype=int32)\n\n  If ``size`` is too large, then the output is padded with ``fill_value``:\n\n  >>> jit_union1d(ar1, ar2, size=8, fill_value=0)\n  Array([1, 2, 3, 4, 5, 6, 0, 0], dtype=int32)",
    "std_args": [
      {"name": "ar1", "type": "Any"},
      {"name": "ar2", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "fill_value", "type": "Any"},
    ],
    "variants": {},
  },
  "unique_all": {
    "description": "Returns the unique elements of an input array ``x``, the first occurring indices for each unique element in ``x``, the indices from the set of unique elements that reconstruct ``x``, and the corresponding counts for each unique element in ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "unique_counts": {
    "description": "Returns the unique elements of an input array ``x`` and the corresponding counts for each unique element in ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "unique_inverse": {
    "description": "Returns the unique elements of an input array ``x`` and the indices from the set of unique elements that reconstruct ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "unique_values": {
    "description": "Returns the unique elements of an input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
    ],
    "variants": {},
  },
  "unravel_index": {
    "description": "Convert flat indices into multi-dimensional indices.\n\nJAX implementation of :func:`numpy.unravel_index`. The JAX version differs in\nits treatment of out-of-bound indices: unlike NumPy, negative indices are\nsupported, and out-of-bound indices are clipped to the nearest valid value.\n\nArgs:\n  indices: integer array of flat indices\n  shape: shape of multidimensional array to index into\n\nReturns:\n  Tuple of unraveled indices\n\nSee also:\n  :func:`jax.numpy.ravel_multi_index`: Inverse of this function.\n\nExamples:\n  Start with a 1D array values and indices:\n\n  >>> x = jnp.array([2., 3., 4., 5., 6., 7.])\n  >>> indices = jnp.array([1, 3, 5])\n  >>> print(x[indices])\n  [3. 5. 7.]\n\n  Now if ``x`` is reshaped, ``unravel_indices`` can be used to convert\n  the flat indices into a tuple of indices that access the same entries:\n\n  >>> shape = (2, 3)\n  >>> x_2D = x.reshape(shape)\n  >>> indices_2D = jnp.unravel_index(indices, shape)\n  >>> indices_2D\n  (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))\n  >>> print(x_2D[indices_2D])\n  [3. 5. 7.]\n\n  The inverse function, ``ravel_multi_index``, can be used to obtain the\n  original indices:\n\n  >>> jnp.ravel_multi_index(indices_2D, shape)\n  Array([1, 3, 5], dtype=int32)",
    "std_args": [
      {"name": "indices", "type": "Any"},
      {"name": "shape", "type": "Any"},
    ],
    "variants": {},
  },
  "unsignedinteger": {
    "description": "The class representing a Python class.",
    "std_args": [],
    "variants": {},
  },
  "unstack": {
    "description": "Splits an array into a sequence of arrays along the given axis.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "unwrap_random_key_data": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "update": {
    "description": "Update the :class:`~torch.nn.ParameterDict` with key-value pairs from ``parameters``, overwriting existing keys.",
    "std_args": [
      {"name": "self", "type": "Any"},
      {"name": "parameters", "type": "Any"},
    ],
    "variants": {},
  },
  "update_infinity_moment": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "update_moment": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "update_moment_per_elem_norm": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "upper": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "upsample_bilinear": {
    "description": "Upsamples the input, using bilinear upsampling.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "scale_factor", "type": "Any"},
    ],
    "variants": {},
  },
  "upsample_nearest": {
    "description": "Upsamples the input, using nearest neighbours' pixel values.",
    "std_args": [
      {"name": "input", "type": "Any"},
      {"name": "size", "type": "Any"},
      {"name": "scale_factor", "type": "Any"},
    ],
    "variants": {},
  },
  "upscale_factor": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "value": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "value_and_grad": {
    "description": "Evaluates value and gradient.",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "argnums", "type": "Any"},
      {"name": "has_aux", "type": "Any"},
    ],
    "variants": {},
  },
  "values": {
    "description": "Return an iterable of the ParameterDict values.",
    "std_args": [
      {"name": "self", "type": "Any"},
    ],
    "variants": {},
  },
  "vander": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "var": {
    "description": "Calculates the variance of the input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "correction", "type": "Union[int, float]"},
      {"name": "keepdims", "type": "bool"},
    ],
    "variants": {},
  },
  "variant": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "vdot": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "vecdot": {
    "description": "Computes the (vector) dot product of two arrays.",
    "std_args": [
      {"name": "x1", "type": "array"},
      {"name": "x2", "type": "array"},
      {"name": "axis", "type": "int"},
    ],
    "variants": {},
  },
  "vector_norm": {
    "description": "Computes the vector norm of a vector (or batch of vectors) ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "axis", "type": "Optional[Union[int, Tuple[int, Ellipsis]]]"},
      {"name": "keepdims", "type": "bool"},
      {"name": "ord", "type": "Union[int, float, Literal[inf, Any]]"},
    ],
    "variants": {},
  },
  "vectorize": {
    "description": "Define a vectorized function with broadcasting.\n\n:func:`vectorize` is a convenience wrapper for defining vectorized\nfunctions with broadcasting, in the style of NumPy's\n`generalized universal functions <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.\nIt allows for defining functions that are automatically repeated across\nany leading dimensions, without the implementation of the function needing to\nbe concerned about how to handle higher dimensional inputs.\n\n:func:`jax.numpy.vectorize` has the same interface as\n:class:`numpy.vectorize`, but it is syntactic sugar for an auto-batching\ntransformation (:func:`vmap`) rather than a Python loop. This should be\nconsiderably more efficient, but the implementation must be written in terms\nof functions that act on JAX arrays.\n\nArgs:\n  pyfunc: function to vectorize.\n  excluded: optional set of integers representing positional arguments for\n    which the function will not be vectorized. These will be passed directly\n    to ``pyfunc`` unmodified.\n  signature: optional generalized universal function signature, e.g.,\n    ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If\n    provided, ``pyfunc`` will be called with (and expected to return) arrays\n    with shapes given by the size of corresponding core dimensions. By\n    default, pyfunc is assumed to take scalar arrays as input, and if\n    ``signature`` is ``None``, ``pyfunc`` can produce outputs of any shape.\n\nReturns:\n  Vectorized version of the given function.\n\nExamples:\n  Here are a few examples of how one could write vectorized linear algebra\n  routines using :func:`vectorize`:\n\n  >>> from functools import partial\n\n  >>> @partial(jnp.vectorize, signature='(k),(k)->(k)')\n  ... def cross_product(a, b):\n  ...   assert a.shape == b.shape and a.ndim == b.ndim == 1\n  ...   return jnp.array([a[1] * b[2] - a[2] * b[1],\n  ...                     a[2] * b[0] - a[0] * b[2],\n  ...                     a[0] * b[1] - a[1] * b[0]])\n\n  >>> @partial(jnp.vectorize, signature='(n,m),(m)->(n)')\n  ... def matrix_vector_product(matrix, vector):\n  ...   assert matrix.ndim == 2 and matrix.shape[1:] == vector.shape\n  ...   return matrix @ vector\n\n  These functions are only written to handle 1D or 2D arrays (the ``assert``\n  statements will never be violated), but with vectorize they support\n  arbitrary dimensional inputs with NumPy style broadcasting, e.g.,\n\n  >>> cross_product(jnp.ones(3), jnp.ones(3)).shape\n  (3,)\n  >>> cross_product(jnp.ones((2, 3)), jnp.ones(3)).shape\n  (2, 3)\n  >>> cross_product(jnp.ones((1, 2, 3)), jnp.ones((2, 1, 3))).shape\n  (2, 2, 3)\n  >>> matrix_vector_product(jnp.ones(3), jnp.ones(3))  # doctest: +IGNORE_EXCEPTION_DETAIL\n  Traceback (most recent call last):\n  ValueError: input with shape (3,) does not have enough dimensions for all\n  core dimensions ('n', 'k') on vectorized function with excluded=frozenset()\n  and signature='(n,k),(k)->(k)'\n  >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones(3)).shape\n  (2,)\n  >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones((4, 3))).shape\n  (4, 2)\n\n  Note that this has different semantics than `jnp.matmul`:\n\n  >>> jnp.matmul(jnp.ones((2, 3)), jnp.ones((4, 3)))  # doctest: +IGNORE_EXCEPTION_DETAIL\n  Traceback (most recent call last):\n  TypeError: dot_general requires contracting dimensions to have the same shape, got [3] and [4].",
    "std_args": [
      {"name": "pyfunc", "type": "Any"},
      {"name": "excluded", "type": "Any"},
      {"name": "signature", "type": "Any"},
    ],
    "variants": {},
  },
  "vectorized_map": {
    "description": "Parallel map of `function` on axis 0 of tensor(s) `elements`.\n\nSchematically, `vectorized_map` implements the following,\nin the case of a single tensor input `elements`:\n\n```python\ndef vectorized_map(function, elements):\n    outputs = []\n    for e in elements:\n        outputs.append(function(e))\n    return np.stack(outputs)\n```\n\nIn the case of an iterable of tensors `elements`,\nit implements the following:\n\n```python\ndef vectorized_map(function, elements):\n    batch_size = elements[0].shape[0]\n    outputs = []\n    for index in range(batch_size):\n        outputs.append(function([e[index] for e in elements]))\n    return np.stack(outputs)\n```\n\nIn this case, `function` is expected to take as input\na single list of tensor arguments.",
    "std_args": [
      {"name": "function", "type": "Any"},
      {"name": "elements", "type": "Any"},
    ],
    "variants": {},
  },
  "view_as_complex": {
    "description": "Converts a real tensor with shape `(..., 2)` to a complex tensor,\nwhere the last dimension represents the real and imaginary components\nof a complex tensor.\n\nArgs:\n    x: A real tensor with last dimension of size 2.\n\nReturns:\n    A complex tensor with shape `x.shape[:-1]`.\n\nExample:\n\n```\n>>> import numpy as np\n>>> from keras import ops\n\n>>> real_imag = np.array([[1.0, 2.0], [3.0, 4.0]])\n>>> complex_tensor = ops.view_as_complex(real_imag)\n>>> complex_tensor\narray([1.+2.j, 3.+4.j])\n```",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "view_as_real": {
    "description": "Converts a complex tensor to a real tensor with shape `(..., 2)`,\nwhere the last dimension represents the real and imaginary components.\n\nArgs:\n    x: A complex tensor.\n\nReturns:\n    A real tensor where the last dimension contains the\n    real and imaginary parts.\n\nExample:\n```\n>>> import numpy as np\n>>> from keras import ops\n\n>>> complex_tensor = np.array([1 + 2j, 3 + 4j])\n>>> real = ops.view_as_real(complex_tensor)\n>>> real\narray([[1., 2.],\n       [3., 4.]])\n```",
    "std_args": [
      {"name": "x", "type": "Any"},
    ],
    "variants": {},
  },
  "vmap": {
    "description": "Vectorizing map.",
    "std_args": [
      {"name": "func", "type": "Any"},
      {"name": "in_axes", "type": "Any"},
      {"name": "out_axes", "type": "Any"},
      {"name": "randomness", "type": "Any"},
    ],
    "variants": {},
  },
  "vsplit": {
    "description": "Split an array into sub-arrays vertically.\n\nJAX implementation of :func:`numpy.vsplit`.\n\nRefer to the documentation of :func:`jax.numpy.split` for details; ``vsplit`` is\nequivalent to ``split`` with ``axis=0``.\n\nExamples:\n  1D array:\n\n  >>> x = jnp.array([1, 2, 3, 4, 5, 6])\n  >>> x1, x2 = jnp.vsplit(x, 2)\n  >>> print(x1, x2)\n  [1 2 3] [4 5 6]\n\n  2D array:\n\n  >>> x = jnp.array([[1, 2, 3, 4],\n  ...                [5, 6, 7, 8]])\n  >>> x1, x2 = jnp.vsplit(x, 2)\n  >>> print(x1, x2)\n  [[1 2 3 4]] [[5 6 7 8]]\n\nSee also:\n  - :func:`jax.numpy.split`: split an array along any axis.\n  - :func:`jax.numpy.hsplit`: split horizontally, i.e. along axis=1\n  - :func:`jax.numpy.dsplit`: split depth-wise, i.e. along axis=2\n  - :func:`jax.numpy.array_split`: like ``split``, but allows ``indices_or_sections``\n    to be an integer that does not evenly divide the size of the array.",
    "std_args": [
      {"name": "ary", "type": "Any"},
      {"name": "indices_or_sections", "type": "Any"},
    ],
    "variants": {},
  },
  "vstack": {
    "description": "Vertically stack arrays.\n\nJAX implementation of :func:`numpy.vstack`.\n\nFor arrays of two or more dimensions, this is equivalent to\n:func:`jax.numpy.concatenate` with ``axis=0``.\n\nArgs:\n  tup: a sequence of arrays to stack; each must have the same shape along all\n    but the first axis. If a single array is given it will be treated\n    equivalently to `tup = unstack(tup)`, but the implementation will avoid\n    explicit unstacking.\n  dtype: optional dtype of the resulting array. If not specified, the dtype\n    will be determined via type promotion rules described in :ref:`type-promotion`.\n\nReturns:\n  the stacked result.\n\nSee also:\n  - :func:`jax.numpy.stack`: stack along arbitrary axes\n  - :func:`jax.numpy.concatenate`: concatenation along existing axes.\n  - :func:`jax.numpy.hstack`: stack horizontally, i.e. along axis 1.\n  - :func:`jax.numpy.dstack`: stack depth-wise, i.e. along axis 2.\n\nExamples:\n  Scalar values:\n\n  >>> jnp.vstack([1, 2, 3])\n  Array([[1],\n         [2],\n         [3]], dtype=int32, weak_type=True)\n\n  1D arrays:\n\n  >>> x = jnp.arange(4)\n  >>> y = jnp.ones(4)\n  >>> jnp.vstack([x, y])\n  Array([[0., 1., 2., 3.],\n         [1., 1., 1., 1.]], dtype=float32)\n\n  2D arrays:\n\n  >>> x = x.reshape(1, 4)\n  >>> y = y.reshape(1, 4)\n  >>> jnp.vstack([x, y])\n  Array([[0., 1., 2., 3.],\n         [1., 1., 1., 1.]], dtype=float32)",
    "std_args": [
      {"name": "tup", "type": "Any"},
      {"name": "dtype", "type": "Any"},
    ],
    "variants": {},
  },
  "wald": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "warmup_constant_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "warmup_cosine_decay_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "warmup_exponential_decay_schedule": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "weight": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "weight_hh": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "weight_ih": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "where": {
    "description": "Returns elements chosen from ``x1`` or ``x2`` depending on ``condition``.",
    "std_args": [
      {"name": "condition", "type": "array"},
      {"name": "x1", "type": "Union[array, int, float, complex, bool]"},
      {"name": "x2", "type": "Union[array, int, float, complex, bool]"},
    ],
    "variants": {},
  },
  "while_loop": {
    "description": "A Flax NNX transformation of `jax.lax.while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html>`_.",
    "std_args": [
      {"name": "cond_fun", "type": "Any"},
      {"name": "body_fun", "type": "Any"},
      {"name": "init_val", "type": "Any"},
    ],
    "variants": {},
  },
  "xavier_normal": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "xavier_uniform": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "xlog1py": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "zero_grad": {
    "description": "Sets the gradients of all optimized parameters to zero.",
    "std_args": [],
    "variants": {},
  },
  "zero_nans": {
    "description": "The class representing a Python module/class/instance attribute.",
    "std_args": [],
    "variants": {},
  },
  "zeros": {
    "description": "Returns a new array having a specified ``shape`` and filled with zeros.",
    "std_args": [
      {"name": "shape", "type": "Union[int, Tuple[int, Ellipsis]]"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
  "zeros_": {
    "description": "Fill the input Tensor with the scalar value `0`.",
    "std_args": [
      {"name": "tensor", "type": "Any"},
    ],
    "variants": {},
  },
  "zeros_like": {
    "description": "Returns a new array filled with zeros and having the same ``shape`` as an input array ``x``.",
    "std_args": [
      {"name": "x", "type": "array"},
      {"name": "dtype", "type": "Optional[dtype]"},
      {"name": "device", "type": "Optional[device]"},
    ],
    "variants": {},
  },
}
