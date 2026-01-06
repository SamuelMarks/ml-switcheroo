"""
Internal Standards Definition (Hub of the Knowledge Base).

This module defines the "Golden Set" of abstract operations (The Hub).
It defines the **Abstract Schema**: Standard argument names (`std_args`) and docstrings.
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
  "max": {"description": "Element-wise maximum or reduction.", "std_args": ["x"]},
  "min": {"description": "Element-wise minimum or reduction.", "std_args": ["x"]},
  "relu": {"description": "Rectified Linear Unit.", "std_args": ["x"]},
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
  "data_ptr": {"description": "Get valid data pointer or buffer access.", "std_args": []},
  "CastFloat": {"description": "Cast tensor to float32", "std_args": ["x"], "metadata": {"target_type": "Float32"}},
  "CastDouble": {"description": "Cast tensor to float64", "std_args": ["x"], "metadata": {"target_type": "Float64"}},
  "CastHalf": {"description": "Cast tensor to float16", "std_args": ["x"], "metadata": {"target_type": "Float16"}},
  "CastLong": {"description": "Cast tensor to int64", "std_args": ["x"], "metadata": {"target_type": "Int64"}},
  "CastInt": {"description": "Cast tensor to int32", "std_args": ["x"], "metadata": {"target_type": "Int32"}},
  "CastShort": {"description": "Cast tensor to int16", "std_args": ["x"], "metadata": {"target_type": "Int16"}},
  "CastByte": {"description": "Cast tensor to uint8", "std_args": ["x"], "metadata": {"target_type": "UInt8"}},
  "CastBool": {"description": "Cast tensor to bool", "std_args": ["x"], "metadata": {"target_type": "Bool"}},
  "CastChar": {"description": "Cast tensor to int8/char", "std_args": ["x"], "metadata": {"target_type": "Int8"}},
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
    "std_args": ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
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
    "std_args": ["params", "lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad"],
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
  "Resize": {"description": "Resize the input image to the given size.", "std_args": ["size"]},
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
  "CudaAvailable": {  # <--- NEW: Wired Orphan
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
}
