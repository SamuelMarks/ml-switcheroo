# The Limits of Static Analysis in Deep Learning Source-to-Source Compilation

## Introduction: The Fundamental Impasse

Translating deep learning models across frameworks (e.g., PyTorch to JAX) via static code analysis is fundamentally a source-to-source compilation problem. However, this translation is severely constrained by the architectural paradigm of modern "define-by-run" frameworks.

Frameworks like PyTorch embed a domain-specific language (DSL) for tensor operations within a Turing-complete host language (Python). Consequently, determining the exact computational graph—including the operations to be executed, tensor shapes, and data types—is generally **undecidable** at compile-time (prior to execution). This is analogous to the halting problem: a static analyzer evaluating an Abstract Syntax Tree (AST) cannot definitively resolve program state without symbolically executing the program.

Conversely, frameworks like JAX and TensorFlow (in Graph mode) operate on a "define-and-run" or functional paradigm. They expect to trace a function into an intermediate representation (IR, such as HLO or Jaxpr) using static shapes ahead-of-time (AOT) or just-in-time (JIT).

Because a purely static AST converter cannot observe runtime values (data), it fails whenever the host language's syntax depends on the tensor state.

---

## How `ml-switcheroo` Operates (The CST Approach)

To understand these limitations practically, we must look at how `ml-switcheroo` is architected. Unlike tracing compilers (`torch.fx` or `jax.make_jaxpr`) which execute the code with dummy inputs to capture a flattened graph, `ml-switcheroo` operates entirely offline via **Concrete Syntax Trees (CST)** and static semantic rewriting.

The `ml-switcheroo` pipeline generally follows these steps:
1. **Parsing:** Python source is parsed into a CST (preserving comments, formatting, and original variable names).
2. **Semantic Discovery & Rewriting:** The CST is traversed. Plugins and rewrite strategies (`src/ml_switcheroo/rewriter/`) identify framework-specific API calls (e.g., `torch.nn.Linear`) and attempt to map them to an intermediate representation closely aligned with MLIR or StableHLO semantics.
3. **Code Generation:** The abstract representation is emitted as idiomatic Python code for the target framework (`src/ml_switcheroo/codegen/`).

**The Trade-off:** `ml-switcheroo` intentionally chooses this path to preserve **source-code fidelity**. A traced graph destroys `if` statements, unrolls `for` loops, deletes comments, and mangles variable names, resulting in unmaintainable output. `ml-switcheroo` outputs readable, human-maintainable code. However, this CST-first approach means `ml-switcheroo` is inherently blind to dynamic runtime states.

Below are the technical paradigms where `ml-switcheroo`'s static CST rewriting breaks down, necessitating either symbolic execution, runtime tracing, or manual architectural rewrites by the user.

---

## 1. Data-Dependent Shapes (e.g., Boolean Masking, NMS)

In operations like Non-Maximum Suppression (NMS) or boolean masking, the output shape of a tensor depends entirely on the *values* computed during the forward pass, not on the input shape.

**How it breaks `ml-switcheroo`:**
The rewriter can statically infer that `Linear(10, 5)` outputs a shape of `(B, 5)` through simple type inference. However, it cannot predict how many elements will evaluate to `True` in a boolean mask. PyTorch naturally handles dynamic shapes, whereas JAX requires shapes to be statically bound at compile time. `ml-switcheroo` will generate valid-looking JAX syntax for the mask, but the resulting code will immediately fail JAX's compilation constraints at runtime.

```python
import torch
import torch.nn as nn


class DataDependentMasking(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(10, 10)

  def forward(self, x):
    out = self.fc(x)

    # 🚨 DYNAMIC SHAPE: The size of 'filtered_out' is determined by runtime values.
    mask = out > 0
    filtered_out = out[mask]

    # Static CST analysis cannot resolve this branch safely for AOT compilation.
    if filtered_out.numel() > 0:
      return filtered_out.mean()
    else:
      return torch.tensor(0.0)
```

## 2. Value-Dependent Control Flow (e.g., Dynamic Routing, MoE)

In Mixture of Experts (MoE) or dynamic routing networks, the network decides which architectural branches to execute based on intermediate tensor values.

**How it breaks `ml-switcheroo`:**
In eager PyTorch, standard Python `if/else` statements act as the control flow mechanism. `ml-switcheroo` sees a CST `IfNode` and translates it directly to a Python `if/else` block in the target language. However, when JAX traces this generated function for JIT compilation, the condition evaluates to an abstract tracer (representing a future tensor), raising a `ConcretizationTypeError`. To fix this statically, `ml-switcheroo` would need to structurally transform Python `if` statements into `jax.lax.cond`—a highly complex rewrite that often conflicts with variable scoping rules in standard Python.

```python
import torch
import torch.nn as nn


class DynamicRoutingMoE(nn.Module):
  def __init__(self):
    super().__init__()
    self.expert_1 = nn.Linear(10, 5)
    self.expert_2 = nn.Linear(10, 5)
    self.router = nn.Linear(10, 1)

  def forward(self, x):
    route_score = torch.sigmoid(self.router(x))

    # 🚨 DYNAMIC CONTROL FLOW: Branch resolution requires concretized values.
    # Translating this statically to JAX will fail during JIT tracing.
    if route_score.mean() > 0.5:
      return self.expert_1(x)
    else:
      return self.expert_2(x)
```

## 3. Data-Dependent Loops and Autoregressive Generation

Sequence generation tasks (e.g., Beam Search, RNN unrolling based on sequence end tokens) often rely on unbounded loops that terminate when a specific tensor value is reached.

**How it breaks `ml-switcheroo`:**
Similar to `if/else`, Python `while` loops that check a tensor's state cannot be traced by functional JIT compilers. Furthermore, static analysis cannot guarantee loop-carried dependency safety. Translating this to JAX requires a structural rewrite to `jax.lax.while_loop`, which has strict requirements about loop state structures (carry shapes must remain invariant). A CST parser cannot reliably enforce or infer these invariants without executing the loop.

```python
import torch


class AdaptiveComputationTime(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.cell = torch.nn.Linear(10, 10)
    self.halting_layer = torch.nn.Linear(10, 1)

  def forward(self, x):
    state = x
    ponder_cost = 0.0

    # 🚨 UNBOUNDED LOOP: Condition relies on a dynamically computed tensor value.
    while ponder_cost < 0.9:
      state = self.cell(state)
      halt_prob = torch.sigmoid(self.halting_layer(state))
      ponder_cost += halt_prob.item()  # .item() forces a graph break in JIT

    return state
```

## 4. Imperative State Mutations and Aliasing

PyTorch extensively supports in-place operations (`add_()`) and tensor view mutations (`tensor[:, 0] = 1`).

**How it breaks `ml-switcheroo`:**
JAX requires pure functions and immutable data structures (using `tensor.at[...].set(...)`). To translate imperative mutations safely, `ml-switcheroo` would need to perform exhaustive **Alias Analysis** to ensure that mutating one variable doesn't silently alter another variable pointing to the same memory block. Because Python allows highly dynamic aliasing (e.g., passing tensors through nested lists, dicts, or `*args`), perfectly safe static translation of mutations across a CST is computationally intractable.

```python
import torch


def imperative_mutation(x):
  # 'y' is a view of 'x'. They share memory.
  y = x.view(-1)

  # 🚨 IN-PLACE MUTATION: This modifies 'x' implicitly.
  # ml-switcheroo translating line-by-line might miss that
  # 'x' must also be functionally updated and returned in the target framework.
  y[0] = 100.0

  return x.sum()
```

## 5. Autograd in the Forward Pass (Meta-Learning)

In algorithms like WGAN-GP or MAML, the network invokes the automatic differentiation engine *during* the forward pass.

**How it breaks `ml-switcheroo`:**
When `ml-switcheroo` inspects the CST, it only sees the forward network definition; the massive computational graph built dynamically by the Autograd engine is completely invisible to static analysis. Furthermore, translating PyTorch's implicitly global `torch.autograd.grad` to JAX (`jax.grad` or `jax.vjp`) requires converting the targeted sub-graph into an explicitly pure, stateless Python function. This is a major architectural refactoring that goes far beyond syntax translation.

```python
import torch
import torch.nn as nn


class GradientPenaltyNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(10, 50), nn.Linear(50, 1))

  def forward(self, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

    disc_interpolates = self.net(interpolates)

    # 🚨 DYNAMIC GRAPH RESOLUTION: The operations to calculate 'gradients'
    # are not in the AST. They are generated dynamically by traversing the backward graph.
    gradients = torch.autograd.grad(
      outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones_like(disc_interpolates), create_graph=True
    )[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
```

## Conclusion

`ml-switcheroo` is optimized for the ~90% of standard deep learning layers (e.g., Transformers, ResNets) where static topology holds true. By strictly operating on the Concrete Syntax Tree (CST), it successfully preserves code readability, comments, and structure—a critical requirement for maintainable code migrations.

However, static AST/CST converters operate entirely on **Syntax**, whereas advanced deep learning logic often operates on **State & Data**. Whenever the *data* dictates the *syntax* (which layers execute, loop bounds, aliasing, or dynamic graph construction), purely static source-to-source translation reaches its theoretical limit. Bridging this remaining gap inherently requires either manual user intervention or abandoning code readability in favor of runtime graph tracing.
