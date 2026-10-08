# nn.Module Fundamentals

## Goal

I already understand:

```text
Tensor
    ↓
Autograd
    ↓
Gradients
```

Now I want to understand how PyTorch organizes an actual neural network.

The goal is not just to know how to write:

```python
class Model(nn.Module):
    ...
```

I want to understand:

- Why `nn.Module` exists.
- What a Module represents.
- What a Submodule is.
- What `nn.Parameter` is.
- Why `requires_grad=True` and `nn.Parameter` are different concepts.
- How Parameters are registered.
- Why `super().__init__()` is required.
- What `forward()` does.
- What happens when I call `model(x)`.
- How `nn.Linear` works.
- How to derive Linear weight and bias shapes.
- What `parameters()` returns.
- What `named_parameters()` returns.
- What a Buffer is.
- Parameter vs Buffer.
- What Module "state" means.
- What "registration" means.
- What `state_dict()` represents.

The most important goal is to have clear terminology.

---

# 1. Why Do We Need `nn.Module`?

I can already create a trainable tensor using:

```python
w = torch.tensor(
    1.0,
    requires_grad=True
)
```

Autograd can calculate:

```text
w.grad
```

without requiring `nn.Module`.

For a tiny model:

```text
y = wx + b
```

I could manually create:

```python
w = torch.tensor(
    1.0,
    requires_grad=True
)

b = torch.tensor(
    0.0,
    requires_grad=True
)
```

Then perform:

```text
forward
    ↓
loss
    ↓
backward
    ↓
w.grad
b.grad
```

This works.

The problem becomes organization as models become larger.

A neural network can contain:

```text
weight1
bias1

weight2
bias2

weight3
bias3

...

millions or billions of parameters
```

I need a structured way to:

- Group parameters.
- Group layers.
- Define forward computation.
- Discover parameters.
- Move model state between devices.
- Save and restore model state.
- Compose smaller components into larger models.

This is the problem `nn.Module` helps solve.

---

# 2. First Mental Model of a Module

My simplest mental model is:

> A Module packages computation together with the state needed by that
> computation.

Conceptually:

```text
                    nn.Module
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
       Computation                 State
            │                       │
            ▼                       ▼
        forward()           things the Module
                              must remember
```

For a Linear layer:

```text
                     Linear
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
       Computation               State

       y = xWᵀ + b          weight + bias
```

This is my main anchor for understanding `nn.Module`.

---

# 3. What Is a Module?

`nn.Module` is PyTorch's base abstraction for neural-network layers and
models.

Examples of Modules include:

```python
nn.Linear(...)
nn.Conv2d(...)
nn.ReLU(...)
```

My own model can also be a Module:

```python
class MyModel(nn.Module):
    ...
```

So conceptually:

```text
Module
=
building block of a PyTorch model
```

A Module can be:

```text
one layer
```

or:

```text
an entire neural network
```

or:

```text
a component containing many other Modules
```

---

# 4. Modules Can Contain Modules

Consider:

```python
class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(5, 3)
        self.layer2 = nn.Linear(3, 1)
```

Here:

```text
MyModel
```

is a Module.

But:

```text
layer1
layer2
```

are also Modules because `nn.Linear` itself is an `nn.Module`.

Therefore:

```text
                   MyModel
                   Module
                  /      \
                 /        \
                ▼          ▼
             layer1      layer2
             Module      Module
```

A Module contained inside another Module is called a:

```text
Submodule
```

Therefore:

> A Submodule is simply a Module contained inside another Module.

---

# 5. Module Tree

Because Modules can contain other Modules, a neural network naturally forms
a tree.

Example:

```text
                         MyModel
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
          layer1                          layer2
        nn.Linear                       nn.Linear
```

Each Linear layer itself contains state:

```text
                         MyModel
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
          layer1                          layer2
            │                               │
       ┌────┴────┐                     ┌────┴────┐
       │         │                     │         │
       ▼         ▼                     ▼         ▼
    weight      bias                weight      bias
```

This tree structure is fundamental to how PyTorch manages large models.

---

# 6. Parameters

Consider:

```python
layer = nn.Linear(5, 3)
```

This layer has:

```text
weight
bias
```

These are the learnable state of the layer.

PyTorch represents this kind of state using:

```python
nn.Parameter
```

My mental model is:

> A Parameter is tensor state that is registered as a model parameter.

Usually Parameters represent learnable quantities such as:

```text
weights
biases
```

So:

```text
Linear
│
├── weight → Parameter
└── bias   → Parameter
```

---

# 7. `requires_grad` vs `nn.Parameter`

This distinction is very important.

Consider:

```python
w = torch.tensor(
    1.0,
    requires_grad=True
)
```

Autograd understands:

```text
I need gradients with respect to w.
```

Therefore:

```text
requires_grad
      ↓
Autograd concern
```

Now consider:

```python
w = nn.Parameter(
    torch.tensor(1.0)
)
```

This communicates:

```text
This tensor represents a Parameter
belonging to a Module.
```

Therefore:

```text
nn.Parameter
      ↓
Module-registration concern
```

These are different concepts.

---

# 8. Autograd and Module Registration Are Different Systems

A useful separation is:

```text
AUTOGRAD

requires_grad=True
        ↓
track computation
        ↓
calculate gradients
        ↓
tensor.grad
```

versus:

```text
MODULE SYSTEM

nn.Parameter
        ↓
register as model Parameter
        ↓
discover through model.parameters()
        ↓
can be passed to optimizer
```

Therefore:

```text
requires_grad=True
```

does NOT automatically mean:

```text
registered Module Parameter
```

---

# 9. Why Not Just Use Normal Tensors?

Consider:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.weight = torch.tensor(
            1.0,
            requires_grad=True
        )
```

Autograd understands that gradients can be calculated for `weight`.

But simply assigning an ordinary Tensor does not make that Tensor a
registered Parameter of the Module.

Compare:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor(1.0)
        )
```

Now PyTorch understands:

```text
weight
   │
   ├── participates in Autograd
   │
   └── is registered as a Module Parameter
```

This registration allows Parameter discovery through the Module hierarchy.

---

# 10. `nn.Parameter` Defaults to Gradient Tracking

Normally:

```python
p = nn.Parameter(
    torch.tensor(1.0)
)
```

has:

```text
p.requires_grad = True
```

This makes sense because Parameters generally represent learnable model
quantities.

However:

```python
p = nn.Parameter(
    torch.tensor(1.0),
    requires_grad=False
)
```

is also valid.

This leads to an important distinction:

```text
Parameter?
     │
     └── What role does this tensor have inside the Module?


requires_grad?
     │
     └── Should Autograd compute gradients with respect to it?
```

These are independent concepts.

---

# 11. Parameter Does Not Mean Optimizer Automatically Updates It

One confusion I had was:

> If I create an `nn.Parameter`, the optimizer automatically finds it and
> updates it.

That is not quite right.

Usually I explicitly write something like:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

The chain is:

```text
nn.Parameter
      ↓
registered with Module
      ↓
model.parameters()
      ↓
passed to optimizer
      ↓
optimizer now knows which Parameters to manage
```

The optimizer does not randomly scan all Python objects looking for
Parameters.

---

# 12. What Does `forward()` Do?

Consider:

```python
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(
            torch.tensor(1.0)
        )

        self.b = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, x):
        return self.w * x + self.b
```

The `forward()` method defines the Module's computation.

Conceptually:

```text
input x
   │
   ▼
w*x + b
   │
   ▼
output
```
*The Module's state is:

```text
w
*
```

The Module's computation is:*
```text
w*x + b
```

---

# 13. F*rward Does Not Mean Backward

An i*portant distinction:

```text
forw*rd()
    ↓
calculate output values*```

If Autograd is enabled:

```t*xt
forward()
    ↓
calculate value*
+
record operations needed for ba*kward
```

But `forward()` itself *oes not compute:

```text
∂loss
──*───
∂weight
```

Gradient computat*on occurs later through:

```pytho*
loss.backward()
```

So:

```text*FORWARD

compute values
+
possibly*build Autograd graph
```

versus:
*```text
BACKWARD

traverse Autogra* graph
+
compute gradients
```

--*

# 14. `model(x)` vs `model.forwa*d(x)`

Normally models are called *ike:

```python
output = model(x)
*``

rather than:

```python
output*= model.forward(x)
```

When:

```*ython
model(x)
```

is executed, P*thon invokes the Module's callable*machinery.

Conceptually:

```text*model(x)
    │
    ▼
nn.Module.__c*ll__()
    │
    ├── PyTorch Modul* machinery
    │
    ▼
forward(x)
*   │
    ▼
output
```

The importa*t mental model is:

> `forward()` *efines the computation, while call*ng `model(x)` goes through
> `nn.M*dule`'s call machinery before/arou*d executing `forward()`.

This dis*inction matters because Module cal* machinery supports behavior
aroun* the forward call, such as hooks.
*Therefore normal PyTorch usage sho*ld be:

```python
output = model(x*
```

---

# 15. Registered Submod*le Does Not Mean Automatically Exe*uted

Suppose:

```python
class Mo*el(nn.Module):

    def __init__(s*lf):
        super().__init__()

        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):

        x = self.layer1(x)

        return x
```

Even though:

```text
layer2
```

is a registered Submodule, it is not executed because `forward()` never calls
it.

Therefore:

> Registration tells PyTorch that the Module belongs to the model.

It does NOT mean:

> PyTorch automatically executes every registered Submodule.

`forward()` defines the actual computation path.

---

# 16. Understanding `nn.Linear`

Consider:

```python
layer = nn.Linear(
    in_features=3,
    out_features=2
)
```

The layer accepts:

```text
3 input features
```

and produces:

```text
2 output features
```

Conceptually:

```text
[x1, x2, x3]
      │
      ▼
   Linear
      │
      ▼
   [y1, y2]
```

Each output is a linear/affine combination of all input features:

```text
y1 = w11*x1 + w12*x2 + w13*x3 + b1

y2 = w21*x1 + w22*x2 + w23*x3 + b2
```

---

# 17. How to Derive Linear Weight Shape

I should NOT blindly memorize:

```text
weight.shape = (out_features, in_features)
```

Instead, derive it.

For:

```python
nn.Linear(3, 2)
```

ask:

```text
How many outputs?

2
```

Each output needs:

```text
one weight for every input feature

= 3 weights
```

Therefore:

```text
output 1 → 3 weights

output 2 → 3 weights
```

Stack the two sets of weights:

```text
weight.shape = (2, 3)
```

Or:

```text
weight.shape
=
(out_features, in_features)
```

The reasoning is:

```text
OUT outputs

×

IN weights per output
```

---

# 18. How to Derive Bias Shape

Each output feature gets one bias.

For:

```python
nn.Linear(3, 2)
```

there are:

```text
2 output features
```

Therefore:

```text
bias.shape = (2,)
```

General rule:

```text
bias.shape = (out_features,)
```

---

# 19. Linear Mathematics and PyTorch Weight Orientation

Mathematically, I might naturally write:

```text
x @ W + b
```

and choose:

```text
W.shape = (in_features, out_features)
```

For example:

```text
x.shape = (3,)

W.shape = (3,2)

x @ W → (2,)
```

This is mathematically valid.

PyTorch's `nn.Linear` stores weights as:

```text
(out_features, in_features)
```

Therefore the operation can be thought of conceptually as:

```text
x @ weight.T + bias
```

For:

```text
x.shape      = (3,)
weight.shape = (2,3)
weight.T     = (3,2)

(3,) @ (3,2)
      ↓
     (2,)
```

This explains the PyTorch weight orientation instead of treating it as a
random convention.

---

# 20. Linear With a Batch

Consider:

```python
layer = nn.Linear(3, 2)

x = torch.randn(32, 3)
```

The input means:

```text
32 samples
each having 3 features
```

So:

```text
x.shape = (32,3)
```

Conceptually:

```text
x
(32,3)

@

weight.T
(3,2)

=

output
(32,2)
```

The batch dimension remains:

```text
32
```

while the feature dimension changes:

```text
3 → 2
```

---

# 21. Linear Bias and Broadcasting

For:

```python
nn.Linear(3,2)
```

the bias is:

```text
bias.shape = (2,)
```

The output before bias has shape:

```text
(32,2)
```

So:

```text
(32,2)
+
(2,)
```

uses broadcasting.

The bias logically applies to every sample in the batch.

This connects `nn.Linear` to my previous broadcasting knowledge.

---

# 22. Counting Linear Parameters

For:

```python
nn.Linear(IN, OUT)
```

weights:

```text
OUT × IN
```

biases:

```text
OUT
```

Total:

```text
IN × OUT + OUT
```

assuming the layer uses bias.

For:

```python
nn.Linear(5,3)
```

weights:

```text
5 × 3 = 15
```

biases:

```text
3
```

total:

```text
18
```

---

# 23. Multi-Layer Shape Example

Consider:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(5, 4)
        self.layer2 = nn.Linear(4, 3)
        self.layer3 = nn.Linear(3, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
```

For:

```python
x = torch.randn(32,5)
```

the shapes are:

```text
Input

(32,5)
```

Layer 1:

```text
weight = (4,5)
bias   = (4,)

output = (32,4)
```

Layer 2:

```text
weight = (3,4)
bias   = (3,)

output = (32,3)
```

Layer 3:

```text
weight = (1,3)
bias   = (1,)

output = (32,1)
```

---

# 24. Counting Parameters in the Multi-Layer Model

Layer 1:

```text
weights = 4 × 5 = 20
biases  = 4

total = 24
```

Layer 2:

```text
weights = 3 × 4 = 12
biases  = 3

total = 15
```

Layer 3:

```text
weights = 1 × 3 = 3
biases  = 1

total = 4
```

Entire model:

```text
24 + 15 + 4

=

43 trainable scalar Parameters
```

---

# 25. Why `super().__init__()`?

Consider:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(5,3)
```

Our class inherits from:

```text
nn.Module
```

Therefore:

```text
nn.Module
    ↑
    │
  Model
```

The base `nn.Module` has initialization work to perform.

My mental model is:

```text
Model.__init__()
       │
       ▼
super().__init__()
       │
       ▼
initialize base nn.Module machinery
       │
       ▼
now assign/register model components
```

So:

```python
super().__init__()
```

must happen before assigning child Modules and Parameters.

---

# 26. Why Forgetting `super().__init__()` Is a Problem

Suppose:

```python
class BrokenModel(nn.Module):

    def __init__(self):

        self.layer = nn.Linear(5,3)
```

My initial intuition was:

> Maybe `layer` will simply behave like an ordinary Python attribute but won't
> be registered.

A better mental model is:

```text
self.layer = nn.Linear(...)
       │
       ▼
PyTorch's Module machinery recognizes
that a child Module is being assigned
       │
       ▼
tries to register child Module
       │
       ▼
base Module state has not been initialized
       │
       ▼
problem
```

Therefore:

> `super().__init__()` initializes the `nn.Module` foundation before we begin
> registering model components.

---

# 27. What Does Registration Mean?

This was one of the terms I found confusing.

My simplified definition is:

> Registration means PyTorch knows that an object belongs to a Module in a
> particular role.

For example:

```python
self.layer = nn.Linear(...)
```

means PyTorch can know:

```text
layer
=
child Module
```

When:

```python
self.weight = nn.Parameter(...)
```

PyTorch can know:

```text
weight
=
Parameter
```

When:

```python
self.register_buffer(
    "running_mean",
    tensor
)
```

PyTorch can know:

```text
running_mean
=
Buffer
```

So registration is not mysterious.

It establishes relationships:

```text
                  Model
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   Submodules   Parameters    Buffers
```

---

# 28. Why Registration Matters

Once PyTorch knows the Module hierarchy, it can manage things recursively.

Example:

```text
Model
│
├── layer1
│   ├── weight
│   └── bias
│
└── layer2
    ├── weight
    └── bias
```

PyTorch can traverse:

```text
Model
 ↓
child Modules
 ↓
their Parameters
```

This allows APIs such as:

```python
model.parameters()
```

to discover Parameters recursively.

It also allows Module transformations such as:

```python
model.to("cuda")
```

to operate recursively on registered Module tensor state such as Parameters
and Buffers.

---

# 29. `parameters()`

Consider:

```python
model.parameters()
```

My mental model is:

> Recursively traverse the Module hierarchy and yield registered Parameters.

For:

```text
Model
│
├── layer1
│   ├── weight
│   └── bias
│
└── layer2
    ├── weight
    └── bias
```

conceptually:

```text
model.parameters()

→ layer1.weight Parameter
→ layer1.bias Parameter
→ layer2.weight Parameter
→ layer2.bias Parameter
```

Important:

`parameters()` does not literally return a Python list.

It provides an iterator over Parameters.

---

# 30. `named_parameters()`

`named_parameters()` also traverses registered Parameters recursively, but
provides the Parameter's qualified name.

Conceptually:

```text
"layer1.weight" → Parameter
"layer1.bias"   → Parameter
"layer2.weight" → Parameter
"layer2.bias"   → Parameter
```

Therefore:

```text
parameters()
      ↓
Parameter objects


named_parameters()
      ↓
(name, Parameter) pairs
```

---

# 31. Why `named_parameters()` Is Useful

Names add context.

For example:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.grad)
```

might let me inspect:

```text
layer1.weight → gradient
layer1.bias   → gradient

layer2.weight → gradient
layer2.bias   → gradient
```

This can be useful for:

```text
debugging
gradient inspection
logging
freezing selected Parameters
model analysis
```

---

# 32. What Does "State" Mean?

This word initially felt vague.

My simple definition is:

> State is information a Module needs to remember between calls.

Consider:

```python
class Model(nn.Module):

    def forward(self, x):
        ...
```

The input:

```text
x
```

is temporary.

The Module receives it during the forward call.

But things such as:

```text
weight
bias
running statistics
```

remain with the model between calls.

Therefore:

```text
state
=
things the Module remembers
```

---

# 33. Model State Has Different Categories

The main tensor state categories are:

```text
                         Module State
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
               Parameters             Buffers
                    │                   │
                    ▼                   ▼
             parameter state       non-parameter
                                    tensor state
```

This distinction is important.

---

# 34. Buffer

Sometimes a Module needs to remember a tensor that should belong to the
model but should not be treated as a model Parameter.

This is a:

```text
Buffer
```

My mental model:

> A Buffer is managed tensor state belonging to a Module that is not a model
> Parameter.

One example is running statistics used by some layers.

Conceptually:

```text
Parameters
    ↓
model Parameters


Buffers
    ↓
other managed tensor state
```

---

# 35. Why Not Use an Ordinary Tensor?

Suppose:

```python
self.running_value = torch.zeros(3)
```

This is a Tensor.

But PyTorch does not automatically know that it should be treated as managed
Module state merely because it is a Tensor attribute.

Instead:

```python
self.register_buffer(
    "running_value",
    torch.zeros(3)
)
```

communicates:

> This Tensor belongs to this Module as a Buffer.

Conceptually:

```text
ordinary Tensor attribute
        │
        ▼
Python knows about it


registered Buffer
        │
        ▼
Python knows about it
+
PyTorch Module knows its role
```

---

# 36. Parameter vs Buffer

This is the clean distinction I want to remember:

```text
                     Module State
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
          Parameter                  Buffer
              │                       │
              ▼                       ▼
      parameter of model       non-parameter
                                 tensor state
```

Do NOT reduce this to:

```text
Parameter = requires_grad=True

Buffer = requires_grad=False
```

That is incorrect.

---

# 37. `requires_grad` Is a Separate Axis

Consider:

```python
p = nn.Parameter(
    torch.tensor(1.0),
    requires_grad=False
)
```

This object is:

```text
Parameter?       YES

requires_grad?   NO

Buffer?          NO
```

Therefore:

```text
Parameter / Buffer
       ↓
Module-state classification


requires_grad
       ↓
Autograd behavior
```

These answer different questions.

---

# 38. A Clean Three-Way Separation

This is one of the most important mental models in this section:

```text
                         Tensor
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼

    requires_grad        Parameter           Buffer

          │                 │                 │
          ▼                 ▼                 ▼

       Autograd           Module            Module
       concept            concept           concept

          │                 │                 │
          ▼                 ▼                 ▼

Should gradients       Is this model      Is this managed
be calculated?         parameter state?   non-parameter state?
```

This clears up a lot of terminology confusion.

---

# 39. Buffers and Device Movement

Registered Parameters and Buffers are managed by the Module.

This becomes useful when:

```python
model.to("cuda")
```

Conceptually:

```text
Before:

Model
│
├── Parameter → CPU
└── Buffer    → CPU


model.to("cuda")


After:

Model
│
├── Parameter → GPU
└── Buffer    → GPU
```

An ordinary Tensor attribute does not automatically receive the same Module
management merely because it is attached to the Python object.

---

# 40. Buffer Persistence Is a Separate Concept

There are two separate questions:

```text
Is this Tensor a registered Buffer?

                vs

Should this Buffer be persisted
when saving model state?
```

A Buffer can be registered as persistent or non-persistent.

Conceptually:

```text
Buffer
│
├── persistent
│      └── included in state_dict
│
└── non-persistent
       └── not included in state_dict
```

This distinction is separate from whether the Tensor is a Buffer.

---

# 41. `state_dict()`

The name can be understood literally:

```text
state
+
dictionary
```

My mental model is:

> `state_dict()` represents the persistent registered tensor state required
> to save and restore the Module.

It includes:

```text
registered Parameters

+

persistent registered Buffers
```

recursively across the Module hierarchy.

---

# 42. Example `state_dict`

Suppose:

```text
Model
│
├── layer
│   ├── weight → Parameter
│   └── bias   → Parameter
│
└── running_sum → persistent Buffer
```

Then conceptually:

```text
state_dict

"layer.weight" → Tensor
"layer.bias"   → Tensor
"running_sum"  → Tensor
```

Notice:

```text
state_dict
```

is about:

```text
saving/restoring persistent model state
```

not simply:

```text
which tensors should optimizer update?
```

---

# 43. `parameters()` vs `state_dict()`

This is an important distinction.

## `parameters()`

Question:

> Which registered Parameters does the model have?

Conceptually:

```text
model.parameters()
        ↓
Parameters
```

This is useful when giving Parameters to an optimizer.

---

## `state_dict()`

Question:

> What persistent tensor state should be saved/restored?

Conceptually:

```text
state_dict()
      │
      ├── Parameters
      │
      └── persistent Buffers
```

Therefore `state_dict()` can contain more than `parameters()`.

---

# 44. `named_parameters()` vs `named_buffers()` vs `state_dict()`

Suppose:

```python
class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(3, 3)
        )

        self.register_buffer(
            "running_sum",
            torch.zeros(3)
        )

        self.temp = torch.ones(3)
```

Then conceptually:

```text
named_parameters()

"weight" → Parameter
```

```text
named_buffers()

"running_sum" → Buffer
```

```text
state_dict()

"weight"      → Tensor

"running_sum" → Tensor
```

But:

```text
temp
```

is merely an ordinary Tensor attribute.

It is not automatically a registered Parameter or Buffer.

---

# 45. Registration Does Not Mean "Everything Is Registered"

This was another terminology confusion.

I should NOT say:

> When the Model is created, PyTorch registers all its attributes and state.

Instead:

> `nn.Module` initializes the machinery required to manage registered
> components. As I assign/register child Modules, Parameters, and Buffers,
> PyTorch records them in their corresponding roles.

This is much more precise.

---

# 46. The Clean Module Mental Model

This is the diagram I want to remember:

```text
                            nn.Module
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
        Computation                               State
             │                                     │
             ▼                       ┌─────────────┴─────────────┐
         forward()                   │                           │
                                     ▼                           ▼
                                Parameters                     Buffers
                                     │                           │
                                     ▼                           ▼
                              parameter state             non-parameter
                                                           tensor state
```

And Modules can contain Modules:

```text
                            MyModel
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
            layer1                            layer2
           Submodule                         Submodule
              │                                 │
       ┌──────┴──────┐                  ┌───────┴──────┐
       │             │                  │              │
       ▼             ▼                  ▼              ▼
    weight          bias             weight           bias
   Parameter      Parameter         Parameter       Parameter
```

---

# 47. Registration Mental Model

I should remember:

```text
Registration
     =
PyTorch knows:

1. this object belongs to the Module

AND

2. what role it has
```

Example:

```text
fc
↓
registered Submodule


weight
↓
registered Parameter


running_mean
↓
registered Buffer
```

Registration allows PyTorch to recursively manage the model.

---

# 48. `nn.Module` Interview Answer

If asked:

> What is `nn.Module`?

My answer would be:

> `nn.Module` is PyTorch's base abstraction for models and layers. I think
> of a Module as packaging computation, usually implemented in `forward()`,
> together with the state needed by that computation.
>
> A Module can contain child Modules, called Submodules. Its state can include
> Parameters, which represent model parameter state, and Buffers, which
> represent managed non-parameter tensor state.
>
> PyTorch registers these objects with the Module so that it understands their
> role and place in the Module hierarchy and can recursively discover and
> manage the model.

---

# 49. Parameter Interview Answer

If asked:

> What is `nn.Parameter`?

My answer would be:

> `nn.Parameter` is a Tensor type used to mark a tensor as a Parameter of an
> `nn.Module`. When assigned to a Module, it is registered so Module APIs can
> discover and manage it. Parameters typically represent learnable model
> state such as weights and biases.

---

# 50. Buffer Interview Answer

If asked:

> What is a Buffer?

My answer would be:

> A Buffer is managed tensor state belonging to a Module that is not treated
> as a model Parameter. It is useful for tensors the Module needs to retain
> and manage, but which should not semantically be Parameters. Persistent
> Buffers are also included in the Module's saved state.

---

# 51. Why Buffer Instead of `Parameter(requires_grad=False)`?

A Parameter with:

```python
requires_grad=False
```

is still semantically:

```text
a Parameter
```

A Buffer communicates:

```text
this is model state

but

this is NOT model Parameter state
```

Therefore:

```text
Parameter vs Buffer
```

is about Module semantics.

```text
requires_grad
```

is about Autograd behavior.

---

# 52. `state_dict()` Interview Answer

If asked:

> What is `state_dict()`?

My answer would be:

> A Module's `state_dict()` is a dictionary representing its persistent
> registered tensor state. It contains registered Parameters and persistent
> registered Buffers recursively across the Module hierarchy, and is commonly
> used when saving and restoring model state.

---

# 53. What I Initially Found Confusing

Several terms originally blurred together:

```text
Module
Submodule
Parameter
Buffer
State
Registration
requires_grad
state_dict
```

The confusion came from treating them as if they all described the same
thing.

They do not.

My clearer model is:

```text
Module
    ↓
container/building block for computation + state


Submodule
    ↓
Module inside another Module


State
    ↓
information Module remembers


Parameter
    ↓
model Parameter state


Buffer
    ↓
managed non-Parameter tensor state


Registration
    ↓
PyTorch knows an object's role
and relationship to Module


requires_grad
    ↓
Autograd should compute gradient?


state_dict
    ↓
persistent registered tensor state
for save/restore
```

---

# 54. Final Teaching Mental Model

If I had to teach `nn.Module` to someone from scratch, I would start with:

> A PyTorch Module combines computation with state.

Then:

```text
Module
│
├── Computation
│     └── forward()
│
└── State
      ├── Parameters
      └── Buffers
```

Then explain composition:

```text
Model
│
├── Submodule
│
├── Submodule
│
└── Buffer
```

And finally explain registration:

> Registration is how PyTorch knows what belongs to the Module and what role
> each object has.

This allows Module APIs to recursively:

```text
discover Parameters
discover Buffers
manage Submodules
move managed tensor state between devices
save persistent tensor state
restore persistent tensor state
```

That is the mental model I want to carry forward.

---

# 55. Current Neural Network Mental Model

At this point:

```text
                         nn.Module
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     ▼
      Computation                              State
          │                                     │
          ▼                        ┌─────────────┴─────────────┐
      forward()                    │                           │
                                   ▼                           ▼
                              Parameters                     Buffers
                                   │                           │
                             typically                    managed
                             learnable                  non-Parameter
                               state                    tensor state
```

Modules compose recursively:

```text
Module
│
├── Submodule
│    ├── Parameter
│    └── Parameter
│
├── Submodule
│    ├── Parameter
│    └── Parameter
│
└── Buffer
```

And registration makes PyTorch aware of this hierarchy.

---

# 56. What Comes Next

The next concepts should build on this foundation rather than introduce more
terminology all at once.

Topics still to learn include:

```text
model.train()
vs
model.eval()

loss functions

optimizers

full training loop

activation functions

nn.Sequential

saving/loading state_dict

freezing parameters

hooks

and eventually deeper model architectures
```

Before moving forward, the key terminology that should remain clear is:

```text
Module      → computation + state

Submodule   → Module inside Module

Parameter   → model Parameter state

Buffer      → managed non-Parameter tensor state

State       → information Module remembers

Registration
            → PyTorch knows ownership + role

requires_grad
            → Autograd behavior

state_dict
            → persistent registered tensor state
              for save/restore
```
