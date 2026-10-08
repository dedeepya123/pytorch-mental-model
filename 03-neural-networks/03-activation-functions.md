# Activation Functions in PyTorch

## Goal

I already understand conceptually why neural networks need nonlinear activation
functions.

In this section, my goal is specifically to understand activation functions
from the PyTorch perspective:

- How activations appear in a PyTorch model.
- `nn.ReLU()` vs `F.relu()`.
- Activation Modules vs functional operations.
- Whether activation Modules contain Parameters.
- How activations affect tensor shapes.
- How activations participate in Autograd.
- Why Module hierarchy and Autograd graph are different concepts.
- Why Module traversal and Parameter traversal are different.
- How ReLU, GELU, and SiLU fit into the same PyTorch mental model.

---

# 1. Activation Functions Inside a PyTorch Model

Consider:

```python
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
```

The model can be visualized as:

```text
Input
  │
  ▼
Linear(5,4)
  │
  ▼
ReLU
  │
  ▼
Linear(4,2)
  │
  ▼
Output
```

Its Module hierarchy is:

```text
Model
│
├── fc1  → nn.Linear
├── relu → nn.ReLU
└── fc2  → nn.Linear
```

Therefore an activation can itself be represented as an `nn.Module`.

---

# 2. Activation Modules Can Be Stateless

One important realization is:

> Being an `nn.Module` does not mean a Module must contain trainable
> Parameters.

For example:

```python
nn.Linear(5,4)
```

contains:

```text
weight
bias
```

These are Parameters.

But:

```python
nn.ReLU()
```

does not need:

```text
weight
bias
```

The ReLU computation is simply an elementwise transformation.

Conceptually:

```text
ReLU(x) = max(0, x)
```

Therefore:

```text
                 nn.Module
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
     Stateful                Stateless
      Module                  Module
         │                     │
         ▼                     ▼
     nn.Linear               nn.ReLU
         │
         ▼
     Parameters
```

So:

```text
Module
≠
must have Parameters
```

---

# 3. Shape Through an Activation Function

Suppose:

```python
x = torch.randn(32, 5)
```

Then:

```python
x = self.fc1(x)
```

changes:

```text
(32,5)
   ↓
Linear(5,4)
   ↓
(32,4)
```

Now:

```python
x = self.relu(x)
```

ReLU processes each element individually.

Therefore:

```text
(32,4)
   ↓
ReLU
   ↓
(32,4)
```

The shape does not change.

Finally:

```python
x = self.fc2(x)
```

gives:

```text
(32,4)
   ↓
Linear(4,2)
   ↓
(32,2)
```

The complete shape flow is:

```text
Input
(32,5)
   │
   ▼
Linear(5,4)
   │
   ▼
(32,4)
   │
   ▼
ReLU
   │
   ▼
(32,4)
   │
   ▼
Linear(4,2)
   │
   ▼
Output
(32,2)
```

This gives me a useful shape heuristic:

```text
Linear
   ↓
usually changes the feature dimension


Pointwise activation
   ↓
usually preserves the tensor shape
```

---

# 4. ReLU

ReLU performs the elementwise operation:

```text
ReLU(x) = max(0, x)
```

For example:

```text
input:

[-2, 1, -4, 3]

     ↓ ReLU

output:

[0, 1, 0, 3]
```

ReLU changes values but not shape.

If:

```text
input.shape = (32, 128)
```

then:

```text
ReLU(input).shape = (32, 128)
```

---

# 5. Activation Functions Participate in Autograd

An important realization is:

> An operation does not need trainable Parameters to participate in Autograd.

Suppose:

```python
z = self.fc1(x)

a = self.relu(z)

y = self.fc2(a)
```

The forward computation is:

```text
x
│
▼
Linear
│
▼
z
│
▼
ReLU
│
▼
a
│
▼
Linear
│
▼
y
```

If gradient tracking is enabled, Autograd records these relevant tensor
operations.

Later:

```python
loss.backward()
```

has to propagate gradients backward through the entire computation:

```text
loss
  │
  ▼
fc2
  │
  ▼
ReLU
  │
  ▼
fc1
```

For ReLU, conceptually:

```text
input > 0
    ↓
local derivative behaves like 1


input < 0
    ↓
local derivative behaves like 0
```

Therefore ReLU has:

```text
trainable Parameters? NO

participates in Autograd? YES
```

These are different concepts.

---

# 6. Parameter Ownership vs Autograd Participation

This gives me an important distinction:

```text
Does something have Parameters?
        │
        ▼
Does this Module own learnable state?


Does something participate in Autograd?
        │
        ▼
Is its tensor operation part of the differentiable computation?
```

For example:

```text
nn.Linear
│
├── Parameters? YES
└── Autograd?   YES


nn.ReLU
│
├── Parameters? NO
└── Autograd?   YES
```

Therefore:

```text
No Parameters
≠
No gradient flow
```

---

# 7. Module API vs Functional API

PyTorch commonly exposes neural-network operations in two styles.

For ReLU, I may see:

```python
nn.ReLU()
```

or:

```python
torch.nn.functional.relu(...)
```

Usually:

```python
import torch.nn.functional as F
```

allows:

```python
F.relu(x)
```

These represent the same basic mathematical operation but are packaged
differently in PyTorch.

---

# 8. Module-Style ReLU

Example:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(5, 4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc(x)

        x = self.relu(x)

        return x
```

Here:

```python
self.relu
```

is an `nn.Module`.

Because it is assigned as a Module attribute, it becomes part of the Module
hierarchy.

Conceptually:

```text
Model
│
├── fc
└── relu
```

---

# 9. Functional ReLU

The same computation can be written:

```python
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(5, 4)

    def forward(self, x):

        x = self.fc(x)

        x = F.relu(x)

        return x
```

Now:

```text
F.relu
```

is a function call.

It is not a child Module stored inside `Model`.

Therefore:

```text
nn.ReLU()
    ↓
Module representation
```

while:

```text
F.relu()
    ↓
functional representation
```

---

# 10. Important Correction: Functional Operations Still Participate in Autograd

I initially thought:

> `F.relu()` is only a function, so maybe Autograd does not track it.

That is incorrect.

Autograd tracks relevant **tensor operations**.

It does not require every operation to be an `nn.Module`.

I already saw this earlier with operations such as:

```python
y = x ** 2
```

The power operation is not an `nn.Module`, but Autograd tracks it.

Similarly:

```python
x = F.relu(x)
```

can absolutely participate in the Autograd graph.

Therefore:

```text
F.relu

Module?          NO

Submodule?       NO

Parameters?      NO

Autograd?        YES
```

when it is part of a gradient-tracked computation.

---

# 11. `nn.ReLU` vs `F.relu`

My mental comparison is:

```text
nn.ReLU()
│
├── Module?                 YES
├── Can be registered
│   as Submodule?           YES
├── Parameters?             NO
└── Autograd operation?     YES
```

versus:

```text
F.relu(x)
│
├── Module?                 NO
├── Registered Submodule?   NO
├── Parameters?             NO
└── Autograd operation?     YES
```

The key difference is therefore not:

```text
Autograd vs no Autograd
```

The key difference is:

```text
Module representation

vs

functional operation
```

---

# 12. Module Tree Is Different From Autograd Graph

This distinction is extremely important.

Suppose:

```python
class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(5,4)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc(x)

        x = self.relu(x)

        x = F.relu(x)

        return x
```

The Module tree is:

```text
Model
│
├── fc
└── relu
```

`F.relu` does not appear here.

But the Autograd computation can conceptually contain:

```text
input
  │
  ▼
Linear operation
  │
  ▼
ReLU operation
  │
  ▼
Functional ReLU operation
  │
  ▼
output
```

Therefore:

```text
Module hierarchy
≠
Autograd computation graph
```

They describe different things.

---

# 13. Module Hierarchy

The Module hierarchy answers:

> What Modules does this model structurally contain?

For example:

```text
Model
│
├── fc1
├── act1
├── fc2
├── act2
└── fc3
```

This structure comes from registered `nn.Module` objects.

---

# 14. Autograd Graph

The Autograd graph answers:

> What differentiable tensor operations actually executed to produce this
> output?

Conceptually:

```text
Tensor
  ↓
matmul
  ↓
add
  ↓
ReLU
  ↓
matmul
  ↓
add
  ↓
GELU
  ↓
...
```

The Autograd graph is based on executed tensor operations.

It is not simply a copy of the Module hierarchy.

---

# 15. Parameter Traversal Is Also Different

Consider:

```text
Model
│
├── fc1
├── act1
├── fc2
├── act2
└── fc3
```

All five are child Modules.

But only:

```text
fc1
fc2
fc3
```

contain Parameters.

Therefore conceptual parameter traversal gives:

```text
fc1.weight
fc1.bias

fc2.weight
fc2.bias

fc3.weight
fc3.bias
```

There is no:

```text
act1.weight
act1.bias

act2.weight
act2.bias
```

for ordinary ReLU/GELU Modules.

So:

```text
Module traversal
≠
Parameter traversal
```

---

# 16. Three Independent Questions

Whenever I see something in PyTorch model code, I should ask three
independent questions.

## Question 1

```text
Is it an nn.Module?
```

This tells me whether it can be structurally represented in the Module
hierarchy.

---

## Question 2

```text
Does it own Parameters?
```

This tells me whether it contributes learnable model state.

---

## Question 3

```text
Does its tensor operation participate in Autograd?
```

This tells me whether gradients need to propagate through the operation.

These questions must not be mixed together.

---

# 17. Comparing Common Examples

Conceptually:

```text
nn.Linear

Module?                YES
Parameters?            YES
Autograd participation? YES
```

```text
nn.ReLU

Module?                YES
Parameters?            NO
Autograd participation? YES
```

```text
F.relu

Module?                NO
Parameters?            NO
Autograd participation? YES
```

```text
x ** 2

Module?                NO
Parameters?            NO
Autograd participation? YES
```

```text
x + 10

Module?                NO
Parameters?            NO
Autograd participation? YES
```

provided the operations occur in a gradient-tracked computation.

---

# 18. GELU

GELU is another activation I commonly see in PyTorch models.

Module form:

```python
self.act = nn.GELU()
```

Functional forms may also appear depending on the code.

For my current mental model, the important properties are:

```text
GELU
│
├── nonlinear activation
├── can be represented as a Module
├── elementwise
├── preserves input shape
├── normally has no trainable Parameters
└── participates in Autograd
```

I do not need to memorize the exact mathematical formula to understand its
role in PyTorch architecture.

---

# 19. Important GELU Correction

I initially described GELU as:

```text
elementwise sigmoid
```

That is incorrect.

GELU and Sigmoid are different activation functions.

The useful PyTorch-level takeaway for now is:

```text
GELU
    ↓
