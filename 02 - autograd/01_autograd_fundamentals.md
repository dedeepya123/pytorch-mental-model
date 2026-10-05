# Autograd Fundamentals

## Goal

So far I have learned how tensors represent data:

```text
Tensor
├── shape
├── stride
├── dtype
├── device
├── views
├── broadcasting
└── storage
```

The next question is:

> How does PyTorch compute gradients automatically?

The answers to that question form the foundation of neural-network training.

In this section I want to understand:

- What `requires_grad` means
- What autograd tracks
- What a computation graph is
- What `grad_fn` is
- Leaf vs non-leaf tensors
- What `backward()` does
- How gradients are stored
- Why gradients accumulate
- Why `zero_grad()` exists
- Scalar vs non-scalar backward
- `retain_grad()`
- `retain_graph=True`
- `torch.no_grad()`
- `detach()`
- In-place operations and autograd
- How autograd ultimately trains a model

The goal is to understand:

```python
loss.backward()
```

instead of merely memorizing it.

---

# 1. The Problem Autograd Solves

Consider:

```text
y = x²
```

At:

```text
x = 3
```

we know:

```text
dy/dx = 2x = 6
```

Now suppose:

```text
a = x²

b = a + 2

y = 3b
```

or:

```text
y = 3(x² + 2)
```

If:

```text
x = 2
```

then:

```text
a = 4
b = 6
y = 18
```

Computing:

```text
y = 18
```

is easy.

The difficult question is:

> How does y change if x changes slightly?

That is:

```text
dy/dx
```

Neural-network training depends on repeatedly computing such derivatives with
respect to millions or billions of parameters.

Autograd automates this process.

---

# 2. `requires_grad`

Consider:

```python
import torch

x = torch.tensor(
    2.0,
    requires_grad=True
)
```

My mental model is:

> `requires_grad=True` tells PyTorch that I may later need gradients with
> respect to this tensor.

Therefore:

```text
track relevant operations
        ↓
build autograd graph
        ↓
allow gradient computation later
```

An important refinement:

`requires_grad=True` does NOT mean:

```text
immediately calculate gradients
```

Instead it means:

```text
track computation
so gradients can later be computed
```

---

# 3. Forward Pass

Consider:

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

a = x ** 2

b = a + 2

y = b * 3
```

Numerically:

```text
x = 2

a = 4

b = 6

y = 18
```

A useful mental model is:

```text
Forward pass has two jobs

1. Compute values

2. Build computation history
```

Conceptually:

```text
x
│
│ square
▼
a
│
│ +2
▼
b
│
│ ×3
▼
y
```

This recorded history is the computation graph.

---

# 4. Computation Graph

The graph captures:

```text
What operation produced this tensor?
```

For the example above:

```text
x
│
│ square
▼
a
│
│ add
▼
b
│
│ multiply
▼
y
```

The graph is created dynamically during the forward pass.

My mental model is:

```text
forward pass
       ↓
build graph
       ↓
backward() can later walk graph
```

---

# 5. `grad_fn`

Every tensor produced by a tracked operation has a:

```python
tensor.grad_fn
```

Conceptually:

> `grad_fn` tells me what operation produced this tensor and provides the
> connection into the autograd graph.

Example:

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

a = x ** 2
```

Then:

```text
x.grad_fn = None

a.grad_fn = Power operation
```

The exact class names are unimportant.

The important question is:

```text
What operation created this tensor?
```

---

# 6. Leaf vs Non-Leaf Tensors

A tensor created directly by the user:

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)
```

is a leaf tensor.

Conceptually:

```text
x

requires_grad=True

grad_fn=None

leaf=True
```

A tensor created by an operation:

```python
a = x ** 2
```

is a non-leaf tensor.

Conceptually:

```text
a

requires_grad=True

grad_fn exists

leaf=False
```

Therefore:

```text
Created directly
      ↓
leaf

Created by tracked operation
      ↓
non-leaf
```

---

# 7. Why `backward()` Exists

Forward gives:

```text
y = 18
```

But training needs:

```text
dy/dx
```

Autograd computes this through:

```python
y.backward()
```

Conceptually:

```text
FORWARD

x
│
▼
a
│
▼
b
│
▼
y


BACKWARD

y
▲
│
b
▲
│
a
▲
│
x
```

Backward walks through the graph in reverse using the chain rule.

---

# 8. Manual Gradient Example

We have:

```text
a = x²

b = a + 2

y = 3b
```

We want:

```text
dy/dx
```

Compute local derivatives:

```text
dy/db = 3

db/da = 1

da/dx = 2x
```

Chain rule:

```text
dy/dx

=
dy/db × db/da × da/dx
```

At:

```text
x = 2
```

we get:

```text
3 × 1 × 4

=
12
```

Therefore:

```text
dy/dx = 12
```

---

# 9. `.grad`

Before:

```python
y.backward()
```

we have:

```python
x.grad
```

```text
None
```

After:

```python
y.backward()
```

we have:

```text
x.grad = 12
```

This gives the correct mental model:

```text
.grad
    ↓
accumulated gradient
computed during backward
```

Importantly:

```text
grad_fn
      ≠
grad
```

```text
grad_fn
    ↓
How was this tensor produced?


.grad
    ↓
What gradient was accumulated
for this tensor?
```

---

# 10. What Does `.grad` Mean?

This question caused an important clarification.

Suppose:

```python
y.backward()
```

Then:

```python
x.grad
```

means:

```text
∂y
──
∂x
```

If:

```python
loss.backward()
```

then:

```python
weight.grad
```

means:

```text
∂loss
──────
∂weight
```

The output used in:

```python
backward()
```

defines what gradient is being propagated.

---

# 11. Gradient Computation vs Gradient Retention

Consider:

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

a = x ** 2

b = a + 2

y = b * 3
```

During backward:

```text
dy/dy = 1

dy/db = 3

dy/da = 3

dy/dx = 12
```

Thus gradients are computed throughout the graph.

However:

```text
gradient computed
      ≠
gradient stored in .grad
```

By default:

```text
x.grad = 12

a.grad = None

b.grad = None

y.grad = None
```

Non-leaf gradients are used during backward but normally are not retained.

---

# 12. `retain_grad()`

Suppose:

```python
a.retain_grad()

b.retain_grad()

y.retain_grad()
```

before:

```python
y.backward()
```

Then:

```text
x.grad = 12

a.grad = 3

b.grad = 3

y.grad = 1
```

The key insight:

> `retain_grad()` does not cause gradients to be computed.

They were already computed.

It changes:

```text
Compute gradient
       vs
Store gradient
```

for non-leaf tensors.

---

# 13. Scalar Outputs

Autograd is easiest to understand with scalar outputs.

Example:

```python
x = torch.tensor(
    [[1., 2.],
     [3., 4.]],
    requires_grad=True
)

loss = (x ** 2).sum()
```

Now:

```text
loss is scalar
```

and:

```python
loss.backward()
```

computes:

```text
∂loss
──────
∂x
```

Result:

```text
[[2, 4],
 [6, 8]]
```

Notice:

```text
x.grad.shape
=
x.shape
```

Each element means:

```text
∂loss
──────
∂x[i,j]
```

---

# 14. Non-Scalar Outputs

Consider:

```python
x = torch.tensor(
    [1., 2., 3.],
    requires_grad=True
)

y = x ** 2
```

Now:

```text
y.shape = (3,)
```

and:

```text
x.shape = (3,)
```

The derivative becomes a matrix (Jacobian).

Therefore:

```python
y.backward()
```

is ambiguous because:

```text
y is not scalar
```

---

# 15. Incoming Gradient

PyTorch allows:

```python
y.backward(
    gradient=torch.tensor(
        [1., 1., 1.]
    )
)
```

The supplied argument is the incoming gradient.

Conceptually:

```text
incoming gradient
        ×
local derivative
        ↓
gradient propagated backward
```

Example:

```python
x = [2,3]

y = x²

incoming = [5,10]
```

Local derivative:

```text
[4,6]
```

Result:

```text
x.grad

=

[20,60]
```

because:

```text
[5×4, 10×6]
```

---

# 16. Why Loss Is Usually Scalar

This is one reason training commonly uses:

```text
loss
```

as a scalar.

Then:

```python
loss.backward()
```

has an obvious meaning:

```text
∂loss
──────
∂parameter
```

with no need to supply an explicit incoming gradient.

---

# 17. Gradient Accumulation

Suppose:

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x ** 2

y.backward()
```

Then:

```text
x.grad = 4
```

Now:

```python
z = x ** 2

z.backward()
```

produces another gradient:

```text
4
```

The result becomes:

```text
x.grad

=

4 + 4

=

8
```

Therefore:

```text
.grad accumulates
```

instead of being overwritten.

---

# 18. Why Accumulation Exists

Within one graph:

```text
     x
   /   \
 x²     3x
   \   /
     y
```

both branches contribute gradient.

Those contributions must be summed.

This same accumulation mechanism is also used across multiple backward calls.

---

# 19. Clearing Gradients

Because:

```text
.grad accumulates
```

training loops usually clear gradients.

Example:

```python
x.grad.zero_()
```

This means:

```text
set gradient buffer to zero
```

It does NOT:

```text
remove requires_grad

destroy graph

modify x
```

It only clears:

```text
x.grad
```

---

# 20. Graph Lifetime

Forward:

```python
y = x ** 2
```

creates:

```text
graph
```

Then:

```python
y.backward()
```

uses the graph.

My mental model:

```text
forward
    ↓
build graph

backward
    ↓
use graph

backward complete
    ↓
graph normally released
```

This is why:

```python
y.backward()
y.backward()
```

typically fails.

---

# 21. `retain_graph=True`

Example:

```python
y.backward(
    retain_graph=True
)

y.backward()
```

Now the graph remains available after the first backward pass.

This allows the second backward call.

However:

> `retain_graph=True` should not be used casually because it keeps graph
> resources alive.

---

# 22. `torch.no_grad()`

Suppose:

```python
with torch.no_grad():
    y = x ** 2
```

The computation still happens:

```text
y = 4
```

But:

```text
autograd graph
```

is not built.

My mental model:

```text
compute value ✅

build graph ❌
```

Therefore:

```text
y.requires_grad = False

y.grad_fn = None
```

---

# 23. `detach()`

Consider:

```python
y = x ** 2

z = y.detach()
```

The graph already exists:

```text
x
│
▼
y
```

Then:

```text
detach()
```

creates:

```text
z
```

which is disconnected from the graph.

Conceptually:

```text
storage

      value
      /   \
     /     \
    y       z

tracked   untracked
```

Therefore:

```text
z.requires_grad = False

z.grad_fn = None
```

---

# 24. `no_grad()` vs `detach()`

A useful comparison:

```python
with torch.no_grad():
    y = x ** 2
```

means:

```text
graph never exists
```

Whereas:

```python
y = x ** 2

z = y.detach()
```

means:

```text
graph exists first

then z disconnects from it
```

This distinction is important.

---

# 25. In-Place Operations and Autograd

Consider:

```python
x = torch.tensor(
    3.0,
    requires_grad=True
)

y = x ** 2
```

Backward needs:

```text
x = 3
```

because:

```text
dy/dx = 2x
```

Now suppose:

```python
x += 10
```

Before:

```python
y.backward()
```

Conceptually:

```text
forward used:

x = 3


current storage:

x = 13
```

Autograd may have saved values needed from forward.

An in-place modification can invalidate those saved values.

Rather than silently producing an incorrect gradient, PyTorch often raises
an error.

---

# 26. In-Place vs Out-Of-Place

In-place:

```python
x += 10
```

```text
modify same storage
```

Out-of-place:

```python
z = x + 10
```

```text
new tensor
new storage
```

This distinction matters because autograd may need information saved during
the original forward pass.

---

# 27. Manual Training Example

Suppose:

```text
prediction = w × x
```

where:

```text
w = trainable weight
```

Choose:

```text
x = 2

target = 10

w = 1
```

Prediction:

```text
2
```

Loss:

```text
(prediction-target)^2

=
64
```

---

# 28. Compute Gradient

Model:

```text
prediction = wx
```

Loss:

```text
L = (wx - target)^2
```

Derivative:

```text
dL/dw

=
2(wx-target)x
```

Substitute values:

```text
=
2(2-10)(2)

=
-32
```

Therefore:

```text
w.grad = -32
```

after:

```python
loss.backward()
```

---

# 29. Parameter Update

Suppose:

```text
learning_rate = 0.1
```

Gradient descent:

```text
new_w

=
old_w - lr × gradient
```

Substitute:

```text
1 - 0.1(-32)

=
4.2
```

Thus:

```text
w

1.0 → 4.2
```

The gradient told us how to change the parameter to reduce loss.

---

# 30. Why `optimizer.zero_grad()` Exists

Training loops commonly look like:

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Now this makes sense.

```text
zero_grad()
    ↓
clear accumulated gradients


backward()
    ↓
compute parameter gradients


step()
    ↓
update parameters
```

---

# 31. Final Autograd Mental Model

```text
requires_grad=True
          │
          ▼
track operations
          │
          ▼
build dynamic graph
          │
          ▼
forward computes values
          │
          ▼
loss
          │
          ▼
backward()
          │
          ▼
chain rule
          │
          ▼
gradient propagation
          │
          ▼
parameter.grad
          │
          ▼
optimizer updates parameters
```

---

# 32. What I Understand Now

I understand:

```text
requires_grad
grad_fn
leaf tensors
non-leaf tensors
backward()
.grad
retain_grad()
gradient accumulation
zero_grad()
graph lifetime
retain_graph
scalar outputs
non-scalar outputs
incoming gradients
no_grad()
detach()
in-place mutation
manual gradient descent
```

Autograd can now be summarized as:

> During the forward pass, PyTorch computes tensor values while building a
> dynamic computation graph. During backward, PyTorch traverses that graph in
> reverse using the chain rule, computes gradients, accumulates them into
> relevant leaf tensors, and enables gradient-based parameter updates.
