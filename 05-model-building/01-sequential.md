# Sequential Models in PyTorch

## Goal

Before learning custom architectures, we need to understand the simplest way
to build a neural network in PyTorch:

```python
nn.Sequential
```

The goals of this chapter are:

- Understand why Sequential exists.
- Understand what problem it solves.
- Understand how tensors flow through Sequential.
- Understand shape propagation.
- Understand when Sequential is sufficient.
- Understand when a custom Module becomes necessary.

---

# 1. Where We Are In The Learning Journey

So far we understand:

```text
Dataset
    ↓
DataLoader
    ↓
Batch
    ↓
Model
    ↓
Loss
    ↓
Backward
    ↓
Optimizer
```

But we have not yet discussed:

```text
How is the Model actually built?
```

Sequential is our first answer to that question.

---

# 2. The Simplest Neural Network

Consider:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

This is one of the most common patterns in deep learning.

Every layer feeds directly into the next layer.

---

# 3. Manual Implementation

Without Sequential, we might write:

```python
x = linear1(x)

x = relu(x)

x = linear2(x)
```

This works.

However, PyTorch noticed that this pattern occurs constantly.

---

# 4. Enter Sequential

PyTorch provides:

```python
nn.Sequential
```

which means:

```text
Run these modules
one after another.
```

---

# 5. Mental Model

```python
nn.Sequential(
    layer1,
    layer2,
    layer3
)
```

should immediately make you think:

```text
Input
  ↓
layer1
  ↓
layer2
  ↓
layer3
  ↓
Output
```

---

# 6. First Example

```python
model = nn.Sequential(

    nn.Linear(10,20),

    nn.ReLU(),

    nn.Linear(20,3)
)
```

Visualized:

```text
Input
  ↓
Linear(10→20)
  ↓
ReLU
  ↓
Linear(20→3)
  ↓
Output
```

---

# 7. Shape Propagation

Suppose:

```text
Input Shape

(32,10)
```

Batch size:

```text
32
```

Features:

```text
10
```

---

# 8. First Layer

```python
nn.Linear(10,20)
```

Transforms:

```text
(B,10)

↓

(B,20)
```

Result:

```text
(32,20)
```

---

# 9. ReLU Layer

```python
nn.ReLU()
```

Changes values.

Does not change shape.

```text
(32,20)

↓

(32,20)
```

---

# 10. Final Layer

```python
nn.Linear(20,3)
```

Transforms:

```text
(B,20)

↓

(B,3)
```

Result:

```text
(32,3)
```

---

# 11. Full Shape Trace

```text
(32,10)

      ↓

Linear(10,20)

      ↓

(32,20)

      ↓

ReLU

      ↓

(32,20)

      ↓

Linear(20,3)

      ↓

(32,3)
```

---

# 12. Sequential Internals

Conceptually:

```python
model(x)
```

is equivalent
