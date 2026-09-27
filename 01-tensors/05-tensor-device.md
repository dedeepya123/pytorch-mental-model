# Tensor Device

## Goal

So far, my tensor mental model includes:

```text
Tensor
├── shape
├── stride
├── dtype
└── device
```

I understand:

```text
shape
  ↓
What is the logical structure of the tensor?

stride
  ↓
How do I traverse its underlying storage?

dtype
  ↓
How is each individual element represented?

device
  ↓
Where is the tensor's data allocated?
```

In this section, I want to understand:

- What a tensor device means.
- CPU tensors vs GPU tensors.
- What happens when a tensor moves between devices.
- Where tensor operations execute.
- Why tensors participating in an operation normally need compatible devices.
- Why CPU-GPU data transfers matter for performance.
- How tensor devices eventually connect to `model.to(device)`.

---

# 1. What Is a Device?

Every tensor has a device associated with it.

For example:

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

print(x.device)
```

Normally this tensor is created on the CPU.

Conceptually:

```text
x
│
├── shape  = (3,)
├── dtype  = float32
└── device = cpu
```

My mental model is:

> The device tells me where the tensor's data is allocated.

For example:

```text
device = cpu
    ↓
tensor data is allocated in CPU-accessible memory
```

or:

```text
device = cuda
    ↓
tensor data is allocated on a CUDA device
```

The location of the tensor's data also determines where operations involving
that tensor normally execute.

---

# 2. CPU and GPU

A simplified machine model is:

```text
                   Computer
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
            CPU               GPU
             │                 │
             ▼                 ▼
         CPU memory         GPU memory
```

This distinction matters because CPU memory and GPU memory represent
different device locations.

A CPU tensor might look conceptually like:

```text
CPU

CPU memory
┌─────────────────┐
│ 1.0  2.0  3.0   │
└─────────────────┘
        │
        ▼
        x

device = cpu
```

A CUDA tensor might look like:

```text
GPU

GPU memory
┌─────────────────┐
│ 1.0  2.0  3.0   │
└─────────────────┘
        │
        ▼
        y

device = cuda
```

---

# 3. Creating a Tensor on a Device

A tensor can be created directly on a device.

For example:

```python
x = torch.tensor(
    [1.0, 2.0, 3.0],
    device="cuda"
)
```

Conceptually:

```text
GPU
 │
 ▼
allocate tensor data
 │
 ▼
1.0 2.0 3.0
 │
 ▼
x.device = cuda
```

A common pattern is:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor(
    [1.0, 2.0, 3.0],
    device=device
)
```

This allows the code to select CUDA when available and otherwise use CPU.

---

# 4. Moving a Tensor Between Devices

Consider:

```python
x = torch.tensor([1.0, 2.0, 3.0])

y = x.to("cuda")
```

Initially:

```text
x.device = cpu
```

After:

```python
y = x.to("cuda")
```

we have:

```text
y.device = cuda
```

My important mental model is:

> Moving a tensor between devices requires its data to become available in
> storage on the destination device.

Conceptually:

```text
CPU                               GPU

CPU memory                        GPU memory

1.0 2.0 3.0    ── transfer ──►   1.0 2.0 3.0
     │                                │
     ▼                                ▼
     x                                y

device=cpu                       device=cuda
```

---

# 5. Device Transfer vs Transpose

This is very different from something like:

```python
y = x.T
```

When transpose creates a view, the underlying data can remain shared.

Conceptually:

```text
transpose

same device
same underlying data

        storage
           │
      ┌────┴────┐
      │         │
      ▼         ▼
      x         y

different metadata
```

But:

```python
y = x.to("cuda")
```

is different.

```text
CPU storage
    │
    │ transfer
    ▼
GPU storage
```

So:

```text
transpose
    ↓
can change metadata
while sharing data
```

whereas:

```text
CPU → GPU
    ↓
requires data on
another device
```

This gives me another important distinction:

```text
Metadata change
        vs
Device transfer
```

---

# 6. Moving Back to CPU

Consider:

```python
x = torch.tensor([1.0, 2.0, 3.0])

y = x.to("cuda")

z = y.to("cpu")
```

Conceptually:

```text
Step 1

CPU

1.0 2.0 3.0
     │
     x
```

Then:

```text
Step 2

CPU                         GPU

1.0 2.0 3.0                1.0 2.0 3.0
     │                           │
     x                           y
```

Then:

```text
Step 3

CPU                         GPU

1.0 2.0 3.0                1.0 2.0 3.0
     │                           │
     x                           y


1.0 2.0 3.0
     │
     z
```

The important mental model is:

> Moving `y` back to the CPU does not mean PyTorch "returns to" `x` and
> automatically reconnects the result to `x`'s original storage.

Instead:

```text
x CPU storage
     │
     │ transfer
     ▼
y GPU storage
     │
     │ transfer
     ▼
z CPU storage
```

The values may be equal, but that does not mean the tensors share the same
storage.

---

# 7. `.to()` Does Not Always Mean "Copy"

I should avoid memorizing:

> `.to()` always copies a tensor.

Suppose:

```python
x = torch.tensor([1.0, 2.0, 3.0])

y = x.to("cpu")
```

If `x` is already on the CPU with the requested dtype, there may be no
device or dtype conversion to perform.

A better mental model is:

```text
                 x.to(target)
                      │
                      ▼
       Are device and dtype already
          compatible with target?
                 /          \
               YES          NO
                │            │
                ▼            ▼
             may reuse    conversion /
             original      transfer
```

So:

> `.to()` describes the desired device/dtype, not a requirement that a new
> independent copy must always be produced.

---

# 8. Where Does Computation Happen?

Consider:

```python
x = torch.tensor(
    [1.0, 2.0, 3.0],
    device="cuda"
)

y = x * 2
```

Since `x` is a CUDA tensor, the multiplication executes using the CUDA
backend.

Conceptually:

```text
GPU
 │
 ▼
x
