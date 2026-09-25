# Tensor Memory Model

## Goal

So far, my mental model of a tensor has been:

```text
Tensor
├── dimensions
├── shape
├── numel
└── indexing
```

This explains the logical structure of a tensor, but it does not explain how
the tensor's elements are actually represented in memory.

In this section, I want to understand:

- How multidimensional tensors relate to underlying storage.
- How PyTorch maps multidimensional indices to storage locations.
- What strides mean.
- Why transpose does not necessarily require moving data.
- How multiple tensors can share the same underlying data.
- What contiguous and non-contiguous mean.
- When PyTorch may need to copy data.

The goal is to move from:

> A tensor is an n-dimensional array.

toward a deeper mental model:

```text
Tensor
   │
   ├── underlying data
   │
   └── metadata describing how to interpret that data
```

---

# 1. Logical Tensor vs Physical Storage

Consider:

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])
```

Logically, I see:

```text
10  20  30
40  50  60
```

with:

```text
shape = (2, 3)
```

However, the actual values can be thought of as living in underlying storage.

For a simple contiguous tensor, a useful conceptual picture is:

```text
storage position:

 0   1   2   3   4   5
 │   │   │   │   │   │
 ▼   ▼   ▼   ▼   ▼   ▼
10  20  30  40  50  60
```

This gives me two different ways of thinking about the same tensor:

```text
Logical representation

    10  20  30
    40  50  60

         │
         │ metadata maps logical indices
         │ to storage locations
         ▼

Underlying storage

    10  20  30  40  50  60
```

The important realization is:

> The logical multidimensional representation of a tensor and the underlying
> storage of its elements are related, but they are not the same concept.

---

# 2. The Problem: How Does Indexing Reach Memory?

Consider:

```python
x[1, 2]
```

Logically, this means:

```text
row 1
column 2
```

which gives:

```text
60
```

But underneath, PyTorch needs some way of translating:

```text
(1, 2)
```

into the appropriate location in storage:

```text
storage position 5
```

For our tensor:

```text
x[0, 0] → storage position 0
x[0, 1] → storage position 1
x[0, 2] → storage position 2

x[1, 0] → storage position 3
x[1, 1] → storage position 4
x[1, 2] → storage position 5
```

This leads to the concept of **stride**.

---

# 3. Stride

My intuitive definition of stride is:

> Stride tells me how many storage elements I need to move when I advance
> by one index along a particular tensor dimension.

Consider again:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])
```

We have:

```text
shape = (2, 3)
```

### Moving along dimension 1

Going from:

```text
x[0, 0] → 10
```

to:

```text
x[0, 1] → 20
```

moves one position in storage.

Therefore:

```text
stride along dimension 1 = 1
```

### Moving along dimension 0

Going from:

```text
x[0, 0] → 10
```

to:

```text
x[1, 0] → 40
```

moves three storage positions.

Therefore:

```text
stride along dimension 0 = 3
```

So:

```text
shape  = (2, 3)
stride = (3, 1)
```

I can inspect this in PyTorch using:

```python
x.shape
x.stride()
```

---

# 4. Index to Storage Mapping

For this simple tensor:

```text
shape  = (2, 3)
stride = (3, 1)
```

the storage position can be reasoned about using:

```text
storage_position =
      index_dim0 × stride_dim0
    + index_dim1 × stride_dim1
```

For:

```python
x[1, 2]
```

we have:

```text
index  = (1, 2)
stride = (3, 1)
```

Therefore:

```text
storage position
= 1 × 3 + 2 × 1
= 3 + 2
= 5
```

Storage position 5 contains:

```text
60
```

So conceptually:

```text
logical index
    (1, 2)
       │
       ▼
1 × stride[0] + 2 × stride[1]
       │
       ▼
1 × 3 + 2 × 1
       │
       ▼
       5
       │
       ▼
storage[5]
       │
       ▼
      60
```

This is a much stronger mental model than simply thinking of tensors as
nested Python lists.

---

# 5. Strides for Higher-Dimensional Tensors

Consider a normal contiguous tensor with:

```text
shape = (2, 3, 4)
```

A typical stride would be:

```text
stride = (12, 4, 1)
```

Why?

Moving one position along dimension 2 moves by one element:

```text
dim 2 stride = 1
```

Moving one position along dimension 1 skips a group of 4 elements:

```text
dim 1 stride = 4
```

Moving one position along dimension 0 skips:

```text
3 × 4 = 12 elements
```

Therefore:

```text
dim 0 stride = 12
dim 1 stride = 4
dim 2 stride = 1
```

So:

```text
shape   = ( 2, 3, 4)
stride  = (12, 4, 1)
```

For:

```python
x[1, 2, 3]
```

the storage position is:

```text
1 × 12
+ 2 × 4
+ 3 × 1

= 12 + 8 + 3

= 23
```

---

# 6. Shape Alone Is Not Enough

Initially, I thought I could calculate memory positions using only the shape.

For a basic contiguous tensor, it can look like:

```text
index_i × size of later dimensions
```

This intuition works because the strides of a normal contiguous tensor can
be derived from its shape.

However, this is not the general rule.

The more general mental model is:

```text
index_i × stride_i
```

This distinction becomes important when tensor dimensions are rearranged.

For example, transpose can change strides without moving the underlying data.

---

# 7. Transpose

Consider:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])
```

For `x`:

```text
shape  = (2, 3)
stride = (3, 1)
```

Logically:

```text
10  20  30
40  50  60
```

Now:

```python
y = x.T
```

Logically, `y` looks like:

```text
10  40
20  50
30  60
```

Its shape becomes:

```text
shape = (3, 2)
```

My important realization was:

> PyTorch does not need to physically rearrange all the values just to
> represent this transpose.

The underlying data can remain in the same storage:

```text
10  20  30  40  50  60
```

Instead, the metadata changes.

Conceptually:

```text
x

shape  = (2, 3)
stride = (3, 1)


          transpose
              │
              ▼


y

shape  = (3, 2)
stride = (1, 3)
```

The storage can remain shared:

```text
                Storage
       10 20 30 40 50 60
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
          x             y

shape   (2, 3)        (3, 2)
stride  (3, 1)        (1, 3)
```

---

# 8. Why the New Stride Works

After transpose:

```text
y.shape  = (3, 2)
y.stride = (1, 3)
```

Logically:

```text
10  40
20  50
30  60
```

Suppose I ask for:

```python
y[1, 0]
```

Using the strides:

```text
storage position
= 1 × 1 + 0 × 3
= 1
```

Storage position 1 contains:

```text
20
```

Therefore:

```python
y[1, 0]
```

returns:

```text
20
```

Similarly:

```python
y[2, 1]
```

maps to:

```text
2 × 1 + 1 × 3
= 2 + 3
= 5
```

Storage position 5 contains:

```text
60
```

Therefore:

```python
y[2, 1]
```

returns:

```text
60
```

This proves why changing metadata can produce a different logical tensor
without rearranging the underlying values.

---

# 9. Shared Storage

Since `x` and `y` can refer to the same underlying data, modifying the data
through one tensor can affect what is observed through the other tensor.

Consider:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

y = x.T
```

Now:

```python
y[0, 1] = 999
```

Logically, `y` becomes:

```text
10  999
20   50
30   60
```

But:

```text
y[0, 1]
```

and:

```text
x[1, 0]
```

refer to the same underlying element.

Therefore `x` becomes:

```text
10   20   30
999  50   60
```

This gives me an important mental model:

> Two tensors can have different shapes and strides while referring to the
> same underlying data.

This also introduces the idea of a **view**.

For now, my intuitive understanding of a view is:

> A tensor can provide a different logical interpretation of existing data
> without necessarily owning an independent copy of that data.

I will explore `view()` and related operations separately.

---

# 10. Contiguous Tensors

Consider the original tensor:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])
```

Logical traversal gives:

```text
10 → 20 → 30 → 40 → 50 → 60
```

The underlying storage order is also:

```text
10 → 20 → 30 → 40 → 50 → 60
```

So the logical traversal follows the expected contiguous storage layout.

For this tensor:

```text
shape  = (2, 3)
stride = (3, 1)
```

and:

```python
x.is_contiguous()
```

returns:

```text
True
```

My intuitive definition is:

> A tensor is contiguous when its logical layout corresponds to a standard
> contiguous arrangement of its elements in memory.

---

# 11. Why a Transposed Tensor Can Be Non-Contiguous

After:

```python
y = x.T
```

the logical tensor is:

```text
10  40
20  50
30  60
```

Logical traversal is therefore:

```text
10 → 40 → 20 → 50 → 30 → 60
```

But the shared storage remains:

```text
10 → 20 → 30 → 40 → 50 → 60
```

So the logical layout no longer corresponds to the standard contiguous
layout for a tensor of `y`'s shape.

Therefore:

```python
y.is_contiguous()
```

returns:

```text
False
```

The important point is not to memorize:

> transpose = non-contiguous

Instead, I want to understand why it can happen:

```text
transpose
    │
    ▼
dimensions/strides change
    │
    ▼
same underlying storage
    │
    ▼
new logical layout may no longer be contiguous
```

---

# 12. Making a Tensor Contiguous

Suppose:

```python
y = x.T
```

and `y` is non-contiguous.

Now:

```python
z = y.contiguous()
```

For our example, PyTorch needs a contiguous representation corresponding
to `y`'s logical layout.

`y` logically contains:

```text
10  40
20  50
30  60
```

A contiguous arrangement corresponding to this logical layout can be
conceptually pictured as:

```text
10  40  20  50  30  60
```

So conceptually:

```text
Original/shared data

10  20  30  40  50  60
        │
        │ y.contiguous()
        ▼

Contiguous representation for y's layout

10  40  20  50  30  60
```

This means that making a non-contiguous tensor contiguous can require
copying/rearranging its data.

---

# 13. Metadata Operation vs Data Copy

This introduced an important distinction.

Some tensor operations can be performed largely by changing metadata:

```text
same data
   +
different shape/stride interpretation
```

Other operations may require creating a new physical arrangement of the data.

Conceptually:

```text
               Tensor Operation
                      │
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
    Metadata change          Data movement
           │                     │
    reuse existing              may need
       data                   new storage
```

For example, in our current example:

```text
x.T
```

can reuse the underlying data with different metadata.

Whereas converting that non-contiguous transpose into a contiguous
representation may require data movement.

This distinction will become important when learning about:

- `view()`
- `reshape()`
- `transpose()`
- `permute()`
- `contiguous()`
- performance implications

---

# 14. Hands-On Experiment

I can verify the current mental model using:

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

y = x.T

print("x")
print(x)
print("shape:", x.shape)
print("stride:", x.stride())
print("contiguous:", x.is_contiguous())

print()

print("y")
print(y)
print("shape:", y.shape)
print("stride:", y.stride())
print("contiguous:", y.is_contiguous())
```

Before running this code, I should predict:

```text
x:

shape      = (2, 3)
stride     = (3, 1)
contiguous = True


y:

shape      = (3, 2)
stride     = (1, 3)
contiguous = False
```

---

# 15. Hands-On Experiment: Shared Data

I can also verify whether changing `y` affects `x`.

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

y = x.T

print("Before modification:")

print("x:")
print(x)

print("y:")
print(y)

y[0, 1] = 999

print("\nAfter modifying y[0, 1]:")

print("x:")
print(x)

print("y:")
print(y)
```

My prediction is:

```text
x:

10   20   30
999  50   60


y:

10   999
20    50
30    60
```

because:

```text
y[0, 1]
```

and:

```text
x[1, 0]
```

map to the same underlying element.

---

# 16. My Mistakes and Refinements

## Initial Thought

Initially, my tensor mental model was mostly:

> A tensor is an n-dimensional array.

That is useful, but incomplete.

## Refined Understanding

A more useful low-level mental model is:

```text
                    Tensor
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     underlying data          metadata
                                  │
                           ┌──────┴──────┐
                           │             │
                         shape         stride
```

The metadata tells PyTorch how to logically interpret and access the
underlying data.

---

# 17. Interview Answer: What Is a Stride?

If asked:

> What is a stride in PyTorch?

My current answer would be:

> A stride tells us how many elements in the underlying storage we need to
> move to advance by one position along a particular tensor dimension.

For example:

```text
shape  = (2, 3)
stride = (3, 1)
```

means:

- Moving one index along dimension 0 moves 3 storage elements.
- Moving one index along dimension 1 moves 1 storage element.

---

# 18. Interview Answer: Does Transpose Copy Data?

If asked:

> Does transposing a tensor necessarily rearrange all of its data?

My answer would be:

> No. A transpose can often be represented by changing tensor metadata,
> particularly the shape and strides, while referring to the same underlying
> data. Because the resulting logical layout can differ from the standard
> contiguous layout, the transposed tensor may be non-contiguous.

---

# 19. Interview Answer: What Does Contiguous Mean?

If asked:

> What does it mean for a tensor to be contiguous?

My current answer would be:

> A contiguous tensor has a memory layout that corresponds to the standard
> contiguous layout expected for its shape. Its strides describe that
> contiguous arrangement. Operations such as transpose can create a
> non-contiguous view because they can change how the same underlying data is
> interpreted without physically rearranging it.

---

# 20. Teach It to Someone

If I had to explain the memory model to another engineer:

> A tensor should not be thought of only as a multidimensional nested array.
> There is underlying data, and the tensor has metadata that describes how
> that data should be interpreted.
>
> Shape describes the logical size along each dimension.
>
> Stride tells us how far to move through the underlying storage when moving
> one position along a dimension.
>
> Because logical interpretation is separated from the underlying data,
> operations such as transpose can sometimes create a different tensor view
> without physically rearranging the data.
>
> This is also why two tensors can have different shapes and strides but
> still refer to the same underlying data.

---

# 21. Current Mental Model

My tensor mental model has now evolved from:

```text
Tensor
    =
N-dimensional array
```

to:

```text
                         Tensor
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      underlying data                  metadata
                                             │
                               ┌─────────────┼─────────────┐
                               │             │             │
                             shape         stride        ...
                               │             │
                               │             ▼
                               │       how to move through
                               │       underlying storage
                               │
                               ▼
                       logical dimensions
```

This distinction explains how:

```text
same underlying data
         │
         ├── tensor x
         │   shape  = (2, 3)
         │   stride = (3, 1)
         │
         └── tensor y
             shape  = (3, 2)
             stride = (1, 3)
```

can provide different logical representations of the same values.

---

# 22. What Comes Next

I now have a basic understanding of:

```text
Tensor memory model
├── logical tensor
├── underlying data
├── shape
├── stride
├── logical index → storage position
├── transpose
├── shared data
├── contiguous
└── non-contiguous
```
