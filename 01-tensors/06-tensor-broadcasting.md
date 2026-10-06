# Tensor Broadcasting

## Goal

So far, I have learned how to reason about:

```text
Tensor
├── dimensions
├── shape
├── indexing
├── storage
├── stride
├── contiguity
├── views
├── reshape
├── dtype
└── device
```

Now I want to understand how PyTorch performs operations between tensors
whose shapes are not exactly the same.

This is where **broadcasting** comes in.

The goals of this section are to understand:

- What broadcasting is.
- Why tensors with different shapes can sometimes participate in the same operation.
- How broadcasting compatibility is determined.
- Why dimensions are compared from the right.
- What happens to dimensions of size 1.
- How missing dimensions are handled.
- How broadcasting can work without physically copying data.
- How stride 0 can represent logical expansion.
- Why mutation of expanded tensors requires care.

---

# 1. The Problem Broadcasting Solves

Consider:

```python
import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

y = torch.tensor([10, 20, 30])
```

Their shapes are:

```text
x.shape = (2, 3)
y.shape = (3,)
```

The shapes are different.

But PyTorch allows:

```python
z = x + y
```

and logically produces:

```text
11 22 33
14 25 36
```

Conceptually, it looks as if:

```text
y =

10 20 30
```

behaved like:

```text
10 20 30
10 20 30
```

during the operation.

This behavior is called **broadcasting**.

---

# 2. Initial Mental Model

My initial intuition was:

> If one tensor is smaller than another tensor, PyTorch can repeat values from
> the smaller tensor so that the shapes match.

This is useful for visualization, but it can be misleading.

The important refinement is:

> Broadcasting is a logical expansion. It does not necessarily mean PyTorch
> physically copies and duplicates all the values of the smaller tensor.

Before understanding the memory behavior, I first need to understand when
two shapes are broadcast-compatible.

---

# 3. Broadcasting Starts With Shapes

When reasoning about broadcasting, I should avoid starting with phrases such
as:

> values along dimension 0 are broadcast along dimension 1

This quickly becomes confusing.

Instead, I should always begin with the tensor shapes.

The process is:

```text
Tensor shapes
     │
     ▼
right-align them
     │
     ▼
compare corresponding dimensions
     │
     ▼
determine compatibility
     │
     ▼
determine result shape
```

---

# 4. Broadcasting Compatibility Rule

When comparing two tensor shapes, start from the **rightmost dimension**.

Two corresponding dimensions are compatible when:

```text
1. Their sizes are equal

OR

2. One of them has size 1
```

Missing leading dimensions can conceptually be treated as dimensions of
size 1.

Therefore:

```text
same size
   ↓
compatible


one dimension = 1
   ↓
compatible through broadcasting


different sizes and neither is 1
   ↓
incompatible
```

---

# 5. Example: `(2, 3)` + `(3,)`

Consider:

```python
x = torch.zeros(2, 3)
y = torch.zeros(3)
```

Shapes:

```text
x → (2, 3)
y →    (3,)
```

The correct approach is to align them from the right:

```text
x → (2, 3)
y → (1, 3)
```

The leading `1` is conceptual and helps with reasoning about the missing
dimension.

Now compare:

```text
          dim 0    dim 1

x            2        3
y            1        3
             │        │
             │        └── 3 == 3
             │
             └─────────── 1 broadcasts to 2
```

Therefore:

```text
WORKS

result shape = (2, 3)
```

Conceptually:

```text
y:

10 20 30
```

behaves like:

```text
10 20 30
10 20 30
```

during the operation.

---

# 6. Example: `(2, 3)` + `(2, 1)`

Consider:

```python
x = torch.zeros(2, 3)

y = torch.tensor([
    [10],
    [20]
])
```

Shapes:

```text
x → (2, 3)
y → (2, 1)
```

Compare corresponding dimensions:

```text
          dim 0    dim 1

x            2        3
y            2        1
             │        │
             │        └── 1 broadcasts to 3
             │
             └─────────── 2 == 2
```

Therefore:

```text
WORKS

result shape = (2, 3)
```

Conceptually:

```text
y:

10
20
```

behaves during the operation like:

```text
10 10 10
20 20 20
```

A precise explanation is:

> Dimension 1 of `y` has size 1, so it can broadcast to size 3 to match
> dimension 1 of `x`.

---

# 7. An Important Mistake: Aligning From the Left

Consider:

```python
x = torch.zeros(2, 3)
y = torch.zeros(2)
```

Shapes:

```text
x → (2, 3)
y →    (2,)
```

Initially, I reasoned:

```text
2 matches 2

and the remaining dimension can be treated as 1
```

and predicted that broadcasting would work.

This was incorrect.

The mistake was aligning dimensions from the **left**.

Broadcasting aligns dimensions from the **right**.

Correct alignment:

```text
x → (2, 3)
y → (1, 2)
```

Now compare:

```text
          dim 0    dim 1

x            2        3
y            1        2
             │        │
             │        └── 3 vs 2
             │             incompatible
             │
             └─────────── 2 vs 1
                           compatible
```

The final dimension fails because:

```text
3 != 2
```

and:

```text
neither dimension has size 1
```

Therefore:

```text
(2, 3) + (2,)
```

fails.

This gives me an important rule:

> Missing dimensions are leading dimensions because shapes are aligned from
> the right.

---

# 8. Higher-Dimensional Example

Consider:

```python
x = torch.zeros(4, 3, 2)
y = torch.zeros(3, 1)
```

Shapes:

```text
x → (4, 3, 2)
y → (   3, 1)
```

Right-align them:

```text
x → (4, 3, 2)
y → (1, 3, 1)
```

Now compare:

```text
          dim 0    dim 1    dim 2

x            4        3        2
y            1        3        1
             │        │        │
             │        │        └── 1 broadcasts to 2
             │        │
             │        └─────────── 3 == 3
             │
             └──────────────────── 1 broadcasts to 4
```

Therefore:

```text
WORKS

result shape = (4, 3, 2)
```

---

# 9. General Shape Reasoning Process

Whenever I see an operation involving tensors with different shapes, I
should reason using this process:

```text
Step 1
Write both shapes.

        ↓

Step 2
Right-align them.

        ↓

Step 3
Treat missing leading dimensions
as size 1 for reasoning.

        ↓

Step 4
Compare each corresponding dimension.

        ↓

same size?
    → compatible

one is 1?
    → compatible through broadcasting

otherwise?
    → incompatible

        ↓

Step 5
Determine the resulting dimension size.
```

For compatible dimensions, the resulting size is the larger compatible
dimension.

---

# 10. Broadcasting Does Not Mean Physical Duplication

Consider:

```python
x = torch.zeros(1_000_000, 3)

y = torch.tensor([10.0, 20.0, 30.0])

z = x + y
```

Shapes:

```text
x → (1_000_000, 3)
y → (            3,)
```

Right-aligning:

```text
x → (1_000_000, 3)
y → (        1, 3)
```

Shape reasoning suggests that `y` behaves as though it were:

```text
10 20 30
10 20 30
10 20 30
...
```

one million times.

A naive implementation could physically create one million copies.

But that would be unnecessary.

My deeper mental model is:

> Broadcasting represents logical expansion and does not necessarily require
> physically duplicating the broadcasted tensor's data.

This connects broadcasting directly to the storage and stride concepts I
learned earlier.

---

# 11. Broadcasting and Strides

Start with:

```python
y = torch.tensor([10, 20, 30])
```

We have:

```text
shape  = (3,)
stride = (1,)
```

The storage can be visualized as:

```text
storage index:    0    1    2

value:           10   20   30
```

Now suppose we want `y` to logically behave like:

```text
10 20 30
10 20 30
```

with:

```text
shape = (2, 3)
```

Ask the stride question:

> How many storage positions should I move when I advance one logical index
> along each dimension?

---

# 12. Stride Along Dimension 1

Moving from:

```text
10 → 20
```

requires moving one storage position.

Therefore:

```text
stride along dimension 1 = 1
```

---

# 13. Stride Along the Broadcasted Dimension

Now move from:

```text
logical[0, 0]
```

to:

```text
logical[1, 0]
```

Both values should be:

```text
10
```

We want to access exactly the same physical storage position.

Therefore:

```text
stride along dimension 0 = 0
```

The logically expanded tensor can therefore have:

```text
shape  = (2, 3)
stride = (0, 1)
```

This introduces an important concept:

> A stride of 0 means that moving along that logical dimension does not move
> to another position in the underlying storage.

---

# 14. Proving Stride 0 With the Index Formula

The expanded tensor has:

```text
shape  = (2, 3)
stride = (0, 1)
```

Using the storage-position formula:

```text
storage_position =
      index_dim0 × stride_dim0
    + index_dim1 × stride_dim1
```

For:

```text
expanded[0, 0]
```

we get:

```text
0 × 0 + 0 × 1
=
0
```

Therefore:

```text
expanded[0, 0] → storage[0] → 10
```

For:

```text
expanded[1, 0]
```

we get:

```text
1 × 0 + 0 × 1
=
0
```

Therefore:

```text
expanded[1, 0] → storage[0] → 10
```

Both logical positions map to the same physical storage location.

Similarly:

```text
expanded[0, 1]

0 × 0 + 1 × 1
=
1
```

and:

```text
expanded[1, 1]

1 × 0 + 1 × 1
=
1
```

Both map to:

```text
storage[1] → 20
```

So logically:

```text
10 20 30
10 20 30
```

can still use physical storage containing only:

```text
10 20 30
```

---

# 15. Broadcasting as Logical Expansion

This gives me a deeper mental model for broadcasting:

```text
small tensor storage

10 20 30
    │
    ▼

logical expansion

10 20 30
10 20 30
10 20 30
...
```

The logical expansion does not necessarily mean:

```text
physical storage

10 20 30 10 20 30 10 20 30 ...
```

Instead, repeated logical access can reuse the same storage.

Conceptually:

```text
                 Broadcasting
                      │
                      ▼
             compatible dimensions
                      │
                      ▼
              logical expansion
                      │
                      ▼
       physical duplication not required
                      │
                      ▼
              stride 0 can represent
              repeated logical access
```

---

# 16. `expand()`

PyTorch exposes this idea directly through `expand()`.

Consider:

```python
import torch

y = torch.tensor([10, 20, 30])

expanded = y.expand(2, 3)

print(expanded)

print("y shape:", y.shape)
print("y stride:", y.stride())

print()

print("expanded shape:", expanded.shape)
print("expanded stride:", expanded.stride())
```

My prediction is:

```text
y.shape  = (3,)
y.stride = (1,)
```

and:

```text
expanded.shape  = (2, 3)
expanded.stride = (0, 1)
```

Logically:

```text
expanded =

10 20 30
10 20 30
```

But this does not require six independent stored values.

---

# 17. Expanded Tensors Can Alias the Same Storage

Because:

```text
expanded.stride = (0, 1)
```

multiple logical positions can point to the same physical element.

For example:

```text
expanded[0,0] ─────┐
                    │
                    ▼
                 storage[0]
                    ▲
                    │
expanded[1,0] ─────┘
```

This means that:

```python
expanded[0, 0]
```

and:

```python
expanded[1, 0]
```

both access the same underlying storage element.

---

# 18. Mutation of an Expanded Tensor

Suppose:

```python
y = torch.tensor([10, 20, 30])

expanded = y.expand(2, 3)
```

Then:

```python
expanded[0, 0] = 999
```

changes the underlying element:

```text
storage[0]
```

Because both:

```text
expanded[0, 0]
```

and:

```text
expanded[1, 0]
```

map to that storage position, the logical result becomes:

```text
999 20 30
999 20 30
```

The original tensor also observes the change:

```text
y = [999, 20, 30]
```

This gives me an important lesson:

> Expanded views can contain multiple logical indices that refer to the same
> physical storage location.

Therefore, in-place mutation of expanded tensors requires special care.

---

# 19. Broadcasting Does Not Mean the Result Is Free

An important distinction is between:

```text
broadcasted input
```

and:

```text
operation result
```

Consider:

```python
x = torch.zeros(1_000_000, 3)

y = torch.tensor([10.0, 20.0, 30.0])

z = x + y
```

`y` does not need to be physically duplicated one million times simply to
participate in the operation.

However:

```text
z.shape = (1_000_000, 3)
```

The result itself contains three million logical result elements.

So I should not interpret broadcasting as:

> Broadcasting means the entire operation requires no additional memory.

A better distinction is:

```text
broadcasted operand

small storage
     │
     ▼
logical expansion
     │
     ▼
physical duplication unnecessary
```

versus:

```text
operation output

result shape
     │
     ▼
actual result values
     │
     ▼
result representation/storage required
```

---

# 20. Larger Broadcasting Challenge

Consider:

```python
a = torch.zeros(8, 1, 6, 1)

b = torch.zeros(7, 1, 5)
```

Shapes:

```text
a → (8, 1, 6, 1)
b → (   7, 1, 5)
```

Right-align:

```text
a → (8, 1, 6, 1)
b → (1, 7, 1, 5)
```

Compare:

```text
dim 0:

8 vs 1
→ compatible
→ b broadcasts from 1 to 8


dim 1:

1 vs 7
→ compatible
→ a broadcasts from 1 to 7


dim 2:

6 vs 1
→ compatible
→ b broadcasts from 1 to 6


dim 3:

1 vs 5
→ compatible
→ a broadcasts from 1 to 5
```

Therefore:

```text
a + b
```

works.

The result shape is:

```text
(8, 7, 6, 5)
```

A precise explanation is:

> `a` broadcasts along dimensions 1 and 3. After accounting for the missing
> leading dimension, `b` broadcasts along dimensions 0 and 2.

---

# 21. Interview Answer: What Is Broadcasting?

If asked:

> What is broadcasting in PyTorch?

My current answer would be:

> Broadcasting allows tensor operations to work on certain tensors whose
> shapes are different but compatible. PyTorch compares their dimensions
> starting from the right. Corresponding dimensions are compatible when they
> have the same size or when one has size 1. Missing leading dimensions can
> conceptually be treated as size 1. Dimensions of size 1 can then logically
> expand to match the corresponding dimension of the other tensor.

---

# 22. Interview Answer: Does Broadcasting Copy Data?

If asked:

> Does broadcasting physically copy the smaller tensor multiple times?

My answer would be:

> Not necessarily. Broadcasting is primarily a logical expansion. A
> broadcasted or expanded representation can reuse the same underlying
> storage rather than physically duplicating values. A stride of zero can
> represent a broadcasted dimension because moving along that logical
> dimension continues to access the same underlying storage position.

---

# 23. Interview Answer: What Does a Stride of 0 Mean?

If asked:

> What does stride 0 mean?

My answer would be:

> A stride of zero means that advancing an index along that tensor dimension
> does not advance through the underlying storage. Multiple logical positions
> along that dimension can therefore refer to the same physical element. This
> is useful for representing logical expansion during broadcasting without
> physically duplicating data.

---

# 24. Interview Answer: Why Compare Shapes From the Right?

If asked:

> How do you determine whether two tensors can broadcast?

I would answer:

> I right-align the shapes and compare corresponding dimensions from the
> rightmost dimension. Each pair must either have equal sizes or one of them
> must have size 1. Missing leading dimensions can be treated as size 1 for
> reasoning. If any corresponding pair has different sizes and neither is 1,
> the shapes are not broadcast-compatible.

For example:

```text
(2, 3)
   (2,)
```

becomes:

```text
(2, 3)
(1, 2)
```

and fails because:

```text
3 vs 2
```

are different and neither is 1.

---

# 25. Teach It to Someone

If I were teaching broadcasting to another engineer, I would say:

> Broadcasting lets PyTorch perform elementwise operations between tensors
> whose shapes are different but compatible.
>
> Don't start by imagining values being copied.
>
> Start with the shapes.
>
> Right-align the shapes and compare each dimension. Equal sizes are
> compatible. A dimension of size 1 can expand to match the other dimension.
> Missing leading dimensions can be treated as size 1.
>
> The important low-level insight is that this expansion can be logical
> rather than physical. PyTorch can reuse the same underlying values rather
> than storing repeated copies.
>
> One way to represent this is with stride 0. If a dimension has stride 0,
> moving along that logical dimension keeps accessing the same underlying
> storage location.
>
> This is why broadcasting can be memory-efficient, but it is also why
> modifying expanded views requires care.

---

# 26. Mistakes and Corrections

## Mistake 1: Aligning Shapes From the Left

Given:

```text
x = (2, 3)
y = (2,)
```

I initially reasoned:

```text
2 matches 2
```

and assumed the remaining dimension could broadcast.

That was incorrect.

The shapes must be aligned from the right:

```text
x → (2, 3)
y → (1, 2)
```

Now:

```text
3 vs 2
```

is incompatible.

### Lesson

> Broadcasting shape comparison starts from the rightmost dimensions.

---

## Mistake 2: Describing Broadcasting Using "Values Along Dimensions"

I initially used explanations like:

> Values along dimension 0 get broadcast along dimension 1.

This became confusing.

A clearer explanation is:

> Dimension 1 has size 1 and broadcasts to size 3.

### Lesson

When reasoning about broadcasting:

```text
talk about:

dimension
+
dimension size
+
corresponding dimension
```

rather than trying to describe blocks, rows, or values being copied.

---

## Mistake 3: Thinking Broadcasting Means Copying Values

A simple visualization makes it look like:

```text
[10, 20, 30]

becomes

[10, 20, 30]
[10, 20, 30]
```

This is logically useful but does not imply physical duplication.

The same storage can be logically reused.

### Lesson

> Logical expansion and physical duplication are different concepts.

---

# 27. Current Mental Model

My current broadcasting mental model has two layers.

## Shape Layer

```text
                       Broadcasting
                            │
                            ▼
                     right-align shapes
                            │
                            ▼
                 compare each dimension
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
            equal       one is 1       otherwise
              │             │             │
              ▼             ▼             ▼
         compatible     broadcast      incompatible
```

## Memory Layer

```text
                Broadcasted dimension
                         │
                         ▼
                 logical expansion
                         │
                         ▼
               reuse underlying data
                         │
                         ▼
                     stride 0
                         │
                         ▼
           multiple logical positions can
           reference the same physical value
```

---

# 28. Connection to Previous Concepts

Broadcasting now connects directly to concepts I already learned:

```text
shape
  │
  └── determines broadcasting compatibility


stride
  │
  └── stride 0 can represent repeated logical access


storage
  │
  └── values do not necessarily need to be duplicated


views
  │
  └── logical representation can differ from physical storage


expand
  │
  └── exposes logical expansion explicitly
```

This reinforces one of the most important ideas I have learned about tensors:

> The logical tensor I see is not necessarily a direct representation of how
> all of its apparent values are physically arranged in memory.

---

# 29. Tensor Mental Model So Far

At this point, my tensor mental model has grown considerably.

```text
                              Tensor
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
      shape                    stride                   dtype
        │                        │                        │
        ▼                        ▼                        ▼
 logical dimensions       storage traversal       value representation

                                 │
                                 ▼
                               device
                                 │
                                 ▼
                          where data lives
```

On top of those properties, tensor operations introduce behaviors such as:

```text
Tensor Operations
│
├── indexing
├── slicing
├── transpose
├── views
├── reshape
├── device transfer
└── broadcasting
```

The common theme is:

```text
What does the tensor look like logically?

            vs

How is its data actually represented,
stored, accessed, and moved?
```

---

# 30. What I Understand Now

I can now reason about broadcasting using:

```text
1. Write shapes.

2. Right-align shapes.

3. Missing leading dimensions behave like size 1.

4. Compare corresponding dimensions.

5. Equal sizes are compatible.

6. Size 1 can broadcast.

7. Different sizes where neither is 1 are incompatible.

8. Determine the resulting shape.

9. Remember that logical expansion does not necessarily imply data copies.

10. Remember that stride 0 can represent repeated access to the same
    underlying value.
```

---

# 31. What Comes Next

The tensor foundation now includes:

```text
Tensor
├── dimensions
├── shape
├── indexing
├── slicing
├── storage
├── strides
├── contiguity
├── views
├── reshape
├── dtype
├── device
└── broadcasting
```

The next major topic is **Autograd**.

This will move from:

```text
How does a tensor represent data?
```

to:

```text
How does PyTorch track computations involving tensors?
```

Questions I want to answer include:

```text
What is requires_grad?

What exactly does PyTorch track?

What is a computation graph?

When is that graph created?

What is grad_fn?

What is a leaf tensor?

What does backward() actually do?

Where are gradients stored?

Why do gradients accumulate?

Why do we normally call zero_grad()?

What happens if we modify tensors in-place?

What does torch.no_grad() do?

How does all of this eventually train a neural network?
```

The goal will not be to memorize:

```python
loss.backward()
```

Instead, I want to understand exactly what changes inside PyTorch before and
after that line executes.
