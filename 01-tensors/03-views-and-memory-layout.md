# Views and Memory Layout

## Goal

In the previous section, I learned that a tensor can be thought of as:

```text
                         Tensor
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      underlying data                  metadata
                                             │
                                      ┌──────┴──────┐
                                      │             │
                                    shape         stride
```

This means that changing the logical representation of a tensor does not
always require moving or copying its underlying data.

In this section, I want to understand:

- What a view is.
- How multiple tensors can share the same underlying data.
- How `view()` works.
- How transpose differs from reshape/view.
- Why some `view()` operations fail.
- Why `.contiguous()` can be needed.
- How `reshape()` differs from `view()`.
- How to reason about metadata-only operations versus data copies.

The main question is:

> When can PyTorch create a different logical tensor by changing metadata,
> and when does it need to copy or rearrange data?

---

# 1. What Is a View?

My current mental model is:

> A view is a tensor that gives a different logical interpretation of existing
> underlying data without necessarily creating an independent copy of that data.

Conceptually:

```text
                    Underlying Data
                          │
                 10 20 30 40 50 60
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
           Tensor A    Tensor B    Tensor C
              │           │           │
           metadata    metadata    metadata
```

The different tensors can have different metadata while referring to the
same underlying data.

This means:

```text
modify through one view
          │
          ▼
underlying data changes
          │
          ▼
change may be visible through
other tensors sharing that data
```

---

# 2. `view()`

Consider:

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])
```

For `x`:

```text
shape      = (2, 3)
stride     = (3, 1)
contiguous = True
```

Its underlying data can be visualized as:

```text
10 20 30 40 50 60
```

Now:

```python
a = x.view(3, 2)
```

Logically, `a` becomes:

```text
10 20
30 40
50 60
```

The requested logical traversal is still:

```text
10 → 20 → 30 → 40 → 50 → 60
```

which matches the existing data layout.

PyTorch therefore does not need to rearrange the elements.

Conceptually:

```text
                    underlying data

                 10 20 30 40 50 60
                       /     \
                      /       \
                     ▼         ▼

                     x         a

shape              (2,3)     (3,2)

stride             (3,1)     (2,1)
```

The tensors have different metadata but can refer to the same underlying
data.

---

# 3. Sharing Data Through a View

Because `a` and `x` share the underlying data:

```python
a[0, 0] = 999
```

also changes the value observed through:

```python
x[0, 0]
```

Conceptually:

```text
              x                  a
              │                  │
              └────────┬─────────┘
                       │
                       ▼

             underlying data

          999 20 30 40 50 60
```

This gives me an important rule:

> A view is not necessarily an independent copy.

Therefore, modifying a view can modify data visible through another tensor
sharing the same underlying storage.

---

# 4. `view()` Is Not the Same as `transpose()`

This distinction is important.

Start with:

```python
x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])
```

Then:

```python
a = x.view(3, 2)
```

gives:

```text
10 20
30 40
50 60
```

But:

```python
y = x.T
```

gives:

```text
10 40
20 50
30 60
```

Both tensors have:

```text
shape = (3, 2)
```

but they contain different logical arrangements.

```text
x.view(3, 2)             x.T

10 20                    10 40
30 40                    20 50
50 60                    30 60
```

This reinforces an important lesson:

> Shape alone does not completely describe how a tensor interprets its
> underlying data.

Strides matter.

For example:

```text
x.view(3,2)

shape  = (3,2)
stride = (2,1)
```

whereas:

```text
x.T

shape  = (3,2)
stride = (1,3)
```

---

# 5. Why Transpose Is Different

Start with:

```text
x:

10 20 30
40 50 60

shape  = (2,3)
stride = (3,1)
```

After:

```python
y = x.T
```

the logical tensor becomes:

```text
10 40
20 50
30 60
```

but its underlying data does not need to be rearranged.

It can remain:

```text
10 20 30 40 50 60
```

Instead, the metadata changes:

```text
x

shape  = (2,3)
stride = (3,1)

        │
        │ transpose
        ▼

y

shape  = (3,2)
stride = (1,3)
```

This makes `y` a different logical interpretation of the same underlying
data.

---

# 6. Why `view()` Can Fail

Now consider:

```python
y = x.T
```

Logically:

```text
y =

10 40
20 50
30 60
```

So the logical traversal of `y` is:

```text
10 → 40 → 20 → 50 → 30 → 60
```

But its shared underlying data is still arranged as:

```text
10 → 20 → 30 → 40 → 50 → 60
```

Now suppose I ask for:

```python
y.view(6)
```

The requested 1-D tensor would logically need to be:

```text
[10, 40, 20, 50, 30, 60]
```

But the existing data is laid out as:

```text
[10, 20, 30, 40, 50, 60]
```

For the requested 1-D view, there is no appropriate single stride that
produces:

```text
10 → 40 → 20 → 50 → 30 → 60
```

from the existing layout.

The storage jumps would effectively look like:

```text
10 → 40    +3
40 → 20    -2
20 → 50    +3
50 → 30    -2
30 → 60    +3
```

A simple 1-D view needs a consistent stride.

Therefore the requested view cannot be represented using the existing
layout without rearranging the data.

---

# 7. The Important `view()` Mental Model

A useful beginner mental model is:

> `view()` changes the logical shape while continuing to use the same
> underlying data. It does not silently solve an incompatible layout by
> copying and rearranging the elements.

Therefore, conceptually:

```text
                    view(...)
                       │
                       ▼

       Can the requested view be represented
        using the existing memory layout
            without copying the data?

                 /             \
               YES             NO
                │               │
                ▼               ▼

          create a view        fail
```

For example:

```python
x.view(3, 2)
```

works for our contiguous `x`.

But:

```python
y = x.T
y.view(6)
```

fails for this example because `y`'s existing layout cannot represent the
requested flattened view.

---

# 8. A Useful but Incomplete Shortcut

Initially, I thought:

> `view()` requires a contiguous tensor.

That is a useful shortcut for understanding many common cases, but it is not
the deepest rule.

The stronger mental model is:

> The requested view must be compatible with the tensor's existing size and
> stride layout because `view()` does not copy the underlying data to make
> the requested layout possible.

Contiguous tensors make many common `view()` operations straightforward.

Non-contiguous tensors make some requested views impossible.

Therefore I should reason about:

```text
existing shape
       +
existing strides
       +
requested shape
       │
       ▼
Can this be represented without copying?
```

rather than memorizing only:

```text
non-contiguous → view always impossible
```

---

# 9. Making a Tensor Contiguous

For:

```python
y = x.T
```

we have logically:

```text
10 40
20 50
30 60
```

but the shared underlying data is arranged as:

```text
10 20 30 40 50 60
```

Now:

```python
c = y.contiguous()
```

can produce a contiguous representation matching `y`'s logical order:

```text
10 40 20 50 30 60
```

Conceptually:

```text
y logical order

10 40
20 50
30 60

       │
       │ contiguous()
       ▼

contiguous representation

10 40 20 50 30 60
```

Now:

```python
c.view(6)
```

can produce:

```text
[10, 40, 20, 50, 30, 60]
```

because the requested 1-D representation now matches the contiguous data
layout.

---

# 10. `reshape()`

Now consider:

```python
y = x.T
```

We learned that:

```python
y.view(6)
```

can fail because producing the requested logical representation would
require data rearrangement.

PyTorch also provides:

```python
y.reshape(6)
```

My mental model for `reshape()` is:

> `reshape()` is more flexible than `view()`. If the requested shape can be
> represented without copying, it can use a view-like result. If the
> requested shape cannot be represented using the existing layout, it may
> create a copy.

Conceptually:

```text
                   reshape(...)
                       │
                       ▼

       Can the requested shape be represented
               without copying?

                 /             \
               YES             NO
                │               │
                ▼               ▼

          may use a view    may create copy
```

This is different from relying on `view()`:

```text
view()

possible without copy?
    │
    ├── yes → view
    │
    └── no  → fail


reshape()

possible without copy?
    │
    ├── yes → may use view
    │
    └── no  → may copy
```

---

# 11. `reshape()` Does Not Always Copy

An important mistake to avoid is saying:

> `reshape()` creates a copy.

That is not always true.

A better statement is:

> `reshape()` may return a view when possible and may copy when necessary.

Therefore, I should not write code assuming that a result from `reshape()`
always has independent data.

---

# 12. Example: `reshape()` Requiring a Copy

Consider:

```python
x = torch.arange(6).reshape(2, 3)
```

Then:

```text
x =

0 1 2
3 4 5

shape  = (2,3)
stride = (3,1)
```

Now:

```python
y = x.T
```

produces:

```text
y =

0 3
1 4
2 5

shape  = (3,2)
stride = (1,3)
```

Notice that transpose does NOT produce:

```text
0 1
2 3
4 5
```

That would correspond to reshaping sequential values.

Transpose changes which axes are used when interpreting the same underlying
data.

Now:

```python
a = y.reshape(6)
```

logically needs to produce:

```text
[0, 3, 1, 4, 2, 5]
```

For this case, the requested flattened representation cannot be represented
as the required 1-D view of `y`'s existing layout.

Therefore, a copy is needed.

Conceptually:

```text
       x and y share underlying data

             0 1 2 3 4 5
               /     \
              ▼       ▼

              x       y


                  reshape(6)
                       │
                       │ copy needed
                       ▼

                separate data

               0 3 1 4 2 5
                       │
                       ▼
                       a
```

---

# 13. Copy vs Shared Data

In the previous example:

```python
a = y.reshape(6)
```

required a copy.

Therefore:

```python
a[0] = 999
```

changes `a`:

```text
a = [999, 3, 1, 4, 2, 5]
```

but does not modify the original `x` or `y`.

They remain logically:

```text
x =

0 1 2
3 4 5
```

and:

```text
y =

0 3
1 4
2 5
```

because `a` uses separate underlying data in this case.

---

# 14. Hands-On Experiment: `view()`

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

a = x.view(3, 2)

print("x:")
print(x)
print("shape:", x.shape)
print("stride:", x.stride())
print("contiguous:", x.is_contiguous())

print()

print("a:")
print(a)
print("shape:", a.shape)
print("stride:", a.stride())
print("contiguous:", a.is_contiguous())

a[0, 0] = 999

print("\nAfter modifying a:")
print("x:")
print(x)

print("a:")
print(a)
```

Before running this, my prediction is:

```text
x.shape  = (2,3)
x.stride = (3,1)

a.shape  = (3,2)
a.stride = (2,1)
```

Because `a` is a view sharing the underlying data, modifying:

```python
a[0,0]
```

will also change:

```python
x[0,0]
```

---

# 15. Hands-On Experiment: Breaking `view()`

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

y = x.T

print("y:")
print(y)

print("shape:", y.shape)
print("stride:", y.stride())
print("contiguous:", y.is_contiguous())

z = y.view(6)
```

My prediction is that the final operation fails.

The reasoning is:

```text
y logical traversal:

10 → 40 → 20 → 50 → 30 → 60

existing data layout:

10 → 20 → 30 → 40 → 50 → 60
```

The requested 1-D view cannot represent `y`'s logical ordering using its
existing layout without rearranging data.

`view()` does not perform that copy automatically.

---

# 16. Hands-On Experiment: `contiguous().view()`

```python
import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]
])

y = x.T

z = y.contiguous().view(6)

print(z)
print("shape:", z.shape)
print("stride:", z.stride())
print("contiguous:", z.is_contiguous())
```

My predicted result is:

```text
tensor([10, 40, 20, 50, 30, 60])
```

The reasoning is:

```text
y
│
│ non-contiguous logical layout
│
▼
contiguous()
│
│ produce contiguous representation
│
▼
10 40 20 50 30 60
│
│ view(6)
▼
[10, 40, 20, 50, 30, 60]
```

---

# 17. Hands-On Experiment: `reshape()`

```python
import torch

x = torch.arange(6).reshape(2, 3)

y = x.T

a = y.reshape(6)

print("x:")
print(x)

print("\ny:")
print(y)

print("\na:")
print(a)

a[0] = 999

print("\nAfter modifying a:")

print("x:")
print(x)

print("\ny:")
print(y)

print("\na:")
print(a)
```

For this specific example, my prediction is:

```text
x:

0 1 2
3 4 5
```

```text
y:

0 3
1 4
2 5
```

```text
a:

[0, 3, 1, 4, 2, 5]
```

After:

```python
a[0] = 999
```

I expect:

```text
a:

[999, 3, 1, 4, 2, 5]
```

while `x` and `y` remain unchanged because this particular reshape requires
separate data.

---

# 18. An Important Mistake I Made

When reasoning about:

```python
x = torch.arange(6).reshape(2, 3)

y = x.T
```

I initially thought `y` would be:

```text
0 1
2 3
4 5
```

That was incorrect.

This would correspond to interpreting the sequential elements using a
different shape.

The actual transpose is:

```text
0 3
1 4
2 5
```

The lesson is:

> Transpose and reshape are fundamentally different operations even when
> they produce the same shape.

For example:

```text
x.reshape(3,2)             x.T

0 1                        0 3
2 3                        1 4
4 5                        2 5

shape = (3,2)              shape = (3,2)
```

Same shape does not imply the same tensor contents or strides.

---

# 19. Interview Answer: What Is a View?

If asked:

> What is a tensor view?

My current answer would be:

> A view is a tensor that provides a different logical interpretation of
> existing underlying data without requiring an independent copy. Because
> views can share underlying data, modifying the data through one view can
> affect values observed through another tensor sharing that data.

---

# 20. Interview Answer: `view()` vs `reshape()`

If asked:

> What's the difference between `view()` and `reshape()`?

My current answer would be:

> `view()` changes the tensor's shape while sharing the underlying data, so
> the requested view must be compatible with the tensor's existing size and
> stride layout. It does not silently copy the data to make an incompatible
> layout work.
>
> `reshape()` is more flexible. When the requested shape can be represented
> without copying, it may return a view. When that is not possible, it may
> create a copy.
>
> Therefore, I should not assume that a tensor returned by `reshape()` always
> shares data or always owns a copy.

---

# 21. Interview Answer: `view()` vs Transpose

If asked:

> If `x.view(3,2)` and `x.T` both produce shape `(3,2)`, are they equivalent?

My answer is:

> No. Shape alone does not determine a tensor's logical layout. `view(3,2)`
> reinterprets the existing element sequence with a different shape when the
> layout permits it. Transpose swaps dimensions and changes strides, producing
> a different logical ordering while potentially sharing the same underlying
> data.

For example:

```text
x =

10 20 30
40 50 60
```

```text
x.view(3,2)             x.T

10 20                   10 40
30 40                   20 50
50 60                   30 60
```

---

# 22. Teach It to Someone

If I were teaching this concept to another engineer, I would say:

> Don't think of every tensor operation as creating a completely new block of
> data.
>
> A tensor has underlying data plus metadata such as shape and strides.
> Because of this separation, PyTorch can sometimes create a new logical
> tensor simply by changing how existing data is interpreted.
>
> This is the idea behind views.
>
> `view()` tries to provide the requested shape while sharing the existing
> data. If that requested representation cannot be described using the
> existing layout, the operation fails instead of copying the elements.
>
> `reshape()` is more flexible because it can use a view when possible and
> can copy when necessary.
>
> Operations such as transpose can also share underlying data while changing
> strides, which can produce a non-contiguous tensor.

---

# 23. Current Mental Model

My mental model has evolved into:

```text
                         Tensor
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        underlying data                metadata
                                          │
                                   ┌──────┴──────┐
                                   │             │
                                 shape         stride
```

Multiple tensors can sometimes reference the same data:

```text
                    underlying data
                           │
                 ┌─────────┼─────────┐
                 │         │         │
                 ▼         ▼         ▼
                 x       view      transpose
```

An operation may therefore involve:

```text
                    Tensor operation
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       metadata change                data copy
             │                           │
             ▼                           ▼
      potentially cheap        new/rearranged data
      and shares data
```

The key question I should ask when reasoning about an operation is:

> Can this logical representation be expressed using the existing data and
> metadata, or does the data itself need to be rearranged?

---

# 24. What I Understand Now

At this point I understand the relationship between:

```text
Tensor
│
├── underlying data
├── metadata
│   ├── shape
│   └── stride
│
├── view
│   └── shares existing data
│
├── transpose
│   ├── changes dimensions/strides
│   └── can share data
│
├── contiguous
│   └── standard contiguous layout for a shape
│
├── view()
│   ├── no silent copy
│   └── requested view must be compatible with layout
│
└── reshape()
    ├── may return a view
    └── may copy when necessary

transpose()
    ↓
keep data
change metadata/strides
    ↓
can become non-contiguous


view(new_shape)
    ↓
Can new shape be represented
without copying?
    ├── YES → view
    └── NO  → fail


reshape(new_shape)
    ↓
Can new shape be represented
without copying?
    ├── YES → may use view
    └── NO  → may copy
```
