# Tensor dtype

## Goal

So far, my tensor mental model has evolved into:

```text
Tensor
│
├── underlying data
│
└── metadata
    ├── shape
    ├── stride
    └── ...
```

But knowing the values and their shape is not enough.

PyTorch also needs to know:

> How should each value be represented and interpreted?

This is where `dtype` comes in.

In this section, I want to understand:

- What `dtype` means.
- Why tensors need a dtype.
- How dtype affects memory usage.
- How dtype affects numerical precision and range.
- Why floating-point values are important for neural-network training.
- The trade-offs between FP32, FP16, and BF16.
- How dtype decisions affect deep-learning workloads.

---

# 1. What Is `dtype`?

`dtype` stands for **data type**.

My current mental model is:

> `dtype` describes how each element of a tensor is represented and
> interpreted.

For example, a tensor may contain:

```text
integers
floating-point numbers
booleans
etc.
```

Examples:

```python
import torch

a = torch.tensor([1, 2, 3])

b = torch.tensor([1.0, 2.0, 3.0])

c = torch.tensor([True, False, True])
```

These tensors may have the same shape, but the values are represented using
different data types.

I can inspect the dtype using:

```python
tensor.dtype
```

So my tensor mental model becomes:

```text
                    Tensor
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   underlying data               metadata
                                      │
                           ┌──────────┼──────────┐
                           │          │          │
                         shape      stride     dtype
```

---

# 2. dtype Is More Than "Type of Number"

Initially, I could think:

```text
dtype
  ↓
integer / float / boolean
```

But dtype is more important than just categorizing the number.

My deeper mental model is:

```text
                         dtype
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          number type   representation  bits used
                                          │
                              ┌───────────┼───────────┐
                              │           │           │
                              ▼           ▼           ▼
                            memory       range      precision
```

The representation chosen by the dtype affects:

- How much memory each element requires.
- What numerical values can be represented.
- The representable numerical range.
- Numerical precision.
- The numerical behavior of operations.
- Computational efficiency depending on hardware.

---

# 3. dtype Determines Element Size

An important refinement to my initial thinking is the direction of the
relationship.

It is better to think:

```text
dtype
   ↓
representation
   ↓
number of bits
   ↓
bytes per element
```

rather than:

```text
bytes per element
   ↓
dtype
```

For example:

```text
torch.int64
    ↓
64 bits
    ↓
8 bytes per element
```

and:

```text
torch.float32
    ↓
32 bits
    ↓
4 bytes per element
```

The dtype determines the representation, and the representation determines
how much raw storage each element requires.

---

# 4. dtype Inference

Consider:

```python
a = torch.tensor([1, 2, 3])

b = torch.tensor([1.0, 2.0, 3.0])
```

Under the usual PyTorch defaults:

```text
a.dtype → torch.int64
b.dtype → torch.float32
```

This was an important correction to my initial prediction.

I initially thought:

```text
Python floating values
        ↓
torch.float64
```

But PyTorch normally uses its default floating-point dtype when creating a
tensor from Python floating-point values, which is commonly `torch.float32`.

Therefore I should not infer PyTorch dtype purely from how Python itself
represents numbers.

---

# 5. Number of Elements vs Element Size

I already learned:

```python
tensor.numel()
```

tells me the total number of tensor elements.

Now I can also inspect:

```python
tensor.element_size()
```

which tells me how many bytes are used by each element.

This gives the useful relationship:

```text
raw element storage
≈
numel × element_size
```

This refers to the raw storage required for tensor elements and does not
include every possible runtime or allocator overhead.

---

# 6. Hands-On Experiment: dtype and Memory

Consider:

```python
import torch

a = torch.tensor([1, 2, 3])

b = torch.tensor([1.0, 2.0, 3.0])

print("a")
print("dtype:", a.dtype)
print("numel:", a.numel())
print("element size:", a.element_size())
print("data bytes:", a.numel() * a.element_size())

print()

print("b")
print("dtype:", b.dtype)
print("numel:", b.numel())
print("element size:", b.element_size())
print("data bytes:", b.numel() * b.element_size())
```

My prediction is:

```text
a:

dtype        = torch.int64
numel        = 3
element_size = 8 bytes

raw data bytes
= 3 × 8
= 24 bytes
```

For `b`:

```text
b:

dtype        = torch.float32
numel        = 3
element_size = 4 bytes

raw data bytes
= 3 × 4
= 12 bytes
```

This demonstrates that:

> Two tensors with exactly the same number of elements can require different
> amounts of raw element storage because their dtypes are different.

---

# 7. Floating-Point Precision

Consider:

```python
x = torch.tensor(
    [1.23456789],
    dtype=torch.float32
)

y = torch.tensor(
    [1.23456789],
    dtype=torch.float16
)
```

Both represent floating-point values, but they use different numbers of bits.

```text
float32
   ↓
32 bits

float16
   ↓
16 bits
```

With fewer bits available for the representation, FP16 generally has less
numerical precision than FP32.

Conceptually:

```text
desired numerical value
         │
         ▼
    1.23456789
       /    \
      /      \
     ▼        ▼

   FP32      FP16

more information can
be retained in FP32
```

The exact represented values depend on the floating-point format, but the
important mental model is:

```text
fewer representation bits
          ↓
different precision/range trade-offs
```

---

# 8. Range vs Precision

Two important concepts need to be separated:

```text
Precision
   ↓
How finely can values be distinguished?


Range
   ↓
How large or small in magnitude can represented values be?
```

These are related to how the available bits in a floating-point format are
used.

A simplified floating-point mental model is:

```text
┌────────┬────────────┬────────────────────┐
│  sign  │  exponent  │ fraction/significand │
└────────┴────────────┴────────────────────┘
```

At a high level:

```text
exponent
   ↓
strongly affects range


fraction/significand
   ↓
strongly affects precision
```

With a fixed number of bits, there is a trade-off in how those bits are used.

---

# 9. Overflow, Underflow, and NaN Are Different

Initially, I thought that if a number was too large or too small for a
floating-point dtype, it might simply become `NaN`.

That is too broad.

I should distinguish between different numerical situations.

```text
Numerical issue
│
├── Overflow
│   │
│   └── magnitude becomes too large
│
├── Underflow
│   │
│   └── magnitude becomes extremely small
│       and information can be lost
│
└── NaN
    │
    └── represents a not-a-number result
        arising from certain invalid/undefined
        numerical computations
```

Therefore:

> Overflow, underflow, and NaN are related numerical concerns, but they are
> not interchangeable terms.

---

# 10. Why Floating Point Matters for Training

Initially, I thought integers might be problematic for neural-network
training because they could cause unstable updates or exploding gradients.

That is not the fundamental reason.

The more important reasoning comes directly from gradient descent.

Suppose:

```text
weight = 3
```

and:

```text
gradient = 2
learning_rate = 0.001
```

A gradient-descent update would be:

```text
new_weight
=
weight - learning_rate × gradient
```

Therefore:

```text
new_weight
=
3 - 0.001 × 2
=
2.998
```

The updated parameter is fractional.

More generally:

```text
weight
    ↓
1.3472

gradient
    ↓
0.0073

learning rate
    ↓
0.001

update
    ↓
0.0000073

new weight
    ↓
1.3471927
```

Gradient-based optimization commonly relies on many small numerical updates.

Floating-point representations are naturally suited to representing:

- Fractional model parameters.
- Fractional activations.
- Gradients.
- Small parameter updates.
- Intermediate numerical results.

Therefore my better mental model is:

```text
gradient-based optimization
            │
            ▼
fractional quantities +
small numerical updates
            │
            ▼
floating-point representation
is naturally suited to this
```

I should NOT think:

```text
integer weights
    ↓
exploding gradients
```

Those are separate concepts.

---

# 11. Why Use Lower Precision?

If FP32 provides greater numerical capability than lower-precision formats,
why would I use FP16 or BF16?

Because dtype is an engineering trade-off.

```text
                         dtype
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           memory       precision       range
                                           │
                                           ▼
                                    numerical behavior
```

Lower precision can provide important benefits such as:

```text
fewer bytes per element
          ↓
lower memory use
```

and, when supported efficiently by the hardware:

```text
lower-precision computation
          ↓
potential computational efficiency
```

Therefore I should not think:

> FP16 is always faster.

A better statement is:

> Lower-precision formats can reduce memory requirements and can improve
> computational efficiency when the hardware and operations efficiently
> support those formats.

---

# 12. FP32 vs FP16

At a high level:

```text
FP32
│
├── 32 bits
├── 4 bytes per element
└── greater numerical capability
    than FP16
```

```text
FP16
│
├── 16 bits
├── 2 bytes per element
├── lower memory requirement
├── reduced precision
└── smaller dynamic range than BF16
```

This creates an engineering trade-off:

```text
                FP32
                  │
             more bits
                  │
                  ▼
       greater numerical capability

                  │
             trade-off
                  │

                FP16
                  │
              fewer bits
                  │
           ┌──────┴──────┐
           ▼             ▼
      less memory    reduced numerical
                       capability
```

---

# 13. FP16 vs BF16

FP16 and BF16 both use:

```text
16 bits
=
2 bytes per element
```

Therefore they have the same raw memory requirement per element.

However, they use their bit budgets differently.

My high-level mental model is:

```text
                  16-bit budget
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼

Our Tensor Model has now become stronger
Tensor
├── What does it logically look like? → shape
├── How do I traverse its data?       → stride
├── How is each value represented?    → dtype
└── Where does that data live?        → device ← NEXT
