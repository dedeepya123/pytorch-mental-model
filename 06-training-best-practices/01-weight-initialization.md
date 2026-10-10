# Weight Initialization

## Goal

Before training begins, every neural network parameter must have an initial value.

This process is called:

```text
Weight Initialization
```

Weight initialization is one of the earliest and most important steps in the
lifecycle of a neural network.

The goals of this chapter are:

- Understand what initialization is.
- Understand why initialization is necessary.
- Understand why zero initialization fails.
- Understand exploding and vanishing signals.
- Understand the purpose of Xavier and Kaiming initialization.
- Understand PyTorch's default initialization behavior.
- Understand initialization from a PyTorch engineering perspective.

---

# 1. What Happens Before Training?

When we create:

```python
layer = nn.Linear(
    100,
    50
)
```

training has not started yet.

No:

```text
Forward Pass

Backward Pass

Optimizer Step
```

has occurred.

Yet the layer already contains:

```text
Weights

Biases
```

with actual values.

---

# 2. Where Do These Values Come From?

PyTorch creates:

```python
self.weight

self.bias
```

and then initializes them.

Conceptually:

```text
Create Parameters

↓

Initialize Parameters

↓

Ready For Training
```

Training begins later.

---

# 3. The Hidden Question

Before training starts:

```text
What should the weights be?
```

Possible answers:

```text
All Zeros

Very Large Random Numbers

Very Small Random Numbers

Carefully Chosen Random Numbers
```

Only one of these is useful.

---

# 4. Why Not Initialize To Zero?

Suppose:

```text
All Weights = 0
```

Every neuron produces:

```text
Exactly The Same Output
```

during the forward pass.

---

Example:

```text
Neuron A

Neuron B

Neuron C
```

all behave identically.

---

# 5. The Symmetry Problem

During backpropagation:

```text
Neuron A

Neuron B

Neuron C
```

receive identical gradients.

---

Parameter updates become:

```text
Identical
```

for every neuron.

---

Result:

```text
All Neurons Learn
The Same Function
```

This destroys one of the main advantages of neural networks.

---

# 6. Why Random Initialization Exists

Neurons should begin differently.

Example:

```text
Neuron A = 0.13

Neuron B = -0.05

Neuron C = 0.42
```

Now neurons can specialize and learn different patterns.

---

# 7. Why Not Use Huge Random Values?

Suppose weights are:

```text
1000

500

-700
```

The network becomes unstable.

---

# 8. Exploding Activations

Forward pass:

```text
Input

↓

Large Weights

↓

Huge Activations
```

---

Deeper layers receive increasingly larger values.

This can destabilize training.

---

# 9. Exploding Gradients

The same issue can occur during backpropagation.

Gradients may become:

```text
Extremely Large
```

leading to unstable updates.

---

# 10. Why Not Use Tiny Values?

Suppose weights are:

```text
0.0000001
```

everywhere.

---

Forward pass:

```text
Input

↓

Tiny Activations

↓

Nearly Zero Activations
```

---

# 11. Vanishing Signals

As information moves through layers:

```text
Signal Strength
```

becomes increasingly small.

---

Eventually:

```text
Nearly Zero
```

information reaches deeper layers.

---

# 12. Vanishing Gradients

Backpropagation experiences the same issue.

Gradients may become:

```text
Almost Zero
```

making learning
