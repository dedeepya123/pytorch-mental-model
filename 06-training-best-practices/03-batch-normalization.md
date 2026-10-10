# Batch Normalization

## Goal

As neural networks become deeper, training can become unstable.

Activations flowing through the network may become:

```text
Too Large

Too Small

Highly Variable
```

from one layer to another.

Batch Normalization was introduced to stabilize training by keeping activations well-behaved as they pass through the network.

The goals of this chapter are:

- Understand the training instability problem.
- Understand activation normalization.
- Understand BatchNorm intuition.
- Understand running statistics.
- Understand train() vs eval().
- Understand BatchNorm in MLPs and CNNs.
- Understand BatchNorm's trainable parameters.
- Understand where BatchNorm fits into PyTorch architectures.

---

# 1. The Problem

Consider a deep network:

```text
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
ReLU
 ↓
Linear
```

As training progresses, the distribution of activations inside the network can change.

Example:

```text
Layer 1

Mean ≈ 0
Std ≈ 1
```

---

Later:

```text
Layer 2

Mean ≈ 100
Std ≈ 50
```

---

Even later:

```text
Layer 3

Mean ≈ -300
Std ≈ 200
```

Layers continuously receive inputs with changing scales and distributions.

---

# 2. Why This Is A Problem

Neural networks learn more easily when inputs are:

```text
Stable

Predictable

Well Scaled
```

Large shifts in activation distributions can make optimization harder.

---

# 3. The Core Idea

Instead of passing activations directly to the next layer:

```text
Layer Output
      ↓
Next Layer
```

BatchNorm introduces an intermediate step:

```text
Layer Output
      ↓
Normalize
      ↓
Next Layer
```

---

# 4. What BatchNorm Does

Conceptually:

```text
Activation
      ↓
Normalize
      ↓
Mean ≈ 0
Std ≈ 1
```

The next layer receives more stable inputs.

---

# 5. Mental Model

Think:

```text
BatchNorm
        ↓
Keeps Activations
Well Behaved
```

---

# 6. Why The Name?

The normalization is computed using:

```text
The Current Mini-Batch
```

Hence:

```text
Batch Normalization
```

---

# 7. Example Mini-Batch

Suppose:

```text
Batch Size = 32
```

During training BatchNorm examines:

```text
32 Examples
```

and computes:

```text
Mean

Variance
```

for normalization.

---

# 8. Batch Statistics

For each batch BatchNorm computes:

```text
Batch Mean

Batch Variance
```

and uses them to normalize activations.

---

# 9. MLP Version

For fully connected networks:

```python
nn.BatchNorm1d(128)
```

Example:

```python
nn.Sequential(

    nn.Linear(100,128),

    nn.BatchNorm1d(128),

    nn.ReLU()
)
```

---

# 10. Typical MLP Block

A common pattern:

```text
Linear
 ↓
BatchNorm
 ↓
ReLU
```

Interpretation:

```text
Linear
 ↓
Stabilize Activations
 ↓
Apply Nonlinearity
```

---

# 11. CNN Version

For image models:

```python
nn.BatchNorm2d(64)
```

Example:

```python
nn.Conv2d(
    3,
    64,
    3,
    padding=1
)

nn.BatchNorm2d(64)

nn.ReLU()
```

---

# 12. Why 64?

Because the convolution outputs:

```text
64 Channels
```

and BatchNorm normalizes each channel.

---

# 13. BatchNorm In CNNs

For:

```text
(B,C,H,W)
```

BatchNorm2d normalizes:

```text
Each Channel
```

separately.

---

# 14. An Important Surprise

BatchNorm contains:

```text
Trainable Parameters
```

Many beginners assume it is purely a mathematical normalization layer.

It is not.

---

# 15. Gamma And Beta

BatchNorm learns:

```text
Gamma (Scale)

Beta (Shift)
```

parameters.

---

Conceptually:

```text
Normalize
      ↓
Scale
      ↓
Shift
```

---

# 16. Why Learn Scale And Shift?

Pure normalization may remove useful information.

After normalization BatchNorm learns:

```text
How Much Scaling?

How Much Shifting?
```

produces the best results.

---

# 17. BatchNorm Has State

BatchNorm stores information.

Examples:

```text
Running Mean

Running Variance
```

These values become important during inference.

---

# 18. Running Statistics

During training every batch contributes to:

```text
running_mean

running_var
```

---

Over time these become estimates of:

```text
Dataset Statistics
```

---

# 19. The train() Behavior

When:

```python
model.train()
```

BatchNorm uses:

```text
Current Batch Statistics
```

for normalization.

---

It also updates:

```text
running_mean

running_var
```

---

# 20. The eval() Behavior

When:

```python
model.eval()
```

BatchNorm stops using:

```text
Current Batch Statistics
```

---

Instead it uses:

```text
Stored Running Statistics
```

collected during training.

---

# 21. Why Does eval() Exist?

Imagine inference with:

```text
1 Example
```

or:

```text
2 Examples
```

Batch statistics would be unreliable.

Using stored dataset-level estimates is more stable.

---

# 22. One Of The Most Common Bugs

Forgetting:

```python
model.eval()
```

during evaluation.

---

This can cause:

```text
Unstable Predictions

Reduced Accuracy

Inconsistent Results
```

especially when BatchNorm is present.

---

# 23. Mental Model For Modes

Training:

```text
train()

↓

Use Batch Statistics

↓

Update Running Statistics
```

---

Evaluation:

```text
eval()

↓

Use Running Statistics

↓

Do Not Update Statistics
```

---

# 24. How BatchNorm Helps

BatchNorm often leads to:

```text
More Stable Training

Faster Optimization

Better Gradient Flow
```

especially in deeper networks.

---

# 25. Where BatchNorm Lives

BatchNorm is part of:

```text
Model Architecture
```

---

Not:

```text
Optimizer

Dataset

Training Loop
```

---

# 26. Typical Placement In MLPs

Common structure:

```text
Linear
 ↓
BatchNorm
 ↓
ReLU
```

---

# 27. Typical Placement In CNNs

Common structure:

```text
Conv
 ↓
BatchNorm
 ↓
ReLU
```

---

# 28. Example CNN Block

```python
nn.Sequential(

    nn.Conv2d(
        3,
        64,
        3,
        padding=1
    ),

    nn.BatchNorm2d(64),

    nn.ReLU()
)
```

---

# 29. BatchNorm And Shape

BatchNorm does not change tensor shape.

Example:

Input:

```text
(32,64,128,128)
```

Output:

```text
(32,64,128,128)
```

---

Only values change.

---

# 30. BatchNorm And Shape Reasoning

Remember:

```text
BatchNorm

Changes Values

Not Shapes
```

---

# 31. Comparison With Dropout

### BatchNorm

Goal:

```text
Stabilize Activations
```

---

### Dropout

Goal:

```text
Reduce Overfitting
```

---

Both behave differently in:

```text
train()

eval()
```

but solve different problems.

---

# 32. Common Beginner Misconception

Incorrect:

```text
BatchNorm Prevents Overfitting
```

---

Better understanding:

```text
BatchNorm Primarily Improves
Optimization Stability
```

Regularization effects may occur, but stabilization is the main purpose.

---

# 33. Common Debugging Clue

Symptoms:

```text
Training Looks Fine

Inference Looks Strange
```

One of the first things to check:

```python
model.eval()
```

especially when the model contains:

```python
BatchNorm
```

or

```python
Dropout
```

---

# 34. Interview Question

Why do we use BatchNorm?

Strong answer:

> BatchNorm normalizes activations during training, helping maintain stable signal flow and making optimization more efficient.

---

# 35. Interview Question

What statistics does BatchNorm compute?

Strong answer:

> During training BatchNorm computes batch mean and batch variance and uses them to normalize activations.

---

# 36. Interview Question

Why does BatchNorm behave differently in train() and eval()?

Strong answer:

> During training BatchNorm uses current batch statistics and updates running statistics. During evaluation it uses stored running statistics to produce stable predictions.

---

# 37. Interview Question

Why does BatchNorm2d(64) use 64?

Strong answer:

> The layer expects 64 channels and normalizes each channel independently.

---

# 38. Teach It To Someone

If I were teaching BatchNorm:

> BatchNorm stabilizes neural network training by normalizing activations using mini-batch statistics. During training it learns dataset statistics, and during inference it uses those learned statistics to produce consistent predictions.

---

# 39. Master Mental Model

```text
Linear / Conv

↓

BatchNorm

↓

ReLU

↓

Stable Signal Flow

↓

Better Optimization
```

---

# 40. PyTorch Mental Model

```text
BatchNorm
│
├── Lives Inside The Model
├── Has Trainable Parameters
├── Stores Running Statistics
├── Uses Batch Statistics In train()
├── Uses Running Statistics In eval()
└── Preserves Tensor Shapes
```

---

# 41. What I Understand Now

```text
Batch Normalization
│
├── Activation Stabilization
├── Batch Statistics
├── Running Statistics
├── Gamma
├── Beta
├── BatchNorm1d
├── BatchNorm2d
├── train() Behavior
├── eval() Behavior
└── Placement In Architectures
```

---

# 42. Relationship To Previous Chapters

```text
Initialization
      ↓

BatchNorm
      ↓

ReLU
      ↓

Optimizer
```

All four influence signal flow and training stability.

---

# 43. What Comes Next

The next chapter introduces:

```text
Saving And Loading Models
```

and answers:

```text
What state_dict() Really Contains

How To Save Checkpoints

How To Resume Training

How Models Move Between Training And Inference
```
