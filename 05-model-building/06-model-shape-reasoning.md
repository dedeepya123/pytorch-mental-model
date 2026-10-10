# Model Shape Reasoning

## Goal

One of the most important skills in deep learning is the ability to predict
tensor shapes as data flows through a model.

Most PyTorch bugs are not caused by:

```text
Autograd

Optimizers

Loss Functions
```

They are caused by:

```text
Shape Mismatches
```

The goal of this chapter is to develop a systematic approach for reasoning
about tensor shapes before running code.

---

# 1. Why Shape Reasoning Matters

Suppose a layer expects:

```text
(B,128)
```

but receives:

```text
(B,256)
```

The model fails immediately.

Understanding shapes allows us to:

```text
Design Architectures

Debug Models

Answer Interview Questions

Understand Existing Networks
```

---

# 2. The Shape Debugging Mindset

At every layer ask:

```text
What is the shape now?
```

Never guess.

Always trace:

```text
Input

↓

Layer

↓

Output
```

one step at a time.

---

# 3. The Master Rule

For any architecture:

```text
Input

↓

Layer

↓

Layer

↓

Layer

↓

Output
```

there is an equivalent:

```text
Shape

↓

Shape

↓

Shape

↓

Shape
```

pipeline.

Always track both.

---

# 4. Two Shape Worlds

Most models belong to one of two shape families.

---

## MLPs

Track:

```text
(B, Features)
```

Example:

```text
(32,128)
```

---

## CNNs

Track:

```text
(B,C,H,W)
```

Example:

```text
(32,3,224,224)
```

---

# 5. Understanding Batch Dimension

The batch dimension is almost always:

```text
Dimension 0
```

Example:

```text
(32,128)
```

means:

```text
32 Samples

128 Features
```

---

Similarly:

```text
(32,3,224,224)
```

means:

```text
32 Images

3 Channels

224 Height

224 Width
```

---

# 6. MLP Shape Rule

For:

```python
nn.Linear(in_features, out_features)
```

Input:

```text
(B,in_features)
```

Output:

```text
(B,out_features)
```

---

# 7. Example

Layer:

```python
nn.Linear(10,64)
```

Input:

```text
(32,10)
```

Output:

```text
(32,64)
```

---

# 8. MLP Shape Trace

Model:

```python
nn.Sequential(

    nn.Linear(10,64),

    nn.ReLU(),

    nn.Linear(64,32),

    nn.ReLU(),

    nn.Linear(32,3)
)
```

Input:

```text
(32,10)
```

---

Output progression:

```text
(32,10)

↓

(32,64)

↓

(32,64)

↓

(32,32)

↓

(32,32)

↓

(32,3)
```

---

# 9. ReLU Rule

ReLU changes:

```text
Values
```

but not:

```text
Shape
```

---

Input:

```text
(32,64)
```

Output:

```text
(32,64)
```

---

# 10. Dropout Rule

Dropout changes:

```text
Values
```

but not:

```text
Shape
```

---

Input:

```text
(32,64)
```

Output:

```text
(32,64)
```

---

# 11. BatchNorm Rule

BatchNorm typically changes:

```text
Statistics
```

but not:

```text
Shape
```

---

# 12. CNN Shape Tracking

CNNs require tracking:

```text
Channels

Height

Width
```

for every layer.

---

Input:

```text
(B,C,H,W)
```

---

Ask:

```text
What happened to C?

What happened to H?

What happened to W?
```

---

# 13. Conv2d Rule

Layer:

```python
Conv2d(
    in_channels,
    out_channels,
    ...
)
```

Changes:

```text
Channels
```

from:

```text
in_channels

↓

out_channels
```

---

# 14. Example Conv Layer

```python
Conv2d(
    3,
    64,
    kernel_size=3,
    padding=1
)
```

Input:

```text
(32,3,224,224)
```

Output:

```text
(32,64,224,224)
```

---

# 15. Why Channels Changed

The convolution layer learned:

```text
64 Filters
```

which produce:

```text
64 Feature Maps
```

Therefore:

```text
3

↓

64 Channels
```

---

# 16. MaxPool Rule

MaxPool usually changes:

```text
Height

Width
```

but not:

```text
Channels
```

---

# 17. Example MaxPool

Layer:

```python
MaxPool2d(
    2,
    2
)
```

Input:

```text
(32,64,224,224)
```

Output:

```text
(32,64,112,112)
```

---

# 18. Common CNN Pattern

Input:

```text
(32,3,224,224)
```

---

Conv:

```text
(32,64,224,224)
```

---

MaxPool:

```text
(32,64,112,112)
```

---

Conv:

```text
(32,128,112,112)
```

---

MaxPool:

```text
(32,128,56,56)
```

---

# 19. The Channel Pattern

In CNNs:

```text
Channels

Usually Increase
```

Example:

```text
3

↓

32

↓

64

↓

128

↓

256
```

---

# 20. The Spatial Pattern

In CNNs:

```text
Height

Width

Usually Decrease
```

Example:

```text
224

↓

112

↓

56

↓

28
```

---

# 21. Flatten Layer

Eventually:

```text
(B,C,H,W)
```

must become:

```text
(B, Features)
```

for a classifier.

---

# 22. Flatten Rule

```python
torch.flatten(
    x,
    start_dim=1
)
```

converts:

```text
(B,C,H,W)
```

into:

```text
(B,C×H×W)
```

---

# 23. Example Flatten

Input:

```text
(32,128,56,56)
```

Computation:

```text
128 × 56 × 56
=
401,408
```

Output:

```text
(32,401408)
```

---

# 24. Why Flatten Exists

Fully connected layers expect:

```text
(B, Features)
```

not:

```text
(B,C,H,W)
```

Flatten acts as the bridge.

---

# 25. Classification Head Example

Flatten output:

```text
(32,401408)
```

Layer:

```python
nn.Linear(
    401408,
    10
)
```

Output:

```text
(32,10)
```

---

# 26. Complete CNN Shape Trace

Input:

```text
(32,3,128,128)
```

---

Conv:

```python
Conv2d(
    3,
    32,
    3,
    padding=1
)
```

Output:

```text
(32,32,128,128)
```

---

Pool:

```python
