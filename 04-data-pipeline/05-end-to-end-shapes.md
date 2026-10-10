# End-to-End Shape Tracking in PyTorch

## Goal

One of the most common causes of PyTorch bugs is shape mismatch.

Examples:

```text
Expected shape (B,C)

Got shape (B,)
```

or

```text
Expected shape (32,3,224,224)

Got shape (3,224,224)
```

This chapter builds a complete mental model for tracking tensor shapes from:

```text
Raw Data
    ↓
Dataset
    ↓
DataLoader
    ↓
Model
    ↓
Loss Function
```

---

# 1. Why Shapes Matter

Every component in PyTorch expects specific tensor shapes.

Examples:

```text
Conv2d

CrossEntropyLoss

BCEWithLogitsLoss

Linear
```

Incorrect shapes are among the most common runtime errors.

---

# 2. Shape Tracking Philosophy

Whenever a shape bug occurs, trace:

```text
Dataset
   ↓
DataLoader
   ↓
Model
   ↓
Loss
```

and inspect:

```python
print(x.shape)
print(y.shape)
print(output.shape)
```

at each stage.

---

# 3. Single Sample vs Batch

This is the most important distinction.

Dataset returns:

```text
Single Sample
```

DataLoader returns:

```text
Batch
```

---

# 4. Example Image Sample

Suppose a transformed image has:

```text
(3,224,224)
```

Meaning:

```text
Channels = 3

Height = 224

Width = 224
```

This is one sample.

---

# 5. Dataset Shapes

Dataset returns:

```python
dataset[i]
```

Example:

```text
x.shape = (3,224,224)

y = 5
```

Notice:

```text
No batch dimension
```

yet.

---

# 6. DataLoader Adds Batch Dimension

Suppose:

```python
batch_size = 32
```

DataLoader stacks samples during collation.

Result:

```text
(32,3,224,224)
```

---

# 7. DataLoader Mental Model

Dataset:

```text
(C,H,W)
```

DataLoader:

```text
(B,C,H,W)
```

where:

```text
B = batch size
```

---

# 8. Image Classification Pipeline

Start:

```text
Image File
```

Loaded as:

```text
(224,224,3)
```

Many image libraries use:

```text
(H,W,C)
```

ordering.

---

# 9. After Transform

Common image transforms convert:

```text
(H,W,C)
```

to:

```text
(C,H,W)
```

Example:

```text
(224,224,3)

↓

(3,224,224)
```

---

# 10. After DataLoader

Batch size:

```text
32
```

Result:

```text
(32,3,224,224)
```

---

# 11. Input To Model

Training loop receives:

```python
for x, y in loader:
```

Shapes:

```text
x.shape

=

(32,3,224,224)
```

and:

```text
y.shape

=

(32,)
```

---

# 12. Label Shape

For classification:

```text
One label per sample
```

Example:

```python
[
 0,
 1,
 2,
 1,
 ...
]
```

Shape:

```text
(B,)
```

---

# 13. Model Output

Suppose:

```text
3 classes
```

Model output:

```text
(32,3)
```

Meaning:

```text
32 samples

3 logits/sample
```

---

# 14. CrossEntropyLoss Shapes

Input:

```text
(B,C)
```

Example:

```text
(32,3)
```

---

Target:

```text
(B,)
```

Example:

```text
(32,)
```

---

# 15. Why Target Is (B,)

CrossEntropyLoss expects:

```text
Class Indices
```

Example:

```python
[
 0,
 1,
 2,
 0
]
```

One class index per sample.

---

# 16. Why Not (B,C)?

This:

```text
(B,C)
```

would typically represent:

```text
One-Hot Labels
```

Example:

```python
[
 [1,0,0],
 [0,1,0]
]
```

Standard CrossEntropyLoss usage does not require one-hot targets.

---

# 17. CrossEntropy Example

Batch size:

```text
64
```

Classes:

```text
10
```

Logits:

```text
(64,10)
```

Target:

```text
(64,)
```

Loss:

```text
Scalar Tensor
```

---

# 18. Binary Classification Pipeline

Suppose:

```text
Spam

Not Spam
```

Single sample:

```text
(300,)
```

features.

---

# 19. Binary Batch

Batch size:

```text
32
```

Shape:

```text
(32,300)
```

---

# 20. Binary Model Output

Final layer:

```text
1 logit
```

Output:

```text
(32,1)
```

---

# 21. Binary Targets

Common shape:

```text
(32,1)
```

Values:

```text
0.0

1.0
```

---

# 22. BCEWithLogitsLoss Shapes

Input:

```text
(32,1)
```

Target:

```text
(32,1)
```

Matching shapes.

---

# 23. Regression Pipeline

Single sample:

```text
(10,)
```

features.

Target:

```text
150.5
```

---

# 24. Regression Batch

Input:

```text
(32,10)
```

Target:

```text
(32,1)
```

---

# 25. Regression Output

Model output:

```text
(32,1)
```

Loss:

```text
MSELoss
```

---

# 26. Common Shape Bug #1

Input:

```text
(3,224,224)
```

Model expects:

```text
(B,3,224,224)
```

Cause:

```text
Missing Batch Dimension
```

---

# 27. Common Shape Bug #2

CrossEntropyLoss:

```text
logits

(32,10)
```

Target:

```text
(32,10)
```

Problem:

```text
One-Hot Targets
```

Usually target should be:

```text
(32,)
```

---

# 28. Common Shape Bug #3

Binary Classification

Output:

```text
(32,1)
```

Target:

```text
(32,)
```

Potential shape mismatch.

Input and target should typically match.

---

# 29. Common Shape Bug #4

Dataset returns:

```text
Batch
```

instead of:

```text
Single Sample
```

Remember:

```text
Dataset

↓

Single Sample
```

---

```text
DataLoader

↓

Batch
```

---

# 30. Shape Debugging Checklist

When debugging:

```python
print(x.shape)

print(y.shape)

print(output.shape)
```

Check:

```text
Dataset Output

Batch Shape

Model Input

Model Output

Loss Input

Loss Target
```

---

# 31. Full Image Classification Trace

```text
Image File

(224,224,3)

        ↓

Transform

(3,224,224)

        ↓

DataLoader

(32,3,224,224)

        ↓

Model

(32,10)

        ↓

Target

(32,)

        ↓

CrossEntropyLoss

()
```

---

# 32. Full Binary Classification Trace

```text
Dataset

(300,)

       ↓

DataLoader

(32,300)

       ↓

Model

(32,1)

       ↓

Target

(32,1)

       ↓

BCEWithLogitsLoss

()
```

---

# 33. Full Regression Trace

```text
Dataset

(10,)

      ↓

DataLoader

(32,10)

      ↓

Model

(32,1)

      ↓

Target

(32,1)

      ↓

MSELoss

()
```

---

# 34. Master Rule

Whenever confused:

```text
Ask:

What is the shape here?
```

at every stage.

Most PyTorch shape bugs can be found by tracing:

```text
Dataset

↓

DataLoader

↓

Model

↓

Loss
```

one step at a time.

---

# 35. Interview Answer

How do you debug tensor shape issues?

A good answer:

> I trace tensor shapes through the entire pipeline. I verify shapes produced by the Dataset, DataLoader, Model, and Loss Function and compare them with the expected input and output shapes at each stage.

---

# 36. What I Understand Now

I understand:

```text
Shape Tracking
│
├── Dataset Shapes
├── DataLoader Shapes
├── Batch Dimensions
├── Model Input Shapes
├── Model Output Shapes
├── CrossEntropy Shapes
├── BCEWithLogits Shapes
├── Regression Shapes
└── Shape Debugging
```
