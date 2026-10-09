# Loss Functions in PyTorch

## Goal

A neural network produces predictions.

A loss function measures:

> How wrong are those predictions relative to the target?

The loss then becomes the quantity that Autograd differentiates.

The goals of this section are:

- Understand loss functions as PyTorch Modules.
- Understand how losses connect predictions and targets.
- Understand why training usually uses a scalar loss.
- Understand reduction.
- Understand MSELoss.
- Understand logits.
- Understand CrossEntropyLoss.
- Understand BCEWithLogitsLoss.
- Understand logits vs probabilities.
- Understand target shapes and dtypes.
- Understand common loss-function bugs.

---

# 1. Where Loss Fits Into Training

Conceptually:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Target Comparison
  ↓
Loss
  ↓
backward()
  ↓
parameter.grad
```

Loss is the bridge between:

```text
model output
```

and

```text
gradient computation
```

---

# 2. Loss Functions Are Usually Modules

Example:

```python
loss_fn = nn.MSELoss()
```

Notice:

```python
nn.MSELoss()
```

is an `nn.Module`.

This should feel familiar.

Like:

```python
nn.ReLU()
```

a loss function can be a Module even though it may not own trainable
Parameters.

---

# 3. Loss Functions Usually Have No Trainable Parameters

Example:

```python
loss_fn = nn.MSELoss()
```

Conceptually:

```text
Module?      YES

Parameters?  NO
```

The loss function participates in computation but does not normally own model
weights.

This reinforces:

```text
Participates in Autograd
      ≠
Owns Parameters
```

---

# 4. Predictions and Targets

A loss function compares:

```text
Prediction

and

Target
```

Conceptually:

```text
prediction ──┐
             │
             ▼
       Loss Function
             ▲
             │
target ──────┘
```

Result:

```text
loss
```

---

# 5. Example: Mean Squared Error

Suppose:

```python
prediction = [2,4]

target = [3,8]
```

Differences:

```text
[-1,-4]
```

Squared errors:

```text
[1,16]
```

Mean:

```text
(1+16)/2

=

8.5
```

Conceptually:

```python
loss = nn.MSELoss()(
    prediction,
    target
)
```

returns:

```text
8.5
```

---

# 6. Why Training Uses a Scalar Loss

A scalar loss gives a single objective:

```text
many errors
    ↓
single number
```

This is convenient for Autograd.

For a scalar:

```text
d(loss)
────────
d(loss)

=

1
```

Autograd can naturally begin its backward pass from that value.

Therefore training usually looks like:

```text
per-sample losses
       ↓
reduction
       ↓
scalar loss
       ↓
backward()
```

---

# 7. Loss Is Usually Non-Leaf

Example:

```python
prediction = model(x)

loss = loss_fn(
    prediction,
    target
)
```

Conceptually:

```text
prediction
    ↓
loss computation
    ↓
loss
```

Since loss was produced by operations:

```text
loss
```

is typically:

```text
non-leaf tensor
```

while parameters are usually leaves.

---

# 8. Backward Does Not Stop At Prediction

Conceptually:

```text
Parameters
    ↓
Model
    ↓
Prediction
    ↓
Loss
```

Backward:

```text
Loss
 ↑
Prediction
 ↑
Model
 ↑
Parameters
```

Eventually:

```text
parameter.grad

=

∂Loss
──────
∂Parameter
```

---

# 9. Reduction

Many losses support:

```text
reduction
```

Common choices:

```text
none

mean

sum
```

---

## reduction="none"

Returns individual loss values.

Example:

```text
[1,16]
```

---

## reduction="mean"

Returns:

```text
(1+16)/2

=

8.5
```

---

## reduction="sum"

Returns:

```text
17
```

---

# 10. Regression

Regression predicts continuous values.

Examples:

```text
House price

Temperature

Age prediction
```

Typical output:

```python
shape = (B,1)
```

Example:

```text
[125.4]

[232.8]

[118.6]
```

Loss:

```python
nn.MSELoss()
```

or sometimes:

```python
nn.L1Loss()
```

---

# 11. MSE Gradient Intuition

Suppose:

```python
prediction = [2,8]

target = [2,8]
```

Then:

```text
loss = 0
```

and:

```text
gradient = 0
```

This is desirable because:

```text
prediction already matches target
```

Therefore:

```text
no parameter update needed
```

---

# 12. Classification Introduces Logits

Consider:

```python
nn.Linear(hidden_dim, 3)
```

for:

```text
Cat
Dog
Bird
```

The output might be:

```text
[2.1, -0.5, 1.3]
```

These values are called:

```text
logits
```

My mental model:

```text
Raw model scores
       ↓
     Logits
```

Logits are not probabilities.

They:

```text
can be negative

can be larger than 1

do not sum to 1
```

---

# 13. Logits vs Probabilities

Example logits:

```text
[2.0,-1.0,0.5]
```

Possible probability conversion:

```text
Softmax
```

Conceptually:

```text
Logits
   ↓
Softmax
   ↓
Probabilities
```

Probabilities:

```text
sum to 1
```

Unlike logits.

---

# 14. Multiclass Classification

Examples:

```text
Cat
Dog
Bird

one correct class
```

Model output:

```python
shape = (B,C)
```

where:

```text
B = batch size

C = number of classes
```

Example:

```text
(32,10)
```

---

# 15. CrossEntropyLoss

Typical usage:

```python
loss = nn.CrossEntropyLoss()(
    logits,
    target
)
```

Important:

```text
input
=
logits
```

not probabilities.

---

# 16. Common Mistake: Softmax Before CrossEntropyLoss

Wrong:

```python
probs = softmax(
    logits
)

loss = CE(
    probs,
    target
)
```

Correct:

```python
loss = CE(
    logits,
    target
)
```

Mental model:

```text
CrossEntropyLoss
already expects logits
```

---

# 17. CrossEntropy Target Format

This is one of the most important PyTorch conventions.

Correct:

```python
target = [0,2,1]
```

Shape:

```text
(B,)
```

Targets represent:

```text
class indices
```

Example:

```text
0 = Cat

1 = Dog

2 = Bird
```

---

# 18. CrossEntropy Shapes

Input:

```text
(B,C)
```

Target:

```text
(B,)
```

Example:

```python
logits.shape

=
(64,5)
```

```python
target.shape

=
(64,)
```

This immediately suggests:

```text
CrossEntropyLoss
```

---

# 19. Common Mistake: One-Hot Targets

Wrong:

```python
[
 [1,0,0],
 [0,1,0]
]
```

for the most common CrossEntropy use case.

Correct:

```python
[0,1]
```

Class indices are the standard target representation.

---

# 20. CrossEntropy Target Dtype

Targets should typically be:

```python
torch.long
```

because they represent:

```text
class indices
```

not floating-point values.

---

# 21. Binary Classification

Examples:

```text
Spam / Not Spam

Fraud / Not Fraud

Positive / Negative
```

One yes/no decision.

---

# 22. Binary Classification Output

Typical final layer:

```python
nn.Linear(hidden_dim,1)
```

Output shape:

```text
(B,1)
```

Example:

```text
[2.1]

[-1.2]

[0.3]
```

These are:

```text
logits
```

not probabilities.

---

# 23. BCEWithLogitsLoss

Typical usage:

```python
loss = nn.BCEWithLogitsLoss()(
    logits,
    target
)
```

This is the preferred PyTorch pattern.

---

# 24. Common Mistake: Applying Sigmoid First

Wrong:

```python
prob = sigmoid(
    logits
)

loss = BCEWithLogitsLoss(
    prob,
    target
)
```

Correct:

```python
loss = BCEWithLogitsLoss(
    logits,
    target
)
```

Mental model:

```text
BCEWithLogitsLoss
expects logits
```

---

# 25. Binary Classification Target Format

Typical target values:

```text
0.0

or

1.0
```

Shape:

```text
(B,1)
```

Example:

```python
[
 [1.0],
 [0.0],
 [1.0]
]
```

---

# 26. Binary Classification Dtype

Typical binary targets are:

```python
float32
```

not class-index tensors.

This differs from CrossEntropyLoss.

---

# 27. Binary Inference

Training:

```python
loss = BCEWithLogitsLoss(
    logits,
    target
)
```

Inference:

```python
probability = sigmoid(
    logits
)
```

Mental model:

```text
Training
    ↓
Logits → BCEWithLogitsLoss


Inference
    ↓
Logits → Sigmoid → Probability
```

---

# 28. Multiclass Inference

Training:

```python
loss = CrossEntropyLoss(
    logits,
    target
)
```

Inference:

```python
probs = softmax(
    logits,
    dim=1
)
```

Mental model:

```text
Training
    ↓
Logits → CrossEntropyLoss


Inference
    ↓
Logits → Softmax → Probabilities
```

---

# 29. Shapes Cheat Sheet

## Regression

```text
prediction : (B,1)

target     : (B,1)
```

Loss:

```text
MSELoss
```

---

## Binary Classification

```text
logits : (B,1)

target : (B,1)
```

Loss:

```text
BCEWithLogitsLoss
```

Target values:

```text
0.0
1.0
```

---

## Multiclass Classification

```text
logits : (B,C)

target : (B,)
```

Loss:

```text
CrossEntropyLoss
```

Target values:

```text
0..C-1
```

---

# 30. Dtype Cheat Sheet

## MSELoss

```text
prediction : float

target     : float
```

---

## BCEWithLogitsLoss

```text
logits : float

target : float
```

---

## CrossEntropyLoss

```text
logits : float

target : long
```

because target represents:

```text
class indices
```

---

# 31. Common Debugging Errors

Error:

```text
Expected target size (B,)
got (B,C)
```

Likely cause:

```text
One-hot target used with CrossEntropyLoss
```

---

Error:

```text
Expected LongTensor
```

Likely cause:

```text
CrossEntropy target
is not integer class indices
```

---

Error:

```text
Training behaves strangely
```

Possible issue:

```text
Softmax applied before CrossEntropyLoss
```

---

Error:

```text
Training behaves strangely
```

Possible issue:

```text
Sigmoid applied before BCEWithLogitsLoss
```

---

# 32. Final Mental Model

```text
Regression
     ↓
Prediction
     ↓
MSELoss
     ↓
Scalar Loss
```

```text
Binary Classification
     ↓
One Logit
     ↓
BCEWithLogitsLoss
     ↓
Scalar Loss
```

```text
Multiclass Classification
     ↓
C Logits
     ↓
CrossEntropyLoss
     ↓
Scalar Loss
```

---

# 33. Interview Answer: What Is a Loss Function?

A loss function compares model predictions with targets and produces an
objective that can be differentiated by Autograd.

Its output is commonly reduced to a scalar value that becomes the quantity
used during `backward()`.

---

# 34. Interview Answer: Why Is Scalar Loss Convenient?

A scalar loss provides a single optimization objective.

Autograd can naturally begin backward computation using:

```text
d(loss)
────────
d(loss)

=

1
```

which makes gradient propagation straightforward.

---

# 35. Interview Answer: CrossEntropyLoss Input

CrossEntropyLoss normally expects:

```text
Input:
logits

Target:
class indices
```

I should normally pass raw logits directly and not apply Softmax first.

---

# 36. Interview Answer: BCEWithLogitsLoss Input

BCEWithLogitsLoss normally expects:

```text
Input:
logits

Target:
binary float values
```

I should normally pass raw logits directly and not apply Sigmoid first.

---

# 37. What I Understand Now

I understand:

```text
Loss Functions
│
├── prediction + target
├── scalar objective
├── reduction
│
├── MSELoss
├── BCEWithLogitsLoss
├── CrossEntropyLoss
│
├── logits
├── probabilities
│
├── target shapes
├── target dtypes
│
├── logits vs softmax
├── logits vs sigmoid
│
└── common debugging mistakes
```

---

# 38. What Comes Next

The next topic is:

```text
Optimizers
```

which answers:

```text
loss.backward()
      ↓
parameter.grad

then what?
```

Specifically:

```text
SGD

Learning Rate

Momentum

Adam

optimizer.step()

optimizer.zero_grad()
```

and finally how all components come together in a full training loop.
