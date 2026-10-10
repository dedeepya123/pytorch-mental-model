# Training Loop in PyTorch

## Goal

This chapter combines everything learned so far:

```text
Tensors

Autograd

nn.Module

Activations

Loss Functions

Optimizers
```

into a complete end-to-end training workflow.

By the end of this chapter, every line of a standard PyTorch training loop
should be understandable rather than memorized.

---

# 1. What Is Training?

Training is the process of repeatedly:

```text
Make Prediction
      ↓
Measure Error
      ↓
Compute Gradients
      ↓
Update Parameters
      ↓
Improve Model
```

The objective is:

```text
Better Parameters
```

after many iterations.

---

# 2. High-Level Training Pipeline

Conceptually:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Gradient Computation
  ↓
Parameter Update
  ↓
Repeat
```

---

# 3. The Canonical PyTorch Training Loop

```python
for x, y in dataloader:

    optimizer.zero_grad()

    prediction = model(x)

    loss = loss_fn(
        prediction,
        y
    )

    loss.backward()

    optimizer.step()
```

Every line has a specific responsibility.

---

# 4. Step 1 - Get a Batch

```python
for x, y in dataloader:
```

Conceptually:

```text
x
    ↓
Input Batch

y
    ↓
Target Batch
```

Example:

```python
x.shape = (32, 784)

y.shape = (32,)
```

Meaning:

```text
32 samples are being processed together
```

---

# 5. Step 2 - Clear Old Gradients

```python
optimizer.zero_grad()
```

Purpose:

```text
Remove gradients
from previous iterations
```

Reason:

PyTorch accumulates gradients by default.

Without clearing:

```text
old gradient

+

new gradient
```

would remain inside:

```python
parameter.grad
```

---

# 6. Gradient Accumulation Example

Iteration 1:

```text
parameter.grad = 4
```

Iteration 2:

```text
new gradient = 3
```

Without:

```python
zero_grad()
```

Result:

```text
parameter.grad

=

7
```

Most training loops instead want:

```text
current batch gradient only
```

---

# 7. State After zero_grad()

Before:

```text
parameter      = 10

parameter.grad = 4
```

After:

```text
parameter      = 10

parameter.grad = 0
```

Notice:

```text
Parameter unchanged

Gradient cleared
```

---

# 8. Step 3 - Forward Pass

```python
prediction = model(x)
```

This executes the Module hierarchy.

Example:

```text
Input
   ↓
Linear
   ↓
ReLU
   ↓
Linear
   ↓
Prediction
```

---

# 9. What Is Produced During Forward?

Two important things are produced.

### 1. Prediction

Example:

```python
prediction
```

or:

```python
logits
```

depending on the task.

---

### 2. Computation Graph

When gradient tracking is enabled:

```text
Autograd records
relevant tensor operations
```

for the later backward pass.

So forward pass conceptually produces:

```text
Prediction

+

Computation Graph
```

---

# 10. Step 4 - Compute Loss

```python
loss = loss_fn(
    prediction,
    target
)
```

The loss function compares:

```text
Prediction

and

Target
```

and produces a training objective.

---

# 11. Examples of Loss Functions

Regression:

```python
nn.MSELoss()
```

---

Binary Classification:

```python
nn.BCEWithLogitsLoss()
```

---

Multiclass Classification:

```python
nn.CrossEntropyLoss()
```

---

# 12. Loss Is Usually Scalar

Typical result:

```text
loss = 0.83
```

A scalar loss provides:

```text
single optimization objective
```

for Autograd.

---

# 13. Step 5 - Backward Pass

```python
loss.backward()
```

This initiates Autograd.

Conceptually:

```text
Parameters
    ↓
Prediction
    ↓
Loss
```

Backward traverses:

```text
Loss
 ↑
Prediction
 ↑
Parameters
```

and computes gradients.

---

# 14. What Is Computed?

For each Parameter:

```text
∂Loss
──────
∂Parameter
```

These gradients are stored in:

```python
parameter.grad
```

---

# 15. Important Distinction

After:

```python
loss.backward()
```

the gradients change.

The Parameters do not.

Example:

Before:

```text
parameter      = 10

grad           = 0
```

After:

```text
parameter      = 10

grad           = 4
```

The Parameter value is unchanged.

---

# 16. Step 6 - Optimizer Update

```python
optimizer.step()
```

This is where Parameters actually change.

The optimizer reads:

```python
parameter.grad
```

and updates Parameter values.

---

# 17. SGD Mental Model

Conceptually:

```text
parameter

=

parameter
-
learning_rate × gradient
```

Example:

```text
parameter = 10

gradient = 4

lr = 0.1
```

Result:

```text
parameter = 9.6
```

---

# 18. State After step()

Before:

```text
parameter      = 10

parameter.grad = 4
```

After:

```text
parameter      = 9.6

parameter.grad = 4
```

Notice:

```text
Parameter updated

Gradient not automatically cleared
```

---

# 19. Complete Parameter Lifecycle

Initial:

```text
parameter = 10
```

---

After zero_grad():

```text
parameter = 10

grad = 0
```

---

After forward():

```text
prediction computed
```

---

After loss():

```text
loss computed
```

---

After backward():

```text
parameter = 10

grad = 4
```

---

After step():

```text
parameter = 9.6

grad = 4
```

---

# 20. Why Does Training Improve?

Each iteration attempts to move Parameters toward values that reduce loss.

Conceptually:

```text
Current Parameters
       ↓
Current Loss
       ↓
Gradient
       ↓
Updated Parameters
       ↓
Lower Loss
```

Repeated over many iterations.

---

# 21. Mini Training Loop Anatomy

```python
optimizer.zero_grad()

prediction = model(x)

loss = loss_fn(
    prediction,
    y
)

loss.backward()

optimizer.step()
```

Mental interpretation:

```text
Clear Gradients

Forward Pass

Compute Loss

Compute Gradients

Update Parameters
```

---

# 22. Batches

Training usually operates on batches.

Suppose:

```text
Dataset

1000 samples
```

Batch size:

```text
100
```

Each iteration processes:

```text
100 samples
```

at once.

---

# 23. What Is An Iteration?

One execution of:

```python
for x, y in dataloader:
```

is often called:

```text
one iteration
```

or:

```text
one training step
```

---

# 24. What Is An Epoch?

One complete pass through the entire dataset.

Example:

```text
1000 samples

batch size = 100
```

Produces:

```text
10 batches
```

Processing all 10 batches once:

```text
1 epoch
```

---

# 25. Epoch Structure

Typical structure:

```python
for epoch in range(num_epochs):

    for x, y in dataloader:

        ...
```

Outer loop:

```text
Epochs
```

Inner loop:

```text
Batches
```

---

# 26. Why Multiple Epochs?

After one epoch:

```text
Model has seen
every training sample once
```

Usually that is not enough to learn.

Therefore:

```text
Epoch 1

Epoch 2

Epoch 3

...
```

continue refining Parameters.

---

# 27. Training Mode

Before training:

```python
model.train()
```

Purpose:

```text
Place modules
into training mode
```

---

# 28. What Changes In Training Mode?

Important examples:

```text
Dropout

BatchNorm
```

Training mode causes them to use training behavior.

---

# 29. Evaluation Mode

For validation:

```python
model.eval()
```

Purpose:

```text
Place modules
into evaluation mode
```

---

# 30. What Changes In Evaluation Mode?

Dropout:

```text
Disabled
```

BatchNorm:

```text
Uses running statistics
```

instead of batch statistics.

---

# 31. train() vs eval()

Important:

```text
model.train()

and

model.eval()
```

do not control Autograd.

They control Module behavior.

---

# 32. Autograd Control Is Separate

Autograd is controlled using:

```python
torch.no_grad()
```

Example:

```python
with torch.no_grad():
    prediction = model(x)
```

This disables gradient tracking.

---

# 33. Validation Loop

Typical pattern:

```python
model.eval()

with torch.no_grad():

    for x, y in dataloader:

        prediction = model(x)
```

Notice:

```text
No backward()

No optimizer.step()
```

because validation does not update Parameters.

---

# 34. Training vs Validation

Training:

```text
train()

forward

loss

backward

optimizer.step()
```

---

Validation:

```text
eval()

forward

metrics
```

No Parameter updates occur.

---

# 35. Common Misconception: backward() Updates Parameters

Incorrect:

```text
loss.backward()
      ↓
weights change
```

Correct:

```text
loss.backward()
      ↓
compute gradients
```

---

# 36. Common Misconception: step() Computes Gradients

Incorrect:

```text
optimizer.step()
      ↓
compute gradients
```

Correct:

```text
optimizer.step()
      ↓
uses existing gradients
to update Parameters
```

---

# 37. Common Misconception: eval() Disables Autograd

Incorrect:

```text
eval()
      ↓
no gradients
```

Correct:

```text
eval()
      ↓
changes module behavior
```

Autograd control is separate.

---

# 38. Interview Answer: Explain A Training Loop

A training loop repeatedly:

```text
Load Batch
      ↓
Forward Pass
      ↓
Compute Loss
      ↓
Backward Pass
      ↓
Optimizer Update
```

to improve model Parameters over time.

---

# 39. Interview Answer: Why Call zero_grad()?

PyTorch accumulates gradients in:

```python
parameter.grad
```

Therefore earlier gradients are cleared before computing gradients for the current batch.

---

# 40. Interview Answer: What Happens During forward()?

The model executes the Module hierarchy to produce predictions while Autograd records relevant tensor operations for later differentiation.

---

# 41. Interview Answer: What Happens During backward()?

Autograd traverses the computation graph in reverse order and computes gradients of the loss with respect to model Parameters.

---

# 42. Interview Answer: What Happens During step()?

The optimizer reads gradients stored in:

```python
parameter.grad
```

and updates Parameter values according to its optimization algorithm.

---

# 43. Teach It To Someone

If I were teaching the training loop:

> A batch enters the model and produces predictions. The loss function compares predictions with targets and returns a scalar objective. Autograd computes gradients of that loss with respect to model Parameters and stores them in `parameter.grad`. The optimizer then uses those gradients to update Parameter values. Repeating this process over many batches and epochs gradually improves the model.

---

# 44. Final Mental Model

```text
Batch
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Backward
  ↓
parameter.grad
  ↓
Optimizer
  ↓
Updated Parameters
  ↓
Next Batch
```

---

# 45. What I Understand Now

I understand:

```text
Training Loop
│
├── forward()
├── loss()
├── backward()
├── optimizer.step()
├── optimizer.zero_grad()
│
├── iteration
├── batch
├── epoch
│
├── train()
├── eval()
├── no_grad()
│
└── end-to-end training flow
```

---

# 46. Neural Networks Section Complete

I now understand:

```text
Neural Networks
│
├── nn.Module
├── Parameters
├── Buffers
├── Linear
├── train/eval
├── Dropout
├── BatchNorm
├── Activations
├── Loss Functions
├── Optimizers
└── Training Loop
```

This provides a complete mental model of how a PyTorch model learns from data.
