# Model Debugging and Inspection

## Goal

Building a model is only part of the job.

A significant portion of deep learning development involves:

```text
Inspecting Models

Validating Shapes

Checking Parameters

Verifying Gradients

Debugging Training Problems
```

The goal of this chapter is to develop a systematic workflow for understanding and debugging PyTorch models.

---

# 1. Why Debugging Matters

Many beginners assume most failures come from:

```text
Optimizers

Loss Functions

Autograd
```

In practice, the majority of failures come from:

```text
Wrong Shapes

Wrong Dimensions

Missing Gradients

Incorrect Layer Configuration
```

A good engineer can isolate these issues quickly.

---

# 2. A Layered Debugging Strategy

Debug from top to bottom:

```text
Architecture
      ↓
Shapes
      ↓
Parameters
      ↓
Gradients
      ↓
Training Behavior
```

Do not start by changing hyperparameters.

Verify correctness first.

---

# 3. Inspecting Model Architecture

The first check should always be:

```python
print(model)
```

Example:

```python
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

print(model)
```

Output:

```text
Sequential(
  (0): Linear(...)
  (1): ReLU()
  (2): Linear(...)
)
```

Verify:

```text
Correct Layers

Correct Ordering

Correct Dimensions
```

---

# 4. Understanding The Module Tree

A model is not a list of layers.

It is a:

```text
Module Tree
```

Example:

```text
Model
│
├── Block1
│   ├── fc1
│   └── fc2
│
└── Block2
    ├── fc1
    └── fc2
```

Inspection tools operate by traversing this structure.

---

# 5. Inspecting Child Modules

Direct children:

```python
list(model.children())
```

Useful for viewing the immediate structure of a model.

---

# 6. Inspecting All Modules

```python
for name, module in model.named_modules():
    print(name, module)
```

Example:

```text
block1

block1.fc1

block1.relu

block2.fc1
```

Useful for exploring large architectures.

---

# 7. Inspecting Parameters

The most useful inspection API:

```python
for name, param in model.named_parameters():
    print(name)
```

Example:

```text
fc1.weight

fc1.bias

fc2.weight

fc2.bias
```

---

# 8. Inspecting Parameter Shapes

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

Output:

```text
fc1.weight (64,10)

fc1.bias (64,)

fc2.weight (3,64)

fc2.bias (3,)
```

Shapes should match your architecture design.

---

# 9. Counting Parameters

Total parameters:

```python
sum(
    p.numel()
    for p in model.parameters()
)
```

---

Trainable parameters:

```python
sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

---

# 10. Why Parameter Counts Matter

Parameter counts help estimate:

```text
Memory Usage

Model Complexity

Training Cost

Overfitting Risk
```

---

# 11. Parameter Counting Example

Layer:

```python
nn.Linear(10, 64)
```

Weights:

```text
64 × 10 = 640
```

Bias:

```text
64
```

Total:

```text
704 Parameters
```

---

# 12. Shape Inspection During Forward Pass

The most effective debugging technique:

```python
def forward(self, x):

    print(x.shape)

    x = self.fc1(x)
    print(x.shape)

    x = self.fc2(x)
    print(x.shape)

    return x
```

---

# 13. Why Shape Inspection Works

Most runtime errors originate from:

```text
Shape Mismatches
```

Shape tracing usually reveals the issue immediately.

---

# 14. Verifying Gradients

After:

```python
loss.backward()
```

check:

```python
for name, param in model.named_parameters():
    print(
        name,
        param.grad is not None
    )
```

Expected:

```text
True
```

for trainable parameters.

---

# 15. Checking Gradient Magnitudes

```python
for name, param in model.named_parameters():
    print(
        name,
        param.grad.norm()
    )
```

Useful for diagnosing:

```text
Vanishing Gradients

Exploding Gradients
```

---

# 16. Common Bug: Missing Gradients

Symptoms:

```text
Loss Does Not Decrease
```

Potential causes:

```text
Detached Tensor

Missing backward()

requires_grad=False
```

---

# 17. Common Bug: Shape Mismatch

Example:

Model output:

```text
(32,10)
```

Loss expects:

```text
(32,1)
```

Training fails.

Always confirm tensor shapes.

---

# 18. Common Bug: Incorrect Flatten Size

Expected:

```python
nn.Linear(1024, 10)
```

Actual flattened tensor:

```text
2048 Features
```

A very common CNN error.

---

# 19. Common Bug: Oversized Models

Example:

```python
nn.Linear(
    150528,
    5000
)
```

Potential symptoms:

```text
Out Of Memory

Slow Training

Huge Parameter Count
```

---

# 20. Verifying Training Mode

Training:

```python
model.train()
```

Evaluation:

```python
model.eval()
```

Important for:

```text
Dropout

BatchNorm
```

---

# 21. Inspecting State

```python
model.state_dict().keys()
```

Example:

```text
fc1.weight

fc1.bias

fc2.weight

fc2.bias
```

Provides visibility into all registered state.

---

# 22. Practical Debugging Workflow

When a model fails:

### Step 1

Inspect architecture:

```python
print(model)
```

### Step 2

Trace tensor shapes.

### Step 3

Verify parameter counts.

### Step 4

Verify gradients exist.

### Step 5

Check train/eval mode.

### Step 6

Verify loss decreases.

---

# 23. Interview Question

How do you count trainable parameters?

```python
sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

---

# 24. Interview Question

How do you debug shape mismatches?

Strong answer:

> Trace tensor shapes through the forward pass and verify that each layer's output shape matches the next layer's expected input shape.

---

# 25. Interview Question

How do you verify gradients are flowing?

Strong answer:

> After calling `loss.backward()`, inspect parameter gradients and ensure `param.grad` exists and has reasonable magnitude.

---

# 26. Teach It To Someone

If I were teaching model debugging:

> Start with structure, then validate shapes, then verify parameters, then verify gradients. Most model failures become easy to diagnose when inspected in this order.

---

# 27. Master Mental Model

```text
Model
 ↓
Architecture

 ↓
Shapes

 ↓
Parameters

 ↓
Gradients

 ↓
Training Behavior
```

Always debug in this order.

---

# 28. What I Understand Now

I understand:

```text
Model Inspection
│
├── print(model)
├── children()
├── named_modules()
├── named_parameters()
├── Parameter Counts
├── Shape Tracing
├── Gradient Inspection
├── state_dict()
├── train() / eval()
└── Systematic Debugging
```

---

# 29. What Comes Next

The next section focuses on:

```text
Training Best Practices
```

including:

```text
Weight Initialization

Regularization

BatchNorm

Saving & Loading Models

GPU Training

Training Debugging
```

These topics explain how to make models train reliably and efficiently.
