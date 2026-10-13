# GPU Training

## Goal

Modern deep learning workloads are primarily trained on GPUs.

Understanding GPUs in PyTorch requires understanding the concept of:

```text
Devices
```

Every tensor, parameter, buffer, and gradient lives on a device.

The goals of this chapter are:

- Understand PyTorch devices.
- Understand CPU vs GPU execution.
- Move tensors between devices.
- Move models between devices.
- Understand device mismatch errors.
- Understand GPU memory.
- Understand real-world training workflows.

---

# 1. Why GPUs Matter

Neural networks perform large amounts of:

```text
Matrix Multiplication

Convolution

Vector Operations
```

These operations are highly parallelizable.

---

## CPU

Designed for:

```text
General Purpose Computation
```

Characteristics:

```text
Few Powerful Cores
```

---

## GPU

Designed for:

```text
Massively Parallel Computation
```

Characteristics:

```text
Thousands Of Smaller Cores
```

---

Result:

```text
Deep Learning Training
Usually Runs Much Faster On GPUs
```

---

# 2. The Device Mental Model

Every tensor lives somewhere.

Think:

```text
Tensor
  ↓
Device
```

Examples:

```text
CPU

GPU
```

---

# 3. Tensors Have Devices

Example:

```python
x = torch.tensor([1, 2, 3])

print(x.device)
```

Output:

```text
cpu
```

---

Mental model:

```text
Tensor

+

Storage Location
```

---

# 4. Models Have Devices Too

Model parameters are tensors.

Example:

```python
next(model.parameters()).device
```

Possible output:

```text
cpu
```

or:

```text
cuda:0
```

---

# 5. Important Rule

Computations require:

```text
Model Device

=

Tensor Device
```

---

Example:

```text
Model → GPU

Tensor → CPU
```

will fail.

---

# 6. The Most Common Error

Example:

```python
model = model.cuda()

x = x.cpu()

output = model(x)
```

Result:

```text
Device Mismatch Error
```

---

PyTorch cannot perform operations across different devices automatically.

---

# 7. Detecting GPU Availability

Check:

```python
torch.cuda.is_available()
```

Example:

```python
if torch.cuda.is_available():
    print("GPU Available")
```

---

# 8. Selecting A Device

Standard pattern:

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

---

Mental model:

```text
Select Best Available Device
```

at program startup.

---

# 9. Moving Tensors

Use:

```python
x = x.to(device)
```

Example:

```python
x = x.to("cuda")
```

---

Verify:

```python
print(x.device)
```

Output:

```text
cuda:0
```

---

# 10. Moving Models

Exactly the same API.

```python
model = model.to(device)
```

---

# 11. What Does model.to() Move?

PyTorch recursively traverses the module tree.

Moves:

```text
Parameters

Buffers
```

to the requested device.

---

# 12. Parameters And Buffers

Recall:

### Parameters

Examples:

```text
Weights

Biases
```

---

### Buffers

Examples:

```text
running_mean

running_var
```

from BatchNorm.

---

Both are moved.

---

# 13. Why This Matters

A model is more than:

```text
Weights
```

It also contains:

```text
BatchNorm Statistics

Other Registered State
```

All must be moved consistently.

---

# 14. Standard Training Workflow

Before every forward pass:

```python
x = x.to(device)

y = y.to(device)
```

---

Then:

```python
output = model(x)
```

Everything now lives on the same device.

---

# 15. Full Training Template

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = MyModel().to(device)

for x, y in dataloader:

    x = x.to(device)

    y = y.to(device)

    output = model(x)
```

---

# 16. A Useful Debugging Technique

Check tensor location:

```python
print(x.device)
```

---

Check model location:

```python
print(
    next(
        model.parameters()
    ).device
)
```

---

# 17. Device Mismatch Debugging

Whenever you see:

```text
Expected all tensors
to be on the same device
```

check:

```text
Input Device

Target Device

Model Device
```

---

# 18. Mental Rule

Every computation must satisfy:

```text
Inputs

Targets

Parameters

Buffers

Gradients
```

all on the:

```text
Same Device
```

---

# 19. Loading GPU Models On CPU

Suppose:

```text
Checkpoint trained on GPU
```

Current machine:

```text
CPU Only
```

---

Use:

```python
torch.load(
    path,
    map_location="cpu"
)
```

---

Mental model:

```text
Remap Saved State
To CPU
```

during loading.

---

# 20. GPU Memory

GPUs have limited memory.

Common runtime error:

```text
CUDA Out Of Memory
```

---

# 21. Why OOM Happens

Possible causes:

```text
Batch Size Too Large

Model Too Large

Activations Too Large

Limited GPU Memory
```

---

# 22. First Fix For OOM

Most common solution:

```text
Reduce Batch Size
```

Example:

```text
128

↓

64

↓

32
```

---

# 23. Other OOM Fixes

Consider:

```text
Smaller Models

Gradient Accumulation

Mixed Precision Training
```

---

# 24. Batch Size And Memory

Larger batch:

```text
More Samples

↓

More Activations

↓

More Memory
```

---

Smaller batch:

```text
Less Memory
```

---

# 25. Gradients Have Devices Too

Remember:

```python
loss.backward()
```

creates gradients.

Gradients are stored on the same device as their parameters.

---

Example:

```text
Parameter → GPU

Gradient → GPU
```

---

# 26. train() And eval() Still Matter

Device handling does not replace:

```python
model.train()

model.eval()
```

---

These are separate concepts.

---

## train()

Controls:

```text
Dropout

BatchNorm
```

behavior.

---

## Device

Controls:

```text
Where Computation Happens
```

---

# 27. Common Beginner Mistake

Incorrect:

```python
model.eval()

# assume model moved to GPU
```

---

Reality:

```text
Mode

and

Device
```

are independent.

---

# 28. Real Training Loop

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = MyModel().to(device)

model.train()

for x, y in dataloader:

    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(
        output,
        y
    )

    loss.backward()

    optimizer.step()
```

---

# 29. Real Inference Loop

```python
model.load_state_dict(
    torch.load(path)
)

model.to(device)

model.eval()

with torch.no_grad():

    x = x.to(device)

    pred = model(x)
```

---

# 30. The Device Hierarchy

Think:

```text
Tensor
    ↓
Device

Parameter
    ↓
Device

Buffer
    ↓
Device

Gradient
    ↓
Device
```

Everything has a location.

---

# 31. Device Mental Model

The most important PyTorch rule:

```text
Data

↓

Model

↓

Computation
```

must happen on:

```text
One Shared Device
```

---

# 32. Common Beginner Mistakes

### Mistake 1

Move model.

Forget inputs.

---

### Mistake 2

Move inputs.

Forget model.

---

### Mistake 3

Load checkpoint to CPU.

Attempt GPU inference.

---

### Mistake 4

Forget `map_location`.

---

# 33. Interview Question

Why do GPUs accelerate deep learning?

Strong answer:

> GPUs contain many processing cores and are optimized for highly parallel numerical operations such as matrix multiplication and convolution.

---

# 34. Interview Question

What does:

```python
model.to(device)
```

do?

Strong answer:

> It recursively moves all registered parameters and buffers in the model to the specified device.

---

# 35. Interview Question

Why does a device mismatch error occur?

Strong answer:

> PyTorch requires tensors participating in the same computation to reside on the same device.

---

# 36. Interview Question

What is usually the first fix for CUDA Out Of Memory errors?

Strong answer:

> Reduce the batch size because batch size directly impacts activation memory usage.

---

# 37. Teach It To Someone

If I were teaching GPU training:

> Every tensor in PyTorch has a device. Models consist of parameters and buffers that also have devices. Successful GPU training is primarily about ensuring that data, models, gradients, and state all live on the same device during computation.

---

# 38. Master Mental Model

```text
Create Device

↓

Move Model

↓

Move Data

↓

Forward Pass

↓

Backward Pass

↓

Optimization
```

All on the same device.

---

# 39. Relationship To Previous Chapters

```text
Parameters
      ↓

Buffers
      ↓

state_dict()
      ↓

model.to(device)
      ↓

GPU Training
```

The same model state learned during training must be moved to the correct device for computation.

---

# 40. What I Understand Now

```text
GPU Training
│
├── Device Concept
├── CPU vs GPU
├── Tensor Devices
├── Model Devices
├── Parameters
├── Buffers
├── Device Transfers
├── Device Mismatch Errors
├── GPU Memory
├── OOM Debugging
├── Training Workflow
└── Inference Workflow
```

---

# 41. What Comes Next

The next chapter introduces:

```text
Training Debugging
```

and answers:

```text
Why Loss Does Not Decrease

How To Diagnose Training Failures

How To Check Gradients

How To Validate Data Pipelines

How To Systematically Debug Models
```
