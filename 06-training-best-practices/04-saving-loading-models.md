# Saving and Loading Models

## Goal

Training a model can take:

```text
Minutes

Hours

Days

Weeks
```

Once training finishes, we need a way to preserve the learned state of the model.

PyTorch provides mechanisms to:

```text
Save Models

Load Models

Resume Training

Run Inference
```

The goals of this chapter are:

- Understand what should be saved.
- Understand `state_dict()`.
- Understand parameters vs buffers.
- Save and load model state.
- Create training checkpoints.
- Resume interrupted training.
- Understand the model lifecycle.

---

# 1. Why Save Models?

Suppose training takes:

```text
10 Hours
```

and achieves excellent results.

If Python exits:

```text
All Learned Parameters
Are Lost
```

unless they have been saved.

---

# 2. What Does Training Actually Produce?

Training does not modify the architecture.

It modifies:

```text
Parameters

Buffers
```

inside the model.

---

Example:

```python
nn.Linear(
    10,
    64
)
```

contains:

```text
weight

bias
```

These values change during training.

---

# 3. What Do We Need To Save?

The most important component is:

```text
Learned State
```

The learned state consists of:

```text
Parameters

Buffers
```

stored inside the model.

---

# 4. The Most Important API

```python
model.state_dict()
```

This is the primary saving mechanism in PyTorch.

---

# 5. What Is state_dict()?

A model is a module tree.

Example:

```text
Model
│
├── fc1
├── fc2
└── fc3
```

---

`state_dict()` recursively walks the module tree and collects:

```text
Parameters

Buffers
```

from all registered modules.

---

# 6. What Gets Stored?

Examples:

```text
Weights

Biases

Running Means

Running Variances
```

Anything registered as model state becomes part of the state dictionary.

---

# 7. Example State Dict

Suppose:

```python
class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(10,64)

        self.fc2 = nn.Linear(64,3)
```

---

Inspect:

```python
model.state_dict().keys()
```

Possible output:

```text
fc1.weight

fc1.bias

fc2.weight

fc2.bias
```

---

# 8. Parameters vs Buffers

Recall:

## Parameters

Trainable tensors.

Examples:

```text
weight

bias
```

---

## Buffers

Non-trainable state.

Examples:

```text
running_mean

running_var
```

used by BatchNorm.

---

Both are included in:

```python
state_dict()
```

---

# 9. BatchNorm And Saved State

BatchNorm stores:

```text
weight (gamma)

bias (beta)

running_mean

running_var
```

---

These values are critical for inference and must be restored when loading a model.

---

# 10. Saving A Model

Most common approach:

```python
torch.save(
    model.state_dict(),
    "model.pt"
)
```

---

Mental model:

```text
Module Tree

↓

state_dict()

↓

Disk
```

---

# 11. What Is Saved?

Only:

```text
Parameters

Buffers
```

are saved.

---

Not:

```text
Python Source Code

Training Loop

Dataset

Optimizer
```

---

# 12. Why Architecture Is Not Saved

PyTorch assumes your architecture already exists in code.

The saved file contains state, not structure.

---

Think:

```text
Blueprint

+

Learned State

=

Working Model
```

---

# 13. Loading A Model

Step 1:

Create architecture.

```python
model = MLP()
```

---

Step 2:

Load state.

```python
model.load_state_dict(
    torch.load("model.pt")
)
```

---

Step 3:

Switch to evaluation mode.

```python
model.eval()
```

---

# 14. Why Must Architecture Be Recreated?

Because:

```python
load_state_dict()
```

does not create modules.

It only loads:

```text
Parameters

Buffers
```

into existing modules.

---

# 15. Mental Model

Incorrect:

```text
load_state_dict()

↓

creates model
```

---

Correct:

```text
Create Model

↓

Load State

↓

Model Ready
```

---

# 16. Common Loading Error

Suppose:

Saved architecture:

```python
Linear(10,64)
```

---

Current architecture:

```python
Linear(10,128)
```

---

Loading fails due to:

```text
Shape Mismatch
```

---

The model architecture must match the saved state.

---

# 17. Beyond Model Saving

Sometimes we want:

```text
Resume Training
```

instead of only running inference.

---

Additional information becomes important.

---

# 18. Optimizer State

Optimizers also contain state.

Example:

```text
Adam

AdamW

RMSProp
```

maintain internal statistics.

---

Training quality may change if optimizer state is lost.

---

# 19. Inspecting Optimizer State

```python
optimizer.state_dict()
```

returns optimizer state.

---

This state can also be saved and restored.

---

# 20. What Is A Checkpoint?

A checkpoint stores:

```text
More Than Just Model Parameters
```

---

Typical checkpoint:

```text
Model State

Optimizer State

Epoch

Loss

Metrics
```

---

# 21. Saving A Checkpoint

```python
torch.save({

    "epoch": epoch,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "loss": loss

},
"checkpoint.pt")
```

---

# 22. Why Save Checkpoints?

Benefits:

```text
Resume Interrupted Training

Track Progress

Restore Best Model

Recover From Failures
```

---

# 23. Loading A Checkpoint

```python
checkpoint = torch.load(
    "checkpoint.pt"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

epoch = checkpoint["epoch"]
```

---

Training can continue from the saved point.

---

# 24. Training Workflow

```text
Create Model

↓

Train

↓

Save Checkpoint

↓

Continue Later
```

---

# 25. Inference Workflow

```text
Create Model

↓

Load State Dict

↓

model.eval()

↓

Predict
```

---

# 26. Why model.eval() Matters

Some layers behave differently:

```text
Dropout

BatchNorm
```

---

Without:

```python
model.eval()
```

predictions may be inconsistent.

---

# 27. The Model Lifecycle

```text
Create

↓

Initialize

↓

Train

↓

Save

↓

Load

↓

Predict
```

---

Understanding this lifecycle is critical in real-world projects.

---

# 28. Mental Model Of Saved State

Think:

```text
state_dict()

=

Everything Learned
By Training
```

---

But not:

```text
Architecture

Dataset

Training Loop
```

---

# 29. Common Beginner Mistake

Saving only the model state:

```python
model.state_dict()
```

then expecting training to resume exactly.

---

To fully resume training:

```text
Save Optimizer State
```

as well.

---

# 30. Common Beginner Mistake

Loading a model and immediately running inference:

```python
pred = model(x)
```

without:

```python
model.eval()
```

This can produce unexpected results.

---

# 31. Common Beginner Mistake

Changing architecture before loading:

```python
Linear(64,128)
```

instead of:

```python
Linear(64,64)
```

Result:

```text
Shape Mismatch Error
```

---

# 32. Interview Question

What is stored in a state dictionary?

Strong answer:

> A state dictionary contains all registered parameters and buffers required to reconstruct the learned state of a model.

---

# 33. Interview Question

Why doesn't state_dict() save architecture?

Strong answer:

> PyTorch separates model architecture from model state. The architecture is defined by code, while state_dict() stores learned values.

---

# 34. Interview Question

Why save optimizer state?

Strong answer:

> Optimizers maintain internal statistics. Saving optimizer state allows training to resume from the same optimization state rather than starting fresh.

---

# 35. Interview Question

Why call model.eval() after loading a model?

Strong answer:

> Layers such as Dropout and BatchNorm behave differently during training and inference. eval() switches the model into inference mode.

---

# 36. Teach It To Someone

If I were teaching model persistence:

> Training produces learned state, not a new architecture. PyTorch stores this learned state inside a state dictionary. Saving and loading models is fundamentally the process of saving and restoring that state.

---

# 37. Master Mental Model

```text
Architecture

+

State Dict

↓

Working Model

↓

Inference

or

Training
```

---

# 38. PyTorch Mental Model

```text
Model
│
├── Parameters
├── Buffers
└── state_dict()

Optimizer
│
└── optimizer.state_dict()

Checkpoint
│
├── Model State
├── Optimizer State
├── Epoch
└── Metrics
```

---

# 39. Relationship To Previous Chapters

```text
Parameters
      ↓

Module Tree
      ↓

state_dict()
      ↓

Checkpoint
      ↓

Resume Training
```

This chapter ties together model structure, parameters, buffers, and training state.

---

# 40. What I Understand Now

```text
Saving & Loading Models
│
├── state_dict()
├── Parameters
├── Buffers
├── BatchNorm State
├── Model Saving
├── Model Loading
├── Checkpoints
├── Optimizer State
├── Resume Training
└── Inference Workflow
```

---

# 41. What Comes Next

The next chapter introduces:

```text
GPU Training
```

and answers:

```text
How Devices Work

Moving Models To GPUs

Moving Tensors To GPUs

Device Mismatch Errors

Memory Management

Real-World Training Workflow
```
