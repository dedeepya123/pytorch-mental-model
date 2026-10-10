# DataLoader in PyTorch

## Goal

A Dataset provides individual samples.

Training, however, usually operates on batches.

The goal of this chapter is to understand:

- Why DataLoader exists.
- Dataset vs DataLoader.
- batch_size.
- shuffle.
- num_workers.
- drop_last.
- Collation.
- How batches are produced.
- How DataLoader connects to the training loop.

---

# 1. Why DataLoader Exists

Dataset provides:

```python
dataset[i]
```

which returns:

```python
(x, y)
```

Only one sample.

Models typically train on:

```text
Batches
```

Therefore we need:

```text
DataLoader
```

to create batches.

---

# 2. Dataset vs DataLoader

Dataset:

```text
Provides Samples
```

DataLoader:

```text
Provides Batches
```

Mental model:

```text
Dataset
    ↓
Sample

DataLoader
    ↓
Batch
```

---

# 3. Creating A DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32
)
```

Now the DataLoader can generate batches from the Dataset.

---

# 4. What Happens Internally?

Conceptually:

```python
dataset[0]
dataset[1]
dataset[2]
...
dataset[31]
```

are fetched.

Those samples are then combined into one batch.

---

# 5. Single Sample vs Batch

Dataset sample:

```text
x.shape = (784,)
```

DataLoader batch:

```text
x.shape = (32,784)
```

assuming:

```text
batch_size = 32
```

---

# 6. Where Does The Extra Dimension Come From?

The batch dimension is added during:

```text
Collation
```

which combines multiple samples into one tensor.

---

# 7. Collation

Suppose:

```python
(x0,y0)
(x1,y1)
(x2,y2)
(x3,y3)
```

are fetched.

Collation converts them into:

```python
x_batch

y_batch
```

with shapes:

```text
(4, ...)
```

---

# 8. Dataset Calls vs Collation Calls

For:

```text
batch_size = 32
```

DataLoader typically performs:

```text
32 Dataset Calls
```

followed by:

```text
1 Collation Step
```

to create the batch.

---

# 9. The Data Pipeline

```text
Dataset
   ↓
dataset[i]
   ↓
Single Samples
   ↓
Collation
   ↓
Batch Tensor
   ↓
DataLoader
```

---

# 10. batch_size

Controls:

```text
Number Of Samples
Per Batch
```

Example:

```python
batch_size=32
```

means:

```text
32 samples
```

per iteration.

---

# 11. Training Loop Connection

```python
for x, y in loader:
```

returns:

```text
one batch
```

per iteration.

---

# 12. shuffle

Example:

```python
DataLoader(
    dataset,
    shuffle=True
)
```

The traversal order becomes randomized.

---

# 13. What Gets Shuffled?

The Dataset does not change.

Instead:

```text
Indices
```

are shuffled.

Example:

Before:

```text
0 1 2 3 4 5
```

After:

```text
4 1 5 0 2 3
```

---

# 14. Why Shuffling Helps

Without shuffling:

```text
Batch 1
Cats

Batch 2
Dogs

Batch 3
Birds
```

The optimizer may see highly biased batches.

With shuffling:

```text
Cat
Dog
Bird
```

samples are mixed more naturally.

---

# 15. Does Shuffling Occur Every Epoch?

Typically yes.

Each epoch usually receives a new randomized ordering.

---

# 16. Training vs Validation

Common setup:

```python
train_loader
```

```python
shuffle=True
```

---

```python
val_loader
```

```python
shuffle=False
```

Validation does not update Parameters, so shuffling is usually unnecessary.

---

# 17. num_workers

Controls:

```text
How Many Worker Processes
Load Data
```

Example:

```python
num_workers=4
```

---

# 18. Why Workers Exist

Without workers:

```text
Read Data
 ↓
Train
 ↓
Read Data
 ↓
Train
```

Everything happens sequentially.

---

# 19. With Workers

Conceptually:

```text
Train Current Batch

while

Workers Prepare Next Batch
```

Loading and training overlap.

---

# 20. Worker Responsibilities

Workers can perform:

```text
dataset[i]

file reading

image decoding

transforms

batch preparation
```

---

# 21. Worker Non-Responsibilities

Workers do not perform:

```text
forward()

loss computation

backward()

optimizer.step()
```

Those belong to training.

---

# 22. drop_last

Suppose:

```text
Dataset Size = 100

Batch Size = 32
```

---

# 23. drop_last=False

Produces:

```text
32

32

32

4
```

Final partial batch is kept.

---

# 24. drop_last=True

Produces:

```text
32

32

32
```

Final partial batch is discarded.

---

# 25. Why Use drop_last=True?

Sometimes useful when:

```text
Consistent Batch Shapes

BatchNorm Assumptions

Distributed Training
```

are desired.

---

# 26. Complete DataLoader Mental Model

```text
Dataset
    ↓
Individual Samples
    ↓
Workers (optional)
    ↓
Collation
    ↓
Batch Tensor
    ↓
DataLoader
    ↓
Training Loop
```

---

# 27. Relationship To Training

```python
for x, y in loader:
```

provides:

```text
Batch Input

Batch Targets
```

The training loop then takes over.

---

# 28. Responsibility Split

Dataset:

```text
What is sample i?
```

---

DataLoader:

```text
How should samples be batched?
```

---

Model:

```text
Compute Predictions
```

---

Loss Function:

```text
Compute Error
```

---

Optimizer:

```text
Update Parameters
```

---

# 29. Common Misconception

Incorrect:

```text
DataLoader performs training
```

Correct:

```text
DataLoader supplies batches
```

Training is handled elsewhere.

---

# 30. Teach It To Someone

If I were teaching DataLoader:

> DataLoader is a batching and data-delivery mechanism. It repeatedly retrieves samples from a Dataset, optionally shuffles them, optionally loads them using worker processes, collates them into batch tensors, and yields batches to the training loop.

---

# 31. Final Mental Model

```text
Dataset
   ↓
dataset[i]
   ↓
Sample
   ↓
Workers
   ↓
Collation
   ↓
Batch
   ↓
DataLoader
   ↓
for x,y in dataloader
   ↓
Training Loop
```

---

# 32. What I Understand Now

I understand:

```text
DataLoader
│
├── batch_size
├── shuffle
├── num_workers
├── drop_last
├── collation
├── batching
└── training loop integration
```
