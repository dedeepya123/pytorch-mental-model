# Dataset in PyTorch

## Goal

A model cannot train without data.

PyTorch represents data using:

```text
Dataset
```

The goal of this chapter is to understand:

- What a Dataset is.
- Why Datasets exist.
- `__len__()`.
- `__getitem__()`.
- Single samples vs batches.
- How Dataset interacts with DataLoader.
- Why Dataset does not create batches.
- The mental model behind custom datasets.

---

# 1. What Is A Dataset?

A Dataset represents a collection of training samples.

Conceptually:

```text
Dataset
│
├── Sample 0
├── Sample 1
├── Sample 2
├── Sample 3
└── ...
```

Examples:

```text
Images + Labels

Text + Labels

Audio + Labels

Features + Targets
```

---

# 2. Dataset Mental Model

Think of a Dataset like a Python container.

Example:

```python
animals = [
    "cat",
    "dog",
    "bird"
]
```

We can ask:

```python
len(animals)
```

and:

```python
animals[1]
```

Datasets follow the same idea.

---

# 3. Two Fundamental Questions

A Dataset must answer:

```text
1. How many samples exist?

2. Give me sample i.
```

Everything else builds on these two ideas.

---

# 4. __len__()

Used to answer:

```text
How many samples exist?
```

Example:

```python
len(dataset)
```

Possible result:

```text
1000
```

Meaning:

```text
Dataset contains 1000 samples
```

---

# 5. __getitem__()

Used to answer:

```text
Give me sample i
```

Example:

```python
dataset[73]
```

Possible result:

```python
(x73, y73)
```

A single sample.

---

# 6. Minimal Custom Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __len__(self):
        return ...

    def __getitem__(self, idx):
        return ...
```

These two methods are the foundation of the Dataset abstraction.

---

# 7. Example Dataset

Suppose:

```python
features = [
    [1,2],
    [3,4],
    [5,6]
]

labels = [
    0,
    1,
    0
]
```

Then:

```python
dataset[0]
```

returns:

```python
([1,2], 0)
```

---

```python
dataset[1]
```

returns:

```python
([3,4], 1)
```

---

# 8. Dataset Returns A Single Sample

This is extremely important.

Dataset returns:

```text
One Sample
```

not:

```text
One Batch
```

Example:

```python
dataset[20]
```

returns:

```python
(x20, y20)
```

only.

---

# 9. Sample Structure

A sample commonly contains:

```python
(x, y)
```

where:

```text
x = input

y = target
```

Examples:

```text
Image + Class

Audio + Transcript

Features + Label
```

---

# 10. Dataset Responsibility

Dataset is responsible for:

```text
Data Storage

Data Retrieval

Sample Construction
```

It answers:

```text
What is sample i?
```

---

# 11. Dataset Is NOT Responsible For

Dataset does not:

```text
Create Batches

Run Models

Compute Loss

Compute Gradients

Update Parameters
```

Those belong elsewhere.

---

# 12. Dataset vs Model

Dataset:

```text
Represents Data
```

Model:

```text
Represents Computation
```

Different responsibilities.

---

# 13. Dataset vs Loss Function

Dataset:

```text
Provides Training Samples
```

Loss Function:

```text
Measures Prediction Error
```

Completely separate concepts.

---

# 14. Dataset vs Optimizer

Dataset:

```text
Provides Data
```

Optimizer:

```text
Updates Parameters
```

Completely different responsibilities.

---

# 15. Dataset and Shapes

Suppose one image is flattened:

```text
x.shape = (784,)
```

Label:

```text
y = 3
```

Then:

```python
dataset[i]
```

returns:

```text
(784,)
```

not:

```text
(B,784)
```

because datasets return one sample.

---

# 16. Relationship To DataLoader

Dataset provides:

```text
Individual Samples
```

DataLoader provides:

```text
Batches
```

Mental model:

```text
Dataset
    ↓
Single Sample

DataLoader
    ↓
Batch
```

---

# 17. Common Misconception

Incorrect:

```text
Dataset creates batches
```

Correct:

```text
Dataset creates samples

DataLoader creates batches
```

---

# 18. Teach It To Someone

If I were teaching Dataset:

> A Dataset is a collection of training samples. It must tell PyTorch how many samples exist and how to retrieve sample i. A Dataset returns individual samples, while batching is handled later by DataLoader.

---

# 19. Final Mental Model

```text
Dataset
│
├── __len__()
│
├── __getitem__()
│
├── Returns One Sample
│
└── Represents Data
```

---

# 20. What Comes Next

```text
Dataset
    ↓
Single Samples
    ↓
DataLoader
    ↓
Batches
```

The next chapter explains how DataLoader transforms individual samples into batch tensors.
