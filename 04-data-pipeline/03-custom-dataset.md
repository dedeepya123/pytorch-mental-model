# Custom Datasets in PyTorch

## Goal

Real-world data rarely comes pre-packaged in a format that PyTorch can use
directly.

Custom Datasets provide a bridge between:

```text
Raw Data
    ↓
PyTorch Samples
```

The goals of this chapter are:

- Understand why custom Datasets exist.
- Understand `__init__()`.
- Understand `__len__()`.
- Understand `__getitem__()`.
- Understand lazy loading.
- Understand how DataLoader interacts with a Dataset.
- Understand common Dataset design patterns.

---

# 1. Why Custom Datasets Exist

Suppose data lives in:

```text
Images

CSV Files

Audio Files

Databases
```

A model cannot consume these directly.

We need a component that converts:

```text
Raw Files
    ↓
(x, y)
```

samples.

That component is:

```text
Dataset
```

---

# 2. Dataset As A Translator

Think:

```text
Files
    ↓
Dataset
    ↓
PyTorch Samples
```

The Dataset translates storage formats into training samples.

---

# 3. Typical Structure

```python
class MyDataset(Dataset):

    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

---

# 4. __init__()

Purpose:

```text
One-Time Setup
```

Common tasks:

```text
Read CSV

Store Paths

Store Labels

Store Metadata

Store Transforms
```

---

# 5. Example __init__()

```python
def __init__(self):

    self.paths = [
        "a.jpg",
        "b.jpg",
        "c.jpg"
    ]

    self.labels = [
        0,
        1,
        0
    ]
```

The Dataset is now aware of the available samples.

---

# 6. __len__()

Purpose:

```text
How many samples exist?
```

Example:

```python
def __len__(self):
    return len(self.paths)
```

Now:

```python
len(dataset)
```

returns:

```text
3
```

---

# 7. __getitem__()

Purpose:

```text
Give me sample i
```

Example:

```python
def __getitem__(self, idx):

    path = self.paths[idx]

    label = self.labels[idx]

    return path, label
```

---

# 8. Example Retrieval

```python
dataset[1]
```

returns:

```python
("b.jpg", 1)
```

One sample.

---

# 9. Most Important Dataset Idea

A Dataset returns:

```text
One Sample
```

not:

```text
One Batch
```

---

# 10. DataLoader Interaction

When DataLoader needs data:

```python
dataset[i]
```

is called internally.

Conceptually:

```text
DataLoader
      ↓
dataset[i]
      ↓
__getitem__()
      ↓
Single Sample
```

---

# 11. __getitem__ Is Called Repeatedly

`__init__()`:

```text
Usually called once
```

---

`__getitem__()`:

```text
Called repeatedly
during training
```

Potentially millions of times.

---

# 12. Lazy Loading

A critical Dataset concept:

```text
Load data only when needed
```

This is called:

```text
Lazy Loading
```

---

# 13. Small Dataset Approach

```python
self.images = load_all_images()
```

can be acceptable when data is tiny.

---

# 14. Large Dataset Problem

Imagine:

```text
Millions of Images
```

Loading everything in memory may be unrealistic.

---

# 15. Common Pattern

Store:

```text
Paths

Labels

Metadata
```

in `__init__()`.

Load actual files inside:

```python
__getitem__()
```

---

# 16. Example Lazy Loading

```python
def __getitem__(self, idx):

    image_path = self.paths[idx]

    image = read_image(
        image_path
    )

    label = self.labels[idx]

    return image, label
```

Only the requested sample is loaded.

---

# 17. Shape Example

Single sample:

```text
image.shape

=

(3,224,224)
```

---

DataLoader later creates:

```text
(32,3,224,224)
```

for a batch.

Dataset still returns one sample.

---

# 18. Responsibilities

Dataset should:

```text
Know Data

Retrieve Data

Construct Samples
```

---

# 19. Non-Responsibilities

Dataset should not:

```text
Create Batches

Compute Loss

Compute Gradients

Update Parameters
```

---

# 20. Mental Model

```text
Dataset
│
├── __init__()
│
├── __len__()
│
├── __getitem__()
│
└── Returns One Sample
```

---

# 21. Interview Answer

What is the difference between `__init__()` and `__getitem__()`?

A good answer:

> `__init__()` performs one-time Dataset setup such as reading metadata,
> storing paths, labels, and transforms.
>
> `__getitem__()` is called repeatedly by the DataLoader and typically loads,
> processes, and returns a single training sample.

---

# 22. What I Understand Now

I understand:

```text
Custom Dataset
│
├── Setup
├── Metadata
├── Paths
├── Labels
├── __len__()
├── __getitem__()
└── Lazy Loading
```
