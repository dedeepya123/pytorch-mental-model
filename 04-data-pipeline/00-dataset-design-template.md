# Dataset Design Framework

## Goal

Many developers understand:

```python
__len__()

__getitem__()
```

but struggle when faced with a new dataset layout.

The goal of this document is to develop a reusable framework for designing
any PyTorch Dataset from scratch.

After reading this document, I should be able to inspect a new dataset and
systematically determine:

```text
What belongs in __init__()?

What belongs in __len__()?

What belongs in __getitem__()?
```

and how the Dataset integrates with:

```text
Transforms

DataLoader

Training Loop
```

---

# 1. Start With The Data

Before writing code, inspect the dataset.

Ask:

```text
Where does the data live?
```

Examples:

```text
Images

CSV

Parquet

JSON

Audio

Database

Text Files
```

Never start by writing code.

Always start by understanding the data layout.

---

# 2. Define One Sample

Ask:

> What is a single training sample?

Examples:

Image Classification:

```python
(image, label)
```

Sentiment Analysis:

```python
(text, sentiment)
```

Regression:

```python
(features, target)
```

Object Detection:

```python
(image, boxes)
```

---

# 3. Define dataset[idx]

Ask:

> What should this return?

```python
dataset[idx]
```

Examples:

```python
(image, label)
```

or

```python
(text, label)
```

Everything else follows from this answer.

---

# 4. Identify Required Information

Ask:

> What information is required to construct sample idx?

Examples:

```text
File Path

Label

Annotation

Metadata

Transform
```

Usually this information is stored during Dataset initialization.

---

# 5. Cheap Work vs Expensive Work

This is one of the most important Dataset design rules.

---

## Cheap Work

Store in:

```python
__init__()
```

Examples:

```text
Paths

Labels

Metadata

CSV DataFrame

Manifest Files
```

---

## Expensive Work

Perform in:

```python
__getitem__()
```

Examples:

```text
Image Loading

Audio Loading

Decoding

Tokenization

Transforms
```

---

# 6. Why?

Imagine:

```text
5 million images
```

Loading every image during:

```python
__init__()
```

may consume enormous memory.

Instead:

```text
Store paths once

Load requested sample later
```

This pattern is called:

```text
Lazy Loading
```

---

# 7. The Three Dataset Methods

A typical Dataset looks like:

```python
class MyDataset(Dataset):

    def __init__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...
```

Each method has a specific responsibility.

---

# 8. __init__()

Purpose:

```text
One-Time Setup
```

Typical tasks:

```text
Read metadata

Store file paths

Store labels

Store transforms
```

Mental model:

```text
Everything needed
to later construct a sample
```

---

# 9. __len__()

Purpose:

```text
How many samples exist?
```

Template:

```python
def __len__(self):

    return len(self.samples)
```

---

# 10. __getitem__()

Purpose:

```text
Construct sample idx
```

Template:

```python
def __getitem__(self, idx):

    sample = self.samples[idx]

    x = ...

    y = ...

    return x, y
```

---

# 11. Think Backwards

When designing a Dataset, start from:

```python
dataset[idx]
```

Ask:

```text
What should be returned?
```

Then work backwards.

Example:

```python
(image, label)
```

Needed:

```text
Path

Label

Transform
```

Store that information in:

```python
__init__()
```

---

# 12. Universal Dataset Skeleton

```python
class MyDataset(Dataset):

    def __init__(self):

        self.samples = ...

        self.transform = ...

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        x = ...

        y = ...

        if self.transform:
            x = self.transform(x)

        return x, y
```

---

# 13. Example: Image Classification

Suppose:

```text
cats/

dogs/
```

---

Store:

```python
[
    ("cat1.jpg", 0),
    ("cat2.jpg", 0),
    ("dog1.jpg", 1)
]
```

inside:

```python
self.samples
```

---

Retrieve:

```python
path, label = self.samples[idx]
```

Load image:

```python
image = read_image(path)
```

Return:

```python
(image, label)
```

---

# 14. Example: CSV Dataset

CSV:

```text
age,salary,label
23,50000,0
35,80000,1
```

---

Store DataFrame:

```python
self.df
```

---

Retrieve row:

```python
row = self.df.iloc[idx]
```

Return:

```python
features, label
```

---

# 15. Most Common Beginner Mistake

Wrong:

```text
Dataset returns a batch
```

Correct:

```text
Dataset returns one sample
```

---

# 16. Who Creates Batches?

Not Dataset.

DataLoader creates batches.

Mental model:

```text
Dataset
    ↓
One Sample
```

---

```text
DataLoader
    ↓
One Batch
```

---

# 17. Dataset and DataLoader Relationship

Suppose:

```python
batch_size = 32
```

DataLoader may execute:

```python
dataset[0]
dataset[1]
dataset[2]
...
dataset[31]
```

Now it has:

```text
32 Samples
```

---

# 18. Collation

The samples are combined into:

```text
One Batch Tensor
```

through:

```text
Collation
```

---

Conceptually:

```text
Sample
Sample
Sample
Sample
      ↓
Collation
      ↓
Batch
```

---

# 19. Complete Data Pipeline

A custom Dataset participates in:

```text
Raw Files
    ↓
Dataset
    ↓
dataset[idx]
    ↓
Sample
    ↓
Transforms
    ↓
Workers
    ↓
Collation
    ↓
Batch
    ↓
DataLoader
    ↓
Training Loop
```

---

# 20. Connection To Training Loop

Training loop:

```python
for x, y in dataloader:
```

Data does not appear magically.

Behind the scenes:

```text
DataLoader
      ↓
dataset[idx]
      ↓
__getitem__()
      ↓
Transforms
      ↓
Collation
      ↓
Batch
      ↓
Training Loop
```

---

# 21. The Five Questions Framework

Before designing any Dataset:

```text
1. What is one sample?

2. What should dataset[idx] return?

3. What metadata is needed?

4. What can be stored once?

5. What must be loaded per sample?
```

If these five questions are answered, a Dataset implementation usually becomes straightforward.

---

# 22. Teach It To Someone

If I were teaching Datasets:

> A Dataset is an indexable object that knows how many samples exist and how to construct sample i. It typically stores metadata during initialization and loads actual sample content on demand inside __getitem__. DataLoader repeatedly calls __getitem__, applies batching through collation, and feeds batches into the training loop.

---

# 23. Final Mental Model

```text
Data Layout
      ↓
Define Sample
      ↓
dataset[idx]
      ↓
Store Metadata (__init__)
      ↓
Load Sample (__getitem__)
      ↓
Transforms
      ↓
DataLoader
      ↓
Batch
      ↓
Training Loop
```
