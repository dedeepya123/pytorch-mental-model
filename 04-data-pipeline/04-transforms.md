# Transforms in PyTorch

## Goal

Raw data is rarely in the format expected by a model.

Transforms convert:

```text
Raw Sample
     ↓
Model-Ready Sample
```

The goals of this chapter are:

- Understand why transforms exist.
- Understand preprocessing.
- Understand augmentation.
- Understand deterministic vs random transforms.
- Understand training vs validation transforms.
- Understand where transforms are applied.

---

# 1. Why Transforms Exist

Suppose:

```text
Image

4000 x 3000

uint8

pixel range [0,255]
```

A model may expect:

```text
224 x 224

float32

normalized values
```

Transforms perform this conversion.

---

# 2. Mental Model

```text
Raw Data
     ↓
Transforms
     ↓
Model-Ready Data
```

Think:

```text
Transforms
=
Data Preprocessing Pipeline
```

---

# 3. Where Are Transforms Applied?

Common pattern:

```python
def __getitem__(self, idx):

    image = load_image(...)

    if self.transform:
        image = self.transform(
            image
        )

    return image, label
```

Transforms are typically applied inside:

```python
__getitem__()
```

---

# 4. Why Inside __getitem__?

Transforms modify:

```text
Individual Samples
```

and `__getitem__()` produces individual samples.

---

# 5. Categories Of Transforms

Three useful categories:

```text
Format Conversion

Preprocessing

Data Augmentation
```

---

# 6. Format Conversion

Example:

```text
Image File
    ↓
Tensor
```

or:

```text
uint8
    ↓
float32
```

---

# 7. Preprocessing

Examples:

```text
Resize

Crop

Normalize
```

Purpose:

```text
Match model expectations
```

---

# 8. Resize Example

Input:

```text
4000 x 3000
```

Output:

```text
224 x 224
```

The image is made compatible with the model.

---

# 9. Data Augmentation

Data augmentation creates different views of the same sample.

Examples:

```text
Horizontal Flip

Rotation

Random Crop

Color Jitter
```

---

# 10. Why Augmentation?

Suppose:

```text
100 images
```

Without augmentation:

```text
Model may memorize
```

the training data.

---

With augmentation:

```text
One Image
     ↓
Many Variations
```

providing greater diversity.

---

# 11. Augmentation Example

Original:

```text
Cat
```

Transform:

```text
Random Horizontal Flip
```

Possible output:

```text
Flipped Cat
```

Label remains:

```text
Cat
```

---

# 12. Deterministic Transforms

Produces the same output every time.

Example:

```text
Resize
```

Input:

```text
4000 x 3000
```

Output:

```text
224 x 224
```

always.

---

# 13. Random Transforms

Output can change between executions.

Example:

```text
Random Crop
```

Different crop regions may be selected.

---

# 14. Training Transforms

Training often uses:

```text
Resize

Normalize

Random Crop

Random Flip
```

The goal is:

```text
Improve Generalization
```

---

# 15. Validation Transforms

Validation often uses:

```text
Resize

Normalize
```

without randomness.

---

# 16. Why Avoid Randomness In Validation?

Validation aims to measure:

```text
Model Quality
```

consistently.

Random augmentation changes the evaluation input.

---

# 17. Training vs Validation

Training asks:

```text
How can the model learn better?
```

Therefore:

```text
More Diversity
```

is helpful.

---

Validation asks:

```text
How good is the current model?
```

Therefore:

```text
Consistent Evaluation
```

is preferred.

---

# 18. Practical Example

Training:

```text
Image
  ↓
Resize
  ↓
Random Flip
  ↓
Normalize
```

---

Validation:

```text
Image
  ↓
Resize
  ↓
Normalize
```

No randomness.

---

# 19. Can Validation Ever Use Augmentations?

Yes.

Advanced evaluation sometimes uses:

```text
Test-Time Augmentation (TTA)
```

where multiple transformed versions are evaluated and predictions are combined.

However, this is not the standard validation setup.

---

# 20. Relationship To Dataset

Transforms are commonly stored in:

```python
self.transform
```

during Dataset initialization.

---

# 21. Full Data Pipeline

```text
Raw File
    ↓
Load Sample
    ↓
Apply Transform
    ↓
Return Sample
    ↓
DataLoader
    ↓
Batch
    ↓
Training Loop
```

---

# 22. Responsibilities

Transforms are responsible for:

```text
Preprocessing

Normalization

Augmentation

Format Conversion
```

---

# 23. Non-Responsibilities

Transforms do not:

```text
Create Batches

Compute Loss

Compute Gradients

Update Parameters
```

---

# 24. Interview Answer

Why use data augmentation?

A good answer:

> Data augmentation creates diverse variations of training samples while keeping their labels unchanged. This increases data diversity, reduces overfitting, and improves generalization to unseen examples.

---

# 25. Mental Model

```text
Raw Sample
     ↓
Transforms
     ↓
Processed Sample
```

---

# 26. What I Understand Now

I understand:

```text
Transforms
│
├── Preprocessing
├── Augmentation
├── Resize
├── Normalize
├── Random Transformations
├── Training Transforms
└── Validation Transforms
```
