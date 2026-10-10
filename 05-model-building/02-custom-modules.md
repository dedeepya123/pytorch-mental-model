# Custom Modules in PyTorch

## Goal

PyTorch provides many built-in Modules:

```python
nn.Linear

nn.ReLU

nn.Dropout

nn.BatchNorm

nn.Sequential
```

Eventually we need to create our own reusable building blocks.

The goals of this chapter are:

- Understand why Custom Modules exist.
- Understand the responsibilities of `__init__()` and `forward()`.
- Understand Module composition.
- Understand Module trees.
- Understand parameter registration.
- Understand shape flow through a custom Module.
- Understand common mistakes.
- Understand when to use Custom Modules instead of Sequential.

---

# 1. Why Custom Modules Exist

Imagine repeatedly writing:

```python
nn.Linear(128, 256)

nn.ReLU()

nn.Dropout(0.1)
```

across many models.

Instead of writing the same structure repeatedly, we can package it into a reusable Module.

Conceptually:

```text
Several Modules
        +
Custom Logic
        ↓
Custom Module
```

---

# 2. The Most Important Idea

A Custom Module is still a Module.

PyTorch does not care whether the Module was created by:

```python
torch.nn
```

or by:

```python
You
```

Examples:

```python
nn.Linear
```

and

```python
MyBlock
```

are both Modules.

---

# 3. First Custom Module

```python
class MyBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            10,
            20
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            20,
            10
        )

    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x
```

Conceptually:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

---

# 4. Two Parts Of Every Custom Module

Every Custom Module has two major responsibilities:

```text
Build

Execute
```

---

# 5. Build Phase

Implemented in:

```python
__init__()
```

Purpose:

```text
Construct Components
```

Examples:

```python
self.fc1

self.fc2

self.relu

self.dropout
```

Mental model:

```text
Create LEGO Pieces
```

---

# 6. Execute Phase

Implemented in:

```python
forward()
```

Purpose:

```text
Connect Components
```

Example:

```python
x = self.fc1(x)

x = self.relu(x)

x = self.fc2(x)
```

Mental model:

```text
Assemble LEGO Pieces
```

---

# 7. Build vs Execute

Think:

```python
__init__()
```

↓

```text
Runs Once
```

when the model is created.

---

```python
forward()
```

↓

```text
Runs Every
Forward Pass
```

Potentially millions of times during training.

---

# 8. What Happens During model(x)?

Suppose:

```python
output = model(x)
```

PyTorch internally does:

```python
model.forward(x)
```

along with additional framework functionality.

Users typically call:

```python
model(x)
```

not:

```python
model.forward(x)
```

---

# 9. Shape Reasoning

Suppose:

```python
self.fc1 = nn.Linear(
    10,
    20
)
```

Input:

```text
(B,10)
```

Output:

```text
(B,20)
```

---

# 10. Multi-Layer Shape Trace

```python
self.fc1 = nn.Linear(10,20)

self.fc2 = nn.Linear(20,5)
```

Input:

```text
(32,10)
```

---

After fc1:

```text
(32,20)
```

---

After fc2:

```text
(32,5)
