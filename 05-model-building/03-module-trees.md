# Module Trees in PyTorch

## Goal

Many PyTorch features appear automatic:

```python
model.parameters()

model.state_dict()

model.train()

model.eval()
```

Understanding why these work requires understanding one of the most important concepts in PyTorch:

```text
Module Trees
```

The goals of this chapter are:

- Understand what a Module Tree is.
- Understand module registration.
- Understand recursive traversal.
- Understand parameter discovery.
- Understand state_dict collection.
- Understand train/eval propagation.
- Understand how large models are organized.

---

# 1. Why Module Trees Matter

Consider:

```python
class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            10,
            20
        )

        self.fc2 = nn.Linear(
            20,
            5
        )
```

Question:

```text
How does PyTorch know
fc1 and fc2 belong to Model?
```

The answer is:

```text
Module Registration
```

which creates a:

```text
Module Tree
```

---

# 2. The Parent-Child Relationship

When we write:

```python
self.fc1 = nn.Linear(...)
```

PyTorch automatically registers:

```text
fc1
```

as a child Module.

Conceptually:

```text
Model
 │
 └── fc1
```

---

Adding another layer:

```python
self.fc2 = nn.Linear(...)
```

creates:

```text
Model
 │
 ├── fc1
 │
 └── fc2
```

---

# 3. Modules Can Contain Modules

Suppose:

```python
class Block(nn.Module):
```

contains:

```python
self.fc1

self.relu

self.fc2
```

---

And:

```python
class Model(nn.Module):
```

contains:

```python
self.block1

self.block2
```

The structure becomes:

```text
Model
│
├── Block1
│   ├── fc1
│   ├── relu
│   └── fc2
│
└── Block2
    ├── fc1
    ├── relu
    └── fc2
```

This hierarchy is called a:

```text
Module Tree
```

---

# 4. Mental Model

Think:

```text
Models
    ↓
Contain Modules

Modules
    ↓
Contain Modules

Leaf Modules
    ↓
Contain Parameters
```

---

# 5. Parameters Live Inside Modules

Example:

```python
nn.Linear(
    10,
    20
)
```

contains:

```text
weight

bias
```

Conceptually:

```text
Linear
│
├── weight
└── bias
```

---

# 6. Models Are Trees, Not Lists

Incorrect mental model:

```text
Model
 ↓
Parameters
```

---

Correct mental model:

```text
Model
  ↓
Modules
  ↓
Submodules
  ↓
Parameters
```

---

# 7. Recursive Traversal

PyTorch frequently walks the Module Tree.

Conceptually:

```text
Visit Parent

Visit Child

Visit Grandchild

Visit Great-Grandchild
```

until the entire tree has been explored.

This process is called:

```text
Recursive Traversal
```

---

# 8. Why model.parameters() Works

Suppose:

```python
model.parameters()
```

PyTorch recursively traverses:

```text
Model
 ↓
Child Modules
 ↓
Grandchild Modules
 ↓
Parameters
```

collecting every trainable Parameter.

---

# 9. Example Traversal

```text
Model
│
├── Block1
│   ├── fc1
│   └── fc2
│
└── Block2
```

Traversal:

```text
Model
 ↓
Block1
 ↓
fc1
 ↓
weight
 ↓
bias
```

Collect.

Then continue.

---

# 10. Why Optimizers Work

Example:

```python
optimizer = Adam(
    model.parameters()
)
```

The optimizer receives all Parameters discovered through recursive tree traversal.

This is why deeply nested modules still train correctly.

---

# 11. Why state_dict() Works

Example:

```python
model.state_dict()
```

PyTorch again traverses the entire Module Tree.

---

Collected items include:

```text
Parameters

Buffers
```

from every registered Module.

---

# 12. Example state_dict

Conceptually:

```python
{
    "block1.fc1.weight": ...,
    "block1.fc1.bias": ...,
    "block2.fc2.weight": ...
}
```

Notice how names encode the tree structure.

---

# 13. Why train() Works

Suppose:

```python
model.train()
```

We only call it once.

Yet all submodules switch to training mode.

Why?

Because PyTorch recursively traverses the Module Tree.

---

# 14. Propagation Example

Conceptually:

```text
Model
 │
 ├── Block1
 │    ├── Linear
 │    └── Dropout
 │
 └── Block2
```

Calling:

```python
model.train()
```

propagates:

```text
training = True
```

through every node.

---

# 15. Why eval() Works

Exactly the same mechanism.

```python
model.eval()
```

recursively traverses the tree and sets:

```text
training = False
```

for all registered Modules.

---

# 16. The Entire Tree Changes

Important realization:

```python
model.eval()
```

does not affect only:

```text
Top-Level Model
```

It affects:

```text
Entire Module Tree
```

---

# 17. Registration Is Automatic

When we write:

```python
self.fc1 = nn.Linear(...)
```

PyTorch immediately registers:

```text
fc1
```

as a child Module.

---

# 18. Why Registration Matters

Registration enables:

```python
parameters()

state_dict()

train()

eval()
```

to work automatically.

Without registration, PyTorch cannot discover the Module.

---

# 19. Common Beginner Mistake

Incorrect:

```python
fc1 = nn.Linear(...)
```

inside:

```python
__init__()
```

without:

```python
self.fc1
```

---

This creates:

```text
A Local Variable
```

not a registered child Module.

---

# 20. Consequence

PyTorch may fail to discover:

```text
Parameters

Buffers

Submodules
```

inside that object.

---

# 21. Registration Rule

Correct:

```python
self.fc1 = nn.Linear(...)
```

✅ Registered

---

Incorrect:

```python
fc1 = nn.Linear(...)
```

❌ Not registered

---

# 22. How Large Architectures Work

Modern architectures are gigantic Module Trees.

Example:

```text
Transformer
│
├── Embeddings
│
├── Layer1
│   ├── Attention
│   └── MLP
│
├── Layer2
│   ├── Attention
│   └── MLP
│
├── Layer3
│   ├── Attention
│   └── MLP
│
└── Output Head
```

Still just a tree.

---

# 23. CNN Mental Model

A convolutional network is also a Module Tree.

```text
CNN
│
├── Conv Block
│   ├── Conv
│   ├── BatchNorm
│   └── ReLU
│
├── Conv Block
│
└── Classifier
```

---

# 24. Transformer Mental Model

A Transformer is a Module Tree.

```text
Transformer
│
├── Embeddings
├── Encoder Layers
├── Decoder Layers
└── Output Head
```

Nothing fundamentally different.

---

# 25. One Of The Deepest PyTorch Ideas

Everything eventually becomes:

```text
A Module Tree
```

No matter how complex the architecture appears.

---

# 26. Common Interview Question

Why does:

```python
model.parameters()
```

find Parameters from deeply nested modules?

Strong answer:

> Because PyTorch recursively traverses the Module Tree and collects Parameters from all registered child Modules.

---

# 27. Common Interview Question

Why is:

```python
self.fc1 = nn.Linear(...)
```

different from:

```python
fc1 = nn.Linear(...)
```

Strong answer:

> Assigning a Module to `self` registers it as a child Module. Local variables are not registered and therefore are not discovered during recursive traversal.

---

# 28. Common Interview Question

Why does:

```python
model.eval()
```

affect every submodule?

Strong answer:

> PyTorch recursively traverses the entire Module Tree and propagates evaluation mode to all registered child Modules.

---

# 29. Teach It To Someone

If I were teaching Module Trees:

> A PyTorch model is not a flat collection of layers. It is a hierarchy of Modules where parent Modules contain child Modules and leaf Modules contain Parameters. PyTorch recursively traverses this hierarchy to discover Parameters, collect state, and propagate train/eval mode.

---

# 30. Master Mental Model

```text
Model
│
├── Module
│   ├── Module
│   │   └── Parameters
│   │
│   └── Module
│       └── Parameters
│
└── Module
    └── Parameters
```

Functions such as:

```python
parameters()

state_dict()

train()

eval()
```

all operate by recursively traversing this tree.

---

# 31. What I Understand Now

I understand:

```text
Module Trees
│
├── Registration
├── Parent-Child Relationships
├── Recursive Traversal
├── Parameter Discovery
├── state_dict Collection
├── train() Propagation
├── eval() Propagation
└── Large Model Organization
```

---

# 32. What Comes Next

The next chapter introduces:

```text
Multi-Layer Perceptrons (MLPs)
```

which are the first complete neural network architectures built using:

```text
Linear Layers

Activations

Custom Modules

Module Trees
```

and shape reasoning.
