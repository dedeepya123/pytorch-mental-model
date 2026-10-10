# Multi-Layer Perceptrons (MLPs)

## Goal

The Multi-Layer Perceptron (MLP) is the first true neural network architecture
that most practitioners learn.

An MLP combines:

```text
Linear Layers
+
Non-Linear Activations
```

to learn complex relationships in data.

The goals of this chapter are:

- Understand why MLPs were invented.
- Understand why a single linear layer is insufficient.
- Understand why stacking only linear layers does not help.
- Understand the role of activations.
- Understand hidden layers.
- Understand representation learning.
- Understand width and depth.
- Understand shape propagation through MLPs.
- Build MLPs using Sequential and Custom Modules.

---

# 1. The Problem With Linear Models

Suppose we have:

```text
Age

Salary

Experience
```

and want to predict:

```text
Will Customer Buy Product?
```

A simple model could be:

```python
nn.Linear(3, 1)
```

Conceptually:

```text
Input
  ↓
Linear
  ↓
Output
```

This is essentially:

```text
Linear Regression
```

or

```text
Logistic Regression
```

depending on the task.

---

# 2. When Linear Models Work

Linear models work when the relationship in the data is approximately linear.

Example:

```text
x > 5 → Class A

x < 5 → Class B
```

A single linear boundary may separate the classes.

---

# 3. When Linear Models Fail

Imagine:

```text
Inside Circle → Class A

Outside Circle → Class B
```

Visual:

```text
      B B B B

    B A A A B

    B A A A B

      B B B B
```

A single straight line cannot separate:

```text
A

and

B
```

The model lacks expressive power.

---

# 4. The First Idea: More Linear Layers

Suppose we stack layers:

```text
Linear
 ↓
Linear
 ↓
Linear
```

Will this solve the problem?

Surprisingly:

```text
No
```

---

# 5. Why Stacking Linear Layers Doesn't Help

Suppose:

```text
y = W₁x
```

and:

```text
z = W₂y
```

Substituting:

```text
z = W₂(W₁x)
```

which becomes:

```text
z = Wx
```

for some new matrix:

```text
W
```

Therefore:

```text
Linear
 ↓
Linear
 ↓
Linear
```

is mathematically equivalent to:

```text
One Larger Linear Layer
```

---

# 6. The Big Limitation

Without non-linearities:

```text
Depth Adds No New Expressive Power
```

This is one of the most important ideas in deep learning.

---

# 7. Enter Non-Linear Activations

Now consider:

```text
Linear
 ↓
ReLU
 ↓
Linear
```

This changes everything.

Why?

Because:

```text
ReLU
```

is:

```text
Non-Linear
```

---

# 8. Why Nonlinearity Matters

With activations:

```text
Linear
 ↓
Nonlinear
 ↓
Linear
```

cannot be reduced to:

```text
One Linear Transformation
```

The network gains the ability to learn complex relationships.

---

# 9. Birth Of The MLP

A Multi-Layer Perceptron is simply:

```text
Input
 ↓
Linear
 ↓
Activation
 ↓
Linear
 ↓
Activation
 ↓
Linear
 ↓
Output
```

The repeated pattern:

```text
Linear
 +
Activation
```

is the core building block.

---

# 10. Anatomy Of An MLP

An MLP typically consists of:

```text
Input Layer

Hidden Layers

Output Layer
```

---

# 11. Input Layer

The input layer is determined by the number of input features.

Example:

```text
Age

Salary

Experience

Credit Score
```

Input dimension:

```text
4
```

First layer:

```python
nn.Linear(4, 64)
```

---

# 12. Hidden Layers

Hidden layers learn intermediate representations.

Example:

```text
64
 ↓
32
```

These layers transform raw features into more useful features.

---

# 13. What Hidden Layers Actually Learn

Suppose inputs are:

```text
Age

Salary

Experience
```

Hidden layers may internally learn concepts such as:

```text
Buying Power

Career Stability

Risk Profile
```

These features are not explicitly provided.

The network discovers them automatically.

---

# 14. Representation Learning

A powerful mental model:

```text
Raw Features
      ↓
Hidden Features
      ↓
Prediction
```

An MLP is fundamentally a representation-learning machine.

---

# 15. Output Layer

The output layer depends on the task.

---

## Regression

Predict:

```text
House Price
```

Output:

```python
nn.Linear(32, 1)
```

Shape:

```text
(B,1)
```

---

## Binary Classification

Predict:

```text
Spam

Not Spam
```

Output:

```python
nn.Linear(32,1)
```

Shape:

```text
(B,1)
```

---

## Multi-Class Classification

Predict:

```text
10 Classes
```

Output:

```python
nn.Linear(32,10)
```

Shape:

```text
(B,10)
```

---

# 16. Example MLP

```python
model = nn.Sequential(

    nn.Linear(10,64),

    nn.ReLU(),

    nn.Linear(64,32),

    nn.ReLU(),

    nn.Linear(32,3)
)
```

---

# 17. Shape Flow

Input:

```text
(32,10)
```

---

Layer 1:

```text
(32,64)
```

---

ReLU:

```text
(32,64)
```

---

Layer 2:

```text
(32,32)
```

---

ReLU:

```text
(32,32)
```

---

Output:

```text
(32,3)
```

---

# 18. Complete Shape Trace

```text
(32,10)

      ↓

Linear(10→64)

      ↓

(32,64)

      ↓

ReLU

      ↓

(32,64)

      ↓

Linear(64→32)

      ↓

(32,32)

      ↓

ReLU

      ↓

(32,32)

      ↓

Linear(32→3)

      ↓

(32,3)
```

---

# 19. Width

Width refers to:

```text
Number Of Neurons
In A Layer
```

Example:

```python
nn.Linear(10,256)
```

Width:

```text
256
```

---

# 20. Depth

Depth refers to:

```text
Number Of Layers
```

Example:

```text
Linear
 ↓
Linear
 ↓
Linear
```

Depth:

```text
3 Layers
```

---

# 21. Width vs Depth

Increasing width:

```text
More Neurons Per Layer
```

Increasing depth:

```text
More Transformations
```

Both increase model capacity.

---

# 22. Common Beginner Mistake

Thinking:

```text
More Layers
=
Better Model
```

Not necessarily.

Deeper models may introduce:

```text
More Memory Usage

More Computation

Longer Training

More Overfitting
```

---

# 23. MLP Mental Model

Think:

```text
Feature Extraction
      +
Prediction
```

The hidden layers discover useful representations.

The output layer makes the final prediction.

---

# 24. Sequential Implementation

Most simple MLPs can be built with:

```python
nn.Sequential
```

Example:

```python
model = nn.Sequential(

    nn.Linear(10,64),

    nn.ReLU(),

    nn.Linear(64,32),

    nn.ReLU(),

    nn.Linear(32,1)
)
```

---

# 25. Custom Module Implementation

```python
class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(10,64)

        self.fc2 = nn.Linear(64,32)

        self.fc3 = nn.Linear(32,1)

    def forward(self, x):

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = self.fc3(x)

        return x
```

---

# 26. Why Custom Modules?

Custom Modules provide:

```text
Flexibility

Custom Logic

Reusability

Larger Architectures
```

while preserving the same MLP structure.

---

# 27. Common Interview Question

Why doesn't this help?

```text
Linear
 ↓
Linear
 ↓
Linear
```

Strong answer:

> Multiple linear transformations collapse into a single linear transformation, so stacking linear layers without activations does not increase expressive power.

---

# 28. Common Interview Question

Why are activations important?

Strong answer:

> Activations introduce non-linearity, allowing the network to learn relationships that cannot be represented by a single linear transformation.

---

# 29. Common Interview Question

What do hidden layers learn?

Strong answer:

> Hidden layers learn internal feature representations that make prediction easier. These learned features are not explicitly provided in the data.

---

# 30. Teach It To Someone

If I were teaching MLPs:

> An MLP repeatedly applies linear transformations and nonlinear activations. The hidden layers learn increasingly useful representations of the input, and the final layer uses those representations to make predictions.

---

# 31. Master Mental Model

```text
Raw Features
      ↓
Linear
      ↓
Activation
      ↓
Linear
      ↓
Activation
      ↓
Linear
      ↓
Prediction
```

---

# 32. Another Mental Model

```text
Input
     ↓

Representation Learning

     ↓

Prediction
```

The hidden layers perform the representation learning.

---

# 33. Relationship To Previous Chapters

```text
Dataset
     ↓

DataLoader
     ↓

Batch
     ↓

MLP
     ↓

Loss
     ↓

Backward
     ↓

Optimizer
```

This is the first complete end-to-end deep learning pipeline.

---

# 34. What I Understand Now

I understand:

```text
MLPs
│
├── Linear Layers
├── Activations
├── Hidden Layers
├── Representation Learning
├── Width
├── Depth
├── Shape Propagation
├── Sequential Implementation
├── Custom Module Implementation
└── Architecture Design
```

---

# 35. What Comes Next

MLPs work well for:

```text
Tabular Data

Structured Features
```

However, they scale poorly to images.

The next chapter introduces:

```text
Convolutional Neural Networks (CNNs)
```

and answers:

```text
Why MLPs struggle with images

Why convolution was invented

How CNNs learn spatial features
```
