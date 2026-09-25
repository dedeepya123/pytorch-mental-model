# Tensor Fundamentals

## My Initial Mental Model

A tensor is an n-dimensional array, similar to a NumPy array.

A tensor can have any number of dimensions:

- 0-D tensor → scalar
- 1-D tensor → vector
- 2-D tensor → matrix
- N-D tensor → higher-dimensional tensor

An important correction to my initial understanding is that scalar, vector,
and matrix are not separate from tensors. They are all tensors with different
numbers of dimensions.

---

## Why PyTorch Tensor Instead of Just NumPy?

NumPy arrays and PyTorch tensors both provide multidimensional numerical
arrays and mathematical operations.

One important capability PyTorch adds for neural-network training is
automatic differentiation.

PyTorch can track operations performed on tensors when autograd tracking is
enabled. This allows PyTorch to calculate gradients automatically, which is
needed when training neural networks.

PyTorch tensors also support other capabilities useful for deep learning,
such as executing computations on different devices such as CPUs and GPUs.

I will explore autograd and devices separately later.

---

# Dimensions

The number of dimensions, or axes, of a tensor is represented by `ndim`.

Examples:

```python
scalar = torch.tensor(7)
vector = torch.tensor([1, 2, 3])

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

scalar → ndim = 0
vector → ndim = 1
matrix → ndim = 2
A dimension is an axis along which the tensor has some size.
```

# Shape
The shape tells me the size of the tensor along each dimension.
Examples:

```python

x = torch.zeros(2, 3, 4)
ndim  = 3
shape = (2, 3, 4)
This means:
dimension 0 → size 2
dimension 1 → size 3
dimension 2 → size 4
```
``` text
shape = (2, 3, 4)
         ↑  ↑  ↑
         │  │  │
         │  │  └── size along dimension 2
         │  │
         │  └───── size along dimension 1
         │
         └──────── size along dimension 0
```
For Visualization I can think of this as
```text
2 groups
    │
    └── each contains a (3, 4) tensor
            │
            └── 3 groups of 4 elements
```
Each number in shape represents the tensor's size along that dimension.

# Dimensions vs Semantic Meaning

- A dimension does not inherently mean "row", "column", "channel", "batch", or anything else.
- Its meaning comes from what the tensor represents.
``` python
  For example, a tensor may have:
    shape = (32, 3, 224, 224)
    and in an image-processing context the dimensions might represent:
    dimension 0 → batch
    dimension 1 → channels
    dimension 2 → height
    dimension 3 → width
- PyTorch fundamentally sees the dimensions and their sizes. The semantic meaning comes from how the tensor is being used.
```text
                 semantic meaning
                       ↓

dimension 0 ───────── batch
dimension 1 ───────── channels
dimension 2 ───────── height
dimension 3 ───────── width

                       ↓

              (32, 3, 224, 224)
The shape tells PyTorch the sizes.
The context tells us what those dimensions mean.
```
# Scalar vs One-Element Vector
``` python
a = torch.tensor(7)
b = torch.tensor([7])
These both contain one value, but they do not have the same shape
For:
  a = torch.tensor(7)
  ndim = 0
  shape = ()
  numel = 1
  This is a 0-dimensional tensor.
For:
  b = torch.tensor([7])
  we have:
  ndim = 1
  shape = (1,)
  numel = 1
  This is a 1-dimensional tensor whose size along dimension 0 is 1.
```
Therefore:
``` text
torch.tensor(7)
        ↓
      scalar
      ndim = 0
      shape = ()
torch.tensor([7])
        ↓
  one-element vector
      ndim = 1
      shape = (1,)
```
# Number of Elements: numel
The total number of elements in a tensor can be obtained using:
tensor.numel()
``` python
Consider:
x = torch.zeros(2, 3, 4)
2 × 3 × 4 = 24
x.numel() returns 24
```
``` text
ndim
│
└── number of dimensions / axes

shape
└── size along each dimension

numel
└── total number of elements
For:
x = torch.zeros(2, 3, 4)
this means:
ndim = 3
shape = (2, 3, 4)
numel = 24
```
# Hands-On Experiment: Dimensions and Shapes
``` python
import torch

scalar = torch.tensor(7)

vector = torch.tensor([1, 2, 3])

matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

tensor_3d = torch.tensor([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

for name, tensor in [
    ("scalar", scalar),
    ("vector", vector),
    ("matrix", matrix),
    ("tensor_3d", tensor_3d),
]:
    print(f"\n{name}")
    print(tensor)
    print("shape:", tensor.shape)
    print("ndim :", tensor.ndim)
    print("numel:", tensor.numel())

Expected Mental model is
scalar
    ndim  = 0
    shape = ()
    numel = 1

vector
    ndim  = 1
    shape = (3,)
    numel = 3

matrix
    ndim  = 2
    shape = (2, 3)
    numel = 6

tensor_3d
    ndim  = 3
    shape = (2, 2, 2)
    numel = 8
```
# Basic Tensor Indexing
``` python
Consider:
x = torch.zeros(2, 3, 4)
Its sahpe : (2, 3, 4)

Now consider:
x[0] --> I am selecting a specific index along dimension 0.

Before indexing: 
dim 0    dim 1    dim 2
  2        3        4
I select:
dim 0
  ↓
  0
The resulting tensor has: shape = (3, 4)

x.shape → (2, 3, 4)
x[0].shape → (3, 4)
```
# Integer Indexing Removes a Dimension
``` python
Consider:
x = torch.zeros(2, 3, 4)
Then:
x[0] - selects one specific position along dimension 0.

That dimension disappears: 
(2, 3, 4)
 ↑
 select index 0
 ↓
removed

result = (3, 4)

Now:

Python
1
x[0, 1] selects  index 0 along dimension 0 , index 1 along dimension 1

(2, 3, 4)
 ↑  ↑
 0  1
 ↓  ↓
remove both dimensions
result = (4,)


Similarly:
x[0, 1, 2]
selects one position along every dimension:
(2, 3, 4)
↑ ↑ ↑
0 1 2
all dimensions selected
shape = ()
The final result is a 0-D tensor.
```
**My mental rule is:** Integer indexing selects one specific position along a dimension and removes that dimension from the resulting tensor.

# Slicing
``` python
x[:, 1]  - : means that I am selecting all positions along that dimension.
1 means I am selecting one specific position.
For:
x[:, 1]

starting with:

shape = (2, 3, 4)

Show more lines

I can reason:
dim 0    dim 1    dim 2
  2        3        4
  ↓        ↓        ↓
  :        1     untouched
KEEP     REMOVE      KEEP
Therefore:
x[:, 1].shape = (2, 4)
```
# Integer Indexing vs Slicing

Initially, I predicted:

x[:, 1].shape would be: (2, 1, 4) That prediction was incorrect.

The reason is that 1 is an integer index. Integer indexing removes that dimension.
Therefore: x[:, 1] produces: (2, 4)
However: x[:, 1:2] is different. 1:2 is a slice.

A slice preserves the dimension.
Therefore: 
x[:, 1] → (2, 4)
x[:, 1:2] → (2, 1, 4)

``` text
This gives me an important mental rule:
Integer index
↓
select one position
↓
dimension disappears

Slice
↓
select a range
↓
dimension remains
```
# Questions
1. What is a PyTorch tensor?
   A PyTorch tensor is an n-dimensional array used to represent numerical data. It can have zero or more dimensions and has properties such as shape and dtype. PyTorch tensors also integrate with features needed for deep learning, such as automatic differentiation and computation across devices like CPUs and GPUs.
2. What does a tensor shape of (2, 3, 4) mean?
  It means the tensor has three dimensions. Its size along dimension 0 is 2, its size along dimension 1 is 3, and its size along dimension 2 is 4. Therefore, the tensor has 2 × 3 × 4 = 24 elements. The dimensions themselves do not inherently mean things like batch, channel, row, or column. Those semantic meanings depend on what the tensor represents.
3. What is the difference between x[:, 1] and x[:, 1:2]?
  x[:, 1] uses an integer index for dimension 1, so it selects one position and removes that dimension. x[:, 1:2] uses a slice, so the dimension is preserved but its size becomes 1.
``` text
                        Tensor
                           │
          ┌────────────────┼────────────────┐
          │                │                │
         ndim             shape           numel
          │                │                │
          ↓                ↓                ↓
   number of axes    size along each   total number
                         axis           of elements
The dimensions themselves may later acquire semantic meaning:
dimension
    │
    ├── batch
    ├── channel
    ├── height
    ├── width
    ├── sequence
    └── etc.

INTEGER INDEXING

dimension
    ↓
select one position
    ↓
dimension disappears

SLICING

dimension
    ↓
select a range
    ↓
dimension remains
   
