# Convolutional Neural Networks (CNNs)

## Goal

Multi-Layer Perceptrons (MLPs) work well for many tabular problems but scale
poorly to images.

Convolutional Neural Networks (CNNs) were specifically designed to exploit the
spatial structure present in images.

The goals of this chapter are:

- Understand why CNNs were invented.
- Understand why MLPs struggle with images.
- Understand convolutions.
- Understand filters and kernels.
- Understand feature maps.
- Understand channels.
- Understand stride and padding.
- Understand pooling.
- Understand CNN shape propagation.
- Understand how CNNs learn hierarchical visual features.

---

# 1. Why MLPs Struggle With Images

Suppose we have an RGB image:

```text
224 × 224 × 3
```

Total input features:

```text
224 × 224 × 3

=

150,528
```

To use an MLP we typically flatten:

```text
(3,224,224)

↓

(150,528,)
```

This creates several problems.

---

# 2. Problem #1: Parameter Explosion

Consider:

```python
nn.Linear(
    150528,
    1000
)
```

Number of weights:

```text
150,528 × 1000

≈ 150 Million
```

just for a single layer.

This is expensive in both memory and computation.

---

# 3. Problem #2: Loss Of Spatial Structure

Images contain spatial relationships.

Neighboring pixels are often highly related.

For example:

```text
Cat Ear

Eye

Wheel

Edge
```

are formed by groups of nearby pixels.

---

After flattening:

```text
(150,528,)
```

the model no longer knows:

```text
Which pixels are neighbors

Which pixels form shapes

Which pixels belong together
```

---

# 4. Problem #3: Translation Sensitivity

Suppose:

```text
Cat On Left
```

and

```text
Cat On Right
```

Humans know:

```text
Still A Cat
```

An MLP sees:

```text
Completely Different Input Vector
```

because all pixel locations changed.

---

# 5. CNN Motivation

CNNs were designed around one core observation:

```text
Images Have Spatial Structure
```

Instead of looking at the entire image simultaneously, CNNs examine:

```text
Small Local Regions
```

and build larger concepts from them.

---

# 6. The CNN Feature Hierarchy

CNNs learn visual features in stages.

```text
Pixels
  ↓
Edges
  ↓
Corners
  ↓
Textures
  ↓
Parts
  ↓
Objects
  ↓
Predictions
```

This hierarchy is one of the key strengths of CNNs.

---

# 7. What Is A Convolution?

A convolution applies a small pattern detector across an image.

Conceptually:

```text
Image
  ↓
Filter
  ↓
Feature Map
```

---

The filter slides over the image and asks:

```text
Did I find my pattern here?
```

---

# 8. Filters And Kernels

A filter (or kernel) is a small learnable pattern detector.

Example:

```text
1  0 -1

1  0 -1

1  0 -1
```

This might respond strongly to:

```text
Vertical Edges
```

---

Think:

```text
Filter

=

Pattern Detector
```

---

# 9. Sliding Window Intuition

A filter is not applied once.

It moves across the image.

Conceptually:

```text
Position 1

Position 2

Position 3

...
```

At each location the filter produces one value.

---

# 10. Feature Maps

The outputs produced across all locations form a:

```text
Feature Map
```

A feature map answers:

```text
Where Did I Find This Pattern?
```

---

Mental model:

```text
Filter
     ↓
Pattern Detector

Feature Map
     ↓
Pattern Locations
```

---

# 11. One Filter, One Feature Map

Relationship:

```text
1 Filter
     ↓
1 Feature Map
```

---

Example:

```text
32 Filters
     ↓
32 Feature Maps
```

---

# 12. Understanding Conv2d

Example:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3
)
```

Interpretation:

```text
Input:
RGB Image

Learn:
32 Filters

Output:
32 Feature Maps
```

---

# 13. Understanding Channels

Channels represent feature maps.

RGB image:

```text
Red

Green

Blue
```

Therefore:

```python
in_channels=3
```

---

Grayscale image:

```text
1 Channel
```

Therefore:

```python
in_channels=1
```

---

# 14. Output Channels

Example:

```python
out_channels=64
```

means:

```text
Learn 64 Filters

Produce 64 Feature Maps
```

---

# 15. CNN Tensor Shapes

CNNs typically use:

```text
(B,C,H,W)
```

where:

```text
B = Batch Size

C = Channels

H = Height

W = Width
```

---

Example:

```text
(32,3,224,224)
```

means:

```text
32 Images

3 Channels

224 Height

224 Width
```

---

# 16. Local Connectivity

MLPs:

```text
Every Output
connected to
every Input
```

---

CNNs:

```text
Each Filter
looks at
a small region
```

This is called:

```text
Local Connectivity
```

---

# 17. Why Local Connectivity Helps

Benefits:

```text
Fewer Parameters

Less Memory

Less Computation

Better Spatial Awareness
```

---

# 18. Stride

Stride determines:

```text
How Far The Filter Moves
```

during convolution.

---

Default:

```text
Stride = 1
```

Move one position at a time.

---

Example:

```python
stride=2
```

moves two positions at a time.

---

# 19. Intuition For Stride

Larger stride:

```text
More Skipping
```

which leads to:

```text
Smaller Feature Maps
```

---

Mental model:

```text
Stride

=

Downsampling During Convolution
```

---

# 20. Padding

Without padding, feature maps shrink.

Example:

```text
Input:

5 × 5

Kernel:

3 × 3
```

Output:

```text
3 × 3
```

---

Padding adds extra border pixels around the image.

---

# 21. Why Padding Exists

Padding helps:

```text
Preserve Spatial Dimensions

Protect Edge Information

Prevent Rapid Shrinking
```

---

# 22. Common CNN Pattern

Very common:

```python
nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1
)
```

---

# 23. Why Padding = 1?

For:

```text
Kernel = 3

Padding = 1

Stride = 1
```

Spatial dimensions remain unchanged.

Example:

```text
224 × 224

↓

224 × 224
```

---

# 24. Output Shape Formula

For a single spatial dimension:

```text
Output

=

(Input + 2P - K)/S + 1
```

where:

```text
P = Padding

K = Kernel Size

S = Stride
```

---

# 25. Example Calculation

Input:

```text
32
```

Kernel:

```text
3
```

Padding:

```text
1
```

Stride:

```text
1
```

Output:

```text
32
```

Spatial size preserved.

---

# 26. Another Example

Input:

```text
224
```

Kernel:

```text
3
```

Padding:

```text
1
```

Stride:

```text
2
```

Output approximately:

```text
112
```

Spatial dimensions are reduced.

---

# 27. Pooling

Pooling performs:

```text
Downsampling
```

without learning filters.

---

Purpose:

```text
Reduce Spatial Size

Reduce Computation

Keep Important Features
```

---

# 28. Max Pooling

Most common pooling operation.

Example:

```text
1 5

3 2
```

MaxPool:

```text
max(1,5,3,2)

=

5
```

Output:

```text
5
```

---

# 29. MaxPool Mental Model

MaxPool asks:

> What is the strongest feature response in this region?

It keeps the strongest activation.

---

# 30. Common Pooling Layer

```python
nn.MaxPool2d(
    kernel_size=2,
    stride=2
)
```

---

# 31. Pooling Shape Example

Input:

```text
32 × 32
```

Output:

```text
16 × 16
```

---

Easy rule:

```text
2×2 Pool

+

Stride 2

↓

Half Height

Half Width
```

---

# 32. Pooling Does Not Change Channels

Example:

```text
(32,64,224,224)

↓

(32,64,112,112)
```

Notice:

```text
64 Channels
```

remain:

```text
64 Channels
```

---

# 33. Pooling vs Stride

Both reduce spatial dimensions.

---

Stride:

```text
Downsampling
During Convolution
```

---

Pooling:

```text
Separate
Downsampling Step
```

---

# 34. The Classic CNN Block

Traditional CNNs often use:

```text
Conv
 ↓
ReLU
 ↓
MaxPool
```

Interpretation:

```text
Learn Features

↓

Add Nonlinearity

↓

Compress Features
```

---

# 35. Example CNN Block

Input:

```text
(32,3,224,224)
```

---

Convolution:

```python
Conv2d(
    3,
    64,
    3,
    padding=1
)
```

Output:

```text
(32,64,224,224)
```

---

ReLU:

```text
(32,64,224,224)
```

---

MaxPool:

```python
MaxPool2d(
    2,
    2
)
```

Output:

```text
(32,64,112,112)
```

---

# 36. Shape Evolution In CNNs

Typical progression:

```text
(32,3,224,224)

↓

(32,64,224,224)

↓

(32,64,112,112)

↓

(32,128,112,112)

↓

(32,128,56,56)

↓

...
```

Feature maps become:

```text
Deeper
```

while spatial dimensions become:

```text
Smaller
```

---

# 37. CNN vs MLP

MLP:

```text
Flatten Image

↓

Lose Spatial Relationships
```

---

CNN:

```text
Preserve Spatial Structure

↓

Learn Local Patterns

↓

Build Feature Hierarchies
```

---

# 38. Common Interview Question

Why were CNNs invented?

Strong answer:

> CNNs were designed to exploit spatial structure in images through local connectivity and parameter sharing, allowing them to learn visual patterns efficiently while requiring far fewer parameters than fully connected networks.

---

# 39. Common Interview Question

What is a filter?

Strong answer:

> A filter is a learned pattern detector that slides across an image and produces a feature map indicating where that pattern appears.

---

# 40. Common Interview Question

What is the difference between a filter and a feature map?

Strong answer:

> A filter is the learned detector, while the feature map is the output produced when that detector is applied across the image.

---

# 41. Common Interview Question

What does pooling do?

Strong answer:

> Pooling reduces spatial dimensions while retaining important feature information, helping reduce computation and memory requirements.

---

# 42. Teach It To Someone

If I were teaching CNNs:

> CNNs process images by applying small learned filters across local regions. These filters detect patterns such as edges and textures, producing feature maps. Deeper layers combine simpler features into more complex concepts, ultimately allowing the network to recognize objects and make predictions.

---

# 43. Master Mental Model

```text
Image

↓

Filters

↓

Feature Maps

↓

ReLU

↓

Pooling

↓

Higher-Level Features

↓

Prediction
```

---

# 44. The Feature Hierarchy

```text
Pixels

↓

Edges

↓

Corners

↓

Textures

↓

Parts

↓

Objects

↓

Prediction
```

---

# 45. What I Understand Now

I understand:

```text
CNNs
│
├── Convolution
├── Filters
├── Kernels
├── Feature Maps
├── Channels
├── Local Connectivity
├── Stride
├── Padding
├── Pooling
├── Shape Propagation
├── CNN Blocks
└── Feature Hierarchies
```

---

# 46. What Comes Next

The next chapter introduces:

```text
Model Shape Reasoning
```

where we learn how to trace tensor dimensions through:

```text
MLPs

CNNs

Pooling Layers

Classification Heads
```

and build the debugging mindset used by experienced PyTorch engineers.
