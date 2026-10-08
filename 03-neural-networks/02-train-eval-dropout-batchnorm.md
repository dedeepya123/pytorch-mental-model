# Training Mode, Evaluation Mode, Dropout, and BatchNorm

## Goal

In the previous section, I learned that an `nn.Module` packages computation
and state.

My current Module mental model is:

```text
                         nn.Module
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
         Computation                       State
              │                             │
              ▼                ┌────────────┴────────────┐
          forward()            │                         │
                               ▼                         ▼
                          Parameters                   Buffers
```

I also learned that Modules can contain other Modules:

```text
Model
│
├── Submodule
│   ├── Parameter
│   └── Parameter
│
└── Submodule
    ├── Parameter
    └── Parameter
```

Now I want to understand another property of a Module:

> Is the Module currently being used in training mode or evaluation mode?

The goals of this section are to understand:

- What `model.train()` means.
- What `model.eval()` means.
- What these functions do NOT do.
- Why some Modules care about train/eval mode.
- Why `nn.Linear` normally behaves the same in both modes.
- How Dropout behaves differently during training and evaluation.
- Why Dropout scales activations during training.
- How BatchNorm behaves differently during training and evaluation.
- How BatchNorm connects Parameters and Buffers.
- Why `model.eval()` and `torch.no_grad()` are completely different concepts.
- Why inference often uses both.

---

# 1. Why Do Modules Need Training and Evaluation Modes?

Not every layer behaves identically during training and inference.

Imagine:

```text
Training

input
  ↓
model
  ↓
prediction
  ↓
loss
  ↓
backward
  ↓
parameter updates
```

versus:

```text
Inference

input
  ↓
model
  ↓
prediction
```

Some Modules need one behavior while the model is training and another
behavior while it is being evaluated or used for inference.

PyTorch therefore allows a Module hierarchy to have a:

```text
training mode
```

or:

```text
evaluation mode
```

---

# 2. `model.train()`

When I call:

```python
model.train()
```

my mental model is:

> Put the Module and its child Modules into training mode.

Conceptually:

```text
MyModel
│
├── layer1
├── layer2
└── layer3

        │
        │ model.train()
        ▼

MyModel       training=True
│
├── layer1    training=True
├── layer2    training=True
└── layer3    training=True
```

This tells mode-sensitive Modules to use their training behavior.

---

# 3. `model.eval()`

When I call:

```python
model.eval()
```

my mental model is:

> Put the Module and its child Modules into evaluation mode.

Conceptually:

```text
MyModel
│
├── layer1
├── layer2
└── layer3

        │
        │ model.eval()
        ▼

MyModel       training=False
│
├── layer1    training=False
├── layer2    training=False
└── layer3    training=False
```

This tells mode-sensitive Modules to use their evaluation behavior.

---

# 4. `train()` Does Not Perform Training

An important misconception to avoid is:

```python
model.train()
```

does NOT mean:

```text
run forward pass

calculate loss

perform backward

update parameters
```

It simply changes the Module's mode.

My mental model is:

```text
model.train()
      ↓
change Module behavior mode
```

It does not itself:

```text
compute gradients

calculate loss

update parameters
```

---

# 5. `eval()` Does Not Perform Evaluation

Likewise:

```python
model.eval()
```

does NOT:

```text
run validation dataset

calculate evaluation metrics

disable Autograd

freeze Parameters
```

It changes the Module hierarchy into evaluation mode.

So:

```text
model.eval()
      ↓
change Module behavior
```

not:

```text
perform evaluation process
```

---

# 6. Not Every Module Behaves Differently

Consider:

```python
nn.Linear(3, 2)
```

Its computation is conceptually:

```text
y = xW^T + b
```

That mathematical transformation does not need to change simply because the
model is in training or evaluation mode.

Therefore:

```text
Linear

train mode
    ↓
same core affine transformation

eval mode
    ↓
same core affine transformation
```

Some Modules, however, explicitly behave differently depending on mode.

Two important examples are:

```text
Dropout

BatchNorm
```

---

# 7. Dropout

Dropout is an example of a mode-sensitive Module.

Suppose an intermediate activation is:

```text
[2, 4, 6, 8]
```

During training, Dropout randomly suppresses some activations.

Conceptually:

```text
Before Dropout:

[2, 4, 6, 8]

       ↓

Dropout

       ↓

some activations become zero
```

The exact positions dropped are random.

This means successive forward calls during training can produce different
Dropout masks.

---

# 8. Dropout in Training Mode

Suppose:

```python
dropout = nn.Dropout(p=0.5)

dropout.train()
```

Here:

```text
p = 0.5
```

means each activation has:

```text
50% probability of being dropped
```

During training:

```text
activation
    │
    ├── dropped
    │     ↓
    │     0
    │
    └── kept
          ↓
        scaled
```

Therefore two calls:

```python
y1 = dropout(x)
y2 = dropout(x)
```

are not guaranteed to produce identical outputs.

Conceptually:

```text
             x
             │
       ┌─────┴─────┐
       │           │
       ▼           ▼

      call 1      call 2
       │           │
       ▼           ▼

 random mask    random mask
       │           │
       ▼           ▼

      y1          y2
```

So:

```text
y1 == y2
```

is not something I should generally expect during Dropout training behavior.

---

# 9. Dropout in Evaluation Mode

Now:

```python
dropout.eval()
```

Dropout no longer randomly suppresses activations.

Conceptually:

```text
input
  │
  │ Dropout in eval mode
  ▼
output

output = input
```

So Dropout behaves like an identity operation during evaluation.

For the same input:

```python
y1 = dropout(x)
y2 = dropout(x)
```

I expect:

```text
y1 == y2
```

and conceptually:

```text
y1 = x
y2 = x
```

---

# 10. Why Dropout Scaling Is Needed

Suppose:

```python
dropout = nn.Dropout(p=0.5)
```

and consider one activation:

```text
10
```

If Dropout simply behaved like:

```text
50% probability → 0

50% probability → 10
```

then its expected output would be:

```text
0.5 × 0
+
0.5 × 10

=
5
```

But the original activation was:

```text
10
```

Therefore the expected activation would be cut in half during training.

That would create a difference in activation scale between training and
evaluation.

---

# 11. Inverted Dropout Scaling

To compensate, surviving activations during training are scaled by:

```text
1 / (1 - p)
```

For:

```text
p = 0.5
```

this becomes:

```text
1 / 0.5

=

2
```

Therefore:

```text
Dropped:

10 → 0
```

and:

```text
Kept:

10 → 20
```

Expected value:

```text
0.5 × 0
+
0.5 × 20

=

10
```

which matches the original activation.

---

# 12. General Dropout Rule

For:

```python
nn.Dropout(p=p)
```

during training:

```text
DROP

probability = p

output = 0
```

or:

```text
KEEP

probability = 1-p

output = input / (1-p)
```

Therefore the expected activation remains approximately unchanged.

During evaluation:

```text
output = input
```

No additional Dropout scaling is required during evaluation.

---

# 13. Example With `p=0.25`

Suppose:

```text
input = 8

p = 0.25
```

Then:

```text
drop probability
=
0.25
```

and:

```text
keep probability
=
1 - 0.25
=
0.75
```

If dropped:

```text
output = 0
```

If kept:

```text
output

=
8 / 0.75

≈
10.667
```

Expected output:

```text
0.25 × 0
+
0.75 × 10.667

≈
8
```

Therefore:

```text
expected output ≈ original input
```

---

# 14. An Important Dropout Mistake

I initially thought:

> If Dropout probability is 0.25 and the value is dropped, maybe the output
> becomes `value × 0.25`.

That is incorrect.

Dropping means:

```text
output = 0
```

The probability controls whether the value survives.

It does not multiply a dropped activation.

My corrected mental model is:

```text
drop
 ↓
0


keep
 ↓
input / (1-p)
```

---

# 15. `eval()` Does Not Disable Gradients

Suppose:

```python
x = torch.ones(
    10,
    requires_grad=True
)

dropout.eval()

y = dropout(x)

loss = y.sum()

loss.backward()
```

This is valid.

Why?

Because:

```text
dropout.eval()
```

controls:

```text
MODULE MODE
```

while:

```text
requires_grad
```

and:

```text
torch.no_grad()
```

control:

```text
AUTOGRAD BEHAVIOR
```

Therefore:

```text
evaluation mode
```

does NOT mean:

```text
Autograd disabled
```

---

# 16. Module Mode vs Autograd Mode

This is one of the most important distinctions in this section:

```text
                   PyTorch Execution
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     ▼
          Module Mode            Autograd Mode
               │                     │
               ▼                     ▼
         train()/eval()       grad enabled/no_grad()
               │                     │
               ▼                     ▼
     How should mode-sensitive    Should PyTorch
     Modules behave?             record operations
                                 for backward?
```

These are independent systems.

---

# 17. Why Inference Often Uses Both

Inference code commonly looks like:

```python
model.eval()

with torch.no_grad():
    output = model(x)
```

These lines solve different problems.

```text
model.eval()
      │
      ▼
mode-sensitive Modules
use evaluation behavior
```

while:

```text
torch.no_grad()
      │
      ▼
do not construct
Autograd graph
```

Therefore:

```text
                 Inference
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
   Module behavior         Autograd behavior
         │                       │
         ▼                       ▼
   model.eval()            torch.no_grad()
```

---

# 18. BatchNorm

BatchNorm is another mode-sensitive Module.

However, the reason is completely different from Dropout.

Dropout changes:

```text
random dropping behavior
```

BatchNorm changes:

```text
which normalization statistics are used
```

---

# 19. BatchNorm Intuition

Suppose one feature has values:

```text
[2, 4, 6, 8]
```

The mean is:

```text
5
```

BatchNorm conceptually performs a normalization similar to:

```text
x_hat

=

(x - mean)
──────────
sqrt(variance + epsilon)
```

The important question is not initially the exact equation.

The important question is:

> Where do mean and variance come from?

---

# 20. Batch Statistics During Training

Imagine a batch:

```text
                 feature 0   feature 1   feature 2

sample 0             1          10          100

sample 1             2          20          200

sample 2             3          30          300

sample 3             4          40          400
```

BatchNorm computes statistics separately for each feature.

Conceptually:

```text
feature 0
    ↓
mean + variance


feature 1
    ↓
mean + variance


feature 2
    ↓
mean + variance
```

During training, BatchNorm uses statistics from the current training batch
for normalization.

---

# 21. Why Batch Statistics Are Problematic During Inference

Suppose inference receives only one sample.

If BatchNorm always depended on the current batch statistics, the prediction
could depend heavily on:

```text
batch size

or

which other samples happen
to be in the inference batch
```

This is undesirable.

Therefore BatchNorm maintains running estimates of its feature statistics
during training.

---

# 22. Running Statistics

BatchNorm maintains state conceptually like:

```text
running_mean

running_var
```

During training:

```text
current training batch
       │
       ├── calculate batch mean
       │
       └── calculate batch variance
                │
                ▼
        normalize current batch
                │
                +
                ▼
        update running statistics
```

The running statistics provide estimates that can later be used during
evaluation.

---

# 23. Running Mean Is Per Feature

One mistake I initially made was thinking:

```text
running_mean
```

might be a single scalar for the entire BatchNorm layer.

That is incorrect.

For:

```python
nn.BatchNorm1d(3)
```

there are three features.

So BatchNorm needs:

```text
one running mean per feature

and

one running variance per feature
```

Therefore:

```text
running_mean.shape = (3,)

running_var.shape = (3,)
```

Conceptually:

```text
running_mean

[
    mean_feature_0,
    mean_feature_1,
    mean_feature_2
]
```

---

# 24. BatchNorm in Training Mode

Conceptually:

```python
bn.train()
```

makes BatchNorm operate like:

```text
Current training batch
          │
          ├── calculate batch statistics
          │
          ▼
   normalize using
   current statistics
          │
          +
          ▼
   update running statistics
```

Therefore my mental model is:

```text
TRAIN MODE

use current batch statistics

AND

update running statistics
```

---

# 25. BatchNorm in Evaluation Mode

In evaluation mode:

```python
bn.eval()
```

BatchNorm uses stored running statistics.

Conceptually:

```text
input
 │
 ▼
running_mean
+
running_var
 │
 ▼
normalize
 │
 ▼
output
```

So my mental model is:

```text
EVAL MODE

use stored running statistics
```

instead of deriving normalization statistics from the current inference
batch.

---

# 26. Running Statistics Are Estimates

I should avoid saying:

> running_mean is simply the exact mean of all batches seen so far.

A safer mental model is:

> During training, BatchNorm updates running estimates of per-feature
> statistics.

These stored estimates are then used during evaluation.

---

# 27. BatchNorm Also Has Learnable Parameters

BatchNorm does more than normalization.

Conceptually, after normalization:

```text
x
│
▼
normalize
│
▼
x_hat
│
├── scale
│
└── shift
▼
output
```

The final transformation can be thought of as:

```text
output

=

gamma × x_hat + beta
```

where:

```text
gamma
```

is learnable scaling and:

```text
beta
```

is learnable shifting.

---

# 28. BatchNorm Parameters vs Buffers

For:

```python
bn = nn.BatchNorm1d(3)
```

conceptually:

```text
                    BatchNorm1d(3)
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
          Parameters                 Buffers
              │                       │
        ┌─────┴─────┐           ┌─────┴────────┐
        │           │           │              │
        ▼           ▼           ▼              ▼
      weight       bias     running_mean   running_var
      gamma         beta
        │           │           │              │
       (3,)        (3,)        (3,)           (3,)
```

---

# 29. Why Weight and Bias Are Parameters

BatchNorm's:

```text
weight / gamma

bias / beta
```

are learned through optimization.

Therefore:

```text
weight → Parameter

bias → Parameter
```

They can receive gradients and be updated by the optimizer.

For three features:

```text
weight.shape = (3,)

bias.shape = (3,)
```

because each feature gets its own scale and shift.

---

# 30. Why Running Statistics Are Buffers

The model needs to remember:

```text
running_mean

running_var
```

but they are not model Parameters learned through gradient descent.

Therefore:

```text
running_mean → Buffer

running_var → Buffer
```

This is an excellent example of the distinction:

```text
Model State
│
├── Parameters
│   ├── weight
│   └── bias
│
└── Buffers
    ├── running_mean
    └── running_var
```

---

# 31. BatchNorm Connects Several Previous Concepts

BatchNorm gives a real example of why `nn.Module` distinguishes Parameters
and Buffers.

```text
                     BatchNorm
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
      Learnable state          Non-Parameter state
            │                         │
            ▼                         ▼
       Parameters                  Buffers
            │                         │
       gamma, beta          running_mean, running_var
```

It also explains why:

```text
train()
vs
eval()
```

matters.

---

# 32. BatchNorm in Eval Mode Can Still Have Gradients

Suppose:
