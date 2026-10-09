# Optimizers in PyTorch

## Goal

After learning Autograd and Loss Functions, an important question remains:

> Once gradients are computed, who actually changes the model Parameters?

The answer is:

```text
Optimizer
```

The goals of this section are:

- Understand what an optimizer does.
- Understand the relationship between gradients and parameter updates.
- Understand learning rate.
- Understand SGD.
- Understand Momentum.
- Understand Adam.
- Understand optimizer state.
- Understand zero_grad().
- Understand step().
- Understand how optimizers connect to the training loop.

---

# 1. The Optimization Problem

After a forward pass:

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
```

we can compute gradients:

```python
loss.backward()
```

This gives:

```text
parameter.grad

=

∂Loss
──────
∂Parameter
```

But notice:

```text
gradient exists
```

does not mean:

```text
parameter changed
```

These are different things.

---

# 2. What `backward()` Actually Does

Suppose:

```python
loss.backward()
```

Conceptually:

```text
Parameters
    ↓
Forward Computation
    ↓
Loss
    ↓
Backward Computation
    ↓
parameter.grad populated
```

After:

```python
loss.backward()
```

we have:

```text
parameter.grad
```

but parameter values remain unchanged.

---

# 3. Parameter vs Gradient

Conceptually:

```text
Parameter
│
├── value
└── gradient
```

Example:

```text
parameter      = 10

parameter.grad = 4
```

These are not the same thing.

The gradient describes:

```text
How should the parameter change?
```

It is not the parameter itself.

---

# 4. Who Updates Parameters?

The optimizer.

Conceptually:

```text
loss.backward()
       ↓
compute gradients

optimizer.step()
       ↓
update parameters
```

This is the most important optimizer idea.

---

# 5. Gradient Descent

The simplest update rule is:

```text
new_parameter

=

old_parameter
-
learning_rate × gradient
```

or:

```text
w_new

=

w_old
-
lr × grad
```

---

# 6. Example Update

Suppose:

```text
parameter = 10

gradient = 2

learning_rate = 0.1
```

Update:

```text
10

-

0.1 × 2

=

9.8
```

The parameter moved slightly.

---

# 7. Why Is There A Minus Sign?

The gradient points toward:

```text
increasing loss
```

But training wants:

```text
decreasing loss
```

Therefore we move:

```text
opposite the gradient
```

which gives:

```text
parameter
-
lr × gradient
```

---

# 8. Learning Rate

The learning rate determines:

```text
How large should each update be?
```

Mental model:

```text
Gradient
    ↓
Direction

Learning Rate
    ↓
Step Size
```

A useful summary:

```text
Gradient tells WHERE.

Learning rate tells HOW MUCH.
```

---

# 9. Learning Rate Too Large

Suppose:

```text
gradient = 2

lr = 100
```

Update:

```text
parameter

=

10 - 200

=

-190
```

The optimizer dramatically overshoots.

Possible symptoms:

```text
Loss explodes

Loss oscillates

Training diverges

NaNs appear
```

---

# 10. Learning Rate Too Small

Suppose:

```text
lr = 1e-9
```

The parameter barely changes.

Possible symptoms:

```text
Training appears stuck

Loss decreases very slowly
```

Even though gradients exist.

---

# 11. SGD

PyTorch example:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

Mental model:

```text
SGD
│
├── Parameters
└── Learning Rate
```

Conceptually SGD mostly uses:

```text
current gradient
```

to decide the update.

---

# 12. What `optimizer.step()` Does

Conceptually:

```python
optimizer.step()
```

loops through Parameters:

```text
for each parameter:

parameter

=

parameter
-
lr × parameter.grad
```

The real implementation is more sophisticated, but this is the correct mental
model.

---

# 13. Important Distinction

Never confuse:

```text
loss.backward()
```

with:

```text
optimizer.step()
```

---

## `loss.backward()`

Computes gradients.

```text
parameter.grad populated
```

---

## `optimizer.step()`

Updates Parameters.

```text
parameter values change
```

---

# 14. What Stores Gradients?

Gradients are stored in:

```python
parameter.grad
```

Not in:

```python
optimizer
```

and not in:

```python
loss
```

---

# 15. Example State

After:

```python
loss.backward()
```

Possible state:

```text
parameter      = 10

parameter.grad = 4
```

The Parameter has not changed yet.

---

# 16. What Is `zero_grad()`?

PyTorch accumulates gradients.

Example:

```python
loss1.backward()
```

produces:

```text
grad = 2
```

Another call:

```python
loss2.backward()
```

may add:

```text
grad += 3
```

Result:

```text
grad = 5
```

This behavior is called:

```text
gradient accumulation
```

---

# 17. Why Zero Gradients?

Most training loops want fresh gradients every iteration.

Therefore:

```python
optimizer.zero_grad()
```

is commonly used.

Mental model:

```text
clear previous gradients
```

before computing new ones.

---

# 18. What Changes During `zero_grad()`?

Before:

```text
parameter      = 10

parameter.grad = 4
```

After:

```text
parameter      = 10

parameter.grad = 0
```

Notice:

```text
Parameter unchanged

Gradient cleared
```

---

# 19. What Changes During `backward()`?

Before:

```text
parameter      = 10

parameter.grad = 0
```

After:

```text
parameter      = 10

parameter.grad = 4
```

Notice:

```text
Parameter unchanged

Gradient populated
```

---

# 20. What Changes During `step()`?

Before:

```text
parameter      = 10

parameter.grad = 4
```

After:

```text
parameter      = 9.6

parameter.grad = 4
```

Notice:

```text
Parameter updated

Gradient not automatically cleared
```

---

# 21. Full Gradient Lifecycle

```text
parameter.grad = old value
          ↓
optimizer.zero_grad()
          ↓
parameter.grad = 0
          ↓
loss.backward()
          ↓
parameter.grad = new gradient
          ↓
optimizer.step()
          ↓
parameter updated
```

---

# 22. Why Plain SGD Can Be Limited

SGD only looks at:

```text
current gradient
```

Imagine gradients:

```text
10

9

8

7

6
```

All point roughly in the same direction.

SGD does not explicitly remember that history.

---

# 23. Momentum Intuition

Momentum introduces memory.

Conceptually:

```text
current gradient

+

previous update history
```

Mental model:

```text
Heavy ball rolling downhill
```

Instead of:

```text
step
stop

step
stop
```

Momentum maintains velocity.

---

# 24. Momentum Optimizer State

Now the optimizer stores more information.

Conceptually:

```text
Momentum SGD
│
├── Parameters
├── Learning Rate
└── Velocity
```

Velocity represents historical movement direction.

---

# 25. Why Momentum Helps

If gradients consistently point similarly:

```text
8

7

9

8

7
```

Momentum tends to keep moving in that direction.

Benefits can include:

```text
Faster convergence

Smoother optimization
```

---

# 26. Adam

Adam is one of the most commonly used optimizers.

Conceptually:

```text
Adam
│
├── Parameters
├── Learning Rate
├── First Moment
└── Second Moment
```

---

# 27. First Moment Intuition

The first moment can be thought of as:

```text
moving average
of recent gradients
```

Mental model:

```text
recent direction information
```

---

# 28. Second Moment Intuition

The second moment can be thought of as:

```text
moving average
of squared gradients
```

Mental model:

```text
recent gradient magnitude information
```

---

# 29. Adam Mental Model

A useful intuition:

```text
SGD

mostly uses current gradient
```

```text
Adam

uses gradient history
and gradient magnitude history
```

This is not the formal mathematical definition, but it is a useful first
mental model.

---

# 30. Why Adam Became Popular

Different Parameters may experience very different gradient scales.

Example:

```text
Parameter A

grad = 100
```

```text
Parameter B

grad = 0.001
```

Adam adaptively adjusts updates using its historical statistics.

This often makes optimization easier to tune.

---

# 31. Optimizer State

Optimizers can own state.

Conceptually:

```text
Optimizer
│
├── Hyperparameters
│   ├── lr
│   ├── momentum
│   └── weight_decay
│
└── Internal State
    ├── velocity
    ├── first moment
    └── second moment
```

---

# 32. Optimizer Hyperparameters

Examples:

```text
Learning Rate

Momentum

Weight Decay
```

These control optimization behavior.

---

# 33. Optimizer State Dict

Just as models have:

```python
model.state_dict()
```

optimizers also have:

```python
optimizer.state_dict()
```

because optimizers maintain state.

---

# 34. Why Save Optimizer State?

Consider Adam.

If training resumes later and we only restore:

```python
model.state_dict()
```

we lose:

```text
first moment

second moment
```

history.

Training can continue, but the optimizer effectively begins with fresh
statistics.

---

# 35. Typical Optimizer Creation

SGD:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

Momentum SGD:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

Adam:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

# 36. Complete Training Iteration

```python
optimizer.zero_grad()

prediction = model(x)

loss = loss_fn(
    prediction,
    target
)

loss.backward()

optimizer.step()
```

---

# 37. Line-By-Line Interpretation

```python
optimizer.zero_grad()
```

↓

```text
clear old gradients
```

---

```python
prediction = model(x)
```

↓

```text
forward pass
```

---

```python
loss = loss_fn(
    prediction,
    target
)
```

↓

```text
compute objective
```

---

```python
loss.backward()
```

↓

```text
compute gradients

store in parameter.grad
```

---

```python
optimizer.step()
```

↓

```text
update parameters
```

---

# 38. Common Misconception: `backward()` Updates Parameters

Incorrect:

```text
loss.backward()
      ↓
weights updated
```

Correct:

```text
loss.backward()
      ↓
gradients computed
```

Parameter updates happen later.

---

# 39. Common Misconception: Optimizer Stores Gradients

Incorrect:

```text
optimizer owns gradients
```

Correct:

```text
gradients are stored in

parameter.grad
```

The optimizer reads those gradients.

---

# 40. Common Misconception: `step()` Clears Gradients

Incorrect:

```text
optimizer.step()
      ↓
gradients automatically disappear
```

Correct:

```text
optimizer.step()
      ↓
updates Parameters

optimizer.zero_grad()
      ↓
clears gradients
```

---

# 41. Interview Answer: What Does `optimizer.step()` Do?

`optimizer.step()` reads gradients from each Parameter and updates the
Parameter values according to the optimizer algorithm.

For SGD, the mental model is:

```text
parameter

=

parameter
-
lr × gradient
```

---

# 42. Interview Answer: Why Do We Call `zero_grad()`?

PyTorch accumulates gradients across backward passes.

`zero_grad()` clears gradients from earlier iterations so that the new
backward pass computes fresh gradients.

---

# 43. Interview Answer: Difference Between SGD and Adam

SGD primarily uses the current gradient to perform updates.

Adam maintains additional running statistics of recent gradients and recent
gradient magnitudes, allowing adaptive updates.

---

# 44. Teach It To Someone

If I were teaching optimizers:

> Autograd computes gradients and stores them in `parameter.grad`, but it does
> not update the Parameters themselves.
>
> The optimizer is responsible for reading gradients and modifying Parameter
> values.
>
> `optimizer.zero_grad()` clears old gradients,
> `loss.backward()` computes new gradients,
> and `optimizer.step()` updates Parameters.
>
> More advanced optimizers such as Momentum SGD and Adam maintain additional
> optimization state that influences future updates.

---

# 45. Final Mental Model

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
loss.backward()
  ↓
parameter.grad
  ↓
optimizer.step()
  ↓
updated parameters
```

The responsibilities are:

```text
Autograd
    ↓
computes gradients


Optimizer
    ↓
uses gradients
to update Parameters
```

---

# 46. What I Understand Now

I understand:

```text
Optimizer
│
├── Learning Rate
├── SGD
├── Momentum
├── Adam
│
├── optimizer.step()
├── optimizer.zero_grad()
│
├── Parameter updates
├── Gradient accumulation
│
├── Optimizer state
├── Optimizer hyperparameters
└── Optimizer checkpoints
```

---

# 47. What Comes Next

The final chapter of the Neural Networks section is:

```text
Training Loop
```

which combines:

```text
Module

Autograd

Loss Functions

Optimizers

Parameters

Gradients
```

into one complete end-to-end training process.

By that point every line in a PyTorch training loop should feel completely natural rather than a recipe to memorize.
